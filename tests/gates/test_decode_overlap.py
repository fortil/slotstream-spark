"""Step 74 gate: speculative prefetch in the D5 decode loop (1-row graphs, forecast tap, tickets in slotio).

Run through `tools/spark/run.sh --heavy f6-overlap auto 2400 -- tools/spark/py.sh -m pytest -q -m gate
tests/gates/test_decode_overlap.py` after `tools/spark/run.sh -- bash tools/build_slotio.sh`.

8 train-split prompts (two per message kind) x 128 greedy tokens, each prompt run twice from the same cold pool and the same
prefill: lookahead off (graphs without the tap) and lookahead on (`attention-corrected` when the sidecar is present, else
`attention`):
  a  the decode ids of the two arms are bit-identical
  b  the on arm has fewer demand records per token than the off arm (summed over the 8 prompts) and adopted tickets > 0
  c  0 leaked reservations at the end of every request (outstanding tickets 0, issued == adopted + discarded + failed,
     tickets still outstanding when the request ended 0)
Precision (adopted / issued), lateness (adopted that had to wait for the read / adopted) and the per-prompt numbers are printed
and written to $CAP/decode_overlap.json.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.gate, pytest.mark.cuda]

SLOTS = 48 * 48
NEW_TOKENS = 128
PER_KIND = 2
MAX_PROMPT = 800
KINDS = ("reasoning", "multilingual", "structured", "dialogue")
ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "third_party" / "slotstream" / "Tools" / "fixtures" / "expert-lookahead" / "corpus.json"
RES = {"prompts": []}


def say(request, text):
    """A line on the real terminal (pytest -q hides the output of tests that pass)."""
    tr = request.config.pluginmanager.get_plugin("terminalreporter")
    if tr is not None:
        tr.write_line(text)
    else:
        print(text, flush=True)


def prompts(tok):
    corpus = json.loads(CORPUS.read_text())
    out = []
    for kind in KINDS:
        n = 0
        for r in corpus["requests"]:
            if r["split"] != "train" or r["kind"] != kind or "messages" not in r:
                continue
            ids = tok.encode(tok.render_chat(r["messages"], enable_thinking=False))
            if len(ids) > MAX_PROMPT:
                continue
            out.append({"id": r["id"], "kind": kind, "ids": ids})
            n += 1
            if n == PER_KIND:
                break
    return out


@pytest.fixture(scope="module")
def engine(model_dir, packed_dir):
    import gc
    import slotio
    import torch
    from slotstream_spark.engine import lookahead as la
    from slotstream_spark.engine.tokenizer import Tokenizer
    from slotstream_spark.kernels import forecast as fc
    from slotstream_spark.model.config import TextConfig
    from slotstream_spark.model.forward import Model
    from slotstream_spark.model.weights import Embedder, NgramSource, accessor, load_trunk
    from slotstream_spark.runtime.graphs import GraphSet
    from slotstream_spark.runtime.pool import ExpertPool
    from slotstream_spark.runtime.readback import Readback

    cfg = TextConfig.from_checkpoint(model_dir)
    root = os.environ.get("SLOTSTREAM_R")
    st = la.tap_status(Path(root) / "model" if root else Path(model_dir).parent)
    weights = load_trunk(packed_dir / "trunk.safetensors", device="cuda")
    reader = slotio.Reader()
    embed = Embedder(reader, packed_dir / "trunk.safetensors", device="cuda")
    ngram = NgramSource(reader, packed_dir, weights, cfg["eos_token_id"], device="cuda")
    pool = ExpertPool(packed_dir, SLOTS, reader=reader)
    model = Model(cfg, accessor(weights), embed, pool, ngram, device="cuda", gdn="fused", max_ctx=1024)
    model.st.ensure_act(torch.bfloat16)
    corr = None
    if st["present"]:
        corr = fc.TapCorrection.load(Path(model_dir) / fc.SHIPPED_REL, hidden=cfg["hidden_size"],
                                     experts=cfg["num_experts"], targets=47)
    mode = "attention-corrected" if corr is not None else "attention"
    lcfg = la.LookaheadConfig(mode=mode)
    rb = Readback(model.vocab, forecast=True)
    g_off = GraphSet(model, 1, readback=rb)
    g_on = GraphSet(model, 1, readback=rb, pool_handle=g_off.pool_handle, forecast=la.ForecastTap(corr))
    pool.enable_prefetch(lcfg.slot_cap, lcfg.min_evictable)
    e = {"model": model, "pool": pool, "g_off": g_off, "g_on": g_on, "lcfg": lcfg, "la": la,
         "tok": Tokenizer.from_checkpoint(model_dir)}
    RES["mode"] = mode
    RES["tap_correction_status"] = st["line"]
    yield e
    out = os.environ.get("CAP")
    if out:
        os.makedirs(out, exist_ok=True)
        Path(out, "decode_overlap.json").write_text(json.dumps(RES, indent=1, sort_keys=True))
    del g_on, g_off, model, pool, ngram, embed, weights, reader
    gc.collect()
    torch.cuda.empty_cache()


def run_arm(e, g, ids, lookahead):
    """Cold pool, eager prefill, NEW_TOKENS greedy graph steps -> dict(tokens, demand_records, prefetch stats, leaked)."""
    pool, model = e["pool"], e["model"]
    e["g_on"].detach()
    e["g_off"].detach()
    e["g_on"].lookahead = None
    pool.resize_slots(0)
    pool.resize_slots(SLOTS)
    model.st.reset_inplace()
    logits = model.prefill(ids).cpu().numpy()
    base = pool.prefetch_stats()
    g.lookahead = e["la"].Scheduler(pool, e["lcfg"]) if lookahead else None
    toks, timings = [], []
    for _ in range(NEW_TOKENS):
        t = int(np.argmax(logits))
        toks.append(t)
        logits = g.step([t], timings=timings)[0].numpy()
    sched = g.lookahead
    leaked_end = sched.end_request() if sched is not None else 0
    after = pool.prefetch_stats()
    g.lookahead = None
    d = {k: after[k] - base[k] for k in after if k != "outstanding"}
    return {"tokens": toks, "demand_records": sum(int(t[4]) for t in timings), "prefetch": d,
            "outstanding": after["outstanding"], "leaked_at_end": leaked_end}


@pytest.fixture(scope="module")
def runs(engine, request):
    out = []
    for p in prompts(engine["tok"]):
        off = run_arm(engine, engine["g_off"], p["ids"], False)
        on = run_arm(engine, engine["g_on"], p["ids"], True)
        out.append({"prompt": p, "off": off, "on": on})
        say(request, f"{p['id']} {p['kind']:<12} prompt {len(p['ids']):>4}  demand/token off "
              f"{off['demand_records'] / NEW_TOKENS:7.2f} on {on['demand_records'] / NEW_TOKENS:7.2f}  "
              f"issued {on['prefetch']['issued']} adopted {on['prefetch']['adopted']} "
              f"(late {on['prefetch']['adopted_late']}) discarded {on['prefetch']['discarded']} "
              f"failed {on['prefetch']['failed']}")
    return out


def test_a_ids_bit_identical_with_lookahead_on_and_off(runs, request):
    assert len(runs) == len(KINDS) * PER_KIND, f"{len(runs)} prompts selected"
    for r in runs:
        assert r["on"]["tokens"] == r["off"]["tokens"], f"{r['prompt']['id']}: decode ids differ with lookahead on"
    say(request, f"decode ids identical on/off over {len(runs)} prompts x {NEW_TOKENS} tokens")


def test_b_fewer_demand_records_and_adopted(runs, engine, request):
    off = sum(r["off"]["demand_records"] for r in runs)
    on = sum(r["on"]["demand_records"] for r in runs)
    n = len(runs) * NEW_TOKENS
    issued = sum(r["on"]["prefetch"]["issued"] for r in runs)
    adopted = sum(r["on"]["prefetch"]["adopted"] for r in runs)
    late = sum(r["on"]["prefetch"]["adopted_late"] for r in runs)
    RES["demand_per_token_off"], RES["demand_per_token_on"] = off / n, on / n
    RES["issued"], RES["adopted"], RES["adopted_late"] = issued, adopted, late
    RES["precision"] = adopted / issued if issued else 0.0
    RES["lateness"] = late / adopted if adopted else 0.0
    RES["prompts"] = [{"id": r["prompt"]["id"], "kind": r["prompt"]["kind"], "off": r["off"]["demand_records"],
                       "on": r["on"]["demand_records"], "prefetch": r["on"]["prefetch"]} for r in runs]
    say(request, f"mode {RES['mode']}: demand records per token off {off / n:.2f} on {on / n:.2f}  adopted {adopted} of {issued} issued "
                 f"(precision {RES['precision']:.3f}, lateness {RES['lateness']:.3f})")
    assert adopted > 0, "no speculative ticket was adopted"
    assert on < off, f"demand records per token did not drop: off {off / n:.3f} on {on / n:.3f}"


def test_c_no_leaked_reservations(runs, request):
    for r in runs:
        p = r["on"]["prefetch"]
        assert r["on"]["outstanding"] == 0, f"{r['prompt']['id']}: {r['on']['outstanding']} tickets outstanding"
        assert r["on"]["leaked_at_end"] == 0, f"{r['prompt']['id']}: {r['on']['leaked_at_end']} tickets still held at request end"
        assert p["issued"] == p["adopted"] + p["discarded"] + p["failed"], f"{r['prompt']['id']}: {p}"
        assert p["failed"] == 0, f"{r['prompt']['id']}: speculative reads failed: {p}"
    say(request, "0 leaked reservations at the end of all requests")
