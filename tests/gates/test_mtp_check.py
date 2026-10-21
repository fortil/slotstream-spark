"""Step 76 gate (mtp-check): self-speculative decode with the resident MTP head in D5 equals plain decode bit for bit.

Run through `tools/spark/run.sh --heavy f6-mtp-check auto 2700 -- tools/spark/py.sh -m pytest -q -m gate
tests/gates/test_mtp_check.py` after `tools/spark/run.sh -- bash tools/build_slotio.sh`. MTP is forced resident by the
environment (SLOTSTREAM_MTP=resident, read through engine.mtp.policy, the same path the engine takes).

8 train-split prompts (two per message kind) x 64 greedy tokens, each decoded plain (1-row graphs) and then with MTP under three
draft schedules:
  a  the head's own drafts: output ids == plain ids for all 8 prompts; the acceptance per depth is printed
  b  every draft wrong: output ids == plain ids (every round rolls back to keep = 1)
  c  half the drafts wrong (first right and second wrong on odd rounds, the other way round on even rounds): output ids == plain
  d  after every request: 0 record rounds outstanding, 0 lookahead tickets held, and the final model state (position, token
     list) equals the plain run's
Numbers are printed and written to $CAP/mtp_check.json.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.gate, pytest.mark.cuda]

SLOTS = 48 * 48
NEW_TOKENS = 64
PER_KIND = 2
MAX_PROMPT = 800
KINDS = ("reasoning", "multilingual", "structured", "dialogue")
ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "third_party" / "slotstream" / "Tools" / "fixtures" / "expert-lookahead" / "corpus.json"
RES = {"prompts": []}


def say(request, text):
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
    from slotstream_spark.engine import mtp as mtp_mod
    from slotstream_spark.engine.tokenizer import Tokenizer
    from slotstream_spark.model.config import TextConfig
    from slotstream_spark.model.forward import Model
    from slotstream_spark.model.weights import Embedder, NgramSource, accessor, load_trunk
    from slotstream_spark.runtime.graphs import GraphSet
    from slotstream_spark.runtime.pool import ExpertPool

    os.environ["SLOTSTREAM_MTP"] = "resident"                           # forced by the environment
    assert mtp_mod.policy("D5") == "resident"
    cfg = TextConfig.from_checkpoint(model_dir)
    weights = load_trunk(packed_dir / "trunk.safetensors", device="cuda")
    reader = slotio.Reader()
    embed = Embedder(reader, packed_dir / "trunk.safetensors", device="cuda")
    ngram = NgramSource(reader, packed_dir, weights, cfg["eos_token_id"], device="cuda")
    pool = ExpertPool(packed_dir, SLOTS, reader=reader)
    model = Model(cfg, accessor(weights), embed, pool, ngram, device="cuda", gdn="fused", max_ctx=1024)
    model.st.ensure_act(torch.bfloat16)
    head = mtp_mod.MTPHead(cfg, packed_dir / "mtp_trunk.safetensors", packed_dir / "mtp_experts.bin", embed=embed,
                           lm_head=weights["lm_head.weight"], device="cuda", max_ctx=1024)
    g1 = GraphSet(model, 1)
    g3 = GraphSet(model, 3, readback=g1.rb, pool_handle=g1.pool_handle, record=True, multi=True)
    RES["record_ring_bytes"] = g3.record.nbytes
    e = {"model": model, "pool": pool, "head": head, "g1": g1, "g3": g3, "cfg": cfg, "tok": Tokenizer.from_checkpoint(model_dir)}
    yield e
    out = os.environ.get("CAP")
    if out:
        os.makedirs(out, exist_ok=True)
        Path(out, "mtp_check.json").write_text(json.dumps(RES, indent=1, sort_keys=True))
    os.environ.pop("SLOTSTREAM_MTP", None)
    del g3, g1, head, model, pool, ngram, embed, weights, reader
    gc.collect()
    torch.cuda.empty_cache()


def run(e, ids, mtp=False, draft=None):
    from slotstream_spark.engine.generate import generate
    e["g1"].detach()
    e["model"].st.reset_inplace()
    r = generate(e["model"], ids, max_tokens=NEW_TOKENS, greedy=True, graphs=e["g1"],
                 mtp=e["head"] if mtp else None, graphs3=e["g3"] if mtp else None, mtp_draft=draft)
    r["end_pos"], r["end_tokens"] = e["model"].st.pos, list(e["model"].st.tokens)
    return r


def schedule(kind, ref, V):
    """Draft source from the plain output `ref` (the tokens that will be sampled after the pending one); V = vocabulary."""

    def fn(pending, k, n_out):
        want = ref[n_out:n_out + k]
        want = want + [0] * (k - len(want))
        if kind == "all-wrong":
            return [(w + 1) % V for w in want]
        if kind == "half-wrong":
            return [want[0], (want[1] + 1) % V] if (n_out // 3) % 2 else [(want[0] + 1) % V, want[1]]
        raise ValueError(kind)
    return fn


@pytest.fixture(scope="module")
def runs(engine, request):
    out = []
    for p in prompts(engine["tok"]):
        plain = run(engine, p["ids"])
        row = {"prompt": p, "plain": plain}
        row["head"] = run(engine, p["ids"], mtp=True)
        row["all-wrong"] = run(engine, p["ids"], mtp=True, draft=schedule("all-wrong", plain["tokens"], engine["cfg"]["vocab_size"]))
        row["half-wrong"] = run(engine, p["ids"], mtp=True, draft=schedule("half-wrong", plain["tokens"], engine["cfg"]["vocab_size"]))
        out.append(row)
        h = row["head"]["mtp"]
        say(request, f"{p['id']} {p['kind']:<12} prompt {len(p['ids']):>4}  rounds {h['rounds']:>3}  accepted "
                     f"{h['accepted_by_depth']} of {h['attempts_by_depth']}  emitted/round "
                     f"{sum(h['emitted_by_round']) / max(1, h['rounds']):.2f}")
    return out


def test_a_head_drafts_equal_plain(runs, request):
    assert len(runs) == len(KINDS) * PER_KIND, f"{len(runs)} prompts selected"
    for r in runs:
        assert len(r["plain"]["tokens"]) == NEW_TOKENS and not r["plain"]["nan"]
        assert r["head"]["tokens"] == r["plain"]["tokens"], f"{r['prompt']['id']}: MTP output differs from plain"
    att = [sum(r["head"]["mtp"]["attempts_by_depth"][d] for r in runs) for d in (0, 1)]
    acc = [sum(r["head"]["mtp"]["accepted_by_depth"][d] for r in runs) for d in (0, 1)]
    rounds = sum(r["head"]["mtp"]["rounds"] for r in runs)
    emitted = sum(sum(r["head"]["mtp"]["emitted_by_round"]) for r in runs)
    RES["acceptance_by_depth"] = {f"depth{d + 1}": {"accepted": acc[d], "attempts": att[d],
                                                    "rate": acc[d] / att[d] if att[d] else None} for d in (0, 1)}
    RES["rounds"], RES["emitted_per_round"] = rounds, emitted / max(1, rounds)
    RES["prompts"] = [{"id": r["prompt"]["id"], "kind": r["prompt"]["kind"], "mtp": r["head"]["mtp"]} for r in runs]
    say(request, "acceptance per depth: " + ", ".join(
        f"depth {d + 1}: {acc[d]}/{att[d]} = {acc[d] / max(1, att[d]):.3f}" for d in (0, 1)) +
        f"; {emitted / max(1, rounds):.2f} tokens per verify round over {rounds} rounds")
    assert rounds > 0 and att[0] > 0


def test_b_all_wrong_drafts_equal_plain(runs, request):
    for r in runs:
        m = r["all-wrong"]["mtp"]
        assert r["all-wrong"]["tokens"] == r["plain"]["tokens"], f"{r['prompt']['id']}: all-wrong drafts changed the output"
        assert m["accepted"] == 0 and m["rounds"] > 0
    say(request, f"all drafts wrong: ids identical over {len(runs)} prompts x {NEW_TOKENS} tokens, 0 accepted")


def test_c_half_wrong_drafts_equal_plain(runs, request):
    for r in runs:
        m = r["half-wrong"]["mtp"]
        assert r["half-wrong"]["tokens"] == r["plain"]["tokens"], f"{r['prompt']['id']}: half-wrong drafts changed the output"
        assert m["accepted"] > 0 and m["accepted"] < m["drafted"]
    say(request, f"half the drafts wrong: ids identical over {len(runs)} prompts x {NEW_TOKENS} tokens")


def test_d_no_leaks_and_same_final_state(runs, engine, request):
    for r in runs:
        for k in ("head", "all-wrong", "half-wrong"):
            x = r[k]
            assert x["mtp"]["leaked_records"] == 0, f"{r['prompt']['id']} {k}: {x['mtp']['leaked_records']} record rounds leaked"
            assert x["end_pos"] == r["plain"]["end_pos"] and x["end_tokens"] == r["plain"]["end_tokens"], \
                f"{r['prompt']['id']} {k}: final position/token list differ from plain"
    assert engine["g3"].record.outstanding == 0
    assert engine["g3"].lookahead is None
    RES["leaked_records"] = 0
    say(request, "0 leaked record buffers; final position and token list equal plain in all runs")
