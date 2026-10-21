"""Step 73 gate: the forecast tap in the decode graph (D5 engine, 1-row graphs).

Run through `tools/spark/run.sh --heavy f6-forecast auto 1800 -- tools/spark/py.sh -m pytest -q -m gate
tests/gates/test_forecast.py` after `tools/spark/run.sh -- bash tools/fetch_tap_correction.sh`.

4 train-split prompts of the lookahead corpus (messages kinds: reasoning, multilingual, structured, dialogue), rendered with the
checkpoint's chat template, 128 greedy tokens each, every layer l = 0..46 forecasting layer l + 1:
  a  the status file of the optional sidecar and, when the file is present, its header against the model geometry
  b  decode ids are bit-identical with the forecast tap on and off (two graph sets on one readback); the tap's top-24 written by
     the graph equals the one written by the eager run bit for bit; recall of the true routing by the forecast top-10, per layer,
     for the plain tap and, when the correction is present, for the corrected tap (printed, never a pass criterion by itself)
  c  with the correction present: the corrected top-10 set of every (token, layer) pair equals the float64 numpy reference's set
     except for ids the reference cannot rank (margin = [ops] forecast_tie_rel x max|corrected - raw| per row, around the
     reference's 10th logit); the count of excused ids and the fraction of pairs with any difference are printed
  d  negative control: the same check against a reference built with `b` transposed fails (a non-tie difference over the
     4 x 128 x 47 pairs)
Without the sidecar the correction cases skip with reason `tap_correction absent`, and the record copies the status line.
"""
import json
import os
import tomllib
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.gate, pytest.mark.cuda]

SLOTS = 48 * 48
NEW_TOKENS = 128
PROMPT_KINDS = ("reasoning", "multilingual", "structured", "dialogue")
ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "third_party" / "slotstream" / "Tools" / "fixtures" / "expert-lookahead" / "corpus.json"
CONTRACT = ROOT / "slotstream_spark" / "config" / "numeric_contract.toml"
RES = {}
S = {}


def tie_rel():
    return float(tomllib.loads(CONTRACT.read_text())["ops"]["forecast_tie_rel"])


def status(model_dir):
    from slotstream_spark.engine import lookahead as la
    root = os.environ.get("SLOTSTREAM_R")
    return la.tap_status(Path(root) / "model" if root else Path(model_dir).parent)


def prompts(tok):
    corpus = json.loads(CORPUS.read_text())
    out = []
    for kind in PROMPT_KINDS:
        r = next(r for r in corpus["requests"] if r["split"] == "train" and r["kind"] == kind and "messages" in r)
        ids = tok.encode(tok.render_chat(r["messages"], enable_thinking=False))
        out.append({"id": r["id"], "kind": kind, "ids": ids})
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
    st = status(model_dir)
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
    rb = Readback(model.vocab, forecast=True)
    g_off = GraphSet(model, 1, readback=rb)
    tap = la.ForecastTap(corr)
    g_on = GraphSet(model, 1, readback=rb, pool_handle=g_off.pool_handle, forecast=tap)
    e = {"model": model, "pool": pool, "g_off": g_off, "g_on": g_on, "rb": rb, "corr": corr, "status": st, "cfg": cfg,
         "tok": Tokenizer.from_checkpoint(model_dir), "model_dir": Path(model_dir)}
    yield e
    out = os.environ.get("CAP")
    if out:
        os.makedirs(out, exist_ok=True)
        Path(out, "forecast.json").write_text(json.dumps(RES, indent=1, sort_keys=True))
    del g_on, g_off, model, pool, ngram, embed, weights, reader
    gc.collect()
    torch.cuda.empty_cache()


def fresh(e, g, ids):
    g.detach()
    e["g_off"].detach()
    e["model"].st.reset_inplace()
    return e["model"].prefill(ids).cpu().numpy()


def decode_graph(e, g, ids, collect=False):
    """Greedy NEW_TOKENS through graph set `g`; -> tokens (and, with collect, per step the tap's top-24 per layer)."""
    logits = fresh(e, g, ids)
    toks, fcst = [], []
    for _ in range(NEW_TOKENS):
        t = int(np.argmax(logits))
        toks.append(t)
        logits = g.step([t])[0].numpy()
        if collect:
            fcst.append([e["rb"].forecast(L, 1) for L in range(47)])
    return toks, fcst


def decode_eager(e, g, ids, toks):
    """Teacher-forced eager steps over `toks` with the probe: per step, per layer l: (mixed, raw, used) of the tap for l + 1,
    the tap's top-24 from the readback and the true routing ids of every layer."""
    fresh(e, g, ids)
    rows = []
    for t in toks:
        probe = {}
        g.tap_probe = lambda L, mixed, raw, used, _p=probe: _p.__setitem__(
            L, (mixed.float().cpu().numpy()[0], raw.cpu().numpy()[0], used.cpu().numpy()[0]))
        g._run([t], eager=True)
        g.tap_probe = None
        rb = e["rb"]
        rows.append({"probe": probe, "forecast": [rb.forecast(L, 1) for L in range(47)],
                     "actual": [rb.layer_ids(L, 10) for L in range(48)]})
    return rows


@pytest.fixture(scope="module")
def runs(engine):
    e = engine
    out = []
    for p in prompts(e["tok"]):
        off, _ = decode_graph(e, e["g_off"], p["ids"])
        on, fg = decode_graph(e, e["g_on"], p["ids"], collect=True)
        eager = decode_eager(e, e["g_on"], p["ids"], on)
        out.append({"prompt": p, "off": off, "on": on, "graph_forecast": fg, "eager": eager})
    return out


def test_a_status_and_header(engine):
    st = engine["status"]
    RES["tap_correction_status"] = st["line"]
    print(f"tap correction: {st['line']}", flush=True)
    assert st["line"] is not None, "no $R/model/tap-correction.status: run tools/fetch_tap_correction.sh first"
    if st["present"]:
        from slotstream_spark.kernels import forecast as fc
        h = fc.read_header(engine["model_dir"] / fc.SHIPPED_REL)
        cfg = engine["cfg"]
        assert (h["first_target"], h["targets"], h["hidden"], h["experts"]) == (1, 47, cfg["hidden_size"], cfg["num_experts"])
        assert h["file_bytes"] == fc.SHIPPED_BYTES and st["sha256"] == fc.SHIPPED_SHA256


def test_b_ids_identical_and_graph_equals_eager(runs):
    for r in runs:
        assert r["on"] == r["off"], f"{r['prompt']['id']}: decode ids changed with the forecast tap on"
        for step, (g24, row) in enumerate(zip(r["graph_forecast"], r["eager"])):
            for L in range(47):
                gi, gl = g24[L]
                ei, el = row["forecast"][L]
                assert np.array_equal(gi, ei) and gl.tobytes() == el.tobytes(), \
                    f"{r['prompt']['id']} step {step} layer {L}: graph forecast differs from the eager one"
    print(f"decode ids identical with the tap on and off; graph == eager forecast over {len(runs)} x {NEW_TOKENS} x 47", flush=True)


def test_b_recall_per_layer(runs, engine):
    from slotstream_spark.engine import lookahead as la
    raw_r, cor_r = np.zeros(47), np.zeros(47)
    n = 0
    for r in runs:
        for row in r["eager"]:
            n += 1
            for L in range(47):
                _, raw, used = row["probe"][L]
                actual = row["actual"][L + 1]
                raw_r[L] += la.recall(la.top_ids(raw), actual)
                cor_r[L] += la.recall(la.top_ids(used), actual)
    raw_r, cor_r = raw_r / n, cor_r / n
    RES["recall_forecast_top10_per_layer"] = [round(float(x), 4) for x in cor_r]
    RES["recall_plain_tap_per_layer"] = [round(float(x), 4) for x in raw_r]
    print("recall of the true routing by the forecast top-10 (target layers 1..47):")
    print("  forecast tap :", " ".join(f"{x:.3f}" for x in cor_r), f"\n  mean {cor_r.mean():.4f} min {cor_r.min():.4f}")
    if engine["corr"] is not None:
        print("  plain tap    :", " ".join(f"{x:.3f}" for x in raw_r), f"\n  mean {raw_r.mean():.4f}")
    assert np.isfinite(cor_r).all()


def reference(e, probe_row, L, b_override=None):
    """float64 reference (raw, corrected) logits of the tap of layer L (target L + 1) from the engine's captured mixed input."""
    from slotstream_spark.engine import lookahead as la
    from slotstream_spark.kernels import forecast as fc
    c = e["corr"]
    i = L + 1 - c.first_target
    mixed = probe_row["probe"][L][0][None]
    gate = fc.router_weight(e["model"].seg, L + 1).float().cpu().numpy()
    b = c.b[i].float().cpu().numpy() if b_override is None else b_override(c.b[i].float().cpu().numpy())
    return la.reference_logits(mixed, gate, c.a[i].float().cpu().numpy(), b, c.mu[i].cpu().numpy(), c.delta[i].cpu().numpy())


def sets_check(e, runs, b_override=None):
    from slotstream_spark.engine import lookahead as la
    rel = tie_rel()
    pairs = excused = hard = differing = 0
    for r in runs:
        for row in r["eager"]:
            for L in range(47):
                raw, cor = reference(e, row, L, b_override)
                ids24, lg24 = row["forecast"][L]
                eng = la.forecast_top(ids24[0], lg24[0])
                res = la.set_check(eng, cor[0], raw[0], rel)
                pairs += 1
                excused += res["excused"]
                hard += res["hard"]
                differing += 0 if res["equal"] else 1
    return {"pairs": pairs, "excused_ids": excused, "non_tie_ids": hard, "pairs_with_difference": differing,
            "fraction_with_difference": differing / pairs, "tie_rel": rel}


def test_c_corrected_sets_match_float64_reference(runs, engine):
    if engine["corr"] is None:
        pytest.skip("tap_correction absent")
    res = sets_check(engine, runs)
    RES["corrected_sets"] = res
    print(f"corrected top-10 sets vs the float64 reference: {json.dumps(res)}", flush=True)
    assert res["non_tie_ids"] == 0, f"{res['non_tie_ids']} ids differ outside the tie margin: {json.dumps(res)}"


def test_d_negative_control_b_transposed_fails(runs, engine):
    if engine["corr"] is None:
        pytest.skip("tap_correction absent")
    c = engine["corr"]
    res = sets_check(engine, runs, b_override=lambda b: np.ascontiguousarray(b.T).reshape(b.shape))
    RES["control_b_transposed"] = res
    print(f"control (b transposed): {json.dumps(res)}", flush=True)
    assert res["non_tie_ids"] >= 1, "the control did not produce a single non-tie difference over all pairs"
    assert c.rank == 128
