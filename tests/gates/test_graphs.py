"""Step 54: readback + CUDA graphs per segment, 1-row and 3-row variants.

Pool of 48 slots per layer (2,304). Judged here:
  - precondition: one eager decode step lists no cuBLAS/CUTLASS GEMM kernel (every linear is our Triton GEMV), and two
    consecutive graph decodes of the same prompt are bit-identical;
  - graph decode equals eager decode bit for bit (logits and final state) over 2 prompts x 32 tokens;
  - a 3-row step equals three 1-row steps bit for bit in logits, KV (main, ring, compressed) and GDN/conv state;
  - tau p50 in graph mode <= 150 us (wait_seq return -> ensure with all hits -> graph launch return), after a
    >= 16-token warm-up, over 1,024 layer steps.
Writes $CAP/graphs.json when CAP is set. Run: tools/spark/run.sh --heavy graphs 12288 3600 -- tools/spark/py.sh -m pytest
-q -m gate tests/gates/test_graphs.py
Step 56 adds test_e: the n-gram row cache with the prefetch overlapped with G(0) (cache on vs off give the same
tokens and logits; hits > 0 on a repeated-text prompt; the prefetch path is the one the graphs run).
"""
import json
import os
import re
import statistics
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gate

SLOTS = 48 * 48
NEW_TOKENS = 32
TAU_LIMIT_US = 150.0
TAU_WARMUP_TOKENS = 16
TAU_STEPS = 1024
GEMM_RE = re.compile(r"cublas|gemm(?!a)|cutlass|cijk", re.I)      # "gemma" is the Gemma-style RMSNorm, not a GEMM
RES = {}


def prompt_ids(k):
    import make_goldens as mg
    d = mg.find_prompt_dir(mg.DEFAULT_DATA, f"dev{k}")
    return [int(t) for t in json.loads((d / "tokens.json").read_text())["ids"]]


@pytest.fixture(scope="module")
def engine(model_dir, packed_dir):
    import gc
    import slotio
    import torch
    from slotstream_spark.model.config import TextConfig
    from slotstream_spark.model.forward import Model
    from slotstream_spark.model.weights import Embedder, NgramSource, accessor, load_trunk
    from slotstream_spark.runtime.graphs import GraphSet
    from slotstream_spark.runtime.pool import ExpertPool

    cfg = TextConfig.from_checkpoint(model_dir)
    weights = load_trunk(packed_dir / "trunk.safetensors", device="cuda")
    reader = slotio.Reader()
    embed = Embedder(reader, packed_dir / "trunk.safetensors", device="cuda")
    ngram = NgramSource(reader, packed_dir, weights, cfg["eos_token_id"], device="cuda")
    pool = ExpertPool(packed_dir, SLOTS, reader=reader)
    model = Model(cfg, accessor(weights), embed, pool, ngram, device="cuda", gdn="fused", max_ctx=256)
    model.st.ensure_act(torch.bfloat16)
    g1 = GraphSet(model, 1)
    g3 = GraphSet(model, 3, readback=g1.rb, pool_handle=g1.pool_handle)
    yield {"model": model, "pool": pool, "g1": g1, "g3": g3, "ngram": ngram, "reader": reader, "weights": weights,
           "packed_dir": packed_dir, "cfg": cfg}
    out = os.environ.get("CAP")
    if out:
        os.makedirs(out, exist_ok=True)
        Path(out, "graphs.json").write_text(json.dumps(RES, indent=1, sort_keys=True))
    del g3, g1, model, pool, ngram, embed, weights, reader
    gc.collect()
    torch.cuda.empty_cache()


def fresh(e, ids):
    """In-place restart of the sequence (the graphs are bound to its tensors) and eager prefill."""
    e["g1"].detach()
    e["model"].st.reset_inplace()
    return e["model"].prefill(ids).cpu().numpy()


def state_equal(a, b):
    import torch
    out = {"pos": a["pos"] == b["pos"], "tokens": a["tokens"] == b["tokens"], "S": bool(torch.equal(a["S"], b["S"])),
           "conv": bool(torch.equal(a["conv"], b["conv"])), "ple_conv": bool(torch.equal(a["ple_conv"], b["ple_conv"]))}
    for L in a["attn"]:
        for name, x, y in zip(("main_kv", "ring", "comp"), a["attn"][L][:3], b["attn"][L][:3]):
            out[f"{name}[{L}]"] = bool(torch.equal(x, y))
    return out


def eager_run(e, ids):
    from slotstream_spark.runtime.graphs import snapshot_state
    lg = fresh(e, ids)
    logits, toks = [lg], []
    for _ in range(NEW_TOKENS):
        tok = int(np.argmax(lg))
        toks.append(tok)
        lg = e["model"].decode(tok).cpu().numpy()
        logits.append(lg)
    return toks, logits, snapshot_state(e["model"].st)


def graph_run(e, ids):
    from slotstream_spark.runtime.graphs import snapshot_state
    lg = fresh(e, ids)
    logits, toks = [lg], []
    for _ in range(NEW_TOKENS):
        tok = int(np.argmax(lg))
        toks.append(tok)
        lg = e["g1"].step([tok])[0].numpy()
        logits.append(lg)
    return toks, logits, snapshot_state(e["model"].st)


def test_a_no_gemm_kernel_in_eager_decode(engine):
    import torch
    from torch.profiler import ProfilerActivity, profile
    e = engine
    lg = fresh(e, prompt_ids(0))
    tok = int(np.argmax(lg))
    e["model"].decode(tok)                                    # warm
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        e["model"].decode(int(np.argmax(e["model"].decode(tok).cpu().numpy())))
        torch.cuda.synchronize()
    names = sorted({ev.name for ev in prof.events() if ev.device_type == torch.autograd.DeviceType.CUDA})
    bad = [n for n in names if GEMM_RE.search(n)]
    RES["kernel_names"] = len(names)
    RES["gemm_kernels"] = bad
    print(f"eager decode step: {len(names)} distinct CUDA kernels, GEMM-like: {bad}", flush=True)
    assert len(names) > 20, "the profile saw too few kernels to mean anything"
    assert not bad, f"a cuBLAS/CUTLASS kernel runs in the decode step: {bad}"


def test_b_graph_equals_eager_and_repeats(engine):
    e = engine
    res = {}
    for k in (0, 1):
        ids = prompt_ids(k)
        t_e, l_e, s_e = eager_run(e, ids)
        t_g, l_g, s_g = graph_run(e, ids)
        t_g2, l_g2, s_g2 = graph_run(e, ids)
        eq = [bool(np.array_equal(a, b)) for a, b in zip(l_e, l_g)]
        rep = [bool(np.array_equal(a, b)) for a, b in zip(l_g, l_g2)]
        st_eq, st_rep = state_equal(s_e, s_g), state_equal(s_g, s_g2)
        res[f"dev{k}"] = {"tokens_equal": t_e == t_g, "logits_equal": all(eq), "logits_first_diff": eq.index(False) if not all(eq) else None,
                          "state_equal": all(st_eq.values()), "repeat_logits_equal": all(rep), "repeat_state_equal": all(st_rep.values()),
                          "state_bad": [n for n, v in st_eq.items() if not v]}
        print(f"prompt dev{k}: {json.dumps(res[f'dev{k}'])}", flush=True)
    RES["graph_vs_eager"] = res
    for k, r in res.items():
        assert r["tokens_equal"] and r["logits_equal"], f"{k}: graph decode differs from eager decode: {r}"
        assert r["state_equal"], f"{k}: graph state differs from eager state: {r['state_bad']}"
        assert r["repeat_logits_equal"] and r["repeat_state_equal"], f"{k}: two graph decodes of the same prompt differ"


def test_c_three_rows_equal_three_single_rows(engine):
    from slotstream_spark.runtime.graphs import restore_state, snapshot_state
    e = engine
    res = {}
    for k in (0, 1):
        lg = fresh(e, prompt_ids(k))
        toks = []
        for _ in range(8):                                    # leave the prefill state behind
            toks.append(int(np.argmax(lg)))
            lg = e["g1"].step([toks[-1]])[0].numpy()
        snap = snapshot_state(e["model"].st)
        x = [int(np.argmax(lg))]
        rows1 = []
        for i in range(3):
            lgi = e["g1"].step([x[-1]])[0].numpy()
            rows1.append(lgi)
            x.append(int(np.argmax(lgi)))
        s1 = snapshot_state(e["model"].st)
        restore_state(e["model"].st, snap)
        rows3 = e["g3"].step(x[:3]).numpy()
        s3 = snapshot_state(e["model"].st)
        eq = [bool(np.array_equal(rows1[i], rows3[i])) for i in range(3)]
        st_eq = state_equal(s1, s3)
        res[f"dev{k}"] = {"rows_equal": eq, "state_equal": all(st_eq.values()),
                          "state_bad": [n for n, v in st_eq.items() if not v]}
        print(f"prompt dev{k}: {json.dumps(res[f'dev{k}'])}", flush=True)
        assert all(eq), f"dev{k}: 3-row logits differ from 1-row steps: {eq}"
        assert all(st_eq.values()), f"dev{k}: 3-row state differs: {res[f'dev{k}']['state_bad']}"
    RES["three_rows"] = res


def test_d_tau(engine):
    from slotstream_spark.runtime.graphs import restore_state, snapshot_state
    e = engine
    lg = fresh(e, prompt_ids(0))
    for _ in range(TAU_WARMUP_TOKENS):                        # graph decode warm-up
        lg = e["g1"].step([int(np.argmax(lg))])[0].numpy()
    snap = snapshot_state(e["model"].st)
    tok = int(np.argmax(lg))
    samples, cold = [], 0
    for it in range(40):
        restore_state(e["model"].st, snap)
        tm = []
        e["g1"].step([tok], timings=tm)
        if it == 0:
            cold = sum(1 for t in tm if t[4] > 0)
            continue                                           # the first pass loads the experts
        samples += [t for t in tm if t[4] == 0]               # all-hit layer steps, as the M0' definition
        if len(samples) >= TAU_STEPS:
            break
    assert len(samples) >= TAU_STEPS, f"only {len(samples)} all-hit layer steps"
    samples = samples[:TAU_STEPS]
    tau = [(s[2] + s[3]) / 1000 for s in samples]             # ensure + launch, from wait_seq return
    out = {"p50_us": round(statistics.median(tau), 2), "p99_us": round(float(np.percentile(tau, 99)), 2),
           "wait_p50_us": round(statistics.median(s[1] / 1000 for s in samples), 2),
           "ensure_p50_us": round(statistics.median(s[2] / 1000 for s in samples), 2),
           "launch_p50_us": round(statistics.median(s[3] / 1000 for s in samples), 2), "steps": len(samples),
           "warmup_tokens": TAU_WARMUP_TOKENS, "cold_pass_missing_layers": cold}
    RES["tau"] = out
    print("tau:", json.dumps(out), flush=True)
    if out["p50_us"] > TAU_LIMIT_US:
        print(f"DECISION: tau={out['p50_us']} us > {TAU_LIMIT_US:.0f} us -> move the layer loop to C++ / bring F8 "
              "forward (the technology decision)", flush=True)
    assert out["p50_us"] <= TAU_LIMIT_US, f"tau p50 {out['p50_us']} us > {TAU_LIMIT_US} us"


def test_e_ngram_cache_on_vs_off_identical(engine):
    """Cache on and cache off (cache_rows = 0) give identical tokens and logits through the prefetch/wait path;
    the repeated-text prompt makes the cache hit."""
    from slotstream_spark.model.weights import NgramSource
    e = engine
    assert e["g1"].ple_late and e["g3"].ple_late, "the decode loop does not overlap the prefetch with G(0)"
    base = prompt_ids(0)[:20]                                  # prefill `base`, then decode the same text again:
    runs = {}                                                  # every n-gram after the first two tokens is already cached
    seg, orig = e["model"].seg, e["model"].seg.ngram
    try:
        for name, rows in (("on", 400000), ("off", 0)):
            src = NgramSource(e["reader"], e["packed_dir"], e["weights"], e["cfg"]["eos_token_id"], device="cuda",
                              cache_rows=rows)
            seg.ngram = src
            lg = fresh(e, base)
            logits, toks = [lg], []
            for tok in base:                                   # forced tokens: the repeated text
                toks.append(tok)
                lg = e["g1"].step([tok])[0].numpy()
                logits.append(lg)
            for _ in range(8):                                 # then greedy
                toks.append(int(np.argmax(lg)))
                lg = e["g1"].step([toks[-1]])[0].numpy()
                logits.append(lg)
            hits, misses, size = src.rows.cache_stats()
            runs[name] = (toks, logits, (int(hits), int(misses), int(size)))
    finally:
        seg.ngram = orig
    on, off = runs["on"], runs["off"]
    eq = [bool(np.array_equal(a, b)) for a, b in zip(on[1], off[1])]
    RES["ngram_cache"] = {"tokens_equal": on[0] == off[0], "logits_equal": all(eq), "on_hits_misses_size": list(on[2]),
                          "off_hits_misses_size": list(off[2])}
    print("ngram cache:", json.dumps(RES["ngram_cache"]), flush=True)
    assert on[0] == off[0] and all(eq), "cache on differs from cache off"
    assert on[2][0] > 0, f"no cache hit on the repeated-text prompt: {on[2]}"
    assert off[2][0] == 0 and off[2][2] == 0, f"cache_rows = 0 still cached: {off[2]}"
