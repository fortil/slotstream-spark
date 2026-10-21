"""Step 91 (F8): the single per-token graph with host-memory waits equals the F3 segmented path bit for bit.

Judged here, on the Spark (pool of 2,304 slots as in test_graphs.py):
  - structure: the captured graph holds 96 memory-operation nodes (a wait and its re-arm for each of the 48 layer
    boundaries) and no kernel that spins or polls; after a token every `ready` word is 0 and `done[l] == base + l`;
  - 8 prompts (dev0..dev7) x 128 tokens: tokens, fp32 logits and the whole decode state (GDN, conv, KV main/ring/comp) are
    identical to the F3 segmented path;
  - the same with an injected delay of 60 ms in `ensure` at layers 3 and 40: identical, no fallback (the delay is on the
    host side of the wait, the graph simply stays stopped on its memory operation);
  - the host wait bound: with a bound that every token misses, the miss is counted, the next token runs segmented, and the
    output stays identical;
  - 0 CUDA errors: every run ends with a stream synchronize and the kernel log of the run is checked by the guard.
Writes $CAP/token_graph.json when CAP is set. Run: tools/spark/run.sh --heavy token-graph 12288 3600 -- tools/spark/py.sh -m
pytest -q -m gate tests/gates/test_token_graph.py
"""
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pytest
from test_graphs import engine, fresh, prompt_ids, state_equal  # noqa: F401  (engine is a fixture)

pytestmark = pytest.mark.gate

PROMPTS = 8
NEW_TOKENS = 128
SPIN_RE = re.compile(r"spin|poll|busy|wait_flag", re.I)
RES = {}


@pytest.fixture(scope="module", autouse=True)
def _write_result():
    yield
    out = os.environ.get("CAP")
    if out:
        os.makedirs(out, exist_ok=True)
        Path(out, "token_graph.json").write_text(json.dumps(RES, indent=1, sort_keys=True))


@pytest.fixture(scope="module")
def tg(engine):
    from slotstream_spark.runtime.token_graph import TokenGraph
    return TokenGraph(engine["g1"])


def run(e, step, ids, n=NEW_TOKENS, tg=None, check_done=False):
    from slotstream_spark.runtime.graphs import snapshot_state
    lg = fresh(e, ids)
    logits, toks, done_bad = [lg], [], 0
    for _ in range(n):
        tok = int(np.argmax(lg))
        toks.append(tok)
        e["g1"].attach()                      # attach() may move the generation base: read it after
        base = e["g1"].rb.base
        lg = step([tok])[0].numpy()
        logits.append(lg)
        if check_done and tg is not None and not tg._segmented_next and tg.graph_tokens and tg.done_ok(base) is False:
            done_bad += 1
    import torch
    torch.cuda.synchronize()
    return toks, logits, snapshot_state(e["model"].st), done_bad


def compare(a, b):
    eq = [bool(np.array_equal(x, y)) for x, y in zip(a[1], b[1])]
    st = state_equal(a[2], b[2])
    return {"tokens_equal": a[0] == b[0], "logits_equal": all(eq), "first_diff": eq.index(False) if not all(eq) else None,
            "state_equal": all(st.values()), "state_bad": [n for n, v in st.items() if not v]}


def test_a_structure(engine, tg):
    import torch
    from torch.profiler import ProfilerActivity, profile
    e = engine
    lg = fresh(e, prompt_ids(0))
    tok = int(np.argmax(lg))
    e["g1"].attach()
    base = e["g1"].rb.base
    tg.step([tok])
    torch.cuda.synchronize()
    RES["memop_nodes"] = tg.memop_nodes
    assert tg.memop_nodes == 96, f"expected 96 memory-operation nodes (48 waits + 48 re-arms), found {tg.memop_nodes}"
    assert tg.ring.armed() == 0, "a ready word is still raised after the token"
    assert tg.done_ok(base), f"done[l] != base + l after the token: {[int(x) for x in tg.done[:48]]}"
    fresh(e, prompt_ids(0))
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        tg.step([tok])
        torch.cuda.synchronize()
    names = sorted({ev.name for ev in prof.events() if ev.device_type == torch.autograd.DeviceType.CUDA})
    bad = [n for n in names if SPIN_RE.search(n)]
    RES["kernel_names"] = len(names)
    RES["spinning_kernels"] = bad
    assert len(names) > 20, "the profile saw too few kernels to mean anything"
    assert not bad, f"a spinning or polling kernel is in the token graph: {bad}"


def test_b_bit_identical_to_f3(engine, tg):
    e = engine
    res = {}
    for k in range(PROMPTS):
        ids = prompt_ids(k)
        seg = run(e, e["g1"].step, ids)
        g0 = tg.graph_tokens
        gra = run(e, tg.step, ids, tg=tg, check_done=True)
        r = compare(seg, gra)
        r.update({"graph_tokens": tg.graph_tokens - g0, "done_mismatch_tokens": gra[3]})
        res[f"dev{k}"] = r
        print(f"prompt dev{k}: {json.dumps(r)}", flush=True)
    RES["vs_f3"] = res
    RES["tokens_per_prompt"] = NEW_TOKENS
    for k, r in res.items():
        assert r["tokens_equal"] and r["logits_equal"], f"{k}: token graph differs from F3: {r}"
        assert r["state_equal"], f"{k}: token graph state differs from F3: {r['state_bad']}"
        assert r["graph_tokens"] == NEW_TOKENS, f"{k}: {r['graph_tokens']} of {NEW_TOKENS} tokens ran on the graph"
        assert r["done_mismatch_tokens"] == 0, f"{k}: done[l] != base + l on {r['done_mismatch_tokens']} tokens"
    assert tg.fallbacks == 0, f"{tg.fallbacks} fallbacks in the plain runs"


def test_c_injected_delayed_ensure(engine, tg):
    e = engine
    res = {}
    f0 = tg.fallbacks

    def delay(layer):
        if layer in (3, 40):
            time.sleep(0.060)
    try:
        for k in (0, 1):
            ids = prompt_ids(k)
            seg = run(e, e["g1"].step, ids, n=24)
            tg.ensure_hook = delay
            t0 = time.perf_counter()
            gra = run(e, tg.step, ids, n=24, tg=tg, check_done=True)
            dt = time.perf_counter() - t0
            tg.ensure_hook = None
            r = compare(seg, gra)
            r.update({"seconds": round(dt, 2), "done_mismatch_tokens": gra[3]})
            res[f"dev{k}"] = r
            print(f"delayed ensure dev{k}: {json.dumps(r)}", flush=True)
    finally:
        tg.ensure_hook = None
    RES["delayed_ensure"] = res
    RES["delayed_ensure_fallbacks"] = tg.fallbacks - f0
    for k, r in res.items():
        assert r["tokens_equal"] and r["logits_equal"] and r["state_equal"], f"{k}: delayed ensure changed the output: {r}"
        assert r["seconds"] >= 24 * 0.120 * 0.9, f"{k}: the injected delay did not run ({r['seconds']} s)"
        assert r["done_mismatch_tokens"] == 0
    assert tg.fallbacks == f0, "the delay is on the ensure side and must not count as a wait fallback"


def test_d_bounded_wait_falls_back_to_f3(engine, tg):
    e = engine
    ids = prompt_ids(2)
    seg = run(e, e["g1"].step, ids, n=24)
    f0, s0, g0 = tg.fallbacks, tg.segmented_tokens, tg.graph_tokens
    bound, tg.wait_bound_s = tg.wait_bound_s, 1e-9              # every graph token misses the bound at layer 0
    try:
        gra = run(e, tg.step, ids, n=24)
    finally:
        tg.wait_bound_s = bound
    r = compare(seg, gra)
    r.update({"fallbacks": tg.fallbacks - f0, "segmented_tokens": tg.segmented_tokens - s0,
              "graph_tokens": tg.graph_tokens - g0})
    RES["fallback"] = r
    print("fallback:", json.dumps(r), flush=True)
    assert r["tokens_equal"] and r["logits_equal"] and r["state_equal"], f"a fallback changed the output: {r}"
    assert r["fallbacks"] > 0, "no wait missed the 1 ns bound"
    assert r["fallbacks"] == r["graph_tokens"], "a fallback is counted once per token that missed the bound"
    assert r["segmented_tokens"] >= r["fallbacks"] - 1, "the token after a miss did not run on the F3 path"
    assert tg.ring.armed() == 0


def test_e_no_cuda_error_and_graph_still_runs(engine, tg):
    import torch
    e = engine
    lg = fresh(e, prompt_ids(3))
    for _ in range(8):
        lg = tg.step([int(np.argmax(lg))])[0].numpy()
    torch.cuda.synchronize()
    assert tg.ring.armed() == 0
    RES["stats"] = tg.stats()
    print("token graph stats:", json.dumps(tg.stats()), flush=True)
