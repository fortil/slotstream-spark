"""Step 70 gate: the sweep prefill against itself (pool states, read scope) and against the oracle (D5).

Run through `tools/spark/run.sh --heavy f5-sweep-check auto 2700 -- tools/spark/py.sh -m pytest -q -m gate
tests/gates/test_sweep_check.py` (the guard exports SLOTSTREAM_FOOTPRINT_MIB and CAP).

Parts, in file order (the last one releases the engine before the oracle runs, one model process at a time):
  a  a record read through a real slotio Staging slot gives the same MoE bits as the same record in a pool slot
  b  prompts of 2K, 8K and 9K tokens (odd tail): hashes of the final hidden state at every prompt position, the logits of
     the last position and 32 greedy ids are bit-identical between an empty pool, a pool warmed by a previous run, and the
     read scope on versus off; 0 pool writes on non-final passes; bytes requested printed with the scope on and off
  c  the engine dumps of the 2K prompt: the sweep path and the F2 row-group path (Model.teacher_force)
  d  the oracle pair on the same 2K ids (bf16 then fp32, after the engine released the pool): layer_check_seq and
     routing_check_seq per layer on the sweep dump, logits_check_seq (per-sequence KL limits, design decision 18), the sweep-vs-F2 layer_check
     with the same pair; per layer the pair metric and `limit_src` are printed

The hashed quantity is the final hidden state (after the final hyper-connection mixer) of every prompt position, which the
logits are a function of; the full-vocabulary logits are compared bit for bit through the dumps of the 2K prompt (c, d).
"""
import gc
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import contract as ct

pytestmark = [pytest.mark.gate, pytest.mark.cuda]

SLOTS = 19012                         # D5 maximum pool (H2)
LENGTHS = {"2k": 2048, "8k": 8192, "9k": 9000}
SCOPE_PASS = {"2k": 256, "8k": 1024, "9k": 1024}
NEW_TOKENS = 32
MAX_CTX = 9216
LAYERS, HIDDEN, TOPK, VOCAB = 48, 10240, 10, 248320
S = {}                                # the engine and everything the parts hand to each other


def work_dir():
    r = os.environ.get("SLOTSTREAM_R")
    if not r:
        pytest.fail("SLOTSTREAM_R is not set (run through tools/spark/py.sh)", pytrace=False)
    d = Path(r) / "tmp" / "f5-sweep-check"
    d.mkdir(parents=True, exist_ok=True)
    return d


def long_ids(model_dir, n):
    """n token ids of real text: the two perplexity texts, tokenized, each repetition numbered so it is not a copy."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    texts = json.loads((Path(__file__).resolve().parents[1] / "goldens" / "ppl_texts.json").read_text())
    out, k = [], 0
    while len(out) < n:
        for lang in ("en", "es"):
            out += tok.encode(f"Section {k} ({lang}).\n" + texts[lang] + "\n", add_special_tokens=False).ids
            k += 1
    return [int(t) for t in out[:n]]


@pytest.fixture(scope="module")
def engine(model_dir, packed_dir):
    import slotio
    import torch
    from slotstream_spark.engine.prefill import PrefillDriver
    from slotstream_spark.engine import prefill_schedule as ps
    from slotstream_spark.model.config import TextConfig
    from slotstream_spark.model.forward import Model
    from slotstream_spark.model.moe_sweep import SweepMoE
    from slotstream_spark.model.weights import Embedder, NgramSource, accessor, load_trunk
    from slotstream_spark.runtime.pool import ExpertPool

    cfg = TextConfig.from_checkpoint(model_dir)
    weights = load_trunk(packed_dir / "trunk.safetensors", device="cuda")
    reader = slotio.Reader()
    embed = Embedder(reader, packed_dir / "trunk.safetensors", device="cuda")
    ngram = NgramSource(reader, packed_dir, weights, cfg["eos_token_id"], device="cuda")
    pool = ExpertPool(packed_dir, SLOTS, reader=reader)
    model = Model(cfg, accessor(weights), embed, pool, ngram, device="cuda", gdn="fused", max_ctx=MAX_CTX)
    env = ps.envelope()
    moe = SweepMoE(model.seg, pool, max_rows_per_launch=int(env["max_rows_per_launch"]))
    driver = PrefillDriver(model, moe, env)
    eng = {"model": model, "pool": pool, "moe": moe, "driver": driver, "env": env, "torch": torch, "reader": reader,
           "parts": (weights, embed, ngram), "model_dir": model_dir}
    S["eng"] = eng
    del weights, reader, embed, ngram, pool, model, moe, driver     # the dict is the only owner: release() frees the pool
    yield eng
    release()


def release():
    eng = S.pop("eng", None)
    if eng is None:
        return
    torch = eng["torch"]
    eng.clear()
    gc.collect()
    torch.cuda.empty_cache()


def clear_pool(pool):
    """Test-only: every slot unmapped (shrinking to zero unmaps each owner), then the pool grows back."""
    pool.resize_slots(0)
    pool.resize_slots(SLOTS)


class Hasher:
    def __init__(self):
        self.h = hashlib.sha256()
        self.rows = 0

    def __call__(self, pos, fin):
        self.h.update(fin.float().cpu().numpy().tobytes())
        self.rows += fin.shape[0]


def greedy(model, logits, n):
    torch = S["eng"]["torch"]
    from slotstream_spark.model import layers
    out = []
    gemm0 = layers.GEMM_STATS["gemm_calls"]
    for _ in range(n):
        tok = int(torch.argmax(logits))
        out.append(tok)
        logits = model.decode(tok)
    assert layers.GEMM_STATS["gemm_calls"] == gemm0, "decode must never enter the GEMM"
    return out


def sweep_run(ids, plan, *, clear=False, dump=None, decode=NEW_TOKENS):
    from slotstream_spark.engine.prefill import d5_budget
    eng = S["eng"]
    model, driver, pool = eng["model"], eng["driver"], eng["pool"]
    if clear:
        clear_pool(pool)
    model.reset()
    hs = Hasher()
    res = driver.run(ids, model.st, plan=plan, control=d5_budget(), should_continue=lambda: True, on_rows=hs,
                     dump_dir=dump)
    assert res.sweep["gemm_calls"] > 0 and res.sweep["gemv_fallbacks"] == 0, res.sweep
    lg = res.last_logits
    lhash = hashlib.sha256(lg.float().cpu().numpy().tobytes()).hexdigest()
    toks = greedy(model, lg, decode) if decode else []
    assert hs.rows == len(ids)
    return {"hidden": hs.h.hexdigest(), "logits": lhash, "tokens": toks, "res": res}


def show(tag, r):
    res = r["res"]
    print(f"{tag}: passes {res.passes[:6]}{'...' if len(res.passes) > 6 else ''} scopes {len(res.scopes)} "
          f"bytes_requested {res.bytes_requested:,} records {res.records_read} admitted {res.admitted} "
          f"nonfinal pool writes {res.pool_writes_nonfinal} {res.seconds:.1f}s "
          f"gemm_calls {res.sweep['gemm_calls']} gemv_fallbacks {res.sweep['gemv_fallbacks']} "
          f"staging bytes {S['eng']['moe'].staging.bytes:,}", flush=True)


# ------------------------------------------------------------------------------------------------ a
def test_a_staging_slot_equals_pool_slot(engine):
    torch = engine["torch"]
    model, pool, moe = engine["model"], engine["pool"], engine["moe"]
    L = 5
    g = torch.Generator(device="cuda").manual_seed(1729)
    T = 64
    mixed = (torch.randn(T, 2560, generator=g, device="cuda") * 0.5).to(torch.bfloat16)
    experts = np.arange(7, 7 + 24, dtype=np.int32)
    rng = np.random.default_rng(1729)
    ids = rng.choice(experts, size=(T, TOPK)).astype(np.int32)
    w = torch.rand(T, TOPK, generator=g, device="cuda")
    clear_pool(pool)
    before = moe.stats.records_read
    staged = moe.layer(L, mixed, ids, w)
    read = moe.stats.records_read - before
    assert read == len(np.unique(ids)), "with an empty pool every expert comes through a staging slot"
    slots = pool.ensure(L, np.unique(ids).astype(np.int32), 10**6)
    assert len(slots) == len(np.unique(ids))
    before = moe.stats.records_read
    pooled = moe.layer(L, mixed, ids, w)
    assert moe.stats.records_read == before, "now every expert is resident: nothing is read"
    assert torch.equal(staged, pooled), "the MoE bits changed when the record moved from a staging slot to a pool slot"
    print(f"staging slot vs pool slot: {int(len(np.unique(ids)))} experts, {T} tokens, bit-identical", flush=True)


# ------------------------------------------------------------------------------------------------ b
@pytest.mark.parametrize("name", ["2k", "8k", "9k"])
def test_b_cold_warm_scope(engine, name):
    from slotstream_spark.engine.prefill import PrefillPlan
    ids = S.setdefault("ids", {}).setdefault(name, long_ids(engine["model_dir"], LENGTHS[name]))
    env = engine["env"]
    d5 = PrefillPlan.from_envelope(env, prefill_pass=4096, read_scope=8192)
    cold = sweep_run(ids, d5, clear=True)
    show(f"{name} empty pool", cold)
    warm = sweep_run(ids, d5)
    show(f"{name} warmed pool", warm)
    assert cold["hidden"] == warm["hidden"], "prompt hidden states differ between an empty and a warmed pool"
    assert cold["logits"] == warm["logits"], "last-position logits differ between an empty and a warmed pool"
    assert cold["tokens"] == warm["tokens"], "greedy ids differ between an empty and a warmed pool"
    assert cold["res"].admitted > 0 and warm["res"].admitted >= 0
    for r in (cold, warm):
        assert r["res"].pool_writes_nonfinal == 0, "a non-final pass wrote the pool"
    # read scope on vs off at the small pass size where a scope exists (>= 4 identical non-final passes)
    sp = SCOPE_PASS[name]
    on = sweep_run(ids, PrefillPlan.from_envelope(env, prefill_pass=sp, read_scope=8192))
    show(f"{name} scope on (pass {sp})", on)
    off = sweep_run(ids, PrefillPlan.from_envelope(env, prefill_pass=sp, read_scope=0))
    show(f"{name} scope off (pass {sp})", off)
    assert any(len(s) >= 4 for s in on["res"].scopes), "the scope-on run formed no scope of 4 passes"
    assert all(len(s) == 1 for s in off["res"].scopes)
    assert on["hidden"] == off["hidden"], "prompt hidden states differ between scope on and off"
    assert on["logits"] == off["logits"] and on["tokens"] == off["tokens"], "logits or greedy ids differ between scope on and off"
    assert on["res"].pool_writes_nonfinal == 0 and off["res"].pool_writes_nonfinal == 0
    # the two runs start from different pool states (the first one's final pass admitted experts), so the byte counts are
    # printed, not compared; staged_rows is the number of rows that came through staging
    print(f"{name}: bytes requested scope on {on['res'].bytes_requested:,} (staged rows {on['res'].sweep['staged_rows']}) "
          f"off {off['res'].bytes_requested:,} (staged rows {off['res'].sweep['staged_rows']}); "
          f"greedy ids {cold['tokens'][:8]}...", flush=True)
    S.setdefault("summary", {})[name] = {"tokens": LENGTHS[name], "passes_d5": cold["res"].passes,
                                         "bytes_scope_on": on["res"].bytes_requested,
                                         "bytes_scope_off": off["res"].bytes_requested, "hidden_sha256": cold["hidden"],
                                         "ids": cold["tokens"]}


# ------------------------------------------------------------------------------------------------ c
def test_c_engine_dumps_2k(engine):
    from slotstream_spark.engine.prefill import PrefillPlan
    ids = S.setdefault("ids", {}).setdefault("2k", long_ids(engine["model_dir"], LENGTHS["2k"]))
    w = work_dir()
    sweep_dir, f2_dir = w / "engine-sweep", w / "engine-f2"
    for d in (sweep_dir, f2_dir):
        shutil.rmtree(d, ignore_errors=True)
    plan = PrefillPlan.from_envelope(engine["env"], prefill_pass=4096, read_scope=8192)
    r = sweep_run(ids, plan, clear=True, dump=sweep_dir, decode=0)
    show("2k sweep dump", r)
    (sweep_dir / "tokens.json").write_text(json.dumps({"id": "f5-2k", "ids": ids}))
    model = engine["model"]
    with engine["torch"].no_grad():
        model.teacher_force(ids, dump_dir=f2_dir)
    (f2_dir / "tokens.json").write_text(json.dumps({"id": "f5-2k", "ids": ids}))
    S["dumps"] = (sweep_dir, f2_dir)
    print(f"2k F2 row-group dump written ({len(ids)} tokens)", flush=True)


# ------------------------------------------------------------------------------------------------ d
def _arr(d, name, dtype, shape):
    return np.fromfile(Path(d) / name, dtype=dtype).reshape(shape)


def test_d_oracle_pair_and_checks(model_dir):
    import oracle_pair
    release()                                          # the pool is gone before the oracle runs
    gc.collect()
    with open("/proc/meminfo") as f:
        avail = next(int(x.split()[1]) // 1024 for x in f if x.startswith("MemAvailable:"))
    print(f"engine released: MemAvailable {avail} MiB", flush=True)
    assert avail > 40_000, "the engine's pool is still held after release()"
    sweep_dir, f2_dir = S["dumps"]
    ids = S["ids"]["2k"]
    T = len(ids)
    root = Path(os.environ["SLOTSTREAM_R"]) / "tmp" / "oracle-pair-f5"
    bf16_dir, fp32_dir = oracle_pair.pair_dumps(ids, root, "f5-2k")
    for d in (sweep_dir, bf16_dir, fp32_dir):
        assert json.loads((Path(d) / "tokens.json").read_text())["ids"] == ids, d
    c = ct.load()
    bad, rows, worst = [], [], 0.0
    for L in range(LAYERS):
        e = _arr(sweep_dir, f"layer_{L:02d}_out.f32", np.float32, (T, HIDDEN))
        o = _arr(bf16_dir, f"layer_{L:02d}_out.f32", np.float32, (T, HIDDEN))
        o32 = _arr(fp32_dir, f"layer_{L:02d}_out.f32", np.float32, (T, HIDDEN))
        ly = ct.layer_check_seq(L, e, o, o32, c)
        ri_e = _arr(sweep_dir, f"router_{L:02d}_ids.i32", np.int32, (T, TOPK))
        ri_o = _arr(bf16_dir, f"router_{L:02d}_ids.i32", np.int32, (T, TOPK))
        ri_32 = _arr(fp32_dir, f"router_{L:02d}_ids.i32", np.int32, (T, TOPK))
        rt = ct.routing_check_seq(L, ri_e, ri_o, ri_32, c)
        f2 = _arr(f2_dir, f"layer_{L:02d}_out.f32", np.float32, (T, HIDDEN))
        v = ct.rel_l2(e, f2)
        lim, src = ct.seq_limit("layer", L, ly["pair"], c)
        sep = bool(c.layer["separable"][L])
        f2_ok = (v <= lim) if sep else True
        rows.append({"layer": L, "rel_l2": ly["value"], "pair": ly["pair"], "limit": ly["limit_seq"],
                     "limit_src": ly["limit_src"], "separable": ly["separable"], "ok": ly["ok"], "routing": rt["value"],
                     "routing_pair": rt["pair"], "routing_limit": rt["limit_seq"], "routing_limit_src": rt["limit_src"],
                     "routing_ok": rt["ok"], "sweep_vs_f2": v, "sweep_vs_f2_limit": lim, "sweep_vs_f2_ok": bool(f2_ok)})
        print(f"layer {L:02d}: rel_l2 {ly['value']:.4f} (pair {ly['pair']:.4f}, limit {ly['limit_seq']:.4f} "
              f"{ly['limit_src']}, separable {ly['separable']}) routing {rt['value']:.4f} (pair {rt['pair']:.4f}, "
              f"limit {rt['limit_seq']:.4f} {rt['limit_src']}) sweep-vs-F2 {v:.4f} (limit {lim:.4f} {src})", flush=True)
        if not (ly["ok"] and rt["ok"] and f2_ok):
            bad.append(L)
    le = _arr(sweep_dir, "logits.f32", np.float32, (T, VOCAB))
    lo = _arr(bf16_dir, "logits.f32", np.float32, (T, VOCAB))
    l32 = _arr(fp32_dir, "logits.f32", np.float32, (T, VOCAB))
    lc = ct.logits_check_seq(le, lo, l32, c)
    del lo, l32
    print(f"logits over {T} prompt positions: {json.dumps(lc)}", flush=True)
    lf = _arr(f2_dir, "logits.f32", np.float32, (T, VOCAB))
    print(f"sweep vs F2 logits: max abs diff {float(np.abs(le - lf).max()):.4f}", flush=True)
    del le, lf
    cap = os.environ.get("CAP")
    if cap:
        Path(cap).mkdir(parents=True, exist_ok=True)
        Path(cap, "sweep_check.json").write_text(ct.dumps({"layers": rows, "logits": lc, "failing_layers": bad,
                                                           "summary": S.get("summary", {})}))
    assert not bad, f"layers outside the per-sequence rule: {bad}"
    if not lc["judgeable"]:
        pytest.fail(f"DESIGN GATE: the oracle pair exceeds a [logits] ceiling on this sequence — {json.dumps(lc)}", pytrace=False)
    assert lc["ok"], f"logits_check_seq failed: {json.dumps(lc)}"
    shutil.rmtree(work_dir(), ignore_errors=True)
