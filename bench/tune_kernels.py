#!/usr/bin/env python3
"""Step 55: tune the engine GEMV and MoE kernels, freeze the per-shape table, judge the GEMV gate.

  tune_kernels.py [--write PATH] [--out DIR] [--shapes quick|all] [--moe/--no-moe]

For every shape: build the candidate list (<= 256 after pruning, plus the untuned formula and the
step-48 override as baselines), reject configs with `metadata.shared > 101,376` or an error above
1e-4 against the fp64 reference, prune to the 5 fastest by `triton.testing.do_bench`, then decide
with `bench/m0/gemv_bench.py::cold_graph_us` (cold-L2 CUDA graph, user decision 2026-10-11) at R=1.
The config never depends on R; a 3-row call is checked to equal three 1-row calls bit for bit.

The gate is the kernel the layers call (`gemv_affine.py::gemv`), 4-bit gs64, synthetic 2560x2560
weights as in bench/m0prime/run.py::gate_weight, counted with step 23's byte formula: >= 121.5 GB/s.
A miss exits 3 with a DECISION line (the technology decision). Needs torch with CUDA (Spark only).
"""
import argparse
import itertools
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "bench" / "m0"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SMEM_LIMIT = 101_376
GATE_GBS = 121.5
GATE_SHAPE = (2560, 2560, 4, 64)
MAX_CONFIGS = 256
MAX_KITER = 24      # unrolled K iterations per program: Triton compile time grows with it (k=6144: 10 s at 24, 42 s at 48, 191 s at 96)
ERR_LIMIT = 1e-4
CHECK_ROWS = 4096

# (n, k, bits, gs): the shapes of the M0' table (db/records/measurements/m0prime.md) plus the gate shape.
SHAPES = [(2560, 2560, 4, 64), (10240, 2560, 4, 64), (6144, 2560, 4, 64), (2560, 6144, 4, 64),
          (12288, 2560, 4, 64), (512, 2560, 4, 64), (320, 10240, 4, 64), (10240, 320, 4, 64),
          (640, 2560, 4, 64), (2560, 640, 4, 64), (248320, 2560, 4, 64),
          (512, 2560, 16, 0), (640, 2560, 16, 0)]
QUICK = [(2560, 2560, 4, 64), (640, 2560, 4, 64)]

BN_SET, BK_SET, WARPS, STAGES = (4, 8, 16, 32, 64), (64, 128, 256, 512), (2, 4, 8), (1, 2, 3)
SPLIT_SET, SPLIT_SET_HEAD = (1, 2, 4, 8), (1, 2, 4)
TILE_PAIRS = ((2, 2), (4, 2), (4, 3), (8, 3))      # (warps, stages) crossed with each kept tiling


def gemv_bytes(n, k, bits, gs):
    if bits == 16:
        return n * k * 2 + 2 * k + 4 * n
    return n * k * bits / 8 + 2 * (n * k // gs) * 2 + 2 * k + 4 * n


def gbs_of(nbytes, us):
    return nbytes / (us * 1e-6) / 1e9


def make_weight(torch, n, k, bits, gs, seed=0):
    from slotstream_spark.kernels.gemv_affine import AffineWeight, DenseWeight
    g = torch.Generator(device="cuda").manual_seed(seed)
    if bits == 16:
        return DenseWeight((torch.randn(n, k, device="cuda", generator=g) * 0.05).to(torch.bfloat16))
    q = torch.randint(-2**31, 2**31 - 1, (n, k * bits // 32), device="cuda", dtype=torch.int32, generator=g)
    s = (torch.rand(n, k // gs, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    b = (torch.rand(n, k // gs, device="cuda", generator=g) * 0.02 - 0.01).to(torch.bfloat16)
    return AffineWeight(q, s, b, bits, gs, n, k)


def reference(torch, w, x):
    """fp64 reference of the first CHECK_ROWS rows (rows are independent)."""
    from slotstream_spark.kernels import affine_ref
    from slotstream_spark.kernels.gemv_affine import AffineWeight
    m = min(w.n, CHECK_ROWS)
    if isinstance(w, AffineWeight):
        deq = affine_ref.dequant(w.q[:m], w.s[:m], w.b[:m], w.bits, w.gs)
    else:
        deq = w.w[:m]
    return affine_ref.gemv_ref64(x, deq), m


def candidates(n, k, bits, gs):
    """<= MAX_CONFIGS tuned candidates: the tilings closest to 192 programs, each x len(TILE_PAIRS) (warps, stages).
    The v3 body masks the K tail, so there is no k % bk filter; bn * bk <= 16384 is its register budget."""
    from slotstream_spark.kernels import tuning
    splits = SPLIT_SET_HEAD if n > 100000 else SPLIT_SET
    tilings = []
    for bn, bk, sk in itertools.product(BN_SET, BK_SET, splits):
        if bk > k or (bits != 16 and bk % gs) or sk > k // bk or bn * bk > 16384 or -(-(-(-k // bk)) // sk) > MAX_KITER:
            continue
        progs = -(-n // bn) * sk
        if progs < 16:
            continue
        tilings.append((abs(math.log2(progs / tuning.TARGET_PROGRAMS)), -bk, bn, sk, bk))
    tilings.sort()
    out, seen = [], set()
    for _, _, bn, sk, bk in tilings[:MAX_CONFIGS // len(TILE_PAIRS)]:
        for nw, st in TILE_PAIRS:
            out.append({"block_n": bn, "block_k": bk, "split_k": sk, "num_warps": nw, "num_stages": st})
    base = [tuning.formula_config(n, k)]
    if (n, k, bits, gs) in tuning.OVERRIDES:
        base.append(dict(tuning.OVERRIDES[(n, k, bits, gs)]))
    allc = []
    for c in out + base:
        t = tuple(sorted(c.items()))
        if t not in seen:
            seen.add(t)
            allc.append(c)
    return allc, base


def cold_us(torch, gemv_bench, fn):
    return gemv_bench.cold_graph_us(torch, fn)


def tune_shape(torch, tt, gemv_bench, shape, log):
    from slotstream_spark.kernels import gemv_affine as ga
    n, k, bits, gs = shape
    w = make_weight(torch, n, k, bits, gs)
    gx = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(1, k, device="cuda", generator=gx).to(torch.bfloat16)
    x3 = torch.randn(3, k, device="cuda", generator=gx).to(torch.bfloat16)
    ref, m = reference(torch, w, x)
    cands, baselines = candidates(*shape)
    kname = "gemv_dense" if bits == 16 else f"gemv_affine{bits}"
    nbytes = gemv_bytes(*shape)
    scored = []
    for cfg in cands:
        try:
            y = ga.gemv(x, w, cfg=cfg)
            torch.cuda.synchronize()
            shared = ga.SHARED_BYTES.get(kname, 0)
            if shared > SMEM_LIMIT:
                log(f"  reject smem {shared} {cfg}")
                continue
            err = ((y[:, :m].double() - ref).norm() / ref.norm()).item()
            if err > ERR_LIMIT:
                log(f"  reject err {err:.2e} {cfg}")
                continue
            ms = tt.do_bench(lambda: ga.gemv(x, w, cfg=cfg), warmup=10, rep=40)
        except Exception as e:
            log(f"  skip {cfg}: {type(e).__name__}: {str(e)[:90]}")
            continue
        scored.append({"cfg": cfg, "do_bench_us": ms * 1000, "err": err, "shared": shared})
    if not scored:
        raise RuntimeError(f"no valid config for {shape}")
    scored.sort(key=lambda r: r["do_bench_us"])
    top = scored[:5]
    # the baselines are always timed cold as well, so the table can only improve on them
    keys = {tuple(sorted(r["cfg"].items())) for r in top}
    for r in scored[5:]:
        if r["cfg"] in baselines and tuple(sorted(r["cfg"].items())) not in keys:
            top.append(r)
    for r in top:
        r["cold_us"] = cold_us(torch, gemv_bench, lambda c=r["cfg"]: ga.gemv(x, w, cfg=c))
        r["cold_gbs"] = gbs_of(nbytes, r["cold_us"]) if r["cold_us"] > 0 else 0.0
        log(f"  {r['cfg']} do_bench {r['do_bench_us']:.2f} us cold {r['cold_us']:.2f} us {r['cold_gbs']:.1f} GB/s")
    best = max(top, key=lambda r: r["cold_gbs"])
    # confirm: median of three cold timings of the winner
    reps = [best["cold_gbs"]] + [gbs_of(nbytes, cold_us(torch, gemv_bench, lambda: ga.gemv(x, w, cfg=best["cfg"])))
                                 for _ in range(2)]
    gbs = statistics.median(reps)
    # row invariance with this config: 3 rows == three 1-row calls
    y3 = ga.gemv(x3, w, cfg=best["cfg"])
    rows_equal = all(bool(torch.equal(ga.gemv(x3[r:r + 1], w, cfg=best["cfg"])[0], y3[r])) for r in range(3))
    ga.gemv(x, w, cfg=best["cfg"])
    shared = ga.SHARED_BYTES.get(kname, 0)
    base_rows = [r for r in top if r["cfg"] in baselines]
    out = {"config": best["cfg"], "gbs_cold_graph": round(gbs, 2), "cold_graph_us": round(nbytes / (gbs * 1e3), 2),
           "gbs_cold_graph_reps": [round(v, 2) for v in reps], "do_bench_us": round(best["do_bench_us"], 2),
           "do_bench_gbs": round(gbs_of(nbytes, best["do_bench_us"]), 2), "rel_err": best["err"],
           "shared_bytes": shared, "rows3_bitwise": rows_equal, "bytes": int(nbytes), "candidates": len(cands),
           "baselines": [{"config": r["cfg"], "gbs_cold_graph": round(r["cold_gbs"], 2)} for r in base_rows]}
    del w
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------------------------------- MoE
def moe_inputs(torch, r):
    from slotstream_spark.kernels import moe_gemv_affine as moe
    g = torch.Generator().manual_seed(11)
    n_exp, n_slots = 16, 16
    pool = torch.randint(0, 256, (n_slots, moe.RECORD_BYTES), dtype=torch.uint8, generator=g)
    # sane scales/biases: overwrite s and b regions with small bf16 values
    for off, cnt in ((moe.O_GATE_S, 640 * 40), (moe.O_GATE_B, 640 * 40), (moe.O_UP_S, 640 * 40),
                     (moe.O_UP_B, 640 * 40), (moe.O_DOWN_S, 2560 * 10), (moe.O_DOWN_B, 2560 * 10)):
        vals = (torch.randn(n_slots, cnt, generator=g) * 0.02).to(torch.bfloat16).view(torch.uint8).reshape(n_slots, -1)
        pool[:, off:off + cnt * 2] = vals
    pool = pool.pin_memory()
    mp = torch.full((49 * 512,), -1, dtype=torch.int32)
    for e in range(n_exp):
        mp[3 * 512 + e] = e
    ptrs = torch.tensor([pool.data_ptr() + i * moe.RECORD_BYTES for i in range(n_slots)], dtype=torch.int64)
    x = torch.randn(r, 2560, generator=g).to(torch.bfloat16).cuda()
    ids = torch.stack([torch.randperm(n_exp, generator=g)[:10] for _ in range(r)]).to(torch.int32).cuda()
    w = torch.softmax(torch.randn(r, 10, generator=g), dim=-1).cuda()
    return pool, mp.cuda(), ptrs.cuda(), x, ids, w


def tune_moe(torch, tt, gemv_bench, log):
    from slotstream_spark.kernels import moe_gemv_affine as moe
    from slotstream_spark.kernels import gemv_affine as ga
    from slotstream_spark.kernels import tuning
    pool, mp, ptrs, x, ids, w = moe_inputs(torch, 1)
    h = torch.empty(1, 10, 640, device="cuda")
    o = torch.empty(1, 10, 2560, device="cuda")
    out = torch.empty(1, 2560, device="cuda")
    pool3, mp3, ptrs3, x3, ids3, w3 = moe_inputs(torch, 3)
    h3 = torch.empty(3, 10, 640, device="cuda")
    o3 = torch.empty(3, 10, 2560, device="cuda")
    out3 = torch.empty(3, 2560, device="cuda")
    ref_cfg = dict(tuning.MOE_DEFAULT)
    moe.moe_decode(x, ids, w, 3, mp, ptrs, h, o, out, cfg=ref_cfg)
    torch.cuda.synchronize()
    ref = out.clone()
    rec_bytes = 10 * (2 * (640 * 2560 // 2 + 2 * 640 * 40 * 2) + 2560 * 640 // 2 + 2 * 2560 * 10 * 2)
    cands = [dict(bn=bn, br=br, bk=bk, bkd=bkd, num_warps=nw)
             for bn, br, bk, bkd, nw in itertools.product((4, 8, 16), (8, 16, 32), (128, 256, 512), (128, 256), (2, 4))]
    cands.sort(key=lambda c: (c != ref_cfg, abs(math.log2(c["bn"] / ref_cfg["bn"])) + abs(math.log2(c["br"] / ref_cfg["br"]))
                              + abs(math.log2(c["bk"] / ref_cfg["bk"])) + abs(math.log2(c["bkd"] / ref_cfg["bkd"]))
                              + abs(math.log2(c["num_warps"] / ref_cfg["num_warps"]))))
    cands = cands[:MAX_CONFIGS]
    scored = []
    for c in cands:
        try:
            moe.moe_decode(x, ids, w, 3, mp, ptrs, h, o, out, cfg=c)
            torch.cuda.synchronize()
            sh = max(ga.SHARED_BYTES.get("moe_gate_up", 0), ga.SHARED_BYTES.get("moe_down", 0))
            if sh > SMEM_LIMIT:
                log(f"  moe reject smem {sh} {c}")
                continue
            err = ((out.double() - ref.double()).norm() / ref.double().norm()).item()
            if err > ERR_LIMIT:
                log(f"  moe reject err {err:.2e} {c}")
                continue
            ms = tt.do_bench(lambda: moe.moe_decode(x, ids, w, 3, mp, ptrs, h, o, out, cfg=c), warmup=10, rep=40)
        except Exception as e:
            log(f"  moe skip {c}: {type(e).__name__}: {str(e)[:90]}")
            continue
        scored.append({"cfg": c, "do_bench_us": ms * 1000, "err": err, "shared": sh})
    scored.sort(key=lambda r: r["do_bench_us"])
    top = scored[:5]
    if not any(r["cfg"] == ref_cfg for r in top):
        top += [r for r in scored if r["cfg"] == ref_cfg]
    for r in top:
        r["cold_us"] = cold_us(torch, gemv_bench,
                               lambda c=r["cfg"]: moe.moe_decode(x, ids, w, 3, mp, ptrs, h, o, out, cfg=c))
        log(f"  moe {r['cfg']} do_bench {r['do_bench_us']:.2f} us cold {r['cold_us']:.2f} us "
            f"({gbs_of(rec_bytes, r['cold_us']):.1f} GB/s)")
    best = min(top, key=lambda r: r["cold_us"])
    base = next((r for r in top if r["cfg"] == ref_cfg), None)
    c = best["cfg"]
    y3 = moe.moe_decode(x3, ids3, w3, 3, mp3, ptrs3, h3, o3, out3, cfg=c).clone()
    rows_equal = True
    for r in range(3):
        o1 = torch.empty(1, 10, 2560, device="cuda")
        h1 = torch.empty(1, 10, 640, device="cuda")
        out1 = torch.empty(1, 2560, device="cuda")
        moe.moe_decode(x3[r:r + 1].contiguous(), ids3[r:r + 1].contiguous(), w3[r:r + 1].contiguous(), 3, mp3, ptrs3,
                       h1, o1, out1, cfg=c)
        rows_equal &= bool(torch.equal(out1[0], y3[r]))
    return {"config": c, "cold_graph_us": round(best["cold_us"], 2), "gbs_cold_graph": round(gbs_of(rec_bytes, best["cold_us"]), 2),
            "do_bench_us": round(best["do_bench_us"], 2), "rel_err_vs_default": best["err"],
            "shared_bytes": best["shared"], "rows3_bitwise": rows_equal,
            "default": None if base is None else {"config": ref_cfg, "cold_graph_us": round(base["cold_us"], 2)},
            "bytes_per_step": rec_bytes}


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--write")
    ap.add_argument("--out")
    ap.add_argument("--shapes", choices=("quick", "all"), default="all")
    ap.add_argument("--no-moe", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    import triton
    import triton.testing as tt
    import gemv_bench

    def log(s):
        print(s, flush=True)

    shapes = QUICK if a.shapes == "quick" else SHAPES
    res = {"shapes": {}, "versions": {"torch": torch.__version__, "triton": triton.__version__,
                                       "cuda": torch.version.cuda},
           "device": torch.cuda.get_device_name(0), "smem_limit": SMEM_LIMIT, "gate_threshold_gbs": GATE_GBS}
    from slotstream_spark.kernels import tuning
    for shape in shapes:
        log(f"shape {shape}")
        r = tune_shape(torch, tt, gemv_bench, shape, log)
        res["shapes"][tuning.shape_key(*shape)] = r
        log(f"  -> {r['config']} {r['gbs_cold_graph']} GB/s cold-L2 graph (do_bench {r['do_bench_gbs']}), "
            f"rows3_bitwise {r['rows3_bitwise']}, smem {r['shared_bytes']}")
    if not a.no_moe:
        log("moe")
        res["moe"] = tune_moe(torch, tt, gemv_bench, log)
        log(f"  -> {res['moe']['config']} {res['moe']['cold_graph_us']} us")
    gate = res["shapes"].get(tuning.shape_key(*GATE_SHAPE))
    res["gate"] = None if gate is None else {"shape": list(GATE_SHAPE), "gbs_cold_graph": gate["gbs_cold_graph"],
                                              "threshold": GATE_GBS, "pass": gate["gbs_cold_graph"] >= GATE_GBS}
    res["elapsed_s"] = round(time.time() - t0, 1)
    bad = [k for k, v in res["shapes"].items() if not v["rows3_bitwise"] or v["shared_bytes"] > SMEM_LIMIT]
    if res.get("moe") and (not res["moe"]["rows3_bitwise"] or res["moe"]["shared_bytes"] > SMEM_LIMIT):
        bad.append("moe")
    res["invariants_ok"] = not bad
    text = json.dumps(res, indent=1)
    out = a.out or os.environ.get("CAP")
    if out:
        os.makedirs(out, exist_ok=True)
        Path(out, "result.json").write_text(text)
    if a.write and a.shapes == "all" and not bad:
        Path(a.write).write_text(text + "\n")
        print(f"wrote {a.write}")
    print(json.dumps({"gate": res["gate"], "invariants_ok": res["invariants_ok"], "elapsed_s": res["elapsed_s"]}))
    if bad:
        print(f"INVARIANT FAILED: {bad}", file=sys.stderr)
        return 1
    if res["gate"] is not None and not res["gate"]["pass"]:
        print(f"DECISION: GEMV 2560x2560 4-bit = {res['gate']['gbs_cold_graph']} GB/s < {GATE_GBS} after tuning "
              "→ plan B (qmv.cu port) / C (MMVQ Q4_1) (the technology decision)")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
