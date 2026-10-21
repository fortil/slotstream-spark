#!/usr/bin/env python3
"""M0 microbenchmark: affine 4-bit GEMV in Triton (not the F2 kernel) and the MoE read test.

  gemv_bench.py [--quick] [--moe] [--out DIR]

Writes $CAP/result.json (or --out DIR/result.json):
  {shapes: {"NxK": {gbs, frac_of_243, config, rel_err}}, moe_read: {device_gbs, pinned_gbs}, versions}
Needs torch with CUDA and triton (Spark only).
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

PEAK_GBS = 243.0
GROUP = 64
K = 2560
SHAPES_N = (640, 1280, 2560, 248320)
SLOT_BYTES = 2_764_800


def build_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def gemv_affine4_kernel(w_ptr, s_ptr, b_ptr, x_ptr, part_ptr, N, K,
                            GROUP: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                            SPLIT_K: tl.constexpr):
        # y[n] += s[n,g] * sum_{k in g} q[n,k] x[k] + b[n,g] * sum_{k in g} x[k]
        pid_n = tl.program_id(0)
        pid_s = tl.program_id(1)
        rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rmask = rows < N
        k_per = K // SPLIT_K
        k_begin = pid_s * k_per
        NG: tl.constexpr = BLOCK_K // GROUP
        WPG: tl.constexpr = GROUP // 8          # 32-bit words per group
        NW: tl.constexpr = BLOCK_K // 8
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for k0 in range(k_begin, k_begin + k_per, BLOCK_K):
            ws = k0 // 8 + tl.arange(0, NW)
            words = tl.load(w_ptr + rows[:, None] * (K // 8) + ws[None, :],
                            mask=rmask[:, None], other=0).to(tl.uint32)
            gq = tl.zeros([BLOCK_N, NG], dtype=tl.float32)
            gx = tl.zeros([NG], dtype=tl.float32)
            for j in tl.static_range(8):
                xj = tl.load(x_ptr + ws * 8 + j).to(tl.float32)
                qj = ((words >> (4 * j)) & 0xF).to(tl.float32)
                gq += tl.sum(tl.reshape(qj * xj[None, :], [BLOCK_N, NG, WPG]), axis=2)
                gx += tl.sum(tl.reshape(xj, [NG, WPG]), axis=1)
            gs = k0 // GROUP + tl.arange(0, NG)
            sc = tl.load(s_ptr + rows[:, None] * (K // GROUP) + gs[None, :],
                         mask=rmask[:, None], other=0.0).to(tl.float32)
            bi = tl.load(b_ptr + rows[:, None] * (K // GROUP) + gs[None, :],
                         mask=rmask[:, None], other=0.0).to(tl.float32)
            acc += tl.sum(sc * gq + bi * gx[None, :], axis=1)
        tl.store(part_ptr + pid_s * N + rows, acc, mask=rmask)

    @triton.jit
    def gemv_affine4_v2_kernel(w_ptr, s_ptr, b_ptr, x_ptr, part_ptr, N, K,
                               GROUP: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                               SPLIT_K: tl.constexpr):
        # Tile layout: one wide 2D load of the packed words, nibbles expanded in a third axis,
        # tail masked (BLOCK_K need not divide the split). More bytes in flight per program.
        pid_n = tl.program_id(0)
        pid_s = tl.program_id(1)
        rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rmask = rows < N
        k_per = K // SPLIT_K
        k_begin = pid_s * k_per
        k_end = k_begin + k_per
        NG: tl.constexpr = BLOCK_K // GROUP
        NW: tl.constexpr = BLOCK_K // 8
        sh = tl.arange(0, 8) * 4
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for k0 in range(k_begin, k_end, BLOCK_K):
            wi = k0 // 8 + tl.arange(0, NW)
            wmask = wi * 8 < k_end
            words = tl.load(w_ptr + rows[:, None] * (K // 8) + wi[None, :],
                            mask=rmask[:, None] & wmask[None, :], other=0).to(tl.uint32)
            kk = wi[:, None] * 8 + tl.arange(0, 8)[None, :]
            xt = tl.load(x_ptr + kk, mask=wmask[:, None], other=0.0).to(tl.float32)
            q = ((words[:, :, None] >> sh[None, None, :]) & 0xF).to(tl.float32)
            gq = tl.sum(tl.reshape(q * xt[None, :, :], [BLOCK_N, NG, GROUP]), axis=2)
            gx = tl.sum(tl.reshape(xt, [NG, GROUP]), axis=1)
            gs = k0 // GROUP + tl.arange(0, NG)
            gmask = gs * GROUP < k_end
            sc = tl.load(s_ptr + rows[:, None] * (K // GROUP) + gs[None, :],
                         mask=rmask[:, None] & gmask[None, :], other=0.0).to(tl.float32)
            bi = tl.load(b_ptr + rows[:, None] * (K // GROUP) + gs[None, :],
                         mask=rmask[:, None] & gmask[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(sc * gq + bi * gx[None, :], axis=1)
        tl.store(part_ptr + pid_s * N + rows, acc, mask=rmask)

    @triton.jit
    def gemv_affine4_v3_kernel(w_ptr, s_ptr, b_ptr, x_ptr, part_ptr, N, K,
                               GROUP: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                               SPLIT_K: tl.constexpr, KITER: tl.constexpr):
        # v1 body with the K loop fully unrolled (KITER chunks): every chunk's loads are
        # independent of the accumulator, so they can all be in flight before the math starts.
        pid_n = tl.program_id(0)
        pid_s = tl.program_id(1)
        rows = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rmask = rows < N
        k_begin = pid_s * (K // SPLIT_K)
        NG: tl.constexpr = BLOCK_K // GROUP
        WPG: tl.constexpr = GROUP // 8
        NW: tl.constexpr = BLOCK_K // 8
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for it in tl.static_range(KITER):
            k0 = k_begin + it * BLOCK_K
            ws = k0 // 8 + tl.arange(0, NW)
            words = tl.load(w_ptr + rows[:, None] * (K // 8) + ws[None, :],
                            mask=rmask[:, None], other=0).to(tl.uint32)
            gs = k0 // GROUP + tl.arange(0, NG)
            sc = tl.load(s_ptr + rows[:, None] * (K // GROUP) + gs[None, :],
                         mask=rmask[:, None], other=0.0).to(tl.float32)
            bi = tl.load(b_ptr + rows[:, None] * (K // GROUP) + gs[None, :],
                         mask=rmask[:, None], other=0.0).to(tl.float32)
            gq = tl.zeros([BLOCK_N, NG], dtype=tl.float32)
            gx = tl.zeros([NG], dtype=tl.float32)
            for j in tl.static_range(8):
                xj = tl.load(x_ptr + ws * 8 + j).to(tl.float32)
                qj = ((words >> (4 * j)) & 0xF).to(tl.float32)
                gq += tl.sum(tl.reshape(qj * xj[None, :], [BLOCK_N, NG, WPG]), axis=2)
                gx += tl.sum(tl.reshape(xj, [NG, WPG]), axis=1)
            acc += tl.sum(sc * gq + bi * gx[None, :], axis=1)
        tl.store(part_ptr + pid_s * N + rows, acc, mask=rmask)

    @triton.jit
    def gather_kernel(pool_ptr, idx_ptr, out_ptr, SLOT_WORDS: tl.constexpr, NCHUNK: tl.constexpr,
                      BLOCK: tl.constexpr):
        s = tl.program_id(0)
        c = tl.program_id(1)
        slot = tl.load(idx_ptr + s)
        offs = c * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(pool_ptr + slot * SLOT_WORDS + offs, mask=offs < SLOT_WORDS, other=0)
        tl.store(out_ptr + s * NCHUNK + c, tl.xor_sum(v, axis=0))

    return {1: gemv_affine4_kernel, 2: gemv_affine4_v2_kernel, 3: gemv_affine4_v3_kernel}, gather_kernel


def gen_weights(torch, n, k, gen):
    w = torch.randint(-2**31, 2**31 - 1, (n, k // 8), device="cuda", dtype=torch.int32, generator=gen)
    s = (torch.rand(n, k // GROUP, device="cuda", generator=gen) * 0.1).to(torch.bfloat16)
    b = (torch.rand(n, k // GROUP, device="cuda", generator=gen) * 0.1 - 0.05).to(torch.bfloat16)
    x = torch.randn(k, device="cuda", generator=gen).to(torch.bfloat16)
    return w, s, b, x


def run_gemv(kern, w, s, b, x, n, k, cfg):
    import torch
    import triton
    bn, bk, sk, nw = cfg["BLOCK_N"], cfg["BLOCK_K"], cfg["SPLIT_K"], cfg["num_warps"]
    kern = kern[cfg.get("kv", 1)]
    part = torch.empty(sk, n, device="cuda", dtype=torch.float32)

    def launch():
        kern[(triton.cdiv(n, bn), sk)](w, s, b, x, part, n, k, GROUP=GROUP, BLOCK_N=bn,
                                       BLOCK_K=bk, SPLIT_K=sk, num_warps=nw,
                                       num_stages=cfg.get("num_stages", 3),
                                       **({"KITER": k // sk // bk} if cfg.get("kv") == 3 else {}))
        return part[0] if sk == 1 else torch.sum(part, dim=0)
    return launch


def reference_rows(torch, w, s, b, x, rows):
    import numpy as np
    from slotstream_spark.model.mlx_ckpt import dequant_affine_torch
    wn = w[:rows].cpu().numpy().view(np.uint32)
    sn = s[:rows].view(torch.int16).cpu().numpy().view(np.uint16)
    bn = b[:rows].view(torch.int16).cpu().numpy().view(np.uint16)
    deq = dequant_affine_torch(wn, sn, bn, bits=4, device="cuda")
    return x.float() @ deq.T


def candidate_configs(k):
    out = []
    for bn, bk, sk, nw in itertools.product((8, 16, 32, 64), (128, 256, 512), (1, 2, 4, 8, 16), (2, 4, 8)):
        if k % (sk * bk) == 0:
            out.append({"BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": sk, "num_warps": nw})
    # tuning pass (kernel v2): wide tiles, masked tails, split-K that need not divide BLOCK_K
    for bn, bk, sk, nw, st in itertools.product((1, 2, 4, 8, 16, 32), (512, 1024, 2048, 4096),
                                                (1, 2, 4, 5, 10), (2, 4, 8), (1, 3)):
        k_per = k // sk
        if k % sk or k_per % GROUP:
            continue
        if bk > 2 * k_per or bn * bk > 32768:
            continue
        if st == 3 and bk >= k_per:      # no loop to pipeline
            continue
        out.append({"kv": 2, "BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": sk, "num_warps": nw,
                    "num_stages": st})
    for bn, bk, sk, nw in itertools.product((4, 8, 16, 32, 64), (128, 256, 512), (1, 2, 4, 5),
                                            (2, 4, 8)):
        if k % (sk * bk) == 0 and k // sk // bk <= 20:
            out.append({"kv": 3, "BLOCK_N": bn, "BLOCK_K": bk, "SPLIT_K": sk, "num_warps": nw})
    return out


def graph_us(torch, launch, reps=50):
    """Informational: per-call time with `reps` launches replayed from one CUDA graph."""
    launch()
    torch.cuda.synchronize()
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        launch()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            launch()
    g.replay()
    torch.cuda.synchronize()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(5):
        g.replay()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) * 1000 / (5 * reps)


def cold_graph_us(torch, launch, reps=20, replays=15):
    """Gate timing: per-call device time inside a captured CUDA graph, L2 flushed before every call.

    Graph A holds `reps` x (flush, call); graph B holds `reps` x flush. The call time is
    (median A - median B) / reps, so the flush cost cancels.
    """
    flush = torch.empty(128 * 1024 * 1024 // 4, device="cuda", dtype=torch.int32)   # >> L2
    launch()
    torch.cuda.synchronize()
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        launch()
        flush.fill_(1)
    torch.cuda.synchronize()

    def capture(with_call):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps):
                flush.fill_(1)
                if with_call:
                    launch()
        return g

    def timed(g):
        g.replay()
        torch.cuda.synchronize()
        ts = []
        for _ in range(replays):
            t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0.record()
            g.replay()
            t1.record()
            torch.cuda.synchronize()
            ts.append(t0.elapsed_time(t1))
        ts.sort()
        return ts[len(ts) // 2]
    ga, gb = capture(True), capture(False)
    ta, tb = timed(ga), timed(gb)
    del flush
    return (ta - tb) * 1000 / reps


def gbs_of(n, k, ms):
    nbytes = n * k / 2 + 2 * n * k / GROUP * 2 + 2 * k + 4 * n
    return nbytes / (ms * 1e-3) / 1e9


def bench_shape(torch, triton_testing, kern, n, k, quick, gen):
    w, s, b, x = gen_weights(torch, n, k, gen)
    check_rows = min(n, 2048)
    ref = reference_rows(torch, w, s, b, x, check_rows)
    cfgs = candidate_configs(k)
    if quick:
        cfgs = [{"BLOCK_N": 16, "BLOCK_K": 256, "SPLIT_K": 2, "num_warps": 4},
                {"kv": 2, "BLOCK_N": 4, "BLOCK_K": 2048, "SPLIT_K": 1, "num_warps": 4, "num_stages": 1}]
    elif n > 100000:
        cfgs = [c for c in cfgs if c["SPLIT_K"] == 1]
    best, best_ms, best_err = None, float("inf"), None
    scored = []
    for cfg in cfgs:
        try:
            launch = run_gemv(kern, w, s, b, x, n, k, cfg)
            y = launch()
            torch.cuda.synchronize()
            err = ((y[:check_rows] - ref).abs().max() / ref.abs().max()).item()
            if err > 1e-3:
                if quick:
                    raise AssertionError(f"GEMV mismatch {n}x{k} {cfg}: rel err {err:.2e}")
                continue
            ms = triton_testing.do_bench(launch, warmup=10 if quick else 25, rep=30 if quick else 100)
        except AssertionError:
            raise
        except Exception as e:  # config does not compile or launch
            print(f"skip {cfg}: {type(e).__name__}: {str(e)[:100]}", file=sys.stderr)
            continue
        scored.append((ms, cfg, err))
        if ms < best_ms:
            best, best_ms, best_err = cfg, ms, err
    if best is None:
        raise AssertionError(f"no correct GEMV config for {n}x{k}")
    g = gbs_of(n, k, best_ms)
    gus = None
    try:
        gus = graph_us(torch, run_gemv(kern, w, s, b, x, n, k, best))
    except Exception as e:
        print(f"graph timing failed: {type(e).__name__}: {str(e)[:100]}", file=sys.stderr)
    # Gate value: cold-L2 graph timing. Re-time the 5 best do_bench configs, keep the fastest.
    cold_us, cold_cfg, cold_err = None, None, None
    scored.sort(key=lambda r: r[0])
    for _, cfg, err in scored[:5]:
        try:
            us = cold_graph_us(torch, run_gemv(kern, w, s, b, x, n, k, cfg))
        except Exception as e:
            print(f"cold graph failed {cfg}: {type(e).__name__}: {str(e)[:100]}", file=sys.stderr)
            continue
        if us > 0 and (cold_us is None or us < cold_us):
            cold_us, cold_cfg, cold_err = us, cfg, err
    if cold_us is None:
        raise AssertionError(f"no cold-graph timing for {n}x{k}")
    cold_g = gbs_of(n, k, cold_us * 1e-3)
    del w, s, b
    torch.cuda.empty_cache()
    return {"gbs_cold_graph": round(cold_g, 2), "frac_cold_graph": round(cold_g / PEAK_GBS, 4),
            "cold_graph_us_per_call": round(cold_us, 2), "cold_graph_config": cold_cfg,
            "cold_graph_rel_err": cold_err, "gbs": round(g, 2), "frac_of_243": round(g / PEAK_GBS, 4), "config": best,
            "rel_err": best_err, "ms": best_ms,
            "graph_us_per_call": None if gus is None else round(gus, 2),
            "graph_gbs": None if gus is None else round(gbs_of(n, k, gus * 1e-3), 2)}


def bench_moe(torch, triton, triton_testing, gather_kernel, quick):
    words = SLOT_BYTES // 4
    nslots, nread, block = 64, 10, 4096
    nchunk = triton.cdiv(words, block)
    idx = torch.randperm(nslots, device="cuda")[:nread].to(torch.int32)
    out = torch.empty(nread * nchunk, device="cuda", dtype=torch.int32)
    res = {"slot_bytes": SLOT_BYTES, "slots_read": nread}

    def measure(pool):
        def launch():
            gather_kernel[(nread, nchunk)](pool, idx, out, SLOT_WORDS=words, NCHUNK=nchunk, BLOCK=block)
        launch()
        torch.cuda.synchronize()
        ms = triton_testing.do_bench(launch, warmup=10 if quick else 25, rep=30 if quick else 100)
        return nread * SLOT_BYTES / (ms * 1e-3) / 1e9

    dev = torch.randint(-2**31, 2**31 - 1, (nslots * words,), device="cuda", dtype=torch.int32)
    res["device_gbs"] = round(measure(dev), 2)
    del dev
    try:
        pin = torch.empty(nslots * words, dtype=torch.int32, pin_memory=True)
        pin.random_(-2**31, 2**31 - 1)
        res["pinned_gbs"] = round(measure(pin), 2)
    except Exception as e:
        res["pinned_gbs"] = None
        res["pinned_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return res


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--moe", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    t_start = time.time()

    import torch
    import triton
    import triton.testing
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    kern, gather = build_kernels()
    gen = torch.Generator(device="cuda").manual_seed(0)

    shapes = (2560,) if a.quick else SHAPES_N
    result = {"shapes": {}, "versions": {"torch": torch.__version__, "triton": triton.__version__,
                                           "cuda": torch.version.cuda}}
    for n in shapes:
        r = bench_shape(torch, triton.testing, kern, n, K, a.quick, gen)
        result["shapes"][f"{n}x{K}"] = r
        print(f"{n}x{K}: cold-graph {r['gbs_cold_graph']} GB/s ({100 * r['frac_cold_graph']:.1f}% of 243, gate); do_bench {r['gbs']} GB/s ({100 * r['frac_of_243']:.1f}% of 243; graph {r['graph_gbs']}) rel_err {r['rel_err']:.2e} {r['config']}")
    if a.moe or a.quick:
        result["moe_read"] = bench_moe(torch, triton, triton.testing, gather, a.quick)
        print("moe_read:", json.dumps(result["moe_read"]))
    result["elapsed_s"] = round(time.time() - t_start, 1)

    out = a.out or os.environ.get("CAP")
    if out:
        os.makedirs(out, exist_ok=True)
        Path(out, "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("shapes", "versions")}))
    if a.quick and result["elapsed_s"] > 60:
        print(f"quick run took {result['elapsed_s']} s (> 60 s)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
