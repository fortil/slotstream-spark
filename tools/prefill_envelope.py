#!/usr/bin/env python3
"""CUDA transient envelope of the prefill (step 69): what one pass costs in device memory and time.

Grid: pass {256, 512, 1024, 2048, 4096} x context {2K, 4K, 8K, 16K, 32K}. For every cell it measures

  attention  one QSA layer's store + select + sparse attention for the pass at that context (synthetic bf16 inputs; the
             memory and the time of these ops depend on the shapes, not on the values)
  moe        the grouped affine MoE of a pass (R = 10 x pass rows, 32 synthetic records) plus the rank-order combine

as `torch.cuda.max_memory_allocated` delta over the inputs and the NVML process delta, and the CUDA-event duration of
each launch after one untimed warm-up call per shape (the call that compiles Triton is never a sample).

Fits (least squares on the cells, non-negative): attention transient = bytes_per_pair x q x k + attn_bytes_per_token x q,
MoE transient = moe_bytes_per_token x q. `bytes_per_pass_token` is the sum of the two per-token terms.

`qk_product` = the largest product q x k (q = pass, k = context + pass) such that every cell with a product up to it fits
the D5 prefill ledger line (pass_bytes of Ledger('D5'), 5.32 GB) with its measured attention + MoE transient and takes at
most 200 ms per launch (the kernel watchdog is `KernelExecTimeout: 1`). `max_rows_per_launch` = the largest measured
row count of a grouped-MoE launch that takes at most 200 ms, capped by a 1 GiB temporary (81,920 rows).

  tools/prefill_envelope.py --write      writes db/records/measurements/f5-prefill-envelope.json (and $CAP/result.json)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "db" / "records" / "measurements" / "f5-prefill-envelope.json"
PASSES = (256, 512, 1024, 2048, 4096)
CONTEXTS = (2048, 4096, 8192, 16384, 32768)
ROW_COUNTS = (4096, 40960, 81920, 163840)
WATCHDOG_MS = 200.0
ROWS_TEMP_CAP = 81920            # 1 GiB of out_rows + h per launch at 11.5 KB a row


def nvml_used():
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
            if p.pid == os.getpid() and p.usedGpuMemory is not None:
                return int(p.usedGpuMemory)
    except Exception:
        pass
    return 0


class Meter:
    """Runs fn() once untimed (warm-up: compile, allocator), then once measured; -> (alloc_delta, nvml_delta, ms)."""

    def __init__(self, torch):
        self.t = torch

    def run(self, fn, *, setup=None):
        t = self.t
        for _ in range(2):
            state = setup() if setup else None
            fn(state)
            t.cuda.synchronize()
            del state
        t.cuda.empty_cache()
        state = setup() if setup else None
        t.cuda.synchronize()
        base = t.cuda.memory_allocated()
        nv0 = nvml_used()
        t.cuda.reset_peak_memory_stats()
        a, b = t.cuda.Event(enable_timing=True), t.cuda.Event(enable_timing=True)
        a.record()
        fn(state)
        b.record()
        t.cuda.synchronize()
        res = (int(t.cuda.max_memory_allocated() - base), max(0, nvml_used() - nv0), float(a.elapsed_time(b)))
        del state
        return res


def attention_cell(torch, vendored, meter, q, ctx):
    dev = "cuda"
    s = vendored.QSAShapes()
    cache = vendored.QSACache(ctx + q + 64, dev)
    one = lambda n: torch.ones(n, dtype=torch.float32, device=dev)

    def setup():
        g = torch.Generator(device=dev).manual_seed(1729)
        qkv = torch.randn(q, (2 * s.num_q_heads + 2 * s.num_kv_heads) * s.head_dim, generator=g, device=dev).to(torch.bfloat16)
        iq = torch.randn(q, s.index_heads * s.index_dim, generator=g, device=dev).to(torch.bfloat16)
        ik = torch.randn(q, s.index_dim, generator=g, device=dev).to(torch.bfloat16)
        return qkv, iq, ik

    def fn(inp):
        qkv, iq, ik = inp
        cache.pos = ctx
        q_idx, qq, gate = vendored.qsa_store_kv(cache, qkv, iq, ik, ctx, one(s.head_dim), one(s.head_dim),
                                                one(s.index_dim), one(s.index_dim))
        vendored.qsa_prefill(cache, q_idx, qq, gate)

    return meter.run(fn, setup=setup)


def moe_setup(torch, rows_tokens):
    from slotstream_spark.kernels import fused_moe_affine as fm
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(1729)
    records = torch.randint(0, 256, (fm.GROUP_MAX, fm.RECORD_BYTES), dtype=torch.uint8, device=dev, generator=g)
    base = records.data_ptr()
    group_ptr = torch.tensor([base + i * fm.RECORD_BYTES for i in range(fm.GROUP_MAX)], dtype=torch.int64, device=dev)
    expert_map = torch.full((512,), -1, dtype=torch.int32, device=dev)
    expert_map[:fm.GROUP_MAX] = torch.arange(fm.GROUP_MAX, dtype=torch.int32, device=dev)
    return fm, records, group_ptr, expert_map


def moe_inputs(torch, fm, tokens, rows):
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(7)
    x = torch.randn(tokens, fm.HID, generator=g, device=dev).to(torch.bfloat16)
    row_token = (torch.arange(rows, device=dev) % tokens).to(torch.int32)
    row_weight = torch.rand(rows, generator=g, device=dev)
    row_expert = (torch.arange(rows, device=dev) % fm.GROUP_MAX).to(torch.int32)
    return x, row_token, row_weight, row_expert


def moe_cell(torch, meter, q):
    fm, records, group_ptr, expert_map = moe_setup(torch, q)
    rows = q * fm.TOPK
    res = []

    def setup():
        return moe_inputs(torch, fm, q, rows)

    def fn(inp):
        x, row_token, row_weight, row_expert = inp
        out_rows = torch.empty(rows, fm.HID, dtype=torch.float32, device=x.device)
        fm.fused_moe_affine(x, row_token, row_weight, expert_map, group_ptr, out_rows, row_expert=row_expert)
        fm.combine_rank_order(out_rows[:(rows // fm.TOPK) * fm.TOPK], rows // fm.TOPK)

    res = meter.run(fn, setup=setup)
    del records
    return res


def launch_times(torch, rows):
    """Max of 3 timed calls of each of the two launches at `rows` rows (after one untimed warm-up of the shape)."""
    fm, records, group_ptr, expert_map = moe_setup(torch, rows)
    x, row_token, row_weight, row_expert = moe_inputs(torch, fm, min(rows, 8192), rows)
    row_token = (torch.arange(rows, device="cuda") % x.shape[0]).to(torch.int32)
    aligned = fm.align_rows(row_expert, expert_map, fm.GROUP_MAX)
    h = torch.empty(rows, fm.INTER, dtype=torch.bfloat16, device="cuda")
    out = torch.empty(rows, fm.HID, dtype=torch.float32, device="cuda")
    fm.launch_gate_up(x, row_token, aligned, group_ptr, h, rows)
    fm.launch_down(h, row_weight, aligned, group_ptr, out, rows)
    torch.cuda.synchronize()
    worst = {"gate_up": 0.0, "down": 0.0}
    for _ in range(3):
        for name, fn in (("gate_up", lambda: fm.launch_gate_up(x, row_token, aligned, group_ptr, h, rows)),
                         ("down", lambda: fm.launch_down(h, row_weight, aligned, group_ptr, out, rows))):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            worst[name] = max(worst[name], float(a.elapsed_time(b)))
    del records
    return worst


def nnls2(rows, ys):
    """Non-negative least squares for y = a*x1 + b*x2 (two columns), by the active-set shortcut."""
    import numpy as np
    A = np.array(rows, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)
    best = None
    for cols in ((0, 1), (0,), (1,)):
        sol, *_ = np.linalg.lstsq(A[:, cols], y, rcond=None)
        full = np.zeros(2)
        full[list(cols)] = sol
        if (full < 0).any():
            continue
        err = float(((A @ full - y) ** 2).sum())
        if best is None or err < best[0]:
            best = (err, full)
    return [float(v) for v in best[1]]


def measure():
    import torch
    from slotstream_spark.kernels import vendored
    from slotstream_spark.memory.ledger import Ledger
    meter = Meter(torch)
    t0 = time.time()
    cells = []
    for q in PASSES:
        moe = moe_cell(torch, meter, q)
        for ctx in CONTEXTS:
            attn = attention_cell(torch, vendored, meter, q, ctx)
            k = ctx + q
            cells.append({"pass": q, "context": ctx, "k": k, "product": q * k,
                          "attn_alloc_bytes": attn[0], "attn_nvml_bytes": attn[1], "attn_ms": round(attn[2], 3),
                          "moe_alloc_bytes": moe[0], "moe_nvml_bytes": moe[1], "moe_ms": round(moe[2], 3)})
            print(f"pass {q} ctx {ctx}: attn {attn[0] / 1e6:.1f} MB {attn[2]:.1f} ms | moe {moe[0] / 1e6:.1f} MB {moe[2]:.1f} ms",
                  flush=True)
    launches = {}
    for r in ROW_COUNTS:
        launches[str(r)] = launch_times(torch, r)
        print(f"rows {r}: {launches[str(r)]}", flush=True)
    budget = Ledger("D5").pass_bytes
    a_pair, a_tok = nnls2([[c["pass"] * c["k"], c["pass"]] for c in cells], [c["attn_alloc_bytes"] for c in cells])
    moe_cells = {c["pass"]: c["moe_alloc_bytes"] for c in cells}
    m_tok = nnls2([[q, 0.0] for q in moe_cells], list(moe_cells.values()))[0]
    for c in cells:
        c["transient_bytes"] = c["attn_alloc_bytes"] + c["moe_alloc_bytes"]
        c["fits"] = bool(c["transient_bytes"] <= budget and c["attn_ms"] <= WATCHDOG_MS)
    qk = 0
    for c in sorted(cells, key=lambda c: c["product"]):
        if not c["fits"]:
            break
        qk = c["product"]
    ok_rows = [r for r in ROW_COUNTS if max(launches[str(r)].values()) <= WATCHDOG_MS]
    max_rows = min(max(ok_rows), ROWS_TEMP_CAP) if ok_rows else 0
    all_ms = [c["attn_ms"] for c in cells] + [v for d in launches.values() for v in d.values()]
    return {
        "tool": "tools/prefill_envelope.py", "torch": torch.__version__, "budget_bytes": int(budget),
        "watchdog_ms": WATCHDOG_MS, "qk_product": int(qk), "bytes_per_pair": a_pair,
        "attn_bytes_per_token": a_tok, "moe_bytes_per_token": m_tok, "bytes_per_pass_token": a_tok + m_tok,
        "max_rows_per_launch": int(max_rows), "launch_ms_max": max(all_ms),
        "all_launches_within_watchdog": bool(max(all_ms) <= WATCHDOG_MS),
        "cells": cells, "moe_launches_ms": launches, "seconds": round(time.time() - t0, 1),
        "passes": list(PASSES), "contexts": list(CONTEXTS),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help=f"write {OUT.relative_to(ROOT)}")
    ap.add_argument("--out", help="explicit output path")
    a = ap.parse_args()
    res = measure()
    text = json.dumps(res, indent=1, sort_keys=True)
    outs = []
    if a.out:
        outs.append(Path(a.out))
    elif a.write:
        outs.append(OUT)
    if os.environ.get("CAP"):
        outs.append(Path(os.environ["CAP"]) / "result.json")
    for p in outs:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        print("wrote", p, flush=True)
    print(f"qk_product {res['qk_product']} bytes_per_pair {res['bytes_per_pair']:.3f} "
          f"max_rows_per_launch {res['max_rows_per_launch']} launch_ms_max {res['launch_ms_max']:.1f}", flush=True)
    return 0 if res["qk_product"] > 0 and res["max_rows_per_launch"] > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
