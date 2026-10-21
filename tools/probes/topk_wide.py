#!/usr/bin/env python3
"""Probe: does `kernels.topk_stable` compile and select correctly above its MAX_N (16,384 compressed blocks, which is
65,536 token positions at the indexer's compress ratio of 4)? A 131,072-token window needs N up to 32,768.

  tools/spark/run.sh --heavy topk-wide auto 600 -- tools/spark/py.sh tools/probes/topk_wide.py [--ns 16384,32768,65536] [--rows 256]

For every N it lifts the module's MAX_N, runs k = 512 on random rows with ties and a few special values, checks every row
against the CPU reference of tests/t0/test_topk_stable.py (stable keys, ties by lower index) and times the launch. The
result goes to $CAP/result.json when CAP is set, else to stdout only. Exit 0 when every N selected correctly, 1 otherwise;
a compile or launch failure is recorded per N and does not stop the others.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def keys(x):
    import torch
    b = x.cpu().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    key = torch.where(b >= 0x80000000, (~b) & 0xFFFFFFFF, b ^ 0x80000000)
    nan = (b & 0x7FFFFFFF) > 0x7F800000
    return torch.where(nan, torch.zeros_like(key), key)


def ref_row(x, k, nv):
    import torch
    key = keys(x[:nv])
    order = torch.sort(-key, stable=True).indices[:k]
    sel = order if k <= 16 else torch.sort(order).values
    idx = torch.full((k,), -1, dtype=torch.int64)
    idx[:sel.numel()] = sel
    return idx


def probe(n, rows, k, repeat):
    import torch
    from slotstream_spark.kernels import gemv_affine
    from slotstream_spark.kernels import topk_stable as ts
    out = {"n": n, "rows": rows, "k": k}
    g = torch.Generator().manual_seed(n)
    x = torch.randn(rows, n, generator=g)
    x[:, : n // 8] = torch.randint(0, 7, (rows, n // 8), generator=g).float()        # ties
    x[0, 5] = float("nan")
    x[1, 7] = float("inf")
    x[2, 9] = float("-inf")
    nv = torch.full((rows,), n, dtype=torch.int32)
    nv[3] = k - 1                                                                   # the no-sparsity regime
    nv[4] = n // 2
    old = ts.MAX_N
    ts.MAX_N = max(old, n)
    try:
        xc = x.contiguous().cuda()
        nvc = nv.cuda()
        t0 = time.perf_counter()
        idx, _ = ts.topk_stable(xc, k, nvc)
        torch.cuda.synchronize()
        out["first_call_s"] = round(time.perf_counter() - t0, 3)                   # includes the Triton compile
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(repeat):
            ts.topk_stable(xc, k, nvc, out_idx=idx)
        b.record()
        torch.cuda.synchronize()
        out["launch_ms"] = round(float(a.elapsed_time(b)) / repeat, 3)
        out["shared_bytes"] = gemv_affine.SHARED_BYTES.get("topk_stable")
        bad = 0
        got = idx.cpu().to(torch.int64)
        for r in range(rows):
            if not torch.equal(got[r], ref_row(x[r], k, int(nv[r]))):
                bad += 1
        out["rows_wrong"] = bad
        out["ok"] = bad == 0
    except Exception as e:  # noqa: BLE001 - a compile or launch failure is the finding
        out["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        out["ok"] = False
    finally:
        ts.MAX_N = old
        torch.cuda.empty_cache()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="16384,32768,65536")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--repeat", type=int, default=20)
    a = ap.parse_args(argv)
    import torch
    res = {"tool": "tools/probes/topk_wide.py", "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
           "results": []}
    for n in (int(v) for v in a.ns.split(",") if v.strip()):
        r = probe(n, a.rows, a.k, a.repeat)
        res["results"].append(r)
        print(json.dumps(r), flush=True)
    res["ok"] = all(r["ok"] for r in res["results"])
    if os.environ.get("CAP"):
        p = Path(os.environ["CAP"]) / "result.json"
        p.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        print("wrote", p, flush=True)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
