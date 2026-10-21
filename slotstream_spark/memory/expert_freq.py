#!/usr/bin/env python3
"""Expert access-frequency table from router traces (step 77e, DIAG_F6 option P4).

  expert_freq.py TRACES_DIR --out slotstream_spark/config/expert_freq_gb10.json

TRACES_DIR holds trace-*.npz, each with `ids` int16 of shape (steps, layers, topk) (the trace_routing.py format).
Writes counts[layer][expert] (one per occurrence) plus the coverage of the top 388 experts of each layer. Needs numpy.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

LAYERS, EXPERTS, TOPK, COVER_N = 48, 512, 10, 388


def build(traces_dir, layers=None, experts=None):
    """counts int64 [layers, experts], number of traces, number of steps."""
    files = sorted(glob.glob(str(Path(traces_dir) / "trace-*.npz")))
    if not files:
        raise SystemExit(f"expert_freq: no trace-*.npz in {traces_dir}")
    counts, steps = None, 0
    for f in files:
        ids = np.load(f)["ids"]
        if counts is None:
            L = layers or ids.shape[1]
            E = experts or int(ids.max()) + 1
            counts = np.zeros((L, E), dtype=np.int64)
        steps += ids.shape[0]
        for layer in range(counts.shape[0]):
            counts[layer] += np.bincount(ids[:, layer, :].reshape(-1).astype(np.int64), minlength=counts.shape[1])
    return counts, len(files), steps


def select(counts, per_layer):
    """Per layer the ids with count > 0 ordered by (-count, id), truncated to per_layer (the `hottest` rule of sweep.py)."""
    c = np.asarray(counts)
    ids = np.arange(c.shape[1])
    out = []
    for row in c:
        order = np.lexsort((ids, -row))
        order = order[row[order] > 0][:max(0, int(per_layer))]
        out.append(order.astype(np.int32))
    return out


def coverage(counts, n=COVER_N):
    """Share of all accesses covered by the top-n experts of each layer."""
    c = np.asarray(counts)
    total = int(c.sum())
    if total == 0:
        return 0.0
    top = np.sort(c, axis=1)[:, ::-1][:, :n]
    return float(top.sum()) / total


def check_table(d):
    if d.get("layers") != LAYERS or d.get("experts") != EXPERTS:
        raise ValueError(f"expert table must be {LAYERS} x {EXPERTS}, got {d.get('layers')} x {d.get('experts')}")
    rows = d.get("counts")
    if not isinstance(rows, list) or len(rows) != LAYERS:
        raise ValueError(f"expert table needs {LAYERS} rows of counts")
    for r in rows:
        if len(r) != EXPERTS or any((not isinstance(v, int)) or isinstance(v, bool) or v < 0 for v in r):
            raise ValueError(f"expert table rows need {EXPERTS} non-negative ints")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("traces")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    counts, n, steps = build(a.traces, LAYERS, EXPERTS)
    cov = coverage(counts)
    res = {"layers": LAYERS, "experts": EXPERTS, "topk": TOPK, "traces": n, "steps": int(steps),
           "accesses": int(steps) * LAYERS * TOPK, "source": Path(a.traces).resolve().parent.name,
           "coverage_388": round(cov, 6), "counts": counts.tolist()}
    check_table(res)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res) + "\n")
    seen = (counts > 0).sum(axis=1)
    print(f"traces {n}  accesses {res['accesses']}  coverage_388 {cov:.4f}  min experts seen/layer {int(seen.min())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
