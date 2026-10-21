#!/usr/bin/env python3
"""Repack the 128 n-gram embedding shards into ngram.bin: one fixed-size row per n-gram id.

Source shard s: language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_{s}.{weight,scales,biases}
with weight U32[r,20], scales BF16[r,5], biases BF16[r,5]. Global row g = s*r + row. Each output row is
[80 B weight | 10 B scales | 10 B biases | zero pad up to row_bytes]. Written through one AlignedWriter
(O_DIRECT, sequential, unaligned shard tails carried across writes) to ngram.bin.tmp, renamed once every
shard is written, then ngram.manifest.json. Restartable from scratch (no partial state is trusted).

  repack_ngram.py --model DIR --out DIR --row-bytes 100|128 [--max-mbps 2000] [--allow-buffered]
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from repack_experts import Throttle, atomic_json, proc_io  # noqa: E402
from slotstream_spark.model.mlx_ckpt import AlignedWriter, Checkpoint  # noqa: E402

PREFIX = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_"
W_BYTES, S_BYTES, B_BYTES = 80, 10, 10
FIELDS = {"weight": [0, 80], "scales": [80, 90], "biases": [90, 100]}


def build_rows(w, s, b, row_bytes):
    """w U32[r,20], s/b U16[r,5] -> uint8[r,row_bytes]."""
    r = w.shape[0]
    out = np.zeros((r, row_bytes), dtype=np.uint8)
    out[:, 0:80] = np.ascontiguousarray(w, dtype="<u4").view(np.uint8).reshape(r, W_BYTES)
    out[:, 80:90] = np.ascontiguousarray(s, dtype="<u2").view(np.uint8).reshape(r, S_BYTES)
    out[:, 90:100] = np.ascontiguousarray(b, dtype="<u2").view(np.uint8).reshape(r, B_BYTES)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--row-bytes", type=int, required=True, choices=(100, 128))
    ap.add_argument("--max-mbps", type=float, default=2000)
    ap.add_argument("--allow-buffered", action="store_true")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    final, tmp, man_path = out / "ngram.bin", out / "ngram.bin.tmp", out / "ngram.manifest.json"
    ck = Checkpoint(a.model, allow_buffered=a.allow_buffered)
    pat = re.compile(re.escape(PREFIX) + r"(\d+)\.weight$")
    shards = sorted(int(m.group(1)) for k in ck.tensors() if (m := pat.match(k)))
    if not shards or shards != list(range(len(shards))):
        raise SystemExit(f"n-gram shards are not contiguous from 0: {shards[:5]}")
    nsh = len(shards)
    rps = None
    for s in shards:
        for kind, cols in (("weight", 20), ("scales", 5), ("biases", 5)):
            shape = ck.info(f"{PREFIX}{s}.{kind}").shape
            if len(shape) != 2 or shape[1] != cols or (rps is not None and shape[0] != rps):
                raise SystemExit(f"shard_{s}.{kind}: unexpected shape {shape}")
            rps = shape[0] if rps is None else rps
    rows = nsh * rps

    thr = Throttle(a.max_mbps)
    io0 = proc_io()
    t0 = time.time()
    sha = []
    with AlignedWriter(tmp) as wr:
        for s in shards:
            w = ck.read_array(f"{PREFIX}{s}.weight")
            sc = ck.read_array(f"{PREFIX}{s}.scales")
            bi = ck.read_array(f"{PREFIX}{s}.biases")
            thr.spend(w.nbytes + sc.nbytes + bi.nbytes)
            blk = build_rows(w, sc, bi, a.row_bytes)
            sha.append(hashlib.sha256(blk).hexdigest())
            wr.write(blk)
            thr.spend(blk.nbytes)
    total = rows * a.row_bytes
    if os.stat(tmp).st_size != total:
        raise SystemExit(f"ngram.bin.tmp is {os.stat(tmp).st_size} bytes, expected {total}")
    os.replace(tmp, final)
    dfd = os.open(out, os.O_RDONLY)
    os.fsync(dfd)
    os.close(dfd)

    lock = Path(__file__).resolve().parent.parent / "model.lock.json"
    src = {"repo": "unknown", "revision": "unknown"}
    if lock.exists():
        lj = json.load(open(lock))
        src = {"repo": lj["repo"], "revision": lj["revision"]}
    src["index_sha256"] = ck.index_sha256
    manifest = {"version": 1, "layout": "slotstream-spark-ngram-rows-v1", "row_bytes": a.row_bytes,
                "rows": rows, "rows_per_shard": rps, "shards": nsh, "fields": FIELDS, "bits": 4,
                "group_size": 32, "shard_sha256": sha, "total_bytes": total, "source": src}
    atomic_json(man_path, manifest)
    io1 = proc_io()
    summary = {"path": str(final), "rows": rows, "row_bytes": a.row_bytes, "total_bytes": total,
               "seconds": round(time.time() - t0, 1),
               "proc_io": {k: io1.get(k, 0) - io0.get(k, 0) for k in ("rchar", "read_bytes", "wchar", "write_bytes")}}
    s = json.dumps(summary)
    cap = os.environ.get("CAP")
    if cap:
        with open(os.path.join(cap, "repack_ngram.json"), "w") as f:
            f.write(s + "\n")
    print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
