#!/usr/bin/env python3
"""m-lb1: host-to-device copy of 1,024,000-byte pieces taken from GGUF file pages.

Conditions: (a) cold page cache, (b) hot second pass, (c) from an anonymous numpy
copy of each piece, (d) from pinned memory. Seeded random 4096-aligned offsets.
Writes $CAP/result.json (or --out). The aggregate GB/s of each condition is
total bytes over the summed copy time.
"""
import argparse
import json
import os
import time

import numpy as np
import torch


def pct(v, q):
    return float(np.percentile(np.array(v), q))


def summarize(times, nbytes):
    tot = sum(times)
    return {
        "n": len(times),
        "p50_ms": round(pct(times, 50) * 1e3, 4),
        "p99_ms": round(pct(times, 99) * 1e3, 4),
        "gbs": round(len(times) * nbytes / tot / 1e9, 3),
    }


def dontneed(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf-rel", required=True)
    ap.add_argument("--pieces", type=int, default=300)
    ap.add_argument("--piece-bytes", type=int, default=1024000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out")
    a = ap.parse_args()

    root = os.environ.get("SLOTSTREAM_R", os.path.expanduser("~/slotstream-spark"))
    path = a.gguf_rel if os.path.isabs(a.gguf_rel) else os.path.join(root, a.gguf_rel)
    size = os.path.getsize(path)
    n = a.piece_bytes
    rng = np.random.default_rng(a.seed)
    slots = (size - n) // 4096
    offs = [int(x) * 4096 for x in rng.choice(slots, size=a.pieces, replace=False)]

    mm = np.memmap(path, dtype=np.uint8, mode="c")  # copy-on-write mapping: reads only
    torch.zeros(1, device="cuda")
    dev = torch.empty(n, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    for _ in range(5):  # warm up the copy path
        dev.copy_(torch.from_numpy(np.zeros(n, dtype=np.uint8)))
    torch.cuda.synchronize()

    def timed(src_fn):
        out = []
        for o in offs:
            src = src_fn(o)
            t0 = time.perf_counter()
            dev.copy_(src)
            torch.cuda.synchronize()
            out.append(time.perf_counter() - t0)
        return out

    res = {"file": os.path.basename(path), "file_bytes": size, "pieces": a.pieces, "piece_bytes": n}
    dontneed(path)
    res["a_cold"] = summarize(timed(lambda o: torch.from_numpy(mm[o:o + n])), n)
    res["b_hot"] = summarize(timed(lambda o: torch.from_numpy(mm[o:o + n])), n)

    anon_t = []

    def anon(o):
        t0 = time.perf_counter()
        x = np.array(mm[o:o + n])
        anon_t.append(time.perf_counter() - t0)
        return torch.from_numpy(x)

    res["c_anon"] = summarize(timed(anon), n)
    res["c_anon"]["memcpy_to_anon_p50_ms"] = round(pct(anon_t, 50) * 1e3, 4)

    pin = torch.empty(n, dtype=torch.uint8, pin_memory=True)

    def pinned(o):
        pin.numpy()[:] = mm[o:o + n]
        return pin

    res["d_pinned"] = summarize(timed(pinned), n)
    dontneed(path)

    res["rule_input_gbs_hot"] = res["b_hot"]["gbs"]
    out = a.out or (os.path.join(os.environ["CAP"], "result.json") if os.environ.get("CAP") else None)
    s = json.dumps(res, indent=1)
    if out:
        with open(out, "w") as f:
            f.write(s + "\n")
    print(s)


if __name__ == "__main__":
    main()
