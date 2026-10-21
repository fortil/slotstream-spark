#!/usr/bin/env python3
"""Diagnostic: write a GGUF that is BASE with the tensors matching --take replaced (data and ggml type) by DONOR's.

  gguf_overlay.py --base FILE|DIR --donor FILE|DIR --take REGEX --out FILE [--bf16 REGEX] [--max-mbps 2000] [--dry-run]

Both files must hold the same tensor names and shapes (the repack copies the template's). The header comes from BASE (split.*
keys dropped), so the result is one self-contained GGUF. Used to find which tensors of the repack make llama.cpp worse than the
original: take the donor's tensors into the base (--bf16: those that also match the regex are written as BF16, the donor's Q4_1
values dequantized in fp32 and rounded to nearest even, to tell the values from the Q4_1 kernels), score the result with llamacpp_ppl.py --gguf. Reads are advised out of the
page cache, the output goes through mlx_ckpt.AlignedWriter (O_DIRECT). Nothing here is part of the build.
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import repack_q4_1 as rq  # noqa: E402
from slotstream_spark.model import mlx_ckpt  # noqa: E402

CHUNK = 128 << 20


def nbytes_of(t):
    return rq.row_bytes(t.gtype, t.ne[0]) * int(np.prod(t.ne[1:], dtype=np.int64)) if len(t.ne) > 1 else rq.row_bytes(t.gtype, t.ne[0])


def plan(base, donor, take, bf16=None):
    """-> [(name, ne, gtype, nbytes, source tensor, as_bf16)] in BASE order."""
    dby = {t.name: t for t in donor.tensors}
    if set(dby) != {t.name for t in base.tensors}:
        raise SystemExit(f"base and donor hold different tensors: {sorted(set(dby) ^ {t.name for t in base.tensors})[:8]}")
    rx = re.compile(take)
    rb = re.compile(bf16) if bf16 else None
    out, taken = [], 0
    for t in base.tensors:
        src = dby[t.name] if rx.search(t.name) else t
        taken += src is not t
        if tuple(src.ne) != tuple(t.ne):
            raise SystemExit(f"{t.name}: shape {src.ne} (donor) differs from {t.ne} (base)")
        as_bf16 = bool(rb and src is not t and rb.search(t.name))
        if as_bf16:
            if src.gtype not in (rq.GGML_Q4_1, rq.GGML_F32, rq.GGML_F16, rq.GGML_BF16):
                raise SystemExit(f"{t.name}: --bf16 needs a Q4_1/F32/F16/BF16 donor tensor, got type {src.gtype}")
            out.append((t.name, t.ne, rq.GGML_BF16, int(np.prod(t.ne, dtype=np.int64)) * 2, src, True))
        else:
            out.append((t.name, t.ne, src.gtype, nbytes_of(src), src, False))
    return out, taken


def copy_as_bf16(src, w, throttle, rows_per=512):
    """The donor tensor's values (dequantized in fp32 as ggml does) written as BF16, rows_per rows at a time."""
    K = src.ne[0]
    nrows = int(np.prod(src.ne[1:], dtype=np.int64)) if len(src.ne) > 1 else 1
    rbytes = rq.row_bytes(src.gtype, K)
    with open(src.file, "rb", buffering=0) as f:
        for i0 in range(0, nrows, rows_per):
            n = min(rows_per, nrows - i0)
            f.seek(src.offset + i0 * rbytes)
            raw = f.read(n * rbytes)
            if len(raw) != n * rbytes:
                raise SystemExit(f"{src.name}: short read in {src.file}")
            vals = rq.dequant_ggml(src.gtype, raw, n, K)
            w.write(rq.f32_to_bf16_bits(vals).astype("<u2").tobytes())
            if throttle:
                throttle.spend(len(raw) * 2)


def copy_tensor(src, nbytes, w, throttle):
    with open(src.file, "rb", buffering=0) as f:
        fd = f.fileno()
        f.seek(src.offset)
        left, pos = nbytes, src.offset
        while left:
            n = min(left, CHUNK)
            buf = f.read(n)
            if len(buf) != n:
                raise SystemExit(f"{src.name}: short read in {src.file}")
            w.write(buf)
            os.posix_fadvise(fd, pos, n, os.POSIX_FADV_DONTNEED)
            pos += n
            left -= n
            if throttle:
                throttle.spend(2 * n)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--donor", required=True)
    ap.add_argument("--take", required=True, help="regex (re.search) on tensor names that come from the donor")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bf16", help="regex: tensors taken from the donor that are written as BF16 (dequantized values)")
    ap.add_argument("--max-mbps", type=float, default=2000)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    base, donor = rq.read_gguf(rq.template_paths(a.base)), rq.read_gguf(rq.template_paths(a.donor))
    specs, taken = plan(base, donor, a.take, a.bf16)
    total = sum(s[3] for s in specs)
    by_type = {}
    bnames = {t.name: t for t in base.tensors}
    for name, _ne, gtype, nb, src, _bf in specs:
        if src is not bnames[name]:
            by_type[gtype] = by_type.get(gtype, 0) + 1
    print(f"overlay: {taken} of {len(specs)} tensors from the donor (by ggml type {by_type}), {total / 1e9:.1f} GB -> {a.out}", flush=True)
    if a.dry_run:
        return 0
    kv = [(k, raw) for k, raw in base.kv if not k.startswith(rq.DROP_PREFIXES)]
    align = base.alignment
    header = rq.gguf_header(kv, [(n, ne, g, nb) for n, ne, g, nb, _, _bf in specs], align)
    throttle = rq._throttle(a.max_mbps)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    w = mlx_ckpt.AlignedWriter(tmp, o_direct=True)
    t0 = time.time()
    try:
        w.write(header)
        for i, (name, _ne, _g, nb, src, as_bf16) in enumerate(specs):
            if as_bf16:
                copy_as_bf16(src, w, throttle)
            else:
                copy_tensor(src, nb, w, throttle)
            pad = -nb % align
            if pad:
                w.write(b"\0" * pad)
            if (i + 1) % 200 == 0 or i + 1 == len(specs):
                print(f"{i + 1}/{len(specs)} tensors, {time.time() - t0:.0f} s", flush=True)
    finally:
        w.close()
    os.replace(tmp, out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
