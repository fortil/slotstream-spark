#!/usr/bin/env python3
"""Compare two MTP safetensors files (converted vs mirror). Needs numpy.

  mtp_compare.py CONVERTED MIRROR  -> JSON on stdout
Fields: names_equal, dtypes_shapes_equal, identical_tensors, total_tensors,
max_abs_dequant_diff, rel_rms_dequant_diff (U32 weight/scales/biases triplets, gs64, 4 bit).
"""
import json
import struct
import sys

import numpy as np

GS, BITS = 64, 4


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def tensor_bytes(path, h, base, name):
    lo, hi = h[name]["data_offsets"]
    return np.memmap(path, dtype=np.uint8, mode="r", offset=base + lo, shape=(hi - lo,))


def to_f32(raw, dtype):
    if dtype == "BF16":
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    if dtype == "F16":
        return raw.view(np.float16).astype(np.float32)
    if dtype == "F32":
        return raw.view(np.float32)
    raise ValueError(dtype)


def dequant_rows(wq, sc, bi, r0, r1):
    """wq: (R, K/8) uint32, sc/bi: (R, K/GS) float32 -> (r1-r0, K) float32."""
    w = wq[r0:r1]
    shifts = np.arange(0, 32, BITS, dtype=np.uint32)
    vals = ((w[..., None] >> shifts) & 0xF).astype(np.float32)
    vals = vals.reshape(w.shape[0], -1)
    k = vals.shape[1]
    vals = vals.reshape(w.shape[0], k // GS, GS)
    out = vals * sc[r0:r1, :, None] + bi[r0:r1, :, None]
    return out.reshape(w.shape[0], k)


def main():
    a_path, b_path = sys.argv[1:3]
    ha, ba = header(a_path)
    hb, bb = header(b_path)
    names_equal = set(ha) == set(hb)
    shapes_equal = names_equal and all(
        ha[n]["dtype"] == hb[n]["dtype"] and ha[n]["shape"] == hb[n]["shape"] for n in ha)
    identical = 0
    max_abs = 0.0
    sq_diff = 0.0
    sq_ref = 0.0
    common = sorted(set(ha) & set(hb))
    diff_names = []
    for n in common:
        if ha[n]["shape"] != hb[n]["shape"] or ha[n]["dtype"] != hb[n]["dtype"]:
            diff_names.append(n)
            continue
        ra, rb = tensor_bytes(a_path, ha, ba, n), tensor_bytes(b_path, hb, bb, n)
        same = True
        step = 1 << 28
        for off in range(0, len(ra), step):
            if not np.array_equal(ra[off:off + step], rb[off:off + step]):
                same = False
                break
        if same:
            identical += 1
        else:
            diff_names.append(n)
    for n in diff_names:
        if not n.endswith(".weight") or ha[n]["dtype"] != "U32" or n not in hb:
            continue
        base = n[: -len(".weight")]
        sn, bn = base + ".scales", base + ".biases"
        if sn not in ha or bn not in ha or sn not in hb or bn not in hb:
            continue
        shape = ha[n]["shape"]
        lead = int(np.prod(shape[:-1]))
        sh = ha[sn]["shape"]

        def load(path, h, b, name, dt):
            raw = tensor_bytes(path, h, b, name)
            return to_f32(np.asarray(raw), dt)

        wa = np.asarray(tensor_bytes(a_path, ha, ba, n)).view(np.uint32).reshape(lead, shape[-1])
        wb = np.asarray(tensor_bytes(b_path, hb, bb, n)).view(np.uint32).reshape(lead, shape[-1])
        sa = load(a_path, ha, ba, sn, ha[sn]["dtype"]).reshape(lead, sh[-1])
        sb = load(b_path, hb, bb, sn, hb[sn]["dtype"]).reshape(lead, sh[-1])
        ia = load(a_path, ha, ba, bn, ha[bn]["dtype"]).reshape(lead, sh[-1])
        ib = load(b_path, hb, bb, bn, hb[bn]["dtype"]).reshape(lead, sh[-1])
        rows = max(1, (1 << 24) // (shape[-1] * 8))
        for r0 in range(0, lead, rows):
            r1 = min(lead, r0 + rows)
            da = dequant_rows(wa, sa, ia, r0, r1)
            db = dequant_rows(wb, sb, ib, r0, r1)
            d = np.abs(da - db)
            max_abs = max(max_abs, float(d.max()))
            sq_diff += float((d.astype(np.float64) ** 2).sum())
            sq_ref += float((db.astype(np.float64) ** 2).sum())
    rel = (sq_diff / sq_ref) ** 0.5 if sq_ref > 0 else 0.0
    print(json.dumps({
        "names_equal": names_equal,
        "dtypes_shapes_equal": bool(shapes_equal),
        "identical_tensors": identical,
        "total_tensors": len(set(ha) | set(hb)),
        "max_abs_dequant_diff": max_abs,
        "rel_rms_dequant_diff": rel,
    }, indent=1))


if __name__ == "__main__":
    main()
