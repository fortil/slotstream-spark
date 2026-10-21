#!/usr/bin/env python3
"""R0 (bit-exact repack) and E1 (one packed read vs nine unpacked reads).

Full byte comparison, never hashes alone: every record of experts.bin / mtp_experts.bin against its 9 source
slices, every row of ngram.bin against the shards, every tensor of trunk.safetensors / mtp_trunk.safetensors
against the checkpoint. The manifest layer / shard sha256 are checked in addition and reported separately,
so `mismatches` always counts the damaged records, rows or tensors.

  verify_r0.py --model DIR --packed DIR [--e1 N] [--max-mbps 3000] [--json OUT] [--allow-buffered]

Exit 0 iff all_equal.
"""
import argparse
import glob
import hashlib
import json
import mmap
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extract_trunk import keep_main, keep_mtp  # noqa: E402
from repack_experts import Throttle, layer_tensors, proc_io  # noqa: E402
from repack_ngram import PREFIX, build_rows  # noqa: E402
from slotstream_spark.model.mlx_ckpt import ALIGN, O_DIRECT, Checkpoint  # noqa: E402


class RawFile:
    """Read byte ranges of a plain file with aligned-superset O_DIRECT reads."""

    def __init__(self, path, allow_buffered):
        self.path = str(path)
        self.size = os.stat(path).st_size
        self.fd = None
        if O_DIRECT:
            try:
                fd = os.open(self.path, os.O_RDONLY | O_DIRECT)
                probe = mmap.mmap(-1, ALIGN)
                try:
                    os.preadv(fd, [probe], 0)
                    self.fd = fd
                except OSError:
                    os.close(fd)
                finally:
                    probe.close()
            except OSError:
                pass
        if self.fd is None:
            if not allow_buffered:
                raise RuntimeError(f"O_DIRECT unavailable for {path}; pass --allow-buffered")
            self.fd = os.open(self.path, os.O_RDONLY)

    def read(self, off, length):
        """-> (mmap, memoryview of exactly [off, off+length)); caller closes both via release()."""
        start = off - off % ALIGN
        end = -(-(off + length) // ALIGN) * ALIGN
        buf = mmap.mmap(-1, end - start)
        mv = memoryview(buf)
        got = 0
        while got < end - start:
            n = os.preadv(self.fd, [mv[got:]], start + got)
            if n == 0:
                break
            got += n
        if got < off + length - start:
            raise EOFError(f"short read in {self.path}")
        return buf, mv, off - start

    def read_bytes(self, off, length):
        buf, mv, d = self.read(off, length)
        try:
            return bytes(mv[d:d + length])
        finally:
            mv.release()
            buf.close()

    def close(self):
        os.close(self.fd)


def verify_experts(model, packed, mtp, thr, allow_buffered):
    stem = "mtp_experts" if mtp else "experts"
    man = json.load(open(Path(packed) / f"{stem}.manifest.json"))
    ck = Checkpoint(Path(model) / "mtp.safetensors" if mtp else model, allow_buffered=allow_buffered)
    layers = layer_tensors(ck, mtp)
    E, rec, nl = man["experts"], man["record_bytes"], man["layers"]
    sizes, offs = man["pieces"], man["piece_offsets"]
    rf = RawFile(Path(packed) / f"{stem}.bin", allow_buffered)
    mism = sha_bad = 0
    if rf.size != man["total_bytes"] or nl != len(layers):
        mism += 1
    lb = E * rec
    for L in range(min(nl, len(layers))):
        buf, mv, d = rf.read(L * lb, lb)
        try:
            got = np.frombuffer(buf, dtype=np.uint8, count=lb, offset=d).reshape(E, rec)
            thr.spend(lb)
            bad = np.zeros(E, dtype=bool)
            for j, name in enumerate(layers[L]):
                raw = np.frombuffer(ck.read_bytes(name), dtype=np.uint8).reshape(E, sizes[j])
                thr.spend(raw.nbytes)
                bad |= np.any(got[:, offs[j]:offs[j] + sizes[j]] != raw, axis=1)
            mism += int(bad.sum())
            if hashlib.sha256(mv[d:d + lb]).hexdigest() != man["layer_sha256"][L]:
                sha_bad += 1
            del got
        finally:
            mv.release()
            buf.close()
    rf.close()
    return {"records": nl * E, "mismatches": mism, "layer_sha_bad": sha_bad}


def verify_ngram(model, packed, thr, allow_buffered):
    man = json.load(open(Path(packed) / "ngram.manifest.json"))
    ck = Checkpoint(model, allow_buffered=allow_buffered)
    rb, rps = man["row_bytes"], man["rows_per_shard"]
    rf = RawFile(Path(packed) / "ngram.bin", allow_buffered)
    mism = sha_bad = 0
    if rf.size != man["total_bytes"]:
        mism += 1
    for s in range(man["shards"]):
        w = ck.read_array(f"{PREFIX}{s}.weight")
        sc = ck.read_array(f"{PREFIX}{s}.scales")
        bi = ck.read_array(f"{PREFIX}{s}.biases")
        thr.spend(w.nbytes + sc.nbytes + bi.nbytes)
        want = build_rows(w, sc, bi, rb)
        buf, mv, d = rf.read(s * rps * rb, rps * rb)
        try:
            got = np.frombuffer(buf, dtype=np.uint8, count=rps * rb, offset=d).reshape(rps, rb)
            thr.spend(rps * rb)
            mism += int(np.any(got != want, axis=1).sum())
            if hashlib.sha256(mv[d:d + rps * rb]).hexdigest() != man["shard_sha256"][s]:
                sha_bad += 1
            del got
        finally:
            mv.release()
            buf.close()
    rf.close()
    return {"rows": man["rows"], "mismatches": mism, "shard_sha_bad": sha_bad}


def verify_trunk(model, trunk_path, mtp, thr, allow_buffered):
    src = Checkpoint(Path(model) / "mtp.safetensors" if mtp else model, allow_buffered=allow_buffered)
    want = {n for n in src.tensors() if (keep_mtp if mtp else keep_main)(n)}
    got = Checkpoint(trunk_path, allow_buffered=allow_buffered)
    have = set(got.tensors())
    mism = len(want ^ have)
    for n in sorted(want & have):
        a, b = src.info(n), got.info(n)
        if (a.dtype, a.shape, a.nbytes) != (b.dtype, b.shape, b.nbytes):
            mism += 1
            continue
        x, y = src.read_bytes(n), got.read_bytes(n)
        thr.spend(2 * len(x))
        if x != y:
            mism += 1
    return {"tensors": len(want), "mismatches": mism}


def pct(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


def e1(model, packed, n, allow_buffered):
    man = json.load(open(Path(packed) / "experts.manifest.json"))
    ck = Checkpoint(model, allow_buffered=allow_buffered)
    layers = layer_tensors(ck, False)
    E, rec, sizes = man["experts"], man["record_bytes"], man["pieces"]
    rf = RawFile(Path(packed) / "experts.bin", allow_buffered)
    rng = np.random.default_rng(0)
    pairs = [(int(rng.integers(0, man["layers"])), int(rng.integers(0, E))) for _ in range(n)]
    tp, tu, bad = [], [], 0
    for L, e in pairs:
        t0 = time.perf_counter_ns()
        packed_rec = rf.read_bytes((L * E + e) * rec, rec)
        t1 = time.perf_counter_ns()
        parts = [ck.read_bytes(name, e * sizes[j], sizes[j]) for j, name in enumerate(layers[L])]
        t2 = time.perf_counter_ns()
        tp.append((t1 - t0) / 1000)
        tu.append((t2 - t1) / 1000)
        if packed_rec != b"".join(parts):
            bad += 1
    rf.close()
    return {"n": n, "seed": 0, "mismatches": bad,
            "packed_p50_us": pct(tp, 50), "packed_p99_us": pct(tp, 99),
            "unpacked_p50_us": pct(tu, 50), "unpacked_p99_us": pct(tu, 99),
            "ratio_p50": pct(tu, 50) / pct(tp, 50) if pct(tp, 50) else None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--packed", required=True)
    ap.add_argument("--e1", type=int, default=0)
    ap.add_argument("--max-mbps", type=float, default=3000)
    ap.add_argument("--json", default=None)
    ap.add_argument("--allow-buffered", action="store_true")
    a = ap.parse_args(argv)
    thr = Throttle(a.max_mbps)
    io0 = proc_io()
    ab = a.allow_buffered
    res = {
        "experts": verify_experts(a.model, a.packed, False, thr, ab),
        "mtp_experts": verify_experts(a.model, a.packed, True, thr, ab),
        "ngram": verify_ngram(a.model, a.packed, thr, ab),
        "trunk": verify_trunk(a.model, Path(a.packed) / "trunk.safetensors", False, thr, ab),
        "mtp_trunk": verify_trunk(a.model, Path(a.packed) / "mtp_trunk.safetensors", True, thr, ab),
    }
    res["e1"] = e1(a.model, a.packed, a.e1, ab) if a.e1 else {"n": 0, "mismatches": 0}
    io1 = proc_io()
    io = {}
    cap = os.environ.get("CAP")
    for p in sorted(glob.glob(os.path.join(cap, "repack_*.json"))) if cap else []:
        try:
            pio = json.loads(open(p).read().strip().splitlines()[-1]).get("proc_io")
        except (OSError, ValueError, IndexError):
            continue
        if pio:
            io[Path(p).stem] = {"rchar": pio["rchar"], "read_bytes": pio["read_bytes"]}
    io["verify_r0"] = {"rchar": io1.get("rchar", 0) - io0.get("rchar", 0),
                       "read_bytes": io1.get("read_bytes", 0) - io0.get("read_bytes", 0)}
    res["io"] = io
    parts = ("experts", "mtp_experts", "ngram", "trunk", "mtp_trunk", "e1")
    res["all_equal"] = all(res[k]["mismatches"] == 0 for k in parts) and \
        res["experts"]["layer_sha_bad"] == 0 and res["mtp_experts"]["layer_sha_bad"] == 0 and \
        res["ngram"]["shard_sha_bad"] == 0
    s = json.dumps(res, indent=1)
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as f:
            f.write(s + "\n")
    print(s)
    return 0 if res["all_equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
