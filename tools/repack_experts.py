#!/usr/bin/env python3
"""Repack the routed experts into one record per (layer, expert): experts.bin / mtp_experts.bin.

Record = the 9 per-expert pieces in the order gate/up/down x weight/scales/biases. Layer L, expert e
sits at (L*E + e) * record_bytes. Atomic and resumable per layer: data goes to <name>.tmp, a layer is
recorded in <name>.progress.json (layer -> sha256) only after it is written and fsynced, and the tmp
file is renamed only when every layer has its sha. Reads go through mlx_ckpt (O_DIRECT).

  repack_experts.py --model DIR --out DIR [--mtp] [--max-mbps 2000] [--expect-record 2764800] [--allow-buffered]
"""
import argparse
import hashlib
import json
import mmap
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slotstream_spark.model.mlx_ckpt import ALIGN, O_DIRECT, Checkpoint  # noqa: E402

PROJS = ("gate_proj", "up_proj", "down_proj")
KINDS = ("weight", "scales", "biases")
PIECE_NAMES = [f"{p}.{k}" for p in PROJS for k in KINDS]
LAYOUT = "slotstream-pool-layer-expert-piece-v1"


class Throttle:
    """Token bucket: keeps the average rate at or under mbps MB/s (0 = off)."""

    def __init__(self, mbps):
        self.rate = mbps * 1e6
        self.t = time.monotonic()

    def spend(self, nbytes):
        if self.rate <= 0:
            return
        now = time.monotonic()
        self.t = max(self.t, now) + nbytes / self.rate
        if self.t > now:
            time.sleep(self.t - now)


class SlabFile:
    """pwrite into an existing file without truncating it (O_DIRECT when the filesystem allows)."""

    def __init__(self, path, size, allow_buffered):
        flags = os.O_RDWR | os.O_CREAT
        self.fd = None
        self.direct = False
        if O_DIRECT:
            fd = None
            try:
                fd = os.open(path, flags | O_DIRECT, 0o644)
                # probe on a scratch file next to the data: writing the probe to `path` would zero the first
                # block of a file that a resumed run is about to keep
                probe_path = str(path) + ".probe"
                pfd = os.open(probe_path, flags | O_DIRECT, 0o644)
                probe = mmap.mmap(-1, ALIGN)
                try:
                    os.pwrite(pfd, probe, 0)
                finally:
                    probe.close()
                    os.close(pfd)
                    os.unlink(probe_path)
                self.fd, self.direct = fd, True
            except OSError:
                if fd is not None:
                    os.close(fd)
                if not allow_buffered:
                    raise
        if self.fd is None:
            if not allow_buffered:
                raise RuntimeError("O_DIRECT writes unavailable; pass --allow-buffered")
            self.fd = os.open(path, flags, 0o644)
        if os.fstat(self.fd).st_size != size:
            os.ftruncate(self.fd, 0)
            os.posix_fallocate(self.fd, 0, size) if hasattr(os, "posix_fallocate") else os.ftruncate(self.fd, size)
            self.fresh = True
        else:
            self.fresh = False

    def write_at(self, mv, off):
        done = 0
        while done < len(mv):
            done += os.pwrite(self.fd, mv[done:], off + done)

    def sync(self):
        os.fsync(self.fd)

    def close(self):
        os.close(self.fd)


def proc_io():
    out = {}
    try:
        for ln in open("/proc/self/io"):
            k, v = ln.split(":")
            out[k.strip()] = int(v)
    except OSError:
        pass
    return out


def atomic_json(path, obj):
    tmp = str(path) + ".part"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def layer_tensors(ck, mtp):
    """layer id -> list of 9 checkpoint tensor names."""
    T = ck.tensors()
    if mtp:
        pat = re.compile(r"^mtp\.layers\.(\d+)\.mlp\.switch_mlp\.gate_proj\.weight$")
        base = "mtp.layers.{L}.mlp.switch_mlp."
    else:
        pat = re.compile(r"^language_model\.model\.layers\.(\d+)\.mlp\.switch_mlp\.gate_proj\.weight$")
        base = "language_model.model.layers.{L}.mlp.switch_mlp."
    layers = sorted(int(m.group(1)) for k in T if (m := pat.match(k)))
    out = {}
    for L in layers:
        b = base.format(L=L)
        out[L] = [b + n for n in PIECE_NAMES]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--max-mbps", type=float, default=2000)
    ap.add_argument("--expect-record", type=int, default=2764800)
    ap.add_argument("--allow-buffered", action="store_true")
    a = ap.parse_args(argv)

    model = Path(a.model)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = "mtp_experts" if a.mtp else "experts"
    final = out / f"{stem}.bin"
    tmp = out / f"{stem}.bin.tmp"
    prog_path = out / f"{stem}.progress.json"
    man_path = out / f"{stem}.manifest.json"

    ck = Checkpoint(model / "mtp.safetensors" if a.mtp else model, allow_buffered=a.allow_buffered)
    layers = layer_tensors(ck, a.mtp)
    if not layers:
        raise SystemExit("no switch_mlp tensors found")
    first = layers[min(layers)]
    E = ck.info(first[0]).shape[0]
    sizes = []
    for n in first:
        t = ck.info(n)
        if t.shape[0] != E:
            raise SystemExit(f"{n}: expert dim {t.shape[0]} != {E}")
        sizes.append(t.nbytes // E)
    record = sum(sizes)
    if a.expect_record and record != a.expect_record:
        raise SystemExit(f"record is {record} bytes, expected {a.expect_record}")
    if (E * record) % ALIGN:
        raise SystemExit(f"layer size {E} x {record} is not a multiple of {ALIGN}")
    offsets = [int(x) for x in np.concatenate([[0], np.cumsum(sizes)[:-1]])]
    nl = max(layers) + 1
    if sorted(layers) != list(range(nl)):
        raise SystemExit(f"layers are not contiguous from 0: {sorted(layers)[:5]}...")
    layer_bytes = E * record
    total = nl * layer_bytes

    if final.exists() and not tmp.exists() and man_path.exists():
        print(json.dumps({"status": "already done", "path": str(final)}))
        return 0

    progress = {}
    if prog_path.exists() and tmp.exists():
        progress = {int(k): v for k, v in json.load(open(prog_path)).items()}
    slab = SlabFile(str(tmp), total, a.allow_buffered)
    if slab.fresh:
        progress = {}
    thr = Throttle(a.max_mbps)
    io0 = proc_io()
    t0 = time.time()
    buf = mmap.mmap(-1, layer_bytes)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(E, record)
    done_now = []
    for L in range(nl):
        if L in progress:
            continue
        for j, name in enumerate(layers[L]):
            raw = ck.read_bytes(name)
            thr.spend(len(raw))
            arr[:, offsets[j]:offsets[j] + sizes[j]] = np.frombuffer(raw, dtype=np.uint8).reshape(E, sizes[j])
        mv = memoryview(buf)
        slab.write_at(mv, L * layer_bytes)
        sha = hashlib.sha256(mv).hexdigest()
        mv.release()
        thr.spend(layer_bytes)
        slab.sync()
        progress[L] = sha
        atomic_json(prog_path, {str(k): v for k, v in sorted(progress.items())})
        done_now.append(L)
    del arr
    buf.close()
    slab.sync()
    slab.close()

    missing = [L for L in range(nl) if L not in progress]
    if missing:
        raise SystemExit(f"layers without sha: {missing[:5]}")
    os.replace(tmp, final)
    dfd = os.open(out, os.O_RDONLY)
    os.fsync(dfd)
    os.close(dfd)

    st = os.stat(final)
    synced = ""
    sp = Path(__file__).resolve().parent.parent / ".synced"
    if sp.exists():
        synced = sp.read_text().strip()
    lock = Path(__file__).resolve().parent.parent / "model.lock.json"
    src = {"repo": "unknown", "revision": "unknown"}
    if lock.exists():
        lj = json.load(open(lock))
        src = ({"repo": lj["mtp"]["source_repo"], "revision": lj["mtp"]["source_revision"]} if a.mtp
               else {"repo": lj["repo"], "revision": lj["revision"]})
    manifest = {
        "version": 1,
        "layout": LAYOUT,
        "model_identity": ck.index_sha256,
        "layers": nl,
        "experts": E,
        "pieces": sizes,
        "piece_names": PIECE_NAMES,
        "piece_offsets": offsets,
        "record_bytes": record,
        "alignment": ALIGN,
        "layer_sha256": [progress[L] for L in range(nl)],
        "source": src,
        "repacker": {"path": "tools/repack_experts.py", "synced": synced},
        "stamp": {"dev": st.st_dev, "ino": st.st_ino, "size": st.st_size, "mtime_ns": st.st_mtime_ns,
                  "ctime_ns": st.st_ctime_ns},
        "total_bytes": total,
    }
    atomic_json(man_path, manifest)
    io1 = proc_io()
    summary = {"path": str(final), "layers": nl, "experts": E, "record_bytes": record, "total_bytes": total,
               "written_layers": done_now, "seconds": round(time.time() - t0, 1),
               "direct_write": slab.direct,
               "proc_io": {k: io1.get(k, 0) - io0.get(k, 0) for k in ("rchar", "read_bytes", "wchar", "write_bytes")}}
    s = json.dumps(summary)
    cap = os.environ.get("CAP")
    if cap:
        with open(os.path.join(cap, f"repack_{stem}.json"), "w") as f:
            f.write(s + "\n")
    print(s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
