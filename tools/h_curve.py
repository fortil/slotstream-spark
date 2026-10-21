#!/usr/bin/env python3
"""Hit-rate curve h(experts per layer) of the router traces of a session (step 65).

  h_curve.py TRACES_DIR --out slotstream_spark/config/h_curve_gb10.json

TRACES_DIR holds trace-*.npz, each with `ids` of shape (steps, layers, topk). The cache is global across layers, so a
pool of N experts per layer is N * layers records. Replays every trace in order through the LRU of
third_party/slotstream/Tools/cachesim.py and through an O(1) CLOCK simulator (second chance). Needs numpy.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "slotstream" / "Tools"))
from cachesim import LRU, trace_to_keys  # noqa: E402

POINTS = (13, 30, 48, 81, 105, 150, 200, 300, 396, 512)


class Clock:
    """CLOCK with one reference bit per resident record; access is O(1) amortised."""
    name = "CLOCK"

    def __init__(self, cap):
        self.cap = cap
        self.slot = {}          # key -> ring index
        self.keys = [None] * cap
        self.ref = [False] * cap
        self.hand = 0

    def access(self, key):
        i = self.slot.get(key)
        if i is not None:
            self.ref[i] = True
            return True
        while True:
            if self.keys[self.hand] is None or not self.ref[self.hand]:
                break
            self.ref[self.hand] = False
            self.hand = (self.hand + 1) % self.cap
        old = self.keys[self.hand]
        if old is not None:
            del self.slot[old]
        self.keys[self.hand] = key
        self.ref[self.hand] = False
        self.slot[key] = self.hand
        self.hand = (self.hand + 1) % self.cap
        return False


def hit_rate(key_lists, cap, cls):
    """Hit rate over the whole session; the cache persists across traces (one server, many requests)."""
    p = cls(cap)
    hits = total = 0
    for keys in key_lists:
        for k in keys:
            hits += 1 if p.access(int(k)) else 0
        total += len(keys)
    return hits / total if total else 0.0


def load_keys(traces_dir):
    files = sorted(glob.glob(str(Path(traces_dir) / "trace-*.npz")))
    if not files:
        raise SystemExit(f"h_curve: no trace-*.npz in {traces_dir}")
    out, layers = [], None
    for f in files:
        arr = np.load(f)["ids"]
        layers = arr.shape[1]
        out.append(trace_to_keys(arr).tolist())
    return out, layers, len(files)


def curve(key_lists, layers, points=POINTS):
    rows = []
    for n in points:
        cap = n * layers
        rows.append({"experts_per_layer": n, "records": cap,
                     "lru": round(hit_rate(key_lists, cap, LRU), 6),
                     "clock": round(hit_rate(key_lists, cap, Clock), 6)})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("traces")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    keys, layers, n = load_keys(a.traces)
    rows = curve(keys, layers)
    res = {"layers": layers, "traces": n, "accesses": sum(len(k) for k in keys), "curve": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    for r in rows:
        print(f"{r['experts_per_layer']:4d}/layer  lru {r['lru']:.4f}  clock {r['clock']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
