#!/usr/bin/env python3
"""Summarize an IO-2 capture: io2_summary.py CAPDIR > summary.json (stdlib only).

Decision rule (fixed before measuring): with 100-byte rows and 512-byte logical blocks,
24/128 = 18.75% of rows straddle a sector. ratio = p50(c1-1024-qd1) / p50(c1-512-qd1);
row_bytes = 128 if 0.1875 * (ratio - 1) > 0.05 else 100. Rows are always read with O_DIRECT.
"""
import json
import sys
from pathlib import Path

FILE_BYTES = {100: 32_000_153_600, 128: 40_960_196_608}
STRADDLE = 0.1875
THRESHOLD = 0.05


def load_runs(cap):
    runs = {}
    for line in (Path(cap) / "io2.jsonl").read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if "abort" in d:
            raise SystemExit(f"io2.jsonl holds an aborted run: {d}")
        runs[d["label"]] = d
    return runs


def decide(runs):
    p512 = runs["c1-512-qd1"]["lat_us"]["p50"]
    p1024 = runs["c1-1024-qd1"]["lat_us"]["p50"]
    ratio = p1024 / p512
    row_bytes = 128 if STRADDLE * (ratio - 1) > THRESHOLD else 100
    return {"row_bytes": row_bytes, "ratio": round(ratio, 4), "p50_512_us": p512, "p50_1024_us": p1024,
            "straddle_cost": round(STRADDLE * (ratio - 1), 4), "threshold": THRESHOLD,
            "file_bytes": FILE_BYTES[row_bytes]}


def summarize(cap):
    runs = load_runs(cap)
    out = {"n_runs": len(runs), "runs": {}}
    for label, d in runs.items():
        out["runs"][label] = {"mode": d["mode"], "bs": d["bs"], "qd": d["qd"], "GBps": d["GBps"],
                              "IOPS": d["IOPS"], "p50_us": d["lat_us"]["p50"], "p99_us": d["lat_us"]["p99"],
                              "dev_GBps": d.get("dev_GBps"), "dev_avg_req_KiB": d.get("dev_avg_req_KiB")}
    out["decision"] = decide(runs)
    return out


def main(argv):
    if len(argv) != 1:
        print(__doc__)
        return 2
    print(json.dumps(summarize(argv[0]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
