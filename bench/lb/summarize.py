#!/usr/bin/env python3
"""Summarize a bench/lb/yardstick_session.sh capture directory (stdlib only).

  summarize.py CAPDIR > CAPDIR/summary.json

Layout: CAPDIR/<config>/p1, p2 (the two A/A processes), t<N> (thread sweep), each with
client.jsonl, sample20ms.csv (p1/p2), mlb4.json, envelope.json.
Output per config: prompts {id: {decode_tps, ttft_s: {median,min,max}}}, aa_diff (relative
difference of the p1/p2 decode_tps medians per prompt), mlb2, mlb4, envelope_exceeded, threads,
server {op_offload, memfree_stop_mib} (from p1's envelope.json), memfree {min_mib,
root_direct_growth_mib, rows} pooled over the p1/p2 memfree.csv, memfree_stop (any memfree_stop.json).
"""
import csv
import json
import statistics
import sys
from pathlib import Path


def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"median": None, "min": None, "max": None, "n": 0}
    return {"median": statistics.median(vals), "min": min(vals), "max": max(vals), "n": len(vals)}


def read_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def by_prompt(rows):
    out = {}
    for r in rows:
        out.setdefault(r["prompt_id"], []).append(r)
    return out


def mlb2(csv_path):
    p = Path(csv_path)
    if not p.exists():
        return None
    rows = []
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: (float(v) if v != "" else None) for k, v in r.items()})
            except ValueError:
                pass
    peak, total_bytes = 0.0, 0.0
    for a, b in zip(rows, rows[1:]):
        if a["rd_sectors"] is None or b["rd_sectors"] is None:
            continue
        dt = b["t"] - a["t"]
        if dt <= 0:
            continue
        db = (b["rd_sectors"] - a["rd_sectors"]) * 512
        total_bytes += db
        peak = max(peak, db / dt / 1e6)
    psi = [r[k] for r in rows for k in ("psi0", "psi1", "psi2", "psi3") if r.get(k) is not None]
    dur = rows[-1]["t"] - rows[0]["t"] if len(rows) > 1 else 0
    return {"samples": len(rows), "duration_s": round(dur, 2), "peak_read_mb_s": round(peak, 1),
            "mean_read_mb_s": round(total_bytes / dur / 1e6, 1) if dur > 0 else None,
            "max_inflight": max([r["inflight"] for r in rows if r.get("inflight") is not None] or [None]),
            "max_prod_io_psi_avg10": max(psi) if psi else None}


def memfree_stats(paths):
    """Pool memfree.csv rows: min MemFree and the growth of (root_file - children_file) over its first row."""
    mins, growth, n = [], None, 0
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        first = None
        with open(p) as f:
            for r in csv.DictReader(f):
                n += 1
                try:
                    mins.append(float(r["memfree_mib"]))
                except (KeyError, TypeError, ValueError):
                    pass
                try:
                    d = float(r["root_file_mib"]) - float(r["children_file_mib"])
                except (KeyError, TypeError, ValueError):
                    continue
                if first is None:
                    first = d
                growth = max(growth if growth is not None else d - first, d - first)
    return {"min_mib": min(mins) if mins else None,
            "root_direct_growth_mib": growth, "rows": n}


def summarize(cap):
    cap = Path(cap)
    out = {}
    for cdir in sorted(p for p in cap.iterdir() if p.is_dir()):
        procs = {}
        for pdir in sorted(p for p in cdir.iterdir() if p.is_dir()):
            procs[pdir.name] = pdir
        entry = {"prompts": {}, "aa_diff": {}, "threads": {}, "envelope_exceeded": False}
        meds = {}
        pooled = {}
        for name, pdir in procs.items():
            rows = by_prompt(read_jsonl(pdir / "client.jsonl"))
            if name.startswith("t"):
                tps = [r["decode_tps"] for rs in rows.values() for r in rs]
                entry["threads"][name[1:]] = stats(tps)
                continue
            for pid, rs in rows.items():
                pooled.setdefault(pid, []).extend(rs)
                meds.setdefault(pid, {})[name] = statistics.median(
                    [r["decode_tps"] for r in rs if r["decode_tps"] is not None] or [float("nan")])
            ev = pdir / "envelope.json"
            if ev.exists() and json.loads(ev.read_text()).get("envelope_exceeded"):
                entry["envelope_exceeded"] = True
        for pid, rs in pooled.items():
            entry["prompts"][pid] = {"decode_tps": stats([r["decode_tps"] for r in rs]),
                                     "ttft_s": stats([r["ttft_s"] for r in rs])}
        for pid, m in meds.items():
            if "p1" in m and "p2" in m and m["p1"] == m["p1"] and m["p2"] == m["p2"]:
                mean = (m["p1"] + m["p2"]) / 2
                entry["aa_diff"][pid] = round(abs(m["p1"] - m["p2"]) / mean, 4) if mean else None
        entry["memfree_stop"] = any((pd / "memfree_stop.json").exists() for pd in procs.values())
        entry["memfree"] = memfree_stats(pd / "memfree.csv" for n, pd in procs.items() if n in ("p1", "p2"))
        entry["server"] = None
        if "p1" in procs and (procs["p1"] / "envelope.json").exists():
            e1 = json.loads((procs["p1"] / "envelope.json").read_text())
            entry["server"] = {"op_offload": e1.get("op_offload"), "memfree_stop_mib": e1.get("memfree_stop_mib")}
        if "p1" in procs:
            entry["mlb2"] = mlb2(procs["p1"] / "sample20ms.csv")
            m4 = procs["p1"] / "mlb4.json"
            entry["mlb4"] = json.loads(m4.read_text()) if m4.exists() else None
        out[cdir.name] = entry
    return out


def main(argv):
    if len(argv) != 1:
        print(__doc__)
        return 2
    print(json.dumps(summarize(argv[0]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
