#!/usr/bin/env python3
"""Summarize the raw production reads of prod_reads.sh (stdlib only).

Usage: prom_summary.py CAPDIR > CAPDIR/summary.json

Reads CAPDIR/raw/ and prints a JSON object with:
  v1      POSITIVE iff some single file has `psutil.virtual_memory` and
          (`less than desired GPU memory utilization` or `is_integrated`)
  a1      per load: transient_mib = (peak used - median used before T-10)/MiB - 49844, min_free_mib
  m_lb5   per load: first_rise_s (T -> first sample > baseline + 1 GiB), lower_bound_s = first_rise_s - scrape_interval;
          ok_without_operator iff every lower bound >= 5 s
  b1      per job and hour 04-23: median over days of ttft_p50/ttft_p99/e2e_p99/queue_p99/rph,
          hours with < 20 requests/hour, embeddings_ttft_filled
"""
import json
import re
import statistics
import sys
import time
from pathlib import Path

MIB = 1024 * 1024
GIB = 1024 * MIB
A_BASE_MIB = 49844
METRICS = ("ttft_p50", "ttft_p99", "e2e_p99", "queue_p99", "rph")
HOURS = range(4, 24)


def v1_verdict(text):
    """text: output of the recursive grep, lines `path:lineno:content`."""
    by_file = {}
    for line in text.splitlines():
        m = re.match(r"^(.*?):(\d+):(.*)$", line)
        if not m:
            continue
        by_file.setdefault(m.group(1), []).append(m.group(3))
    hits = []
    for path, rows in sorted(by_file.items()):
        blob = "\n".join(rows)
        if "psutil.virtual_memory" in blob and (
                "less than desired GPU memory utilization" in blob or "is_integrated" in blob):
            hits.append(path)
    return {"verdict": "POSITIVE" if hits else "NEGATIVE", "files": hits}


def _series(resp):
    try:
        res = resp["data"]["result"]
    except (KeyError, TypeError):
        return []
    if not res:
        return []
    out = []
    for ts, v in res[0].get("values", []):
        try:
            f = float(v)
        except ValueError:
            continue
        if f == f:
            out.append((float(ts), f))
    return out


def scrape_interval_s(cfg):
    try:
        y = cfg["data"]["yaml"]
    except (KeyError, TypeError):
        return 15.0
    m = re.search(r"scrape_interval:\s*(\d+)(ms|s|m)", y)
    if not m:
        return 15.0
    n = float(m.group(1))
    return {"ms": n / 1000, "s": n, "m": n * 60}[m.group(2)]


def load_analysis(T, used, free, interval):
    """used/free: [(ts, bytes)]. Returns (a1 dict, m_lb5 dict)."""
    before = [v for t, v in used if t < T - 10]
    base = statistics.median(before) if before else None
    a1 = {"T": T, "baseline_mib": None, "transient_mib": None, "min_free_mib": None}
    if free:
        a1["min_free_mib"] = round(min(v for _, v in free) / MIB, 1)
    mlb = {"T": T, "first_rise_s": None, "lower_bound_s": None}
    if base is not None and used:
        a1["baseline_mib"] = round(base / MIB, 1)
        peak = max(v for _, v in used)
        a1["transient_mib"] = round((peak - base) / MIB - A_BASE_MIB, 1)
        for t, v in used:
            if t >= T and v > base + GIB:
                mlb["first_rise_s"] = round(t - T, 1)
                mlb["lower_bound_s"] = round(t - T - interval, 1)
                break
    return a1, mlb


def b1_tables(raw, jobs):
    tables, low = {}, {}
    for job in jobs:
        per_hour = {h: {k: [] for k in METRICS} for h in HOURS}
        for k in METRICS:
            p = raw / f"b1-{job}-{k}.json"
            if not p.exists():
                continue
            for ts, v in _series(json.loads(p.read_text())):
                h = time.localtime(ts).tm_hour
                if h in per_hour:
                    per_hour[h][k].append(v)
        rows = {}
        for h in HOURS:
            rows[str(h)] = {k: (round(statistics.median(per_hour[h][k]), 4) if per_hour[h][k] else None)
                            for k in METRICS}
        tables[job] = rows
        low[job] = [h for h in HOURS if (rows[str(h)]["rph"] or 0) < 20]
    return tables, low


def summarize(cap):
    raw = Path(cap) / "raw"
    out = {"prom_access": "curl"}
    pa = raw / "prom_access.txt"
    if pa.exists():
        out["prom_access"] = pa.read_text().strip() or "curl"
    v1 = raw / "v1-grep.txt"
    out["v1"] = v1_verdict(v1.read_text()) if v1.exists() else {"verdict": "NEGATIVE", "files": []}

    cfgp = raw / "prom-config.json"
    interval = scrape_interval_s(json.loads(cfgp.read_text())) if cfgp.exists() else 15.0
    out["scrape_interval_s"] = interval
    loads = json.loads((raw / "loads.json").read_text()) if (raw / "loads.json").exists() else []
    a1s, mlbs = [], []
    for ld in loads:
        T = ld["T"]
        up, fp = raw / f"a1-{T}-used.json", raw / f"a1-{T}-free.json"
        if not (up.exists() and fp.exists()):
            continue
        a, m = load_analysis(T, _series(json.loads(up.read_text())), _series(json.loads(fp.read_text())), interval)
        a1s.append(a)
        mlbs.append(m)
    trans = [a["transient_mib"] for a in a1s if a["transient_mib"] is not None]
    out["a1"] = {"loads": a1s, "max_transient_mib": max(trans) if trans else None}
    lbs = [m["lower_bound_s"] for m in mlbs if m["lower_bound_s"] is not None]
    out["m_lb5"] = {"loads": mlbs,
                    "ok_without_operator": (all(x >= 5 for x in lbs) if lbs else None)}

    jobs = json.loads((raw / "b1-jobs.json").read_text()) if (raw / "b1-jobs.json").exists() else []
    tables, low = b1_tables(raw, jobs)
    emb = [j for j in jobs if "embed" in j.lower()]
    filled = bool(emb) and all(tables[j][str(h)]["ttft_p50"] is not None for j in emb for h in HOURS)
    out["b1"] = {"jobs": tables, "low_traffic_hours": low, "embeddings_ttft_filled": filled}
    return out


def main(argv):
    if len(argv) != 1:
        print(__doc__)
        return 2
    print(json.dumps(summarize(argv[0]), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
