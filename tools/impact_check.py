#!/usr/bin/env python3
"""Production impact check of a capture (plan section 2.4, stdlib only).

Usage: impact_check.py CAPDIR --baseline B1_SUMMARY

Items:
  1 oom         guard-post.json: kernel log clean (no oom-kill / NV_ERR_NO_MEMORY)
  2 memavail    min MemAvailable in meminfo.csv >= 8192 MiB
  3 latency     prometheus-window.json vs the b1 hour baseline: with >= 20 requests the
                TTFT p99 stays within +20%; with fewer, medians + queue time; no 5xx series
                exists for these jobs, so a low-traffic window is at best NOT_MEASURED
                (errors_5xx "not measured") unless the window carries errors_5xx;
                0 requests, or a baseline hour with no ttft_p50/queue_p99, is NOT_MEASURED
  4 psi         prod-psi.csv: every some avg10 < 10
With production paused by the user ($R/state/PAUSED exists, or guard-post.json says "paused": true) items 3 and 4
report NOT_APPLICABLE "not_applicable: production paused by the user" (they need live production) and do not
count toward the verdict; items 1 and 2 stay gates. The 14-container check lives in the guard (EXPECTED_CONTAINERS).
Prints {"verdict": PASS|FAIL|NOT_MEASURED, "items": {...}}; exit 1 only on FAIL.
"""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slotstream_spark.guard import paused as _paused  # noqa: E402

NOT_APPLICABLE_PAUSED = "not_applicable: production paused by the user"
MIN_AVAIL_MIB = 8192
PSI_LIMIT = 10.0
MIN_REQ = 20
SLACK = 1.2
QUEUE_FLOOR_S = 0.05


def item_oom(cap):
    p = cap / "guard-post.json"
    if not p.exists():
        return {"status": "NOT_MEASURED", "detail": "no guard-post.json"}
    j = json.loads(p.read_text())
    klog = j.get("klog")
    if klog == "clean":
        return {"status": "PASS", "detail": "klog clean"}
    if klog in (None, "unreadable"):
        return {"status": "NOT_MEASURED", "detail": f"klog {klog}"}
    return {"status": "FAIL", "detail": f"klog {klog}"}


def item_memavail(cap):
    p = cap / "meminfo.csv"
    if not p.exists():
        return {"status": "NOT_MEASURED", "detail": "no meminfo.csv"}
    vals = []
    with open(p) as f:
        for row in csv.DictReader(f):
            try:
                vals.append(int(row["MemAvailable_kB"]))
            except (KeyError, ValueError, TypeError):
                pass
    if not vals:
        return {"status": "NOT_MEASURED", "detail": "no samples"}
    mn = min(vals) // 1024
    return {"status": "PASS" if mn >= MIN_AVAIL_MIB else "FAIL", "min_mib": mn}


def capture_hour(cap):
    try:
        v = json.loads((cap / "verdict.json").read_text())
        return datetime.fromisoformat(v["start"]).hour
    except Exception:
        return None


def item_latency(cap, baseline):
    p = cap / "prometheus-window.json"
    if not p.exists():
        return {"status": "NOT_MEASURED", "detail": "no prometheus-window.json"}
    win = json.loads(p.read_text())
    hour = capture_hour(cap)
    base_jobs = baseline.get("b1", {}).get("jobs", {})
    per_job, fail, measured = {}, False, False
    for job, m in win.get("jobs", {}).items():
        req = m.get("requests") or 0
        base = base_jobs.get(job, {}).get(str(hour)) if hour is not None else None
        if req <= 0:
            per_job[job] = {"status": "NOT_MEASURED", "detail": "0 requests"}
            continue
        if not base:
            per_job[job] = {"status": "NOT_MEASURED", "detail": "no baseline for hour"}
            continue
        if req >= MIN_REQ:
            b, v = base.get("ttft_p99"), m.get("ttft_p99")
            if b is None or v is None:
                per_job[job] = {"status": "NOT_MEASURED", "detail": "ttft_p99 missing"}
                continue
            ok = v <= b * SLACK
            per_job[job] = {"status": "PASS" if ok else "FAIL", "ttft_p99": v, "baseline": b, "requests": req}
        else:
            checks = []
            b, v = base.get("ttft_p50"), m.get("ttft_p50")
            if b is not None and v is not None:
                checks.append(v <= b * SLACK)
            bq, vq = base.get("queue_p99"), m.get("queue_p99")
            if bq is not None and vq is not None:
                checks.append(vq <= max(bq * SLACK, bq + QUEUE_FLOOR_S))
            if not checks:
                per_job[job] = {"status": "NOT_MEASURED", "detail": "no usable baseline for hour",
                                "requests": req, "low_traffic": True}
                continue
            e5 = m.get("errors_5xx")
            if e5 is not None:
                checks.append(e5 == 0)
            ok = all(checks)
            if not ok:
                st = "FAIL"
            elif e5 is None:
                st = "NOT_MEASURED"
            else:
                st = "PASS"
            per_job[job] = {"status": st, "requests": req, "low_traffic": True,
                            "errors_5xx": e5 if e5 is not None else "not measured"}
            if st == "NOT_MEASURED":
                continue
        measured = True
        fail = fail or per_job[job]["status"] == "FAIL"
    if fail:
        status = "FAIL"
    elif measured:
        status = "PASS"
    else:
        status = "NOT_MEASURED"
    return {"status": status, "jobs": per_job, "hour": hour}


def item_psi(cap):
    p = cap / "prod-psi.csv"
    if not p.exists():
        return {"status": "NOT_MEASURED", "detail": "no prod-psi.csv"}
    mx = None
    with open(p) as f:
        for row in csv.DictReader(f):
            for k in ("io_some_avg10", "mem_some_avg10"):
                try:
                    v = float(row[k])
                except (KeyError, ValueError, TypeError):
                    continue
                mx = v if mx is None else max(mx, v)
    if mx is None:
        return {"status": "NOT_MEASURED", "detail": "no samples"}
    return {"status": "PASS" if mx < PSI_LIMIT else "FAIL", "max_avg10": mx}


def is_paused(cap):
    """True when the capture was taken under the pause file (guard-post.json) or the state file exists here."""
    p = Path(cap) / "guard-post.json"
    try:
        if json.loads(p.read_text()).get("paused") is True:
            return True
    except (OSError, ValueError):
        pass
    return _paused.paused_active()


def check(cap, baseline, paused=None):
    cap = Path(cap)
    if paused is None:
        paused = is_paused(cap)
    na = {"status": "NOT_APPLICABLE", "detail": NOT_APPLICABLE_PAUSED}
    items = {
        "oom": item_oom(cap),
        "memavail": item_memavail(cap),
        "latency": dict(na) if paused else item_latency(cap, baseline),
        "psi": dict(na) if paused else item_psi(cap),
    }
    sts = [i["status"] for i in items.values() if i["status"] != "NOT_APPLICABLE"]
    if "FAIL" in sts:
        verdict = "FAIL"
    elif "NOT_MEASURED" in sts:
        verdict = "NOT_MEASURED"
    else:
        verdict = "PASS"
    return {"verdict": verdict, "items": items}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("capdir")
    ap.add_argument("--baseline", required=True)
    a = ap.parse_args(argv)
    baseline = json.loads(Path(a.baseline).read_text())
    res = check(a.capdir, baseline)
    print(json.dumps(res, indent=1))
    return 1 if res["verdict"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
