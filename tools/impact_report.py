#!/usr/bin/env python3
"""Production impact of a capture, the F9 report of plan section 2.4 (stdlib only; step 93 builds it, step 95 runs it).

  impact_report.py --label f9-engine [--capture DIR] [--baseline summary.json] [--write [PATH]]

Criteria over the job window (the numbering is the plan's):
  1 oom          no oom-kill / NV_ERR_NO_MEMORY in the kernel log since the job started       (guard-post.json)
  2 memavail     MemAvailable never below 8 GiB                                                (meminfo.csv)
  3 latency      p99 TTFT of the two vLLM services <= 1.20 x the b1 hour baseline; with 1-19 requests the medians and the
                 queue time instead, plus 0 5xx; with 0 requests `not_measured`, which fails without a user decision record
  4 psi          per-cgroup io and memory pressure `some avg10` < 10% for the docker scopes of the ASR and embeddings
                 services, postgres and qdrant (never the system-wide PSI)                     (prod-psi.csv, per container)
  7 quiet window the run did not overlap 00:00-04:00 local time
  8 containers   the expected count of containers Up before and after (EXPECTED_CONTAINERS or the pause file's count; `not_configured` when neither is set), no `slotstream-spark-*`
                 container left, no process of ours on the GPU after the run, oom_score_adj 1000 on every server we started

While the production services are paused by the user (guard-post.json says `paused: true`) criteria 3, 4 and 7 need live
production or its time windows: they are recorded as `not_applicable: production paused by the user` (the user decision of
2026-10-14 17:40) and do not count; 1, 2 and 8 stay gates. Exit 0 = PASS, 1 = a criterion failed or could not be measured
(`... needs a decision`), 3 = unreadable input.
"""
import argparse
import csv
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import db_check          # noqa: E402
import impact_check as ic  # noqa: E402

NA = "not_applicable: production paused by the user"
PSI_FAMILIES = ("asr", "embedding", "postgres", "qdrant")
DEFAULT_RECORD = "db/records/measurements/f9-impact.md"


def record_id(label):
    """`f9-engine` -> `f9-impact`, `w131k-engine` -> `w131k-impact`: one impact record per job label."""
    return (label[:-len("-engine")] if label.endswith("-engine") else label) + "-impact"


def record_path(label):
    return f"db/records/measurements/{record_id(label)}.md"
QUIET_START_H, QUIET_END_H = 0, 4
DEFAULT_EXPECTED = None


class BadInput(Exception):
    pass


def _json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError) as e:
        raise BadInput(f"{p}: {e}")


def guard_post(cap):
    return _json(Path(cap) / "guard-post.json")


def is_paused(cap):
    try:
        return guard_post(cap).get("paused") is True
    except BadInput:
        return False


def item_psi_cgroups(cap):
    """Per-family max of the docker scopes' `some avg10` (io and memory) from prod-psi.csv rows (epoch, container, io, mem)."""
    p = Path(cap) / "prod-psi.csv"
    if not p.exists():
        return {"status": "NOT_MEASURED", "detail": "no prod-psi.csv"}
    fam = {f: None for f in PSI_FAMILIES}
    with open(p) as f:
        for row in csv.DictReader(f):
            name = (row.get("container") or "").lower()
            for fa in PSI_FAMILIES:
                if fa in name:
                    for k in ("io_some_avg10", "mem_some_avg10"):
                        try:
                            v = float(row[k])
                        except (KeyError, ValueError, TypeError):
                            continue
                        fam[fa] = v if fam[fa] is None else max(fam[fa], v)
    seen = {k: v for k, v in fam.items() if v is not None}
    if not seen:
        return {"status": "NOT_MEASURED", "detail": "no row for the ASR, embeddings, postgres or qdrant scopes"}
    worst = max(seen.values())
    return {"status": "PASS" if worst < ic.PSI_LIMIT else "FAIL", "max_avg10": worst, "per_family": seen,
            "missing_families": sorted(k for k, v in fam.items() if v is None)}


def item_quiet(cap):
    try:
        v = _json(Path(cap) / "verdict.json")
        a, b = datetime.fromisoformat(v["start"]), datetime.fromisoformat(v["end"])
    except (BadInput, KeyError, ValueError):
        return {"status": "NOT_MEASURED", "detail": "no start/end in verdict.json"}
    hits = []
    day = a.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= b:
        lo, hi = day.replace(hour=QUIET_START_H), day.replace(hour=QUIET_END_H)
        if a < hi and b > lo:
            hits.append(day.date().isoformat())
        day += timedelta(days=1)
    return {"status": "FAIL" if hits else "PASS", "start": v["start"], "end": v["end"], "overlaps_quiet_window_on": hits}


def server_oom_scores(cap):
    out = []
    for r in sorted(Path(cap).glob("*/result.json")):
        try:
            d = json.loads(r.read_text())
        except ValueError:
            continue
        for s in d.get("samples", []):
            if isinstance(s, dict) and "oom_score_adj" in s:
                out.append((r.parent.name, s.get("sample"), s["oom_score_adj"]))
    return out


def item_containers(cap):
    gp = guard_post(cap)
    expected = gp.get("expected_containers", DEFAULT_EXPECTED)
    expected = None if expected is None else int(expected)       # null or missing: no count was configured
    up = gp.get("containers_up")
    problems = []
    if expected is not None and up != expected:
        problems.append(f"{up} containers Up after, expected {expected}")
    if gp.get("own_containers", 0) != 0:
        problems.append(f"{gp['own_containers']} slotstream-spark container(s) left")
    if gp.get("nvsmi_own_pids", 0) != 0:
        problems.append(f"{gp['nvsmi_own_pids']} process(es) of ours still on the GPU")
    scores = server_oom_scores(cap)
    bad = [x for x in scores if x[2] != 1000]
    if bad:
        problems.append(f"oom_score_adj != 1000 on {len(bad)} server(s): {bad[:3]}")
    status = "FAIL" if problems else ("NOT_CONFIGURED" if expected is None else "PASS")
    return {"status": status, "containers_up": up, "expected": expected,
            "own_containers": gp.get("own_containers"), "servers_checked_oom_score_adj": len(scores),
            "problems": problems}


def check(cap, baseline):
    """-> {"verdict": PASS|FAIL|NOT_MEASURED, "items": {...}, "paused": bool, "failed": [names]}."""
    cap = Path(cap)
    paused = is_paused(cap)
    na = {"status": "NOT_APPLICABLE", "detail": NA}
    items = {
        "1 oom": ic.item_oom(cap),
        "2 memavail": ic.item_memavail(cap),
        "3 latency": dict(na) if paused else ic.item_latency(cap, baseline),
        "4 psi (per cgroup)": dict(na) if paused else item_psi_cgroups(cap),
        "7 quiet window": dict(na) if paused else item_quiet(cap),
        "8 containers": item_containers(cap),
    }
    sts = [i["status"] for i in items.values() if i["status"] != "NOT_APPLICABLE"]
    verdict = "FAIL" if "FAIL" in sts else ("NOT_MEASURED" if "NOT_MEASURED" in sts else "PASS")
    failed = [k for k, i in items.items() if i["status"] in ("FAIL", "NOT_MEASURED")]
    return {"verdict": verdict, "items": items, "paused": paused, "failed": failed}


def detail(item):
    d = {k: v for k, v in item.items() if k != "status"}
    return json.dumps(d, sort_keys=True) if d else ""


def render(res, capture_rel, label):
    lines = [f"# {record_id(label)}: production impact over the {label} job window", f"Date: {date.today().isoformat()}",
             f"Capture: {capture_rel}",
             f"Command: .venv/bin/python tools/impact_report.py --label {label} --write", "",
             "| criterion | status | detail |", "|---|---|---|"]
    for k, i in res["items"].items():
        lines.append(f"| {k} | {i['status']} | {detail(i).replace('|', '/')} |")
    lines.append("")
    if res["paused"]:
        lines.append("- criteria 3, 4 and 7 are recorded as \"" + NA + "\" (user decision of 2026-10-14 17:40: the services were "
                     "stopped by the user, so there is no live production to measure and no window to protect); 1, 2 and 8 "
                     "are gates")
    if res["verdict"] == "PASS":
        lines.append("Verdict: PASS — no oom-kill, MemAvailable >= 8 GiB, expected containers Up, no process of ours left")
    else:
        lines.append("Verdict: FAIL — " + ", ".join(res["failed"]) + " needs a decision")
    return "\n".join(lines) + "\n"


def capture_rel(p):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT)) + "/"
    except ValueError:
        return str(p) + "/"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", default="f9-engine")
    ap.add_argument("--capture")
    ap.add_argument("--baseline")
    ap.add_argument("--write", nargs="?", const="", default=None)
    a = ap.parse_args(argv)
    cap = Path(a.capture) if a.capture else db_check.latest(a.label)
    if cap is None or not Path(cap).is_dir():
        print(f"impact_report: no capture for {a.label}", file=sys.stderr)
        return 3
    base_path = a.baseline
    if not base_path:
        b = db_check.latest("prod-reads")
        base_path = str(b / "summary.json") if b else None
    try:
        baseline = _json(base_path) if base_path else {}
        res = check(cap, baseline)
    except BadInput as e:
        print(f"impact_report: unreadable input: {e}", file=sys.stderr)
        return 3
    for k, i in res["items"].items():
        print(f"{i['status']:14s} {k} {detail(i)}")
    if a.write is not None:
        dest = Path(a.write or record_path(a.label))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(render(res, capture_rel(cap), a.label))
        print(f"wrote {dest}")
    if res["verdict"] != "PASS":
        print(f"impact: {', '.join(res['failed'])} needs a decision ({res['verdict']})")
        return 1
    print("impact: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
