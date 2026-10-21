#!/usr/bin/env python3
"""Validate db records and query capture directories.

Usage:
  db_check.py FILE...        validate records by directory (measurements/decisions)
  db_check.py latest LABEL   print the last db/sources/runs/*/*/*-LABEL dir
  db_check.py ok LABEL       exit 0 if that capture finished with rc 0 and no cut
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = "db/sources/runs"
ARCHIVED = "db/sources/ARCHIVED.txt"
VERDICT_RE = re.compile(r"^Verdict: (PASS|FAIL|INFO) — \S.*$")


def _runs(root):
    return Path(root) / RUNS


def archived(root=ROOT):
    """Capture ids kept offline: listed one per line in db/sources/ARCHIVED.txt (# starts a comment)."""
    p = Path(root) / ARCHIVED
    if not p.is_file():
        return set()
    return {l.strip() for l in p.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")}


def latest(label, root=ROOT):
    found = sorted(_runs(root).glob(f"*/*/*-{label}"), key=lambda p: p.name)
    found = [p for p in found if p.is_dir()]
    return found[-1] if found else None


def footprint_mib(d):
    """Footprint of a capture dict; captures recorded before the rename carry the old key."""
    return d.get("footprint_mib", d.get("huella_mib"))


def ok(label, root=ROOT):
    d = latest(label, root)
    if d is None:
        return False
    gp = d / "guard-post.json"
    if gp.exists():
        j = json.loads(gp.read_text())
        return j.get("rc") == 0 and j.get("cut") is None
    vj = d / "verdict.json"
    if vj.exists():
        return json.loads(vj.read_text()).get("rc") == 0
    return False


def check_measurement(path, root=ROOT):
    errs = []
    lines = Path(path).read_text().splitlines()
    caps = []
    for l in lines:
        if l.startswith(("Capture:", "Captures:")):
            # "Captures:" lists several dirs; a trailing parenthetical note is not part of the path
            for part in l.split(":", 1)[1].split(","):
                if part.strip():
                    caps.append(part.split()[0])
    if not caps:
        errs.append("no Capture: line")
    offline = archived(root)
    for c in caps:
        d = Path(root) / c
        if not d.is_dir():
            if d.name in offline:
                continue  # archived offline: listed in db/sources/ARCHIVED.txt
            errs.append(f"capture dir missing: {c}")
        elif not ((d / "guard-post.json").exists() or (d / "verdict.json").exists()):
            errs.append(f"capture has no guard-post.json or verdict.json: {c}")
    verdicts = [l for l in lines if l.startswith("Verdict:")]
    if len(verdicts) != 1:
        errs.append(f"expected exactly one Verdict: line, found {len(verdicts)}")
    elif not VERDICT_RE.match(verdicts[0]):
        errs.append(f"invalid Verdict: line: {verdicts[0]}")
    return errs


def check_decision(path, root=ROOT):
    errs = []
    lines = Path(path).read_text().splitlines()

    def field(name):
        for l in lines:
            if l.startswith(name + ":"):
                return l.split(":", 1)[1].strip()
        return None

    if not field("Decision"):
        errs.append("no Decision:")
    ev = field("Evidence")
    if not ev:
        errs.append("no Evidence:")
    else:
        paths = [p.strip() for p in re.split(r"[,\s]+", ev) if p.strip()]
        if not any((Path(root) / p).exists() for p in paths):
            errs.append("Evidence: names no existing path")
    if not field("Reversal"):
        errs.append("no Reversal:")
    return errs


def check_file(path, root=ROOT):
    p = Path(path)
    parts = p.parts
    if "measurements" in parts:
        return check_measurement(p, root)
    if "decisions" in parts:
        return check_decision(p, root)
    return [f"not under records/measurements or records/decisions: {path}"]


def main(argv):
    if len(argv) >= 2 and argv[0] == "latest":
        d = latest(argv[1])
        if d is None:
            return 1
        print(d.relative_to(ROOT))
        return 0
    if len(argv) >= 2 and argv[0] == "ok":
        return 0 if ok(argv[1]) else 1
    if not argv:
        print(__doc__)
        return 2
    rc = 0
    for f in argv:
        if Path(f).name == "README.md":
            continue
        errs = check_file(f)
        for e in errs:
            print(f"{f}: {e}")
        rc = rc or (1 if errs else 0)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
