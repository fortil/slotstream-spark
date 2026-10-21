#!/usr/bin/env python3
"""F8 gate steps (plan steps 90-92). Runs on the Mac.

  f8_gate.py step token-graph   step 91: skip branch = verify the entry record and the flag; enter branch = sync,
                                build, run the token-graph gate and start the A/B + soak job
  f8_gate.py step verdict       step 92: skip branch = verify again; enter branch = wait for the job (exit 124 while it runs:
                                run it again), then fetch the A/B and the soak and write db/records/decisions/f8-token-graph.md

The branch comes from db/records/decisions/f8-entry.md (written by `decide.py f8-entry --write`). On skip the record and
`[f8] token_graph = false` in slotstream_spark/config/platform.toml are checked, "F8 skipped (host+waits = x% < 10%)" is
printed and the exit is 0. Exit codes: 0 ok, 1 a check failed, 3 the record is missing or unreadable, 75 the enter
branch cannot start its job in time (it must start by 13:15 to finish before 23:45).
"""
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = "db/records/decisions/f8-entry.md"
PLATFORM = "slotstream_spark/config/platform.toml"
G2 = "db/records/measurements/g2-g3.md"
JOB_DEADLINE = (13, 15)

ENTER_TOKEN_GRAPH = [
    ["tools/spark/sync.sh"],
    ["tools/spark/run.sh", "--", "bash", "tools/build_slotio.sh"],
    ["tools/spark/run.sh", "--heavy", "token-graph", "12288", "3600", "--", "tools/spark/py.sh", "-m", "pytest", "-q",
     "-m", "gate", "tests/gates/test_token_graph.py"],
    ["tools/spark/jobs.sh", "start", "f8-ab-soak", "--heavy", "f8-ab-soak", "auto", "36000", "--",
     "bash", "tools/spark/snap.sh", "f8-ab-soak", "bash", "bench/f8_ab_soak.sh"],
]
ENTER_VERDICT = [
    ["tools/spark/jobs.sh", "wait", "f8-ab-soak", "--timeout", "500"],
    ["bash", "-c", 'A=$(tools/spark/fetch.sh --last f8-ab) && S=$(tools/spark/fetch.sh --last f8-ab-soak) && '
                   '.venv/bin/python tools/decide.py f8-token-graph "$A" --soak "$S" --write'],
]


def read_entry(root=ROOT):
    """(decision, fraction, g2) from the entry record; raises ValueError when it cannot be read."""
    p = Path(root) / RECORD
    if not p.exists():
        raise ValueError(f"{RECORD} is missing: run decide.py f8-entry --write first")
    text = p.read_text()
    m = re.search(r"^Decision: (enter|skip)$", text, re.M)
    f = re.search(r"host-and-wait share = ([0-9.]+) ", text)
    g = re.search(r"^- g2 verdict .*: (PASS|FAIL|INFO)$", text, re.M)
    if not (m and f and g):
        raise ValueError(f"{RECORD}: no Decision / fraction / g2 verdict lines")
    return m.group(1), float(f.group(1)), g.group(1)


def flag(root=ROOT):
    text = (Path(root) / PLATFORM).read_text()
    m = re.search(r"^\[f8\]\s*\n(?:[^\[]*?\n)?token_graph\s*=\s*(true|false)", text, re.M)
    if not m:
        raise ValueError(f"{PLATFORM}: no [f8] token_graph")
    return m.group(1) == "true"


def consistent(decision, fraction, g2):
    """The decision must follow its own numbers: enter iff fraction >= 0.10 and g2 passed."""
    return decision == ("enter" if fraction >= 0.10 and g2 == "PASS" else "skip")


def run_commands(cmds, root=ROOT):
    for c in cmds:
        print("+ " + " ".join(c), flush=True)
        rc = subprocess.run(c, cwd=root).returncode
        if rc != 0:
            print(f"f8_gate: `{' '.join(c)}` exited {rc}", file=sys.stderr)
            return rc
    return 0


def remote_no_windows():
    """Pause mode on the Spark ($R/state/PAUSED, NO_WINDOWS=1). Unreachable or absent -> False; a malformed file raises."""
    sys.path.insert(0, str(ROOT))
    from slotstream_spark.guard import paused
    r = subprocess.run(["tools/spark/run.sh", "--", "bash", "-c", 'cat "$HOME/slotstream-spark/state/PAUSED" 2>/dev/null; true'],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return False
    return paused.parse(r.stdout, "state/PAUSED").no_windows


def step(name, root=ROOT, now=None, runner=run_commands, pause_probe=remote_no_windows):
    try:
        decision, fraction, g2 = read_entry(root)
        token_graph = flag(root)
    except ValueError as e:
        print(f"f8_gate: {e}", file=sys.stderr)
        return 3
    if not consistent(decision, fraction, g2):
        print(f"f8_gate: the record says {decision} but fraction {fraction:.4f} and g2 {g2} say otherwise", file=sys.stderr)
        return 1
    if decision == "skip":
        if token_graph:
            print("f8_gate: the entry record says skip but [f8] token_graph is true", file=sys.stderr)
            return 1
        print(f"F8 skipped (host+waits = {fraction * 100:.1f}% < 10%)" if fraction < 0.10
              else f"F8 skipped (host+waits = {fraction * 100:.1f}%, g2 {g2})")
        return 0
    if name == "token-graph":
        t = now or datetime.now()
        if (t.hour, t.minute) > JOB_DEADLINE:       # the 13:15 start rule does not apply while production is paused
            try:
                lifted = pause_probe()
            except ValueError as e:
                print(f"f8_gate: {e}", file=sys.stderr)
                return 1
            if not lifted:
                print("f8_gate: the f8-ab-soak job must start by 13:15 to end before 23:45; environment window", file=sys.stderr)
                return 75
        return runner(ENTER_TOKEN_GRAPH, root)
    if name == "verdict":
        return runner(ENTER_VERDICT, root)
    print(f"f8_gate: unknown step {name!r}", file=sys.stderr)
    return 2


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2 or argv[0] != "step" or argv[1] not in ("token-graph", "verdict"):
        print("usage: f8_gate.py step token-graph|verdict", file=sys.stderr)
        return 2
    return step(argv[1])


if __name__ == "__main__":
    sys.exit(main())
