#!/usr/bin/env python3
"""Bounded memory hog for guard drills.

  hog.py [--engine-pid PID] [--engine-log F] [--engine-cap-mib N] [--running-url U] [--run-dir D] [--dry-run]

Writes 1000 to oom_score_adj, grows 1 GiB per step (>= 2 s apart) touching every 4 KiB page, and stops, freeing everything, on
the first of: MemAvailable <= 14,336 MiB; the engine log reaching the floor (`<log>.stop` exists or its tail mentions "floor");
the engine process exiting; /running non-empty; `$R/run/STOP` existing; 23:45; or its own size reaching
(engine cap at start - floor). `--dry-run` takes the meminfo, the clock and the other inputs from --fake FILE (JSON) and
allocates nothing real. The stop reason is printed as `STOP <reason>`.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

FLOOR_MIB = 14336
DEADLINE = (23, 45)
PAGE = 4096


class Inputs:
    """Live inputs; the dry-run replaces each with a fake."""

    def __init__(self, a):
        self.a = a

    def mem_available_mib(self):
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
        raise RuntimeError("no MemAvailable")

    def hhmm(self):
        t = time.localtime()
        return t.tm_hour, t.tm_min

    def running_nonempty(self):
        if not self.a.running_url:
            return False
        try:
            return bool(json.load(urllib.request.urlopen(self.a.running_url, timeout=3)).get("running"))
        except Exception:                                       # noqa: BLE001
            return True                                         # unreachable counts as "not provably empty"

    def stop_file(self):
        return bool(self.a.run_dir) and (Path(self.a.run_dir) / "STOP").exists()

    def engine_alive(self):
        if not self.a.engine_pid:
            return True
        try:
            os.kill(self.a.engine_pid, 0)
            return True
        except OSError:
            return False

    def engine_log_floor(self):
        log = self.a.engine_log
        if not log:
            return False
        if Path(str(log) + ".stop").exists():
            return True
        try:
            with open(log, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 4096))
                return re.search(rb"floor", f.read(), re.I) is not None
        except OSError:
            return False


class FakeInputs(Inputs):
    """Dry-run: values come from a dict that the test (or --fake FILE) mutates between steps."""

    def __init__(self, a, state):
        super().__init__(a)
        self.s = state

    def mem_available_mib(self):
        return self.s["mem_available_mib"]

    def hhmm(self):
        return tuple(self.s.get("hhmm", (12, 0)))

    def running_nonempty(self):
        return bool(self.s.get("running"))

    def stop_file(self):
        return bool(self.s.get("stop_file"))

    def engine_alive(self):
        return self.s.get("engine_alive", True)

    def engine_log_floor(self):
        return bool(self.s.get("engine_log_floor"))


def stop_reason(inp, size_mib, size_cap_mib):
    """The first stop rule that holds, or None. Order = severity: memory first."""
    if inp.mem_available_mib() <= FLOOR_MIB:
        return "memavailable_floor"
    if inp.engine_log_floor():
        return "engine_log_floor"
    if not inp.engine_alive():
        return "engine_exited"
    if inp.running_nonempty():
        return "running_nonempty"
    if inp.stop_file():
        return "stop_file"
    if tuple(inp.hhmm()) >= DEADLINE:
        return "deadline"
    if size_cap_mib is not None and size_mib >= size_cap_mib:
        return "size_cap"
    return None


def run(a, inp, *, alloc=None, sleep=time.sleep, max_steps=None, on_step=None):
    """Grow until a stop rule fires; always frees. -> (reason, size_mib, steps)."""
    chunk = a.chunk_mib
    cap = (a.engine_cap_mib - FLOOR_MIB) if a.engine_cap_mib is not None else None
    held, size, steps = [], 0, 0

    def touch(mib):
        b = bytearray(mib << 20)
        for i in range(0, len(b), PAGE):
            b[i] = 1
        return b
    alloc = alloc or touch
    reason = None
    try:
        while True:
            reason = stop_reason(inp, size, cap)
            if reason:
                break
            if max_steps is not None and steps >= max_steps:
                reason = "max_steps"
                break
            n = chunk if cap is None else min(chunk, cap - size)
            held.append(alloc(n))
            size += n
            steps += 1
            if on_step:
                on_step(size)
            sleep(a.interval_s)
    finally:
        held.clear()
    return reason, size, steps


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine-pid", type=int)
    ap.add_argument("--engine-log")
    ap.add_argument("--engine-cap-mib", type=int)
    ap.add_argument("--running-url", default="http://localhost:8080/running")
    ap.add_argument("--run-dir", default=os.environ.get("SLOTSTREAM_R", "") and os.path.join(os.environ["SLOTSTREAM_R"], "run"))
    ap.add_argument("--chunk-mib", type=int, default=1024)
    ap.add_argument("--interval-s", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fake", help="dry-run inputs as JSON")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    if a.dry_run:
        state = json.loads(Path(a.fake).read_text()) if a.fake else {"mem_available_mib": 10 ** 6}
        inp = FakeInputs(a, state)
        reason, size, steps = run(a, inp, alloc=lambda n: n, sleep=lambda s: None, max_steps=int(state.get("max_steps", 64)))
    else:
        try:
            Path("/proc/self/oom_score_adj").write_text("1000")
        except OSError as e:
            print(f"hog: cannot set oom_score_adj: {e}", file=sys.stderr)
            return 2
        reason, size, steps = run(a, Inputs(a))
    print(f"STOP {reason} size_mib={size} steps={steps}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
