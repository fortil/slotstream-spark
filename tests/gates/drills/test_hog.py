"""Step 66: a bounded memory hog against the engine in D5 and D1.

The hog is tools/drills/hog.py's loop run in this process (oom_score_adj 1000): 1 GiB per step, >= 2 s apart, stopping at
its own limits (MemAvailable <= 14,336 MiB, engine exited or at the floor, /running non-empty, STOP file, 23:45, or
engine cap - floor). Every drop of the engine's pool in its own 20 ms series (`footprint-serve.jsonl`, unified `pool`)
is a release: its delay is measured from the latest hog event before it (the start of a 1 GiB step, which is when the
pages begin to be taken, or its end, whichever is later) and must be <= 2 s, and MemAvailable must rise
>= 0.9 GiB in the 20 ms series of this process right after it. MemAvailable never below 8 GiB, no oom in `journalctl -k`.
"""
import os
import time
from argparse import Namespace

import pytest

import hog
from conftest import GIB, oom_lines, stamp, write_record

pytestmark = pytest.mark.gate

RELEASE_MAX_S = 2.0
RISE_MIN_BYTES = int(0.9 * GIB)
AVAIL_MIN_BYTES = 8 * GIB
CHUNK_BYTES = GIB
HOG_LIMITS = {"memavailable_floor", "engine_log_floor", "engine_exited", "running_nonempty", "stop_file", "deadline",
              "size_cap"}


def releases(pool_rows):
    """pool_rows: [(t_ns, pool_bytes)] -> [(t_ns, chunks_released)] at each drop of the pool."""
    out, prev = [], None
    for t, b in pool_rows:
        if prev is not None and b <= prev - CHUNK_BYTES // 2:
            out.append((t, round((prev - b) / CHUNK_BYTES)))
        prev = b
    return out


def analyse(steps_ns, rel, avail_rows):
    """Per release: delay from the latest hog event (step start or end), MemAvailable rise right after it (20 ms series).
    A release in the middle of a step is attributed to that step (the engine saw the pages being touched)."""
    res = []
    for t, k in rel:
        before = [s for s in steps_ns if s <= t]
        delay = (t - before[-1]) / 1e9 if before else None
        pre = [a for (ts, a) in avail_rows if t - 300_000_000 <= ts < t]
        post = [a for (ts, a) in avail_rows if t <= ts <= t + 1_500_000_000]
        rise = (max(post) - min(pre)) if pre and post else None
        res.append({"t_ns": t, "chunks": k, "delay_s": None if delay is None else round(delay, 3),
                    "avail_rise_bytes": rise})
    return res


@pytest.mark.parametrize("mode", ["D5", "D1"])
def test_hog_releases_chunks(mode, engine_factory, stub, series):
    from slotstream_spark.memory import footprint as fp
    since = stamp()
    stub.set()
    eng = engine_factory(f"hog-{mode.lower()}", mode)
    load_s = eng.wait_ready()
    mem = eng.get("/internal/memory")
    cap_mib, chunks0 = int(mem["cap_bytes"]) >> 20, int(mem["chunks"])
    assert chunks0 > 2, f"{mode}: engine started at {chunks0} chunks, nothing to release"
    with open("/proc/self/oom_score_adj", "w") as f:
        f.write("1000")
    a = hog.parse(["--engine-pid", str(eng.proc.pid), "--engine-log", str(eng.log_path), "--engine-cap-mib", str(cap_mib),
                   "--running-url", "http://127.0.0.1:18080/running", "--run-dir", str(eng.dir)])
    steps_ns, starts_ns = [], []

    def alloc(n):                                                   # hog.run's touch(), with the step start stamped
        starts_ns.append(time.monotonic_ns())
        b = bytearray(n << 20)
        for i in range(0, len(b), hog.PAGE):
            b[i] = 1
        return b
    time.sleep(2)                                                   # a quiet baseline in the series
    reason, size_mib, nsteps = hog.run(a, hog.Inputs(a), alloc=alloc,
                                       on_step=lambda size: steps_ns.append(time.monotonic_ns()))
    time.sleep(3)                                                   # let the engine finish what it was doing
    avail_rows = series.snapshot()
    fps, flags = eng.footprint()
    eng.kill()
    pool_rows = [(r["t_ns"], int(r["unified"].get("pool", 0))) for r in fps]
    rel = releases(pool_rows)
    ev = analyse(sorted(steps_ns + starts_ns), rel, avail_rows)
    t0 = starts_ns[0] if starts_ns else 0
    avail_min = min((av for (ts, av) in avail_rows if ts >= t0 - 1_000_000_000), default=None)
    samples = [fp.Sample(str(r["t_ns"]), r["vmrss"], r["vmhwm"], r["nvml"], 0, {k: int(v) for k, v in r["unified"].items()})
               for r in fps]
    pk = fp.peak(samples, fp.flags_from_dict(flags)) if samples else None
    ooms = oom_lines(since)
    write_record(f"hog-{mode.lower()}", {
        "mode": mode, "load_s": round(load_s, 1), "engine_cap_mib": cap_mib, "chunks_start": chunks0,
        "hog_stop_reason": reason, "hog_size_mib": size_mib, "hog_steps": nsteps,
        "releases": ev, "release_delay_limit_s": RELEASE_MAX_S, "avail_rise_min_bytes": RISE_MIN_BYTES,
        "avail_min_bytes": avail_min, "avail_min_limit_bytes": AVAIL_MIN_BYTES,
        "engine_peak_once_bytes": pk.once_bytes if pk else None, "oom_lines": ooms,
        "governor_shrinks": [r for r in eng.governor() if r["decision"].startswith("ShrinkTo")]})
    assert reason in HOG_LIMITS, reason
    assert ev, f"{mode}: the hog ({nsteps} steps, stop {reason}) triggered no release; the drill proved nothing"
    for e in ev:
        assert e["delay_s"] is not None and e["delay_s"] <= RELEASE_MAX_S, e
        assert e["avail_rise_bytes"] is not None and e["avail_rise_bytes"] >= RISE_MIN_BYTES, e
    assert avail_min is not None and avail_min >= AVAIL_MIN_BYTES, f"MemAvailable min {avail_min}"
    assert not ooms
