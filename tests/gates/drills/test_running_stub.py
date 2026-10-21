"""Step 66: the engine against the /running stub (`SLOTSTREAM_RUNNING_URL=http://127.0.0.1:18080/running`).

D5: a resident appears -> the pool goes to the floor (2 chunks) in <= 2 s; still there -> the engine exits.
D1: A -> floor <= 2 s; B -> exit <= 2 s; A `state=ready` and MemAvailable stable -> regrows one chunk per >= 60 s.
D2: A -> floor <= 2 s.
The floor time is the stub write to the end of the shrink in the governor log (poll time + freed_t); both sides use
CLOCK_MONOTONIC. Run: tools/spark/run.sh --heavy drills auto 5400 -- tools/spark/py.sh -m pytest -q -m gate tests/gates/drills
"""
import time

import pytest

from conftest import FLOOR_CHUNKS, oom_lines, stamp, write_record

pytestmark = pytest.mark.gate

FLOOR_MAX_S = 2.0
EXIT_MAX_S = 2.0
D5_EXIT_MAX_S = 15.0             # governor d5 exit_after_s is 5 s at the floor, plus the shrink and the poll
GROW_GAP_MIN_S = 59.9
GROW_WAIT_S = 420


def floor_seconds(eng, t_flip):
    """Seconds from the stub write to the pool at the floor; None when the governor never logged it."""
    for r in eng.governor():
        if r["t"] >= t_flip - 0.3 and r["chunks_after"] <= FLOOR_CHUNKS and r["decision"].startswith("ShrinkTo"):
            return round(r["t"] + (r["freed_t"] or 0.0) - t_flip, 3)
    return None


def wait_floor(eng, t_flip, timeout_s=10.0):
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        v = floor_seconds(eng, t_flip)
        if v is not None:
            return v
        time.sleep(0.05)
    return None


def start_engine(factory, stub, label, mode):
    stub.set()
    eng = factory(label, mode)
    load_s = eng.wait_ready()
    chunks = eng.get("/health")["chunks"]
    assert chunks > FLOOR_CHUNKS, f"{label}: engine started at {chunks} chunks, nothing to shrink"
    return eng, load_s, chunks


def test_d5_flip_floor_then_exit(engine_factory, stub):
    since = stamp()
    eng, load_s, chunks0 = start_engine(engine_factory, stub, "stub-d5", "D5")
    t_flip = stub.set(("large-drill", "loading"))
    floor_s = wait_floor(eng, t_flip)
    t_exit = eng.wait_exit(D5_EXIT_MAX_S + 10)
    exit_s = None if t_exit is None else round(t_exit - t_flip, 3)
    ooms = oom_lines(since)
    write_record("running-stub-d5", {"mode": "D5", "load_s": round(load_s, 1), "chunks_start": chunks0,
                                     "floor_s": floor_s, "floor_limit_s": FLOOR_MAX_S, "exit_s": exit_s,
                                     "exit_limit_s": D5_EXIT_MAX_S, "exit_code": eng.proc.returncode, "oom_lines": ooms})
    assert floor_s is not None and floor_s <= FLOOR_MAX_S, f"D5 floor {floor_s} s"
    assert exit_s is not None and exit_s <= D5_EXIT_MAX_S, f"D5 exit {exit_s} s"
    assert not ooms


def test_d1_a_floor_then_regrow(engine_factory, stub):
    since = stamp()
    eng, load_s, chunks0 = start_engine(engine_factory, stub, "stub-d1-a", "D1")
    t_a = stub.set(("large-drill", "loading"))
    floor_s = wait_floor(eng, t_a)
    assert floor_s is not None and floor_s <= FLOOR_MAX_S, f"D1+A floor {floor_s} s"
    t_ready = stub.set(("large-drill", "ready"))
    grows, end = [], time.monotonic() + GROW_WAIT_S
    while time.monotonic() < end and len(grows) < 2:
        grows = [r for r in eng.governor() if r["t"] >= t_ready and r["decision"].startswith("GrowTo")]
        if eng.proc.poll() is not None:
            break
        time.sleep(1.0)
    gaps = [round(b["t"] - a["t"], 3) for a, b in zip(grows, grows[1:])]
    steps = [g["chunks_after"] - g["chunks_before"] for g in grows]
    ooms = oom_lines(since)
    write_record("running-stub-d1-a", {"mode": "D1", "load_s": round(load_s, 1), "chunks_start": chunks0,
                                       "floor_s": floor_s, "floor_limit_s": FLOOR_MAX_S, "grows": len(grows),
                                       "grow_steps_chunks": steps, "grow_gaps_s": gaps,
                                       "first_grow_after_ready_s": round(grows[0]["t"] - t_ready, 1) if grows else None,
                                       "grow_gap_min_s": GROW_GAP_MIN_S, "oom_lines": ooms})
    assert len(grows) >= 2, f"only {len(grows)} regrow steps in {GROW_WAIT_S} s"
    assert all(s == 1 for s in steps), steps
    assert all(g >= GROW_GAP_MIN_S for g in gaps), gaps
    assert not ooms


def test_d1_b_exit(engine_factory, stub):
    since = stamp()
    eng, load_s, chunks0 = start_engine(engine_factory, stub, "stub-d1-b", "D1")
    t_b = stub.set(("vision-drill", "loading"))
    t_exit = eng.wait_exit(EXIT_MAX_S + 10)
    exit_s = None if t_exit is None else round(t_exit - t_b, 3)
    ooms = oom_lines(since)
    write_record("running-stub-d1-b", {"mode": "D1", "load_s": round(load_s, 1), "chunks_start": chunks0,
                                       "exit_s": exit_s, "exit_limit_s": EXIT_MAX_S, "exit_code": eng.proc.returncode,
                                       "oom_lines": ooms})
    assert exit_s is not None and exit_s <= EXIT_MAX_S, f"D1+B exit {exit_s} s"
    assert not ooms


def test_d2_a_floor(engine_factory, stub):
    since = stamp()
    eng, load_s, chunks0 = start_engine(engine_factory, stub, "stub-d2-a", "D2")
    t_a = stub.set(("large-drill", "loading"))
    floor_s = wait_floor(eng, t_a)
    ooms = oom_lines(since)
    write_record("running-stub-d2-a", {"mode": "D2", "load_s": round(load_s, 1), "chunks_start": chunks0,
                                       "floor_s": floor_s, "floor_limit_s": FLOOR_MAX_S, "oom_lines": ooms})
    assert floor_s is not None and floor_s <= FLOOR_MAX_S, f"D2+A floor {floor_s} s"
    assert not ooms
