"""Signal thread, controller and footprint rule against fakes: no clock, no network, no GPU."""
import io
import json
import threading

import pytest

from slotstream_spark.engine.pressure import Cancelled, PressureBoundary
from slotstream_spark.memory import footprint as fp
from slotstream_spark.memory import governor as G
from slotstream_spark.memory import signals as S
from slotstream_spark.memory.ledger import Ledger

CFG = G.load_config()
GIB = 1 << 30
LEDGER = Ledger("D5", {"runtime_bytes": 2_637_857_704})


class FakeEngine:
    def __init__(self, chunks=20):
        self.chunks = chunks
        self.calls = []
        self.exit_reason = None
        self.gen_active = threading.Event()      # set while a fake generation holds the stream
        self.resized_while_active = False

    def resize_chunks(self, n):
        if self.gen_active.is_set():
            self.resized_while_active = True
        self.calls.append(("resize", n))
        self.chunks = n

    def drop_prefix(self):
        self.calls.append(("drop_prefix",))

    def request_exit(self, reason):
        self.calls.append(("exit", reason))
        self.exit_reason = reason


def make(running, engine=None, avail=70000, decide=G.decide, **kw):
    engine = engine or FakeEngine()
    state = {"running": running}
    log = io.StringIO()
    sig = S.Signals(mode="D5", cfg=CFG, ledger=LEDGER, engine=engine, get_running=lambda: state["running"],
                    read_mem=lambda: (avail, avail), log=log, minutes_now=lambda: 12 * 60, write_oom=False,
                    decide=decide, **kw)
    return sig, engine, state, log


def entries(log):
    return [json.loads(x) for x in log.getvalue().splitlines()]


def test_d5_nonempty_shrinks_on_the_first_poll_and_exits_after_5_s():
    sig, eng, st, log = make((G.Resident("large-x", "loading"),))
    d = sig.poll(0.0)
    assert isinstance(d, G.ShrinkTo) and eng.chunks == CFG.floor_chunks
    first = entries(log)[0]
    assert first["decision"] == f"ShrinkTo({CFG.floor_chunks})" and first["chunks_before"] == 20
    assert first["chunks_after"] == CFG.floor_chunks and first["freed_t"] is not None
    assert set(first) == {"t", "avail", "free", "running", "cap", "footprint", "decision", "chunks_before",
                          "chunks_after", "freed_t"}
    t, last = 0.25, None
    while t <= 5.5 and sig.exited is None:
        last = sig.poll(t)
        t += 0.25
    assert isinstance(last, G.Exit) and sig.exited and eng.exit_reason
    assert 5.0 <= t - 0.25 <= 5.5


def test_poll_is_gated_to_250_ms():
    sig, eng, st, log = make(())
    assert sig.poll(0.0) is not None
    assert sig.poll(0.1) is None
    assert sig.poll(0.25) is not None


def test_oom_score_adj_is_written(tmp_path):
    p = tmp_path / "oom_score_adj"
    p.write_text("0")
    assert S.write_oom_score_adj(str(p)) is True
    assert p.read_text() == "1000"
    eng = FakeEngine()
    S.Signals(mode="D5", cfg=CFG, ledger=LEDGER, engine=eng, get_running=lambda: (), read_mem=lambda: (1, 1),
              oom_path=str(p), minutes_now=lambda: 720)
    assert p.read_text() == "1000"
    assert S.write_oom_score_adj(str(tmp_path / "missing" / "x")) is False


def test_empty_url_means_nothing_running_not_failed():
    assert S.fetch_running("") == () and S.fetch_running(None) == ()


def test_parse_running():
    r = S.parse_running('{"running":[{"model":"large-x","state":"ready"},{"model":"m"}]}')
    assert r == (G.Resident("large-x", "ready"), G.Resident("m", None))
    assert S.parse_running('{"running":[]}') == ()
    assert S.parse_running("not json") is None and S.parse_running('{"x":1}') is None


def test_ticket_cancels_an_inflight_generation_and_the_shrink_waits_for_the_stream():
    eng = FakeEngine()
    ctl = S.Controller(eng, gate_timeout_s=3.0)
    started, outcome = threading.Event(), {}

    def generation():
        assert ctl.gate.acquire(0)
        eng.gen_active.set()
        started.set()
        try:
            for _ in range(10000):
                ctl.boundary.check()
                threading.Event().wait(0.001)
        except Cancelled as e:
            outcome["cancelled"] = str(e)
        finally:
            eng.gen_active.clear()
            ctl.gate.release()

    th = threading.Thread(target=generation)
    th.start()
    assert started.wait(2)
    assert ctl.shrink(2, "running not empty", cancel=True) is True
    th.join(2)
    assert outcome["cancelled"] == "running not empty"
    assert eng.chunks == 2 and eng.resized_while_active is False
    assert ctl.boundary.snapshot() is None            # acknowledged


def test_shrink_without_the_stream_is_not_executed():
    eng = FakeEngine()
    ctl = S.Controller(eng, gate_timeout_s=0.05)
    assert ctl.gate.acquire(0)                        # a generation that never yields
    assert ctl.shrink(2, "cap", cancel=False) is False
    assert eng.chunks == 20 and eng.calls == []


def test_late_acknowledgement_does_not_clear_a_newer_ticket():
    b = PressureBoundary()
    t1 = b.request("a")
    t2 = b.request("b")
    b.acknowledge(t1)
    assert b.snapshot() is t2
    b.acknowledge(t2)
    assert b.snapshot() is None


def _sample(vmrss, nvml, pool):
    return fp.Sample("t", vmrss, vmrss, nvml, 0, {"pool": pool})


def test_governor_gets_the_count_once_footprint_not_the_raw_sum():
    pool = 2 * GIB
    base = 40 * GIB                                   # non-pool bytes in each counter
    s = _sample(base + pool, base + pool, pool)       # the pool shows in both counters
    flags = {"pool": (True, True)}
    raw = s.vmrss + s.nvml
    once = fp.once_bytes(s, flags)
    assert once == raw - pool
    assert S.governor_footprint_bytes(LEDGER, 4, s, flags) == max(LEDGER.expected_peak_bytes(4), once)
    seen = []

    def spy(g, cfg):
        seen.append(g.footprint_mib)
        return G.Hold()

    sig, eng, st, log = make((), decide=spy, read_footprint=lambda: (s, flags))
    sig.poll(0.0)
    assert seen == [-(-once // (1 << 20))] and seen[0] != -(-raw // (1 << 20))


def test_one_more_chunk_moves_the_footprint_by_one_chunk_not_two():
    flags = {"pool": (True, True)}
    base = 30 * GIB
    f = lambda pool: S.governor_footprint_bytes(LEDGER, 1, _sample(base + pool, base + pool, pool), flags)
    assert f(3 * GIB) - f(2 * GIB) == GIB


def test_footprint_from_jsonl(tmp_path):
    p = tmp_path / "fp.jsonl"
    lines = [{"header": True, "period_ms": 20, "flags": {"pool": [True, True], "reader": [True, False]}},
             {"t_ns": 1, "vmrss": 10, "vmhwm": 12, "nvml": 5, "avail": 1, "free": 1,
              "unified": {"pool": 4, "reader": 1}}]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    samples, flags = fp.from_jsonl(p)
    assert flags == {"pool": (True, True), "reader": (True, False)}
    assert len(samples) == 1 and fp.once_bytes(samples[0], flags) == 10 + 5 - (4 + 4 + 1) + 5
    p.write_text(json.dumps(lines[1]) + "\n")
    with pytest.raises(ValueError):
        fp.from_jsonl(p)
    lines[1]["unified"]["mystery"] = 1
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    with pytest.raises(ValueError):
        fp.from_jsonl(p)
