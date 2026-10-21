"""503 admission, the minimal server's single-generation rule and the planner-based sizing."""
import threading
import types

import pytest
from starlette.testclient import TestClient

from slotstream_spark.engine.pressure import Cancelled
from slotstream_spark.memory import admission, planner
from slotstream_spark.memory import signals as S
from slotstream_spark.memory.ledger import Ledger
from slotstream_spark.server import minimal

LEDGER = Ledger("D5", {"runtime_bytes": 2_637_857_704})
PLAN = planner.plan("D5", LEDGER.expected_peak_bytes(10), measured={"runtime_bytes": 2_637_857_704})


def test_admits_when_it_fits():
    assert admission.admit(1000, PLAN, LEDGER, LEDGER.expected_peak_bytes(PLAN.chunks)).ok


def test_503_when_the_prompt_does_not_fit_the_window():
    r = admission.admit(PLAN.window, PLAN, LEDGER, 1 << 60)
    assert (r.ok, r.status, r.body) == (False, 503, {"error": {"type": "insufficient_memory"}})


def test_503_when_the_cap_shrank_below_the_plan():
    r = admission.admit(10, PLAN, LEDGER, LEDGER.expected_peak_bytes(PLAN.chunks) - 1)
    assert not r.ok and r.status == 503 and r.body == admission.INSUFFICIENT


class Eng:
    chunks = PLAN.chunks
    plan = PLAN

    def __init__(self):
        self.release = threading.Event()
        self.entered = threading.Event()

    def generate(self, ids, n, boundary):
        self.entered.set()
        self.release.wait(2)
        boundary.check()
        return {"tokens": list(ids)[:n]}


def server(cap=None, clock=None):
    eng = Eng()
    ctl = S.Controller(eng)
    st = minimal.State(eng, LEDGER, ctl, lambda: cap if cap is not None else 1 << 60, idle_s=1800,
                       **({"clock": clock} if clock else {}))
    st.ready = True
    return eng, st, TestClient(minimal.make_app(st))


def test_health_and_memory():
    eng, st, c = server()
    assert c.get("/health").status_code == 200
    m = c.get("/internal/memory").json()
    assert m["chunks"] == PLAN.chunks and m["expected_peak"] == LEDGER.expected_peak_bytes(PLAN.chunks)
    st.ready = False
    assert c.get("/health").status_code == 503


def test_generate_and_503_when_it_does_not_fit():
    eng, st, c = server()
    eng.release.set()
    r = c.post("/internal/generate", json={"ids": [1, 2, 3], "max_tokens": 2})
    assert r.status_code == 200 and r.json() == {"tokens": [1, 2]}
    eng2, st2, c2 = server(cap=1)
    r = c2.post("/internal/generate", json={"ids": [1]})
    assert r.status_code == 503 and r.json() == {"error": {"type": "insufficient_memory"}}


def test_503_while_busy():
    eng, st, c = server()
    st.controller.gate.acquire(0)                     # a generation holds the stream
    r = c.post("/internal/generate", json={"ids": [1]})
    assert r.status_code == 503 and r.json()["error"]["type"] == "busy"


def test_idle_exit():
    now = [0.0]
    eng, st, c = server(clock=lambda: now[0])
    assert not st.idle_expired()
    now[0] = 1800.0
    assert st.idle_expired()


def test_planner_sizing_and_max_chunks(monkeypatch):
    from slotstream_spark import cli
    monkeypatch.delenv("SLOTSTREAM_MAX_CHUNKS", raising=False)
    cap = 40000
    n = cli.planned_chunks("D5", cap)
    assert n == planner.plan("D5", cap << 20).chunks and n >= 2
    monkeypatch.setenv("SLOTSTREAM_MAX_CHUNKS", "4")
    assert cli.planned_chunks("D5", cap) == 4
    assert cli.planned_chunks("D5", 1000) == 2        # floored at the governor floor


def test_served_context_covers_the_admission_window_and_serve_uses_it():
    import inspect
    from slotstream_spark import cli
    ctx = cli.served_ctx(PLAN)
    assert ctx >= PLAN.window and ctx % 64 == 0 and ctx - PLAN.window < 64
    assert "build_engine(a, a.pool_slots, served_ctx(plan))" in inspect.getsource(cli.cmd_serve)
