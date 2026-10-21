"""Step 66 (light): the 23:59 stop, the timers and the 04:00-23:45 start rule.

Starting `slotstream-spark-stop.service` leaves the socket, the front and the engine inactive within 15 s; the timers
show 04:00 and 23:59; `may_start` refuses before 04:00 and from 23:45; the socket is started only by the 04:00 timer's
service (no `[Install]` in the socket unit), and this test restores it afterwards. It needs the clock inside 04:00-23:40.
Run: tools/spark/run.sh --timeout 1800 -- tools/capture.sh drills-light -- tools/spark/py.sh -m pytest -q -m gate
tests/gates/drills/test_stop.py
"""
import subprocess
import time
from pathlib import Path

import pytest

from conftest import ROOT, write_record

pytestmark = pytest.mark.gate

STOP_MAX_S = 15.0


def sc(*args):
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True)


def active(unit):
    return sc("is-active", unit).stdout.strip()


def calendar(timer):
    out = sc("show", timer, "-p", "TimersCalendar").stdout
    return out.strip()


def test_stop_unit_and_timers():
    from slotstream_spark.memory import governor as G
    now = time.localtime()
    if not (4 * 60 <= now.tm_hour * 60 + now.tm_min < 23 * 60 + 40):
        pytest.fail("environment window: the stop drill runs inside 04:00-23:40", pytrace=False)
    assert sc("is-active", "--quiet", "slotstream-spark.socket").returncode == 0, "the front socket is not active"
    sc("set-environment", "SLOTSTREAM_MAX_CHUNKS=4")
    try:
        sc("start", "--no-block", "slotstream-spark.service")      # an engine in the middle of its load
        time.sleep(2)
        engine_before = active("slotstream-spark.service")
        sc("start", "slotstream-spark-front.service")
        front_before = active("slotstream-spark-front.service")
        t0 = time.monotonic()
        r = sc("start", "slotstream-spark-stop.service")
        deadline = t0 + STOP_MAX_S
        states = {}
        while time.monotonic() < deadline:
            states = {u: active(u) for u in ("slotstream-spark.socket", "slotstream-spark-front.service",
                                             "slotstream-spark.service")}
            if all(v not in ("active", "activating") for v in states.values()):
                break
            time.sleep(0.25)
        stop_s = round(time.monotonic() - t0, 2)
        start_cal, stop_cal = calendar("slotstream-spark-start.timer"), calendar("slotstream-spark-stop.timer")
        socket_unit = (Path.home() / "slotstream-spark" / "src" / "deploy" / "slotstream-spark.socket")
        socket_text = socket_unit.read_text() if socket_unit.exists() else (ROOT / "deploy" / "slotstream-spark.socket").read_text()
        cfg = G.load_config()
        floor = 1
        gates = {hhmm: G.may_start(h * 60 + m, (), 10 ** 6, floor, cfg)
                 for hhmm, (h, m) in {"03:59": (3, 59), "04:00": (4, 0), "23:44": (23, 44), "23:45": (23, 45)}.items()}
        sc("start", "slotstream-spark-start.service")              # what the 04:00 timer runs: restores the socket
        restored = active("slotstream-spark.socket")
        write_record("stop", {
            "engine_before": engine_before, "front_before": front_before, "stop_rc": r.returncode, "stop_s": stop_s,
            "stop_limit_s": STOP_MAX_S, "states_after": states, "start_timer": start_cal, "stop_timer": stop_cal,
            "socket_has_install_section": "[Install]" in socket_text,
            "may_start": {k: [v[0], v[1]] for k, v in gates.items()}, "socket_restored": restored})
        assert engine_before in ("active", "activating"), engine_before
        assert all(v not in ("active", "activating") for v in states.values()) and stop_s <= STOP_MAX_S, (states, stop_s)
        assert "04:00:00" in start_cal and "23:59:00" in stop_cal, (start_cal, stop_cal)
        assert "[Install]" not in socket_text
        assert gates["03:59"] == (False, "before 04:00") and gates["04:00"][0] is True
        assert gates["23:44"][0] is True and gates["23:45"] == (False, "after 23:45")
        assert restored == "active"
    finally:
        sc("unset-environment", "SLOTSTREAM_MAX_CHUNKS")
        sc("start", "slotstream-spark.socket")
