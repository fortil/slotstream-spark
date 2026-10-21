"""Step 66 (light): on-demand restart through the systemd front on 127.0.0.1:18434.

Drill environment through `systemctl --user set-environment SLOTSTREAM_MAX_CHUNKS=4 SLOTSTREAM_IDLE_EXIT_S=60` (plus the
stub URL for the front), unset afterwards. 1) a request starts the engine (503 + Retry-After while it loads, then 200);
2) after the idle exit (60 s) no engine runs; 3) a request starts it again (503 + Retry-After, then 200); 4) with the stub
non-empty a request gets 503 and no engine process exists. Run: tools/spark/run.sh --timeout 1800 -- tools/capture.sh
drills-light -- tools/spark/py.sh -m pytest -q -m gate tests/gates/drills/test_restart.py
"""
import subprocess
import time
import urllib.error
import urllib.request

import pytest

from conftest import STUB_URL, write_record

pytestmark = pytest.mark.gate

FRONT = "http://127.0.0.1:18434/health"
DRILL_ENV = ("SLOTSTREAM_MAX_CHUNKS", "SLOTSTREAM_IDLE_EXIT_S", "SLOTSTREAM_RUNNING_URL")
READY_MAX_S = 900
IDLE_EXIT_MAX_S = 240


def sc(*args):
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True)


def active(unit):
    return sc("is-active", unit).stdout.strip()


def engine_procs():
    r = subprocess.run(["pgrep", "-f", "slotstream_spark.cli serve"], capture_output=True, text=True)
    return [p for p in r.stdout.split() if p]


def get(url=FRONT, timeout=3):
    """-> (status, Retry-After or None)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.headers.get("Retry-After")
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Retry-After")
    except OSError:
        return None, None


def request_until_ready(max_s=READY_MAX_S):
    """First answer, then poll until 200. -> (first_status, first_retry_after, seconds_to_200)."""
    t0 = time.monotonic()
    first = get()
    seen = [first]
    while time.monotonic() - t0 < max_s:
        if seen[-1][0] == 200:
            return first[0], first[1], round(time.monotonic() - t0, 1)
        time.sleep(2.0)
        seen.append(get())
    return first[0], first[1], None


@pytest.fixture
def drill_env(stub):
    assert sc("is-active", "--quiet", "slotstream-spark.socket").returncode == 0, "the front socket is not active"
    assert active("slotstream-spark.service") in ("inactive", "failed"), "the engine unit is not idle"
    sc("stop", "slotstream-spark-front.service")                  # the next connection re-activates it with the drill env
    stub.set()
    sc("set-environment", "SLOTSTREAM_MAX_CHUNKS=4", "SLOTSTREAM_IDLE_EXIT_S=60", f"SLOTSTREAM_RUNNING_URL={STUB_URL}")
    yield stub
    sc("unset-environment", *DRILL_ENV)
    sc("stop", "slotstream-spark.service")
    sc("stop", "slotstream-spark-front.service")


def test_restart_through_the_front(drill_env):
    stub = drill_env
    s1, ra1, t1 = request_until_ready()
    t_idle = time.monotonic()
    idle_s = None
    while time.monotonic() - t_idle < IDLE_EXIT_MAX_S:
        if active("slotstream-spark.service") not in ("active", "activating"):
            idle_s = round(time.monotonic() - t_idle, 1)
            break
        time.sleep(2.0)
    procs_after_idle = engine_procs()
    s2, ra2, t2 = request_until_ready()
    sc("stop", "slotstream-spark.service")
    time.sleep(3)
    sc("stop", "slotstream-spark-front.service")                  # the front re-reads its environment on the next start
    stub.set(("large-drill", "ready"))
    nonempty = get()
    time.sleep(3)
    procs_blocked = engine_procs()
    state_blocked = active("slotstream-spark.service")
    write_record("restart", {
        "first_start": {"first_status": s1, "retry_after": ra1, "seconds_to_200": t1},
        "idle_exit_s": idle_s, "engine_procs_after_idle": procs_after_idle,
        "second_start": {"first_status": s2, "retry_after": ra2, "seconds_to_200": t2},
        "stub_nonempty": {"status": nonempty[0], "engine_procs": procs_blocked, "engine_unit": state_blocked},
        "env": {"SLOTSTREAM_MAX_CHUNKS": "4", "SLOTSTREAM_IDLE_EXIT_S": "60"}})
    assert (s1, ra1) == (503, "10") and t1 is not None, (s1, ra1, t1)
    assert idle_s is not None and not procs_after_idle, (idle_s, procs_after_idle)
    assert (s2, ra2) == (503, "10") and t2 is not None, (s2, ra2, t2)
    assert nonempty[0] == 503 and not procs_blocked and state_blocked not in ("active", "activating")
