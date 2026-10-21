import json
import sys
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "drills"))
import hog  # noqa: E402
import running_stub  # noqa: E402


def args(**kw):
    a = hog.parse(["--chunk-mib", "1", "--running-url", ""])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


BASE = {"mem_available_mib": 100000, "hhmm": (12, 0)}


def go(state, max_steps=10, **kw):
    """Run the hog; `state` is applied from step 3 on, so the rule must fire within one step of the change."""
    s = dict(BASE)
    a = args(**kw)
    inp = hog.FakeInputs(a, s)
    held = []

    def on_step(size):
        if len(held) == 2:
            s.update(state)
        held.append(size)
    reason, size, steps = hog.run(a, inp, alloc=lambda n: n, sleep=lambda x: None, max_steps=max_steps, on_step=on_step)
    return reason, size, steps


@pytest.mark.parametrize("state,reason", [
    ({"mem_available_mib": 14336}, "memavailable_floor"),
    ({"mem_available_mib": 5000}, "memavailable_floor"),
    ({"engine_log_floor": True}, "engine_log_floor"),
    ({"engine_alive": False}, "engine_exited"),
    ({"running": [{"model": "m", "state": "ready"}]}, "running_nonempty"),
    ({"stop_file": True}, "stop_file"),
    ({"hhmm": (23, 45)}, "deadline"),
])
def test_every_stop_rule_fires_within_one_step(state, reason):
    got, size, steps = go(state)
    assert got == reason and steps == 3                       # grew 3 times; the change came after step 3, stops before step 4


def test_size_cap_is_engine_cap_minus_floor():
    reason, size, steps = go({}, max_steps=100, engine_cap_mib=14336 + 5)
    assert reason == "size_cap" and size == 5


def test_no_growth_when_a_rule_holds_at_start():
    a = args()
    reason, size, steps = hog.run(a, hog.FakeInputs(a, {**BASE, "stop_file": True}), alloc=lambda n: n, sleep=lambda x: None)
    assert (reason, size, steps) == ("stop_file", 0, 0)


def test_real_alloc_touches_and_frees():
    a = args(engine_cap_mib=14336 + 2)
    live = hog.Inputs(a)
    live.mem_available_mib = lambda: 100000
    live.hhmm = lambda: (12, 0)
    reason, size, steps = hog.run(a, live, sleep=lambda x: None)
    assert reason == "size_cap" and size == 2 and steps == 2


def test_engine_log_floor_from_marker_and_text(tmp_path):
    log = tmp_path / "e.log"
    log.write_text("running\n")
    a = args(engine_log=str(log))
    inp = hog.Inputs(a)
    assert not inp.engine_log_floor()
    (tmp_path / "e.log.stop").write_text("")
    assert inp.engine_log_floor()
    (tmp_path / "e.log.stop").unlink()
    log.write_text("running\nMemFree below floor\n")
    assert inp.engine_log_floor()


def test_running_stub_serves_state_and_logs_flips(tmp_path):
    st, lg = tmp_path / "state.json", tmp_path / "flips.log"
    srv = running_stub.make_server(0, str(st), str(lg))
    import threading
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/running"
    try:
        assert json.load(urllib.request.urlopen(url)) == {"running": []}
        st.write_text(json.dumps({"running": [{"model": "large-x", "state": "ready"}]}))
        assert json.load(urllib.request.urlopen(url))["running"][0]["model"] == "large-x"
        st.write_text("")
        assert json.load(urllib.request.urlopen(url)) == {"running": []}
    finally:
        srv.shutdown()
    flips = [json.loads(l) for l in lg.read_text().splitlines()]
    assert [bool(f["running"]) for f in flips] == [True, False] and all("t" in f for f in flips)
