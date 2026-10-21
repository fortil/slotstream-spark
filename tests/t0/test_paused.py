"""Pause mode: the PAUSED state-file reader and the pure functions that honor it. No network, no clock."""
import json
import sys
from pathlib import Path

import pytest

from slotstream_spark.guard import paused
from slotstream_spark.memory import governor as G
from slotstream_spark.memory.governor import Exit, GovInputs, decide, may_start
from slotstream_spark import front

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import impact_check  # noqa: E402
from bench import eligibility  # noqa: E402

CFG = G.load_config()


def put(root, text):
    p = Path(root) / "state" / "PAUSED"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


def test_absent_file_means_every_rule_as_before(tmp_path):
    assert paused.read(tmp_path, {}) is None
    assert paused.no_windows(tmp_path, {}) is False
    assert paused.expected_containers(root=tmp_path, env={}) is None
    assert paused.paused_active(tmp_path, {}) is False


def test_present_file_is_read(tmp_path):
    put(tmp_path, "NO_WINDOWS=1\nEXPECTED_CONTAINERS=1\n")
    assert paused.read(tmp_path, {}) == paused.Paused(True, 1)
    assert paused.no_windows(tmp_path, {}) is True
    assert paused.expected_containers(root=tmp_path, env={}) == 1


def test_comments_blank_lines_and_partial_files(tmp_path):
    put(tmp_path, "# why\n\nEXPECTED_CONTAINERS=13\n")
    s = paused.read(tmp_path, {})
    assert s == paused.Paused(False, 13) and paused.no_windows(tmp_path, {}) is False
    put(tmp_path, "NO_WINDOWS=1\n")
    assert paused.expected_containers(root=tmp_path, env={}) is None      # file without the key: the default (no count)


def test_explicit_env_wins_over_file_and_default(tmp_path):
    put(tmp_path, "NO_WINDOWS=1\nEXPECTED_CONTAINERS=1\n")
    assert paused.expected_containers(root=tmp_path, env={"EXPECTED_CONTAINERS": "7"}) == 7
    assert paused.expected_containers(root=tmp_path / "none", env={"EXPECTED_CONTAINERS": "7"}) == 7
    with pytest.raises(paused.PausedError):
        paused.expected_containers(root=tmp_path, env={"EXPECTED_CONTAINERS": "x"})


@pytest.mark.parametrize("text", ["", "\n\n", "# only a comment\n", "NO_WINDOWS\n", "NO_WINDOWS=2\n", "NO_WINDOWS=\n",
                                  "EXPECTED_CONTAINERS=-1\n", "EXPECTED_CONTAINERS=1.5\n", "EXPECTED_CONTAINERS=\n",
                                  "WINDOWS=1\n", "NO_WINDOWS=1\nbogus\n"])
def test_malformed_file_is_an_error_not_a_default(tmp_path, text):
    put(tmp_path, text)
    with pytest.raises(paused.PausedError):
        paused.read(tmp_path, {})
    with pytest.raises(paused.PausedError):
        paused.no_windows(tmp_path, {})
    with pytest.raises(paused.PausedError):
        paused.expected_containers(root=tmp_path, env={})


def test_file_override_by_env(tmp_path):
    p = tmp_path / "other"
    p.write_text("NO_WINDOWS=1\n")
    assert paused.path(env={"SLOTSTREAM_PAUSED_FILE": str(p)}) == p
    assert paused.no_windows(env={"SLOTSTREAM_PAUSED_FILE": str(p)}) is True


# ---- governor / front ---------------------------------------------------------------------------------------
def test_may_start_windows_lifted_only_by_no_windows():
    args = ((), 60000, 10000, CFG)
    assert may_start(2 * 60, *args) == (False, "before 04:00")
    assert may_start(23 * 60 + 50, *args) == (False, "after 23:45")
    assert may_start(2 * 60, *args, no_windows=True) == (True, "ok")
    assert may_start(23 * 60 + 50, *args, no_windows=True) == (True, "ok")
    assert may_start(2 * 60, None, 60000, 10000, CFG, no_windows=True) == (False, "running unreadable")
    assert may_start(2 * 60, (G.Resident("x"),), 60000, 10000, CFG, no_windows=True) == (False, "running not empty")
    assert may_start(2 * 60, (), 100, 10000, CFG, no_windows=True) == (False, "cap below the floor")


def test_decide_exit_at_2359_only_without_no_windows():
    base = dict(mode="D5", now_min=23 * 60 + 59, running=(), chunks=20, footprint_mib=30000, mem_available_mib=60000, mem_free_mib=60000)
    assert isinstance(decide(GovInputs(**base), CFG), Exit)
    assert not (isinstance(d := decide(GovInputs(**base, no_windows=True), CFG), Exit) and "window" in d.reason)
    early = dict(base, now_min=60)
    assert isinstance(decide(GovInputs(**early), CFG), Exit)
    assert not (isinstance(d := decide(GovInputs(**early, no_windows=True), CFG), Exit) and "window" in d.reason)


def test_front_policy_passes_no_windows():
    kw = dict(engine_ready=False, engine_active=False, now_min=2 * 60, running=(), cap_mib=60000, floor_mib=10000, cfg=CFG)
    assert front.policy(**kw)[0] == front.REFUSE
    assert front.policy(**kw, no_windows=True)[0] == front.START


# ---- eligibility / impact -----------------------------------------------------------------------------------
GOOD = dict(running=[], scopes=[], cap_mib=10, cell_cap_mib=1, finish_minute=60, cut=False, gpu_clock_nominal=True,
            sw_power_cap_delta=0, nvme_temp_mc=40000, nvme_temp_limit_mc=70000, hour=2, foreign_read_mbps=0)


def test_eligibility_quiet_window_and_deadline():
    assert eligibility.evaluate(GOOD) == (False, ["quiet_window"])
    assert eligibility.evaluate(dict(GOOD, no_windows=True)) == (True, [])
    late = dict(GOOD, hour=12, finish_minute=23 * 60 + 50)
    assert eligibility.evaluate(late) == (False, ["past_deadline_or_cut"])
    assert eligibility.evaluate(dict(late, no_windows=True)) == (True, [])
    assert eligibility.evaluate(dict(late, no_windows=True, cut=True)) == (False, ["past_deadline_or_cut"])


def _cap(tmp, klog="clean", avail_kb=100_000_000, paused_flag=None):
    (tmp / "meminfo.csv").write_text(f"MemAvailable_kB\n{avail_kb}\n")
    gp = {"klog": klog}
    if paused_flag is not None:
        gp["paused"] = paused_flag
    (tmp / "guard-post.json").write_text(json.dumps(gp))
    (tmp / "prod-psi.csv").write_text("io_some_avg10,mem_some_avg10\n50.0,1.0\n")
    return tmp


def test_impact_check_paused_marks_latency_and_psi_not_applicable(tmp_path, monkeypatch):
    monkeypatch.setenv("SLOTSTREAM_PAUSED_FILE", str(tmp_path / "absent"))
    cap = _cap(tmp_path)
    assert impact_check.check(cap, {})["items"]["psi"]["status"] == "FAIL"               # unpaused: PSI 50 fails
    r = impact_check.check(cap, {}, paused=True)
    assert r["items"]["latency"] == {"status": "NOT_APPLICABLE", "detail": "not_applicable: production paused by the user"}
    assert r["items"]["psi"]["status"] == "NOT_APPLICABLE" and r["verdict"] == "PASS"
    # the capture says it was paused
    cap2 = _cap(tmp_path, paused_flag=True)
    assert impact_check.check(cap2, {})["verdict"] == "PASS"


def test_impact_check_paused_keeps_oom_and_memavailable_gates(tmp_path, monkeypatch):
    monkeypatch.setenv("SLOTSTREAM_PAUSED_FILE", str(tmp_path / "absent"))
    assert impact_check.check(_cap(tmp_path, klog="oom-kill 1"), {}, paused=True)["verdict"] == "FAIL"
    assert impact_check.check(_cap(tmp_path, avail_kb=4_000_000), {}, paused=True)["verdict"] == "FAIL"


def test_impact_check_reads_the_state_file(tmp_path, monkeypatch):
    p = put(tmp_path, "NO_WINDOWS=1\nEXPECTED_CONTAINERS=1\n")
    monkeypatch.setenv("SLOTSTREAM_PAUSED_FILE", str(p))
    assert impact_check.is_paused(tmp_path) is True
    p.unlink()
    assert impact_check.is_paused(tmp_path) is False
