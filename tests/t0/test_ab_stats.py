import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench import aa, ab, eligibility, stats  # noqa: E402


def test_medians_even_and_odd():
    assert stats.true_median([1, 3, 2]) == 2
    assert stats.true_median([1, 2, 3, 10]) == 2.5
    assert stats.paired_ratios([1, 2, 0, 4], [2, 2, 5, None]) == [2.0, 1.0]


def test_gate_and_futility():
    good = [1.10] * 14 + [0.9, 0.9]
    assert stats.gate(good)
    assert not stats.gate([1.04] * 16)                      # median under the bar
    assert not stats.gate([1.2] * 12 + [0.9] * 4)           # 75 % positive
    assert stats.futility([0.9] * 4, 16, 0.80)              # 12 left, 12/16 < 0.8
    assert not stats.futility([0.9] * 3, 16, 0.80)          # 13 left, 13/16 >= 0.8
    assert stats.futility([], 16, 0.80) is False


def test_balanced_order():
    o = stats.balanced_order(16)
    assert len(o) == 16 and o.count("AB") == 8 and o.count("BA") == 8 and o[:4] == ["AB", "BA", "AB", "BA"]


def test_eligibility_rules():
    ok = dict(running=[], scopes=[{"name": "docker-1.scope", "mem_pressure_avg10": 0.0, "io_pressure_pct": 1.0}], cap_mib=60000,
              cell_cap_mib=57344, finish_minute=22 * 60, cut=False, gpu_clock_nominal=True, sw_power_cap_delta=0,
              nvme_temp_mc=40000, nvme_temp_limit_mc=77850, hour=10, foreign_read_mbps=10)
    assert eligibility.evaluate(ok) == (True, [])
    for k, v, why in [("running", ["m"], "running_not_empty"), ("cap_mib", 1000, "cap_below_cell"),
                      ("finish_minute", 23 * 60 + 50, "past_deadline_or_cut"), ("cut", True, "past_deadline_or_cut"),
                      ("gpu_clock_nominal", False, "gpu_clock_not_nominal"), ("sw_power_cap_delta", 1, "sw_power_cap"),
                      ("nvme_temp_mc", 78000, "nvme_hot"), ("nvme_temp_mc", None, "nvme_temp_unknown"),
                      ("nvme_temp_limit_mc", None, "nvme_temp_unknown"), ("hour", 2, "quiet_window"),
                      ("foreign_read_mbps", 500, "foreign_reads")]:
        good, reasons = eligibility.evaluate({**ok, k: v})
        assert not good and why in reasons, (k, reasons)
    bad = {**ok, "scopes": [{"name": "docker-9.scope", "mem_pressure_avg10": 3.0, "io_pressure_pct": 20.0}]}
    assert eligibility.evaluate(bad)[1] == ["memory_pressure:docker-9.scope", "io_pressure:docker-9.scope"]


class _P:
    def __init__(self, rc=0, out=""):
        self.returncode, self.stdout = rc, out


def _runner(base=10.0, factor=1.1, fail=(), noisy=None):
    """Value by command (A -> base, B -> base * factor; the A/A command 'A' always gives base); `fail` = call indexes."""
    calls = {"n": 0}

    def run(cmd, **kw):
        n = calls["n"]
        calls["n"] += 1
        if n in fail:
            return _P(3)
        v = base * (factor if cmd[0] == "B" else 1.0)
        return _P(0, json.dumps({"decode_tps": v}))
    return run


SNAP = dict(running=[], scopes=[], cap_mib=1, cell_cap_mib=0, finish_minute=0, cut=False, gpu_clock_nominal=True,
            sw_power_cap_delta=0, nvme_temp_mc=1, nvme_temp_limit_mc=9, hour=10, foreign_read_mbps=0)


def test_ab_keeps_excluded_pairs(tmp_path):
    snaps = iter([SNAP] * 3 + [{**SNAP, "running": ["x"]}] + [SNAP] * 100)
    res, d = ab.run_ab("A", "B", pairs=16, label="t", snapshot=lambda: next(snaps), runner=_runner(fail={2}), root=tmp_path,
                       tmp=str(tmp_path / "o"))
    assert res["pairs_run"] == 16 and res["pairs_excluded"] == 2
    reasons = {r["pair"]: r["reasons"] for r in res["pairs"] if not r["eligible"]}
    assert reasons[3] == ["running_not_empty"] and any("rc=3" in x for x in reasons[1])
    assert (d / "pairs.jsonl").read_text().count("\n") == 16
    assert (tmp_path / "runs" / "LAST-t").read_text().strip() == str(d)
    assert res["pass"] is False                               # 14 clean of 16 planned


def test_ab_pass_and_aa(tmp_path):
    res, _ = ab.run_ab("A", "B", pairs=16, label="p", snapshot=lambda: SNAP, runner=_runner(), root=tmp_path,
                       tmp=str(tmp_path / "o"))
    assert res["pass"] and abs(res["median_ratio"] - 1.1) < 1e-9
    res, _ = aa.run_aa("A", pairs=4, label="aa", snapshot=lambda: SNAP, runner=_runner(), root=tmp_path,
                       tmp=str(tmp_path / "o"))
    assert res["pass"] and res["arms"]["B"] == "A'"


def test_futility_stops_early(tmp_path):
    res, _ = ab.run_ab("A", "B", pairs=16, label="f", snapshot=lambda: SNAP, runner=_runner(factor=0.9), root=tmp_path,
                       tmp=str(tmp_path / "o"))
    assert res["stopped"] == "futility" and res["pairs_run"] < 16


def test_collect_cap_is_numeric_and_comparable(tmp_path, monkeypatch):
    mi = tmp_path / "meminfo"
    mi.write_text("MemAvailable:   83886080 kB\nMemFree: 1 kB\n")
    cap = eligibility._cap_mib("D5", str(mi))
    assert isinstance(cap, int) and cap > 0
    snap = dict(running=[], scopes=[], cap_mib=cap, cell_cap_mib=cap + 1, finish_minute=600, cut=False, hour=10,
                nvme_temp_mc=40000, nvme_temp_limit_mc=77850)
    assert eligibility.evaluate(snap) == (False, ["cap_below_cell"])
    assert eligibility.evaluate({**snap, "cell_cap_mib": cap}) == (True, [])


def test_nvme_limit_is_the_device_max_minus_margin(tmp_path):
    (tmp_path / "temp1_input").write_text("52850\n")
    (tmp_path / "temp1_max").write_text("82850\n")
    assert eligibility.nvme_limit_mc(str(tmp_path / "temp1_input")) == (77850, "temp1_max-5000")
    d2 = tmp_path / "x"
    d2.mkdir()
    (d2 / "temp1_input").write_text("52850\n")
    assert eligibility.nvme_limit_mc(str(d2 / "temp1_input")) == (None, "unreadable")


def test_collect_has_no_temperature_argument_and_ab_rows_keep_the_snapshot(tmp_path):
    import inspect
    assert "temp_limit_mc" not in inspect.signature(eligibility.collect).parameters
    res, _ = ab.run_ab("A", "B", pairs=2, label="s", snapshot=lambda: SNAP, runner=_runner(), root=tmp_path, tmp=str(tmp_path / "o"))
    assert all("snapshot" in r for r in res["pairs"])
