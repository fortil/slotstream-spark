"""F3 session job logic with fakes (no GPU, no llama.cpp): pair classification, yardstick arm, phases, trace tool."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from bench import session  # noqa: E402
import trace_routing  # noqa: E402

YARD = {"arm": "q41-b-d5", "op_offload": False, "prefill_device": "cpu", "port": 8091}
OK_SNAP = dict(running=[], scopes=[], cap_mib=60000, cell_cap_mib=57344, finish_minute=600, cut=False, hour=10,
               nvme_temp_mc=50000, nvme_temp_limit_mc=77850, memfree_mib=67000, memavailable_mib=83000)


def test_lb_args_parse_the_arm_and_reject_others():
    assert session.lb_args(YARD) == ["--weights", "q41", "--arm", "B", "--mode", "D5"]
    with pytest.raises(SystemExit):
        session.lb_args({**YARD, "arm": "q41-a-d5"})
    with pytest.raises(SystemExit):
        session.lb_args({**YARD, "op_offload": True})
    with pytest.raises(SystemExit):
        session.lb_args({**YARD, "prefill_device": "gpu"})


def recs(cache_n=0, cold=True):
    out = []
    for k, tps, pf in (("short", 32.0, 30.0), ("2k", 30.0, 27.0)):
        for rep in range(3):
            out.append({"prompt_id": k, "rep": rep, "decode_tps": tps + rep * 0.1, "prefill_tps": pf,
                        "cache_n": cache_n, "cold": cold})
    return out


def test_pair_ratio_is_decode_per_prompt_and_cold_pairs_are_clean():
    lb, pf, cache_n, cold = session.lb_tps(recs())
    assert lb == {"short": pytest.approx(32.1), "2k": pytest.approx(30.1)} and cache_n == 0 and cold
    row = session.lb_pair_row(0, {"short": 36.0, "2k": 33.0}, lb, pf, cache_n, cold, [])
    assert row["clean"] and row["reason"] is None
    assert row["ratio_by_prompt"]["short"] == pytest.approx(36.0 / 32.1)
    assert row["ratio"] == pytest.approx(np.median([36.0 / 32.1, 33.0 / 30.1]))
    assert row["prefill_tps_lb"] == {"short": 30.0, "2k": 27.0} and row["cache_n_max"] == 0


def test_warm_prefix_pair_is_not_clean():
    lb, pf, cache_n, cold = session.lb_tps(recs(cache_n=2771, cold=False))
    row = session.lb_pair_row(1, {"short": 36.0, "2k": 33.0}, lb, pf, cache_n, cold, [])
    assert not row["clean"] and "warm_prefix" in row["reason"] and row["cache_n_max"] == 2771
    lb, pf, cache_n, cold = session.lb_tps(recs(cache_n=None, cold=False))      # no timings at all
    assert not session.lb_pair_row(2, {"short": 1.0, "2k": 1.0}, lb, pf, cache_n, cold, [])["clean"]


def test_memfree_stop_and_missing_values_keep_the_pair_with_a_reason():
    row = session.lb_pair_row(3, {"short": 36.0, "2k": None}, {}, {}, 0, True, ["memfree_stop"])
    assert not row["clean"] and row["ratio"] is None
    assert "memfree_stop" in row["reason"] and "missing_tps:short" in row["reason"]


def engine_result(short, k2):
    return {"prompts": {"short": {"tok_s": short}, "2k": {"tok_s": k2}}}


class FakeRunner:
    """Stands in for subprocess.run: engine runs write their --out file; trace runs return 0."""

    def __init__(self, tps=(36.0, 33.0), fail=()):
        self.calls, self.tps, self.fail = [], tps, set(fail)

    def __call__(self, cmd, **kw):
        n = len(self.calls)
        self.calls.append(cmd)
        rc = 3 if n in self.fail else 0
        if "bench" in cmd and "--out" in cmd and rc == 0:
            Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps(engine_result(*self.tps)))
        return type("P", (), {"returncode": rc, "stdout": "", "stderr": ""})()


FULL = dict(cache_mib=43520, cache_table_mib=43520, cache_full=True, memfree_start_mib=72693, memavailable_start_mib=83811,
            memfree_min_mib=16564, memfree_stop_mib=None)
REDUCED = dict(FULL, cache_mib=40040, cache_full=False, memfree_start_mib=66985)


def fake_lb(clean=True, meta=FULL, calls=None):
    def run(pairdir):
        if calls is not None:
            calls.append(pairdir)
        if not clean:
            return {**meta, "records": None, "reason": "memfree_stop", "op_offload": False}
        return {**meta, "records": recs(), "reason": None, "op_offload": False}
    return run


def make(tmp_path, runner=None, lb_clean=True, deadline=None, snap=None, meta=FULL, calls=None, **kw):
    kw.setdefault("meminfo", str(_meminfo(tmp_path, 72693)))
    return session.Session(tmp_path, runner=runner or FakeRunner(), snapshot=lambda *a: snap or dict(OK_SNAP),
                           yardstick=YARD, lb_runner=fake_lb(lb_clean, meta, calls), deadline=deadline, python="python", **kw)


def _meminfo(tmp_path, free_mib):
    p = tmp_path / f"meminfo.{free_mib}"
    p.write_text(f"MemFree: {free_mib * 1024} kB\nMemAvailable: {83000 * 1024} kB\n")
    return p


@pytest.fixture(autouse=True)
def _limit(monkeypatch):
    monkeypatch.setattr(session.eligibility, "nvme_limit_mc", lambda *a, **k: (77850, "temp1_max-5000"))


def test_full_session_with_fakes(tmp_path):
    s = make(tmp_path)
    assert s.run() == 0
    r = json.loads((tmp_path / "result.json").read_text())
    assert (tmp_path / "started.json").exists() and r["complete"] is True
    aa = r["phases"]["aa"]
    assert len(aa["pairs"]) == 16 and aa["pairs_clean"] == 16 and aa["max_abs_dev"] == 0.0
    lb = r["lb"]
    assert lb["arm"] == "q41-b-d5" and lb["prefill_device"] == "cpu" and lb["op_offload"] is False
    assert lb["clean_pairs"] == 5 and len(lb["pairs"]) == 5            # stops at 5 clean pairs
    assert lb["cache_full_pairs"] == 5
    st = json.loads((tmp_path / "started.json").read_text())
    assert st["nvme_temp_limit_mc"] == 77850 and st["cache_rule"]["memfree_full_mib"] == 70465
    for row in aa["pairs"] + lb["pairs"]:
        assert {"nvme_temp_limit_mc", "memfree_mib", "memavailable_mib"} <= set(row["snapshot"])
    assert set(lb["pairs"][0]) >= {"engine_tps", "lb_tps", "ratio", "prefill_tps_lb", "cache_n_max", "clean", "reason", "cache_mib", "memfree_start_mib"}
    assert len(r["phases"]["d1"]["runs"]) == 3 and r["phases"]["traces"]["ok"]
    calls = s.runner.calls
    assert any("trace_routing.py" in " ".join(c) for c in calls)
    d1 = [c for c in calls if "D1" in c]
    assert len(d1) == 3 and all("--pool-slots" in c and "3880" in c for c in d1)
    assert all("--mode" in c for c in calls if "bench" in c)


def test_lb_phase_gives_up_after_8_attempts_with_no_clean_pair(tmp_path):
    s = make(tmp_path, lb_clean=False)
    assert s.run() == 1
    lb = json.loads((tmp_path / "result.json").read_text())["lb"]
    assert len(lb["pairs"]) == 8 and lb["clean_pairs"] == 0
    assert all("memfree_stop" in p["reason"] for p in lb["pairs"])


def test_ineligible_snapshot_excludes_pairs_but_keeps_them(tmp_path):
    s = make(tmp_path, snap={**OK_SNAP, "running": ["large-1"]})
    s.run()
    r = json.loads((tmp_path / "result.json").read_text())
    assert len(r["phases"]["aa"]["pairs"]) == 16 and r["phases"]["aa"]["pairs_clean"] == 0
    assert all("running_not_empty" in p["reasons"] for p in r["phases"]["aa"]["pairs"])


def test_deadline_stops_the_phases(tmp_path):
    s = make(tmp_path, deadline=1000.0)
    s.now = lambda: 990.0
    s.run()
    r = json.loads((tmp_path / "result.json").read_text())
    assert r["phases"]["aa_stopped"] == "deadline" and r["lb"]["pairs"] == []


def test_session_sh_wraps_snap_and_the_start_command_is_documented():
    sh = (ROOT / "bench/session.sh").read_text()
    assert "tools/spark/snap.sh f3-session tools/spark/py.sh bench/session.py" in sh
    assert "jobs.sh start f3-session --heavy f3-session auto 14400 --memcg slotstream-spark-lb -- bash bench/session.sh" in sh


def test_lb_goes_through_the_launcher_only():
    src = (ROOT / "bench/session.py").read_text()
    assert src.count("deploy/lb/llamacpp.sh") >= 2 and "llama-server" not in src
    assert "--no-cache-prompt" in src and "--memfree-log" in src


def test_trace_selection_and_shape():
    corpus = json.loads((ROOT / "third_party/slotstream/Tools/fixtures/expert-lookahead/corpus.json").read_text())
    reqs = trace_routing.select_requests(corpus, 48)
    assert len(reqs) == 48 and all(r["split"] == "train" and r["messages"] for r in reqs)
    assert [r["id"] for r in reqs] == [r["id"] for r in corpus["requests"] if r["split"] == "train" and r.get("messages")][:48]
    assert trace_routing.request_text(reqs[0])
    with pytest.raises(ValueError):
        trace_routing.select_requests(corpus, 131)
    ok = np.zeros((5, 48, 10), dtype=np.int16)
    assert trace_routing.check_trace(ok, 5) is ok
    for bad in (ok.astype(np.int32), ok[:, :47], ok - 1, ok + 512):
        with pytest.raises(ValueError):
            trace_routing.check_trace(bad)


def test_session_has_no_platform_temp_read():
    src = (ROOT / "bench/session.py").read_text()
    assert "nvme_temp_limit" not in src.replace("nvme_temp_limit_mc", "").replace("nvme_temp_limit_src", "") \
        and "platform.toml" not in src and "tomllib" not in src


def test_reduced_cache_pair_is_clean_and_recorded(tmp_path):
    s = make(tmp_path, meta=REDUCED)
    assert s.run() == 0
    lb = json.loads((tmp_path / "result.json").read_text())["lb"]
    assert lb["clean_pairs"] == 5 and lb["cache_full_pairs"] == 0
    assert all(p["cache_mib"] == 40040 and p["clean"] for p in lb["pairs"])


def test_memfree_start_low_skips_the_launch(tmp_path):
    sleeps, calls = [], []
    s = make(tmp_path, meminfo=str(_meminfo(tmp_path, 48000)), sleep=sleeps.append, calls=calls)
    assert s.run() == 1
    lb = json.loads((tmp_path / "result.json").read_text())["lb"]
    assert len(lb["pairs"]) == 8 and all(p["reason"] == "memfree_start_low" and p["cache_mib"] is None for p in lb["pairs"])
    assert calls == [] and sleeps == [120] * 8


def test_preflight_exits_when_limit_unreadable(tmp_path, monkeypatch):
    monkeypatch.setattr(session.eligibility, "nvme_limit_mc", lambda *a, **k: (None, "unreadable"))
    with pytest.raises(SystemExit):
        make(tmp_path).run()
    assert not (tmp_path / "started.json").exists()
