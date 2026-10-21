import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import impact_check  # noqa: E402
import prom_summary  # noqa: E402

MIB = 1024 * 1024
GIB = 1024 * MIB


def _range(vals, t0):
    return {"status": "success", "data": {"result": [{"values": [[t0 + i, str(v)] for i, v in enumerate(vals)]}]}}


def test_v1_positive_needs_both_markers_in_one_file():
    pos = "pkg/a.py:10:x = psutil.virtual_memory()\npkg/a.py:20:raise less than desired GPU memory utilization\n"
    assert prom_summary.v1_verdict(pos)["verdict"] == "POSITIVE"
    split = "pkg/a.py:10:x = psutil.virtual_memory()\npkg/b.py:20:is_integrated\n"
    assert prom_summary.v1_verdict(split)["verdict"] == "NEGATIVE"
    assert prom_summary.v1_verdict("")["verdict"] == "NEGATIVE"


def test_load_analysis_transient_and_rise(tmp_path):
    T = 1000
    base = 10_000 * MIB
    # 150 s before T flat, then a rise at T+8 s to base + 60,000 MiB.
    used = [(T - 120 + i, base) for i in range(120)]
    used += [(T + i, base) for i in range(8)]
    used += [(T + 8 + i, base + 60_000 * MIB) for i in range(100)]
    free = [(t, 30_000 * MIB) for t, _ in used]
    a, m = prom_summary.load_analysis(T, used, free, 15.0)
    assert a["transient_mib"] == 60_000 - 49_844
    assert a["min_free_mib"] == 30_000
    assert m["first_rise_s"] == 8.0
    assert m["lower_bound_s"] == -7.0


def test_scrape_interval():
    cfg = {"data": {"yaml": "global:\n  scrape_interval: 15s\n"}}
    assert prom_summary.scrape_interval_s(cfg) == 15.0
    assert prom_summary.scrape_interval_s({"data": {"yaml": "global:\n  scrape_interval: 1m\n"}}) == 60.0


def test_summarize_fixture(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "v1-grep.txt").write_text("a.py:1:psutil.virtual_memory\na.py:2:is_integrated\n")
    (raw / "prom-config.json").write_text(json.dumps({"data": {"yaml": "scrape_interval: 15s"}}))
    T = 5000
    (raw / "loads.json").write_text(json.dumps([{"T": T, "ready": T + 60}]))
    base = 5_000 * MIB
    used = [base] * 120 + [base + 55_000 * MIB] * 100
    free = [20_000 * MIB] * 220
    (raw / f"a1-{T}-used.json").write_text(json.dumps(_range(used, T - 120)))
    (raw / f"a1-{T}-free.json").write_text(json.dumps(_range(free, T - 120)))
    (raw / "b1-jobs.json").write_text(json.dumps(["svc-embed"]))
    now = int(time.time())
    day = [now - d * 86400 for d in range(3)]
    for k in prom_summary.METRICS:
        vals = []
        for d in day:
            for h in range(24):
                ts = d - (d % 3600) + h * 3600
                vals.append([ts, "30" if k == "rph" else "0.1"])
        (raw / f"b1-svc-embed-{k}.json").write_text(
            json.dumps({"data": {"result": [{"values": vals}]}}))
    s = prom_summary.summarize(tmp_path)
    assert s["v1"]["verdict"] == "POSITIVE"
    assert s["a1"]["max_transient_mib"] == 55_000 - 49_844
    assert s["m_lb5"]["ok_without_operator"] is False
    assert s["b1"]["embeddings_ttft_filled"] is True
    assert s["b1"]["low_traffic_hours"]["svc-embed"] == []


def _cap(tmp_path, klog="clean", avail_kb=40_000_000, psi="0.5", window=None, hour=10):
    d = tmp_path / "cap"
    d.mkdir()
    (d / "guard-post.json").write_text(json.dumps({"rc": 0, "cut": None, "klog": klog}))
    (d / "meminfo.csv").write_text(f"epoch,MemAvailable_kB,MemFree_kB,Cached_kB\n1,{avail_kb},1,1\n2,{avail_kb + 5},1,1\n")
    (d / "prod-psi.csv").write_text(f"epoch,container,io_some_avg10,mem_some_avg10\n1,c,{psi},0.0\n")
    (d / "verdict.json").write_text(json.dumps({"start": f"2026-10-11T{hour:02d}:05:00-05:00"}))
    if window is not None:
        (d / "prometheus-window.json").write_text(json.dumps({"jobs": window}))
    return d


BASE = {"b1": {"jobs": {"svc": {"10": {"ttft_p50": 0.1, "ttft_p99": 0.5, "queue_p99": 0.0, "rph": 100}}}}}


def test_impact_pass(tmp_path):
    d = _cap(tmp_path, window={"svc": {"requests": 50, "ttft_p99": 0.55, "ttft_p50": 0.1, "queue_p99": 0.0}})
    r = impact_check.check(d, BASE)
    assert r["verdict"] == "PASS", r


def test_impact_latency_fail(tmp_path):
    d = _cap(tmp_path, window={"svc": {"requests": 50, "ttft_p99": 0.7, "ttft_p50": 0.1, "queue_p99": 0.0}})
    assert impact_check.check(d, BASE)["verdict"] == "FAIL"


def test_impact_zero_requests_not_measured(tmp_path):
    d = _cap(tmp_path, window={"svc": {"requests": 0}})
    assert impact_check.check(d, BASE)["verdict"] == "NOT_MEASURED"


def test_impact_oom_and_memory_and_psi_fail(tmp_path):
    win = {"svc": {"requests": 50, "ttft_p99": 0.5, "ttft_p50": 0.1, "queue_p99": 0.0}}
    assert impact_check.check(_cap(tmp_path, klog="oom-kill 1", window=win), BASE)["verdict"] == "FAIL"


def test_impact_low_memory_fails(tmp_path):
    win = {"svc": {"requests": 50, "ttft_p99": 0.5, "ttft_p50": 0.1, "queue_p99": 0.0}}
    assert impact_check.check(_cap(tmp_path, avail_kb=4_000_000, window=win), BASE)["verdict"] == "FAIL"


def test_impact_psi_fail(tmp_path):
    win = {"svc": {"requests": 50, "ttft_p99": 0.5, "ttft_p50": 0.1, "queue_p99": 0.0}}
    assert impact_check.check(_cap(tmp_path, psi="12.0", window=win), BASE)["verdict"] == "FAIL"


def _low(**kw):
    w = {"requests": 5, "ttft_p50": 0.1, "queue_p99": 0.0}
    w.update(kw)
    return {"svc": w}


def test_low_traffic_pass_with_baseline_and_5xx_signal(tmp_path):
    r = impact_check.check(_cap(tmp_path, window=_low(errors_5xx=0)), BASE)
    assert r["items"]["latency"]["status"] == "PASS", r


def test_low_traffic_without_5xx_signal_is_not_measured(tmp_path):
    r = impact_check.check(_cap(tmp_path, window=_low()), BASE)
    j = r["items"]["latency"]["jobs"]["svc"]
    assert r["verdict"] == "NOT_MEASURED" and j["errors_5xx"] == "not measured"


def test_low_traffic_median_above_slack_fails(tmp_path):
    r = impact_check.check(_cap(tmp_path, window=_low(ttft_p50=0.2, errors_5xx=0)), BASE)
    assert r["verdict"] == "FAIL"


def test_low_traffic_all_none_baseline_is_not_measured(tmp_path):
    base = {"b1": {"jobs": {"svc": {"10": {"ttft_p50": None, "ttft_p99": None, "queue_p99": None}}}}}
    r = impact_check.check(_cap(tmp_path, window=_low(errors_5xx=0)), base)
    assert r["items"]["latency"]["status"] == "NOT_MEASURED" and r["verdict"] == "NOT_MEASURED"


def test_low_traffic_nonzero_5xx_fails(tmp_path):
    r = impact_check.check(_cap(tmp_path, window=_low(errors_5xx=2)), BASE)
    assert r["verdict"] == "FAIL"
