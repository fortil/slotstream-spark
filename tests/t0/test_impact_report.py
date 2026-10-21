"""tools/impact_report.py: every branch of the F9 impact criteria on synthetic captures (step 93)."""
import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("impact_report_mod", ROOT / "tools/impact_report.py")
ir = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ir)

BASELINE = {"b1": {"jobs": {"job-asr": {"22": {"ttft_p99": 1.0, "ttft_p50": 0.5, "queue_p99": 0.1}},
                            "job-embeddings": {"22": {"ttft_p99": 0.2, "ttft_p50": 0.1, "queue_p99": 0.05}}}}}


def make_cap(tmp_path, *, paused=True, klog="clean", avail_kb=(90_000_000, 80_000_000), up=1, expected=1, own=0, nv=0,
             psi=(("asr-1", 1.0, 0.5), ("qdrant-1", 0.2, 0.1), ("postgres-1", 0.0, 0.0),
                  ("embeddings-1", 0.1, 0.1)),
             start="2026-10-14T20:00:00+00:00", end="2026-10-15T00:30:00+00:00", window=None, oom_scores=(1000, 1000)):
    cap = tmp_path / "cap"
    cap.mkdir()
    (cap / "guard-post.json").write_text(json.dumps({"rc": 0, "cut": None, "klog": klog, "containers_up": up,
                                                      "expected_containers": expected, "own_containers": own,
                                                      "nvsmi_own_pids": nv, "paused": paused}))
    with open(cap / "meminfo.csv", "w") as f:
        f.write("epoch,MemAvailable_kB,MemFree_kB,Cached_kB\n")
        for i, v in enumerate(avail_kb):
            f.write(f"{i},{v},1,1\n")
    with open(cap / "prod-psi.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "container", "io_some_avg10", "mem_some_avg10"])
        for name, io, mem in psi:
            w.writerow([1, name, io, mem])
    (cap / "verdict.json").write_text(json.dumps({"label": "f9-engine", "rc": 0, "start": start, "end": end}))
    if window is not None:
        (cap / "prometheus-window.json").write_text(json.dumps({"jobs": window}))
    cell = cap / "d5-ttft-cold-8k"
    cell.mkdir()
    (cell / "result.json").write_text(json.dumps({"samples": [{"sample": i, "oom_score_adj": v} for i, v in enumerate(oom_scores)]}))
    return cap


def verdict(cap, baseline=BASELINE):
    return ir.check(cap, baseline)


def test_paused_pass_marks_the_live_production_criteria_not_applicable(tmp_path):
    res = verdict(make_cap(tmp_path))
    assert res["verdict"] == "PASS" and res["paused"]
    for k in ("3 latency", "4 psi (per cgroup)", "7 quiet window"):
        assert res["items"][k]["status"] == "NOT_APPLICABLE" and res["items"][k]["detail"] == ir.NA
    for k in ("1 oom", "2 memavail", "8 containers"):
        assert res["items"][k]["status"] == "PASS"
    assert res["items"]["8 containers"]["servers_checked_oom_score_adj"] == 2


def test_oom_in_the_kernel_log_fails(tmp_path):
    res = verdict(make_cap(tmp_path, klog="oom"))
    assert res["verdict"] == "FAIL" and res["failed"] == ["1 oom"]


def test_unreadable_kernel_log_is_not_measured(tmp_path):
    assert verdict(make_cap(tmp_path, klog="unreadable"))["verdict"] == "NOT_MEASURED"


def test_memavailable_below_8_gib_fails(tmp_path):
    res = verdict(make_cap(tmp_path, avail_kb=(90_000_000, 8_000_000)))
    assert res["verdict"] == "FAIL" and res["items"]["2 memavail"]["min_mib"] < 8192


def test_container_count_leftover_container_gpu_process_and_oom_score_each_fail(tmp_path):
    for i, kw in enumerate(({"up": 0}, {"own": 1}, {"nv": 2}, {"oom_scores": (1000, 999)})):
        d = tmp_path / f"c{i}"
        d.mkdir()
        res = verdict(make_cap(d, **kw))
        assert res["verdict"] == "FAIL" and res["failed"] == ["8 containers"], kw


def test_unpaused_expects_14_containers(tmp_path):
    res = verdict(make_cap(tmp_path, paused=False, up=14, expected=14, start="2026-10-14T05:00:00+00:00",
                           end="2026-10-14T09:00:00+00:00", window={}))
    assert res["items"]["8 containers"]["status"] == "PASS"


def test_latency_zero_requests_is_not_measured_and_fails_the_report(tmp_path, capsys):
    cap = make_cap(tmp_path, paused=False, up=14, expected=14, start="2026-10-14T22:00:00+00:00",
                   end="2026-10-14T22:30:00+00:00", window={"job-asr": {"requests": 0}})
    res = verdict(cap)
    assert res["items"]["3 latency"]["status"] == "NOT_MEASURED" and res["verdict"] == "NOT_MEASURED"
    base = tmp_path / "b.json"
    base.write_text(json.dumps(BASELINE))
    assert ir.main(["--capture", str(cap), "--baseline", str(base)]) == 1
    assert "3 latency needs a decision" in capsys.readouterr().out


def test_latency_with_traffic_within_120_percent_passes_and_above_fails(tmp_path):
    ok = {"job-asr": {"requests": 25, "ttft_p99": 1.1}, "job-embeddings": {"requests": 30, "ttft_p99": 0.23}}
    kw = dict(paused=False, up=14, expected=14, start="2026-10-14T22:00:00+00:00", end="2026-10-14T22:30:00+00:00")
    a = tmp_path / "a"
    a.mkdir()
    assert verdict(make_cap(a, window=ok, **kw))["items"]["3 latency"]["status"] == "PASS"
    b = tmp_path / "b"
    b.mkdir()
    bad = dict(ok, **{"job-asr": {"requests": 25, "ttft_p99": 1.3}})
    res = verdict(make_cap(b, window=bad, **kw))
    assert res["items"]["3 latency"]["status"] == "FAIL" and res["verdict"] == "FAIL"


def test_psi_is_per_cgroup_of_the_four_services_and_ignores_other_scopes(tmp_path):
    kw = dict(paused=False, up=14, expected=14, start="2026-10-14T22:00:00+00:00", end="2026-10-14T22:30:00+00:00", window={})
    a = tmp_path / "a"
    a.mkdir()
    res = verdict(make_cap(a, psi=(("qdrant-1", 12.0, 0.0),), **kw))
    assert res["items"]["4 psi (per cgroup)"]["status"] == "FAIL"
    b = tmp_path / "b"
    b.mkdir()
    res = verdict(make_cap(b, psi=(("other-1", 80.0, 80.0), ("qdrant-1", 1.0, 1.0)), **kw))
    it = res["items"]["4 psi (per cgroup)"]
    assert it["status"] == "PASS" and it["per_family"] == {"qdrant": 1.0} and "asr" in it["missing_families"]
    c = tmp_path / "c"
    c.mkdir()
    res = verdict(make_cap(c, psi=(("other-1", 80.0, 80.0),), **kw))
    assert res["items"]["4 psi (per cgroup)"]["status"] == "NOT_MEASURED"


def test_quiet_window_overlap_fails_when_not_paused(tmp_path):
    kw = dict(paused=False, up=14, expected=14, window={})
    a = tmp_path / "a"
    a.mkdir()
    res = verdict(make_cap(a, start="2026-10-14T23:50:00+00:00", end="2026-10-15T00:30:00+00:00", **kw))
    assert res["items"]["7 quiet window"]["status"] == "FAIL"
    b = tmp_path / "b"
    b.mkdir()
    res = verdict(make_cap(b, start="2026-10-14T05:00:00+00:00", end="2026-10-14T09:00:00+00:00", **kw))
    assert res["items"]["7 quiet window"]["status"] == "PASS"


def test_the_system_wide_psi_is_not_a_criterion():
    src = (ROOT / "tools/impact_report.py").read_text()
    assert "/proc/pressure" not in src and "pressure/memory" not in src


def test_record_has_one_verdict_line_and_validates(tmp_path):
    import db_check
    res = verdict(make_cap(tmp_path))
    text = ir.render(res, "cap/", "f9-engine")
    assert text.count("\nVerdict:") == 1 and ir.NA in text and "needs a decision" not in text
    rec = tmp_path / "rec.md"
    rec.write_text(text.replace("Capture: cap/", "Capture: cap"))
    assert db_check.check_measurement(rec, root=tmp_path) == []
    bad = tmp_path / "bad"
    bad.mkdir()
    res = verdict(make_cap(bad, klog="oom"))
    assert "Verdict: FAIL — 1 oom needs a decision" in ir.render(res, "cap/", "f9-engine")


def test_main_exit_codes_and_record(tmp_path, capsys):
    cap = make_cap(tmp_path)
    out = tmp_path / "f9-impact.md"
    base = tmp_path / "b.json"
    base.write_text(json.dumps(BASELINE))
    assert ir.main(["--capture", str(cap), "--baseline", str(base), "--write", str(out)]) == 0
    assert out.read_text().startswith("# f9-impact") and "Verdict: PASS" in out.read_text()
    assert ir.main(["--capture", str(tmp_path / "missing"), "--baseline", str(base)]) == 3
    (cap / "guard-post.json").write_text(json.dumps({"klog": "oom", "paused": True, "containers_up": 1, "expected_containers": 1}))
    assert ir.main(["--capture", str(cap), "--baseline", str(base)]) == 1


def test_no_expected_count_is_not_configured_not_fail(tmp_path):
    cap = make_cap(tmp_path)
    gp = json.loads((cap / "guard-post.json").read_text())
    del gp["expected_containers"]                       # key missing: a guard-post.json without a configured count
    (cap / "guard-post.json").write_text(json.dumps(gp))
    res = verdict(cap)
    it = res["items"]["8 containers"]
    assert it["status"] == "NOT_CONFIGURED" and it["expected"] is None and res["verdict"] == "PASS"
    gp["expected_containers"] = None                    # explicit null, as the guard writes it
    (cap / "guard-post.json").write_text(json.dumps(gp))
    assert verdict(cap)["items"]["8 containers"]["status"] == "NOT_CONFIGURED"
    gp["own_containers"] = 1                            # the other sub-checks still fail it
    (cap / "guard-post.json").write_text(json.dumps(gp))
    assert verdict(cap)["items"]["8 containers"]["status"] == "FAIL"
