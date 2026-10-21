import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gates import records  # noqa: E402
import db_check  # noqa: E402

CAPTURE = ROOT / "db/sources/runs/2026/10/20261012T173240-m0prime/result.json"
CAPTURE_REL = "db/sources/runs/2026/10/20261012T173240-m0prime"


def _base():
    layer = {"mode": "graph", "cold_us_median": 900.0, "cold_us_p99": 1300.0}
    return {
        "gemv": {"2560x2560_b4": {"n": 2560, "k": 2560, "bits": 4, "bytes": 3701760,
                                  "cold_graph_us": 31.31, "gbs_cold_graph": 118.25}},
        "tau": {"p50_us": 14.1, "p99_us": 15.6, "wait_p50_us": 0.35, "ensure_p50_us": 11.5,
                "launch_p50_us": 2.2, "steps": 1024, "warmup": 16},
        "runtime": {"footprint_bytes": 2637857704, "vmrss": 8280911872, "nvml": 7929053184,
                    "nvml_source": "pynvml", "pool_bytes": 6442450944, "pool_in_rss": True,
                    "pool_in_nvml": True, "weights_bytes": 562076760, "state_bytes": 125128704},
        "gdn_ab": {"rel_l2_by_layer": {"0": 0.0038, "1": 0.0053, "2": 0.0073, "3": 0.011},
                   "kl_mean": 1.8e-4, "kl_p99": 2.9e-3, "kl_max": 5.8e-3, "top1": 0.97,
                   "finite": True, "positions": 277},
        "fill": {"records": 2048, "bytes": 5662310400, "seconds": 0.55, "io_gbs": 10.3, "wall_gbs": 10.2,
                 "lanes": 10, "lane_cpus": [0, 1], "t_memcpy_mean_ms": 0.1346, "t_memcpy_p50_ms": 0.09,
                 "t_memcpy_p99_ms": 0.52},
        "timings": {"layer_steps": {"0": dict(layer), "2": dict(layer), "3": dict(layer)},
                    "t_gdn_us": 916.23, "t_qsa_us": 842.46, "t_embed_us": 21.38, "t_lm_head_us": 2533.34,
                    "t_comp_48_ms": 45.648},
        "chunk_free": {"samples": 100, "p50_ns": 17165749, "p99_ns": 18604500, "max_ns": 18730308},
        "versions": {"torch": "2.13.0", "cuda": "13.0"},
    }


def _run(tmp_path, res, *extra):
    p = tmp_path / "result.json"
    p.write_text(json.dumps(res))
    return records.main(["m0prime", str(p), *extra])


def test_measured_values_pass_and_gemv_is_informational(tmp_path, capsys):
    assert _run(tmp_path, _base()) == 0
    out = capsys.readouterr().out
    assert "DECISION:" not in out
    assert "informational" in out and "step 55" in out


def test_footprint_ceiling(tmp_path, capsys):
    r = _base()
    r["runtime"]["footprint_bytes"] = 2_900_000_000
    assert _run(tmp_path, r) == 0
    r["runtime"]["footprint_bytes"] = 2_900_000_001
    assert _run(tmp_path, r) == 1
    assert "DECISION: runtime footprint" in capsys.readouterr().out


def test_gemv_low_value_still_passes(tmp_path, capsys):
    r = _base()
    r["gemv"]["2560x2560_b4"]["gbs_cold_graph"] = 50.0
    assert _run(tmp_path, r) == 0
    out = capsys.readouterr().out
    assert "DECISION: GEMV" not in out
    assert "step 55" in out


def test_tau_over_limit_fails(tmp_path):
    r = _base()
    r["tau"]["p50_us"] = 151
    assert _run(tmp_path, r) == 1


def test_write_record_and_note(tmp_path):
    dest = tmp_path / "m0prime.md"
    assert _run(tmp_path, _base(), "--note", "first note", "--note", "second note", "--write", str(dest)) == 0
    lines = dest.read_text().splitlines()
    i = next(n for n, l in enumerate(lines) if l.startswith("Command:"))
    assert lines[i + 1] == "first note" and lines[i + 2] == "second note"
    assert [l for l in lines if l.startswith("Verdict:")][0].startswith("Verdict: PASS")
    assert not any(l.startswith("DECISION:") for l in lines)
    (tmp_path / "verdict.json").write_text(json.dumps({"rc": 0}))  # the Capture dir is tmp_path
    assert db_check.check_measurement(dest) == []


def test_real_capture(tmp_path):
    if not CAPTURE.exists():
        pytest.skip(f"capture not present locally: {CAPTURE_REL}/result.json (captures are gitignored)")
    assert records.main(["m0prime", str(CAPTURE)]) == 0


# ---- h1 ----
from slotstream_spark.memory import footprint as fp  # noqa: E402

OLD_H1 = ROOT / "db/sources/runs/2026/10/20261012T180045-h1/result.json"
OLD_H1_REL = "db/sources/runs/2026/10/20261012T180045-h1"
_G = 1 << 30


def _h1_base():
    smp = [
        {"tag": "base", "vmrss": 1_048_608_768, "vmhwm": 1_048_608_768, "nvml": 679_985_152, "pinned": 0, "unified": {"pool": 0}},
        {"tag": "fixture", "vmrss": 1_048_608_768 + 2 * _G, "vmhwm": 1_048_608_768 + 2 * _G, "nvml": 679_985_152 + 2 * _G,
         "pinned": 0, "unified": {"pool": 2 * _G}},
        {"tag": "c", "vmrss": 1_048_608_768 + 3 * _G, "vmhwm": 1_048_608_768 + 3 * _G, "nvml": 679_985_152 + 3 * _G,
         "pinned": 0, "unified": {"pool": 3 * _G}},
        {"tag": "d", "vmrss": 1_048_608_768 + 2 * _G, "vmhwm": 1_048_608_768 + 3 * _G, "nvml": 679_985_152 + 2 * _G,
         "pinned": 0, "unified": {"pool": 2 * _G}},
    ]
    flags = {"pool": [True, True]}
    p = fp.peak([fp.sample_from_dict(s) for s in smp], fp.flags_from_dict(flags))
    d = fp.peak_to_dict(p) | {"peak_bytes": p.nonpool_bytes, "vmhwm": max(s["vmhwm"] for s in smp),
                              "nvml_max": max(s["nvml"] for s in smp), "nvml_source": "pynvml", "pinned": 0}
    return {"torch": "x", "samples": smp, "flags": flags, "d": d,
            "a": {"ok": True, "separable": True, "positions": 23, "rel_l2": 0.006, "limit": 0.03, "spread": 0.01,
                  "control": 0.3, "decode_rel_l2": 0.005, "routing_disagreement": 0, "routing_limit": 0.06,
                  "routing_separable": False, "ngram_ok": True},
            "b": {"pools": [13, 88], "bitwise_equal": True, "sha256": "ab"},
            "c": {"ensures": 1000, "errors": 0, "bytes_equal": 1000, "gpu_checksum_ok": True, "max_read_s": 0.01,
                  "pool_slots": 64, "hits": 1, "misses": 2}}


def _run_h1(tmp_path, res, *extra):
    p = tmp_path / "result.json"
    p.write_text(json.dumps(res))
    return records.main(["h1", str(p), *extra])


def test_h1_pass(tmp_path, capsys):
    assert _run_h1(tmp_path, _h1_base()) == 0
    assert "once per counter" in capsys.readouterr().out


def test_h1_recorded_peak_mismatch(tmp_path, capsys):
    r = _h1_base()
    r["d"]["peak_bytes"] += 1
    assert _run_h1(tmp_path, r) == 3
    assert "cannot evaluate" in capsys.readouterr().err


def test_h1_over_limit(tmp_path, capsys):
    r = _h1_base()
    for s in r["samples"]:
        s["vmhwm"] += 2 * _G
        s["vmrss"] += 2 * _G
    r["samples"][-1]["vmrss"] += 0
    sm = [fp.sample_from_dict(s) for s in r["samples"]]
    base = fp.nonpool_bytes(sm[-1], fp.flags_from_dict(r["flags"]))
    add = (4096 << 20) - fp.peak(sm, fp.flags_from_dict(r["flags"])).nonpool_bytes
    for s in r["samples"]:
        s["vmhwm"] += add
        s["vmrss"] += add
    sm = [fp.sample_from_dict(s) for s in r["samples"]]
    r["d"]["peak_bytes"] = fp.peak(sm, fp.flags_from_dict(r["flags"])).nonpool_bytes
    assert r["d"]["peak_bytes"] == 4096 << 20
    assert _run_h1(tmp_path, r) == 1
    assert "DECISION: H1 gate failed: (d)" in capsys.readouterr().out


def test_h1_old_schema(tmp_path):
    r = _h1_base()
    r["d"] = {"total_bytes": 7905071104, "vmhwm": 1, "nvml": 1, "pinned": 0}
    del r["samples"], r["flags"]
    assert _run_h1(tmp_path, r) == 3


def test_h1_write_note(tmp_path):
    dest = tmp_path / "h1.md"
    assert _run_h1(tmp_path, _h1_base(), "--note", "n1", "--write", str(dest)) == 0
    lines = dest.read_text().splitlines()
    i = next(n for n, l in enumerate(lines) if l.startswith("Command:"))
    assert lines[i + 1] == "n1"
    assert any("each byte once" in l for l in lines)
    assert [l for l in lines if l.startswith("Verdict:")][0].startswith("Verdict: PASS")
    (tmp_path / "verdict.json").write_text(json.dumps({"rc": 0}))
    assert db_check.check_measurement(dest) == []


def test_h1_old_capture_refused():
    if not OLD_H1.exists():
        pytest.skip(f"capture not present locally: {OLD_H1_REL}/result.json")
    assert records.main(["h1", str(OLD_H1)]) == 3


# ---- h2 ----
OLD_H2 = ROOT / "db/sources/runs/2026/10/20261012T184122-h2/result.json"
OLD_H2_REL = "db/sources/runs/2026/10/20261012T184122-h2"


def _h2_layers(pair=0.05):
    return [{"layer": k, "rel_l2": 0.01, "limit": 0.03, "separable": True, "ok": True, "routing": 0.02,
             "routing_limit": 0.06, "routing_separable": True, "routing_ok": True, "rel_l2_pair": 0.01,
             "limit_cal": 0.03, "limit_src": "calibrated", "routing_pair": pair, "routing_limit_cal": 0.06,
             "routing_limit_src": "calibrated"} for k in range(48)]


def _h2_base(tmp_path):
    smp = [
        {"tag": "base", "vmrss": 1_048_608_768, "vmhwm": 1_048_608_768, "nvml": 679_985_152, "pinned": 0, "unified": {"pool": 0}},
        {"tag": "big", "vmrss": 1_048_608_768 + 50 * _G, "vmhwm": 1_048_608_768 + 50 * _G,
         "nvml": 679_985_152 + 50 * _G, "pinned": 0, "unified": {"pool": 50 * _G}},
    ]
    flags = {"pool": [True, True]}
    p = fp.peak([fp.sample_from_dict(s) for s in smp], fp.flags_from_dict(flags))
    prompts = {f"dev{k}": {"tokens": 100, "prompt_len": 36, "ngram_ok": True, "failing_layers": [],
                           "layers": _h2_layers(), "logits": {"ok": True}, "ok": True, "worst_routing_ratio": 0.5,
                           "worst_routing_layer": 20, "worst_routing_src": "calibrated", "bound_layers": []}
               for k in range(8)}
    res = {"sizes": {"tokens_equal": True, "logits_equal": True,
                     "per_size": {str(s): {"chunks": 1} for s in (624, 2304, 3880, 5044, 19012)}},
           "resize": {"all_identical": True},
           "fault": {"fired": True, "map_unchanged": True, "pins_restored": True, "retry_identical": True},
           "gdn": {"failing_layers": [], "logits": {"ok": True}},
           "oracle": {"ok": True, "prompts": prompts},
           "samples": smp, "flags": flags, "peak": footprint_dict(p), "cap_bytes": 64 * _G, "chunks_max": 49,
           "exit": {"pytest": 0, "oracle": 0}}
    (tmp_path / "guard-post.json").write_text(json.dumps({"rc": 0, "cut": None, "containers_up": 14}))
    return res


def footprint_dict(p):
    return fp.peak_to_dict(p)


def _run_h2(tmp_path, res, *extra):
    p = tmp_path / "result.json"
    p.write_text(json.dumps(res))
    return records.main(["h2", str(p), *extra])


def test_h2_pass(tmp_path, capsys):
    assert _run_h2(tmp_path, _h2_base(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "Routing rule: seq_rule" in out
    assert "2x32" in out


def test_h2_recorded_peak_mismatch(tmp_path, capsys):
    r = _h2_base(tmp_path)
    r["peak"]["once_bytes"] += 1
    assert _run_h2(tmp_path, r) == 3
    assert "cannot evaluate" in capsys.readouterr().err


def test_h2_once_over_cap(tmp_path, capsys):
    r = _h2_base(tmp_path)
    r["cap_bytes"] = 40 * _G
    assert _run_h2(tmp_path, r) == 1
    assert "DECISION: H2 gate failed: (f)" in capsys.readouterr().out


def test_h2_ledger_over_ceiling(tmp_path, capsys):
    r = _h2_base(tmp_path)
    r["chunks_max"] = 50
    assert _run_h2(tmp_path, r) == 1
    assert "(g)" in capsys.readouterr().out


def test_h2_oracle_prompt_fails(tmp_path, capsys):
    r = _h2_base(tmp_path)
    r["oracle"]["prompts"]["dev7"].update(failing_layers=[43], ok=False)
    r["oracle"]["ok"] = False
    assert _run_h2(tmp_path, r) == 1
    assert "(e)" in capsys.readouterr().out


def test_h2_missing_guard_post(tmp_path):
    r = _h2_base(tmp_path)
    (tmp_path / "guard-post.json").unlink()
    assert _run_h2(tmp_path, r) == 3


def test_h2_old_schema_refused(tmp_path):
    if not OLD_H2.exists():
        pytest.skip(f"capture not present locally: {OLD_H2_REL}/result.json")
    assert records.main(["h2", str(OLD_H2)]) == 3


def test_h2_write_note(tmp_path):
    dest = tmp_path / "h2.md"
    assert _run_h2(tmp_path, _h2_base(tmp_path), "--note", "n1", "--write", str(dest)) == 0
    lines = dest.read_text().splitlines()
    assert "Routing rule: seq_rule (factor 2.0, ceiling c_k/2), design decision 16" in lines
    assert [l for l in lines if l.startswith("Verdict:")][0].startswith("Verdict: PASS")
    (tmp_path / "verdict.json").write_text(json.dumps({"rc": 0}))
    assert db_check.check_measurement(dest) == []


# ---- step 65: h3 and stopgo ----
def _pair(i, ratio, cache=43520, clean=True):
    return {"pair": i, "clean": clean, "reason": None if clean else "memfree_stop", "ratio": ratio, "cache_mib": cache,
            "memfree_start_mib": 70500 if cache >= 43520 else 67000, "memavailable_start_mib": 82000,
            "memfree_min_mib": 15000, "engine_tps": {"short": 30.0 * ratio, "2k": 34.0 * ratio},
            "lb_tps": {"short": 30.0, "2k": 34.0}, "prefill_tps_lb": {"short": 5.0, "2k": 24.0},
            "snapshot": {"nvme_temp_mc": 55000, "nvme_temp_limit_mc": 77850}}


def _sg(ratios, caches=None, drop=()):
    caches = caches or [43520] * len(ratios)
    pairs = [_pair(i, r, c) for i, (r, c) in enumerate(zip(ratios, caches))]
    for k in drop:
        for p in pairs:
            p.pop(k, None)
    aa = [{"eligible": True, "a": {"short": 20.0, "2k": 21.0}, "b": {"short": 20.2, "2k": 21.1},
           "ratio": {"short": 1.01, "2k": 1.005}, "reasons": [], "snapshot": {"nvme_temp_mc": 55000}} for _ in range(16)]
    return {"lb": {"arm": "q41-b-d5", "prefill_device": "cpu", "pairs": pairs,
                   "cache_rule": {"full_mib": 43520, "memfree_full_mib": 70465}},
            "phases": {"aa": {"pairs": aa}, "d1": {"runs": [{"error": None, "tps": {"short": 10.0, "2k": 9.0}}]}}}


@pytest.fixture
def h3_pass(monkeypatch):
    monkeypatch.setattr(records, "_record_verdict", lambda rel: ("Verdict: PASS", "t_copy separate"))


def test_stopgo_go_with_reduced_cache_suffix(h3_pass):
    rc, lines, out = records.stopgo(_sg([1.02] * 5, [43520] * 3 + [40040] * 2))
    assert rc == 0 and lines[0].startswith("Decision: GO ")
    assert "2/5" in lines[0] and "40040" in lines[0]


def test_stopgo_go_full_cache_has_no_suffix(h3_pass):
    rc, lines, _ = records.stopgo(_sg([1.02] * 5))
    assert rc == 0 and "reduced cache" not in lines[0]


def test_stopgo_missing_cache_mib_is_exit_3(h3_pass):
    rc, lines, _ = records.stopgo(_sg([1.02] * 5, drop=("cache_mib",)))
    assert rc == 3 and "cache_mib" in lines[0]


def test_stopgo_decision_then_user_answer(h3_pass):
    res = _sg([0.62] * 5, [40040] * 5)
    rc, lines, out = records.stopgo(res)
    assert rc == 1 and lines[0].startswith("DECISION: engine D5 decode") and "continue-on-ttft | stop" in lines[0]
    assert records.stopgo(res, "continue-on-ttft", None)[0] == 2
    rc, lines, _ = records.stopgo(res, "continue-on-ttft", "BRIEF.md ## User answers 2026-10-13")
    assert rc == 0 and lines[0].startswith("Decision: GO_ON_TTFT")
    assert records.stopgo(res, "stop", "ref")[0] == 1


def test_stopgo_user_decision_refused_when_median_ge_1(h3_pass):
    assert records.stopgo(_sg([1.02] * 5), "continue-on-ttft", "ref")[0] == 2


def test_stopgo_user_decision_refused_below_half_line(h3_pass):
    rc, lines, _ = records.stopgo(_sg([0.4] * 5), "continue-on-ttft", "ref")
    assert rc == 2 and "half-yardstick stop line" in lines[0]


def test_stopgo_user_decision_refused_when_h3_not_pass(monkeypatch):
    monkeypatch.setattr(records, "_record_verdict", lambda rel: ("Verdict: FAIL", "x"))
    rc, lines, _ = records.stopgo(_sg([0.62] * 5, [40040] * 5), "continue-on-ttft", "ref")
    assert rc == 2 and "h3.md is not PASS" in lines[0]


def test_h4_grow_gap_label_matches_constant():
    rec = {"floor_s": 1.0, "grows": 2, "grow_steps_chunks": [1, 1], "grow_gaps_s": [records.H4_GROW_GAP_S]}
    rows, _, _ = records.evaluate_h4({"running-stub-d1-a": rec})
    row = [r for r in rows if r[0].startswith("D1+A ready")][0]
    assert row[3] is True and "59.9" in row[2] and "tolerance" in row[0]


def test_stopgo_below_half_stops(h3_pass):
    rc, lines, _ = records.stopgo(_sg([0.4] * 5))
    assert rc == 1 and "does not reach half" in lines[0]


def test_stopgo_needs_five_clean_pairs(h3_pass):
    rc, lines, _ = records.stopgo(_sg([1.02] * 4))
    assert rc == 1 and lines[0].startswith("DECISION")


def test_h3_aa_block_keys(tmp_path):
    res = _sg([1.0] * 5)
    rows, decisions, aa, d1 = records.evaluate_h3(res)
    dest = tmp_path / "h3.md"
    records.write_h3(res, rows, decisions, aa, d1, str(tmp_path / "result.json"), str(dest))
    import yardstick
    blk = yardstick.load_record(str(dest))["aa"]
    for k in ("pairs", "median_abs_dev", "max_abs_dev", "nvme_temp_c_min", "nvme_temp_c_max", "excluded_reasons"):
        assert k in blk
    assert blk["pairs"] == 16


def test_h3_low_decode_prints_decision():
    res = _sg([1.0] * 5)
    for p in res["phases"]["aa"]["pairs"]:
        p["a"] = p["b"] = {"short": 10.0, "2k": 11.0}
    _, decisions, _, _ = records.evaluate_h3(res)
    assert decisions and decisions[0].startswith("DECISION: D5 decode") and "stop and ask" in decisions[0]


# ---- step 66: H4 drills -------------------------------------------------------------------------------------------
def _h4_dir(tmp_path, **over):
    GiB = 1 << 30
    rel = [{"t_ns": 1, "chunks": 1, "delay_s": 0.6, "avail_rise_bytes": 1 * GiB}]
    base = {
        "running-stub-d5": {"floor_s": 0.4, "exit_s": 6.0, "oom_lines": []},
        "running-stub-d1-a": {"floor_s": 0.5, "grows": 2, "grow_steps_chunks": [1, 1], "grow_gaps_s": [61.0], "oom_lines": []},
        "running-stub-d1-b": {"exit_s": 1.2, "oom_lines": []},
        "running-stub-d2-a": {"floor_s": 0.6, "oom_lines": []},
        "hog-d5": {"releases": rel, "avail_min_bytes": 12 * GiB, "hog_stop_reason": "memavailable_floor", "hog_size_mib": 9000,
                   "hog_steps": 9, "engine_cap_mib": 40000, "chunks_start": 30, "engine_peak_once_bytes": 30 * GiB, "oom_lines": []},
        "hog-d1": {"releases": rel, "avail_min_bytes": 12 * GiB, "hog_stop_reason": "size_cap", "hog_size_mib": 5000,
                   "hog_steps": 5, "engine_cap_mib": 20000, "chunks_start": 10, "engine_peak_once_bytes": 15 * GiB, "oom_lines": []},
        "restart": {"first_start": {"first_status": 503, "retry_after": "10", "seconds_to_200": 80.0}, "idle_exit_s": 70.0,
                    "engine_procs_after_idle": [], "second_start": {"first_status": 503, "retry_after": "10", "seconds_to_200": 75.0},
                    "stub_nonempty": {"status": 503, "engine_procs": [], "engine_unit": "inactive"}},
        "stop": {"stop_s": 1.5, "start_timer": "TimersCalendar={ OnCalendar=*-*-* 04:00:00 }",
                 "stop_timer": "TimersCalendar={ OnCalendar=*-*-* 23:59:00 }", "socket_has_install_section": False,
                 "may_start": {"03:59": [False, "before 04:00"], "04:00": [True, "ok"], "23:44": [True, "ok"],
                               "23:45": [False, "after 23:45"]}, "socket_restored": "active"},
        "arbiter-1": {"throttle_c": 77.85, "source": "temp1_max-5000", "caps_at_construction": [12, 12]},
    }
    base.update(over)
    for k, v in base.items():
        (tmp_path / f"h4-{k}.json").write_text(json.dumps(v))
    return tmp_path


def test_h4_all_pass_writes_record(tmp_path):
    d = _h4_dir(tmp_path)
    dest = tmp_path / "h4.md"
    assert records.main(["h4", str(d), "--write", str(dest)]) == 0
    txt = dest.read_text()
    assert "Verdict: PASS" in txt and "77.85" in txt


def test_h4_slow_release_fails(tmp_path):
    GiB = 1 << 30
    slow = {"releases": [{"t_ns": 1, "chunks": 1, "delay_s": 2.6, "avail_rise_bytes": GiB}], "avail_min_bytes": 12 * GiB,
            "hog_stop_reason": "size_cap", "hog_size_mib": 1, "hog_steps": 1, "engine_cap_mib": 1, "chunks_start": 3,
            "engine_peak_once_bytes": 0, "oom_lines": []}
    d = _h4_dir(tmp_path, **{"hog-d5": slow})
    assert records.main(["h4", str(d)]) == 1


def test_h4_missing_drill_and_oom_fail(tmp_path):
    d = _h4_dir(tmp_path, **{"running-stub-d2-a": {"floor_s": 0.6, "oom_lines": ["Out of memory: Killed process 1"]}})
    (d / "h4-restart.json").unlink()
    rows, decisions, _ = records.evaluate_h4(records.load_h4([d]))
    failed = {r[0] for r in rows if not r[3]}
    assert "restart record" in failed and "oom-kill or NV_ERR_NO_MEMORY lines in journalctl -k" in failed


def test_h4_without_directory_is_exit_3():
    assert records.main(["h4", "--note", "x"]) == 3


def test_serve_wires_sampler_and_arbiter():
    import inspect
    sys.path.insert(0, str(ROOT))
    from slotstream_spark import cli
    src = inspect.getsource(cli.cmd_serve)
    assert "FootprintFeed(pool" in src and "read_footprint=feed.read" in src and "start_arbiter(pool.reader)" in src
    assert "80.0" not in inspect.getsource(cli.start_arbiter) + inspect.getsource(cli.nvme_throttle_c)


# ---- step 70a: sweep-check ----
def _sweep_base(tmp_path, kl_max=1.4930166431466012, pair_kl_max=1.1491823956944724, with_pair=True):
    layers = [{"layer": k, "ok": True, "routing_ok": True, "sweep_vs_f2_ok": True, "limit_src": "calibrated"}
              for k in range(48)]
    lg = {"top1": 0.99267578125, "kl_mean": 0.004348508799017182, "kl_p99": 0.04749969207287939, "kl_max": kl_max,
          "fixed_limits_ok": False, "ok": True,
          "limits_seq": {"top1_min": 0.89, "kl_mean": 0.04, "kl_p99": 0.43, "kl_max": 3.4475}}
    if with_pair:
        lg["pair"] = {"top1": 0.994140625, "kl_mean": 0.004408653354052665, "kl_p99": 0.06475244648703134,
                      "kl_max": pair_kl_max}
    res = {"layers": layers, "failing_layers": [], "logits": lg,
           "summary": {k: {"hidden_sha256": "ab" * 32, "ids": list(range(32))} for k in ("2k", "8k", "9k")}}
    (tmp_path / "guard-post.json").write_text(json.dumps({"rc": 0, "cut": None, "containers_up": 14}))
    return res


def _run_sweep(tmp_path, res, *extra):
    p = tmp_path / "sweep_check.json"
    p.write_text(json.dumps(res))
    return records.main(["sweep-check", str(p), *extra])


def test_sweep_check_pass(tmp_path, capsys):
    assert _run_sweep(tmp_path, _sweep_base(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "Logits rule: seq KL" in out and "ceiling" in out and "stored bridge (not used)" in out


def test_sweep_check_kl_max_over_limit(tmp_path, capsys):
    assert _run_sweep(tmp_path, _sweep_base(tmp_path, kl_max=1.537)) == 1
    assert "DECISION:" in capsys.readouterr().out


def test_sweep_check_pair_over_ceiling(tmp_path, capsys):
    assert _run_sweep(tmp_path, _sweep_base(tmp_path, pair_kl_max=1.6)) == 1
    assert "DESIGN GATE:" in capsys.readouterr().out


def test_sweep_check_old_schema_refused(tmp_path):
    assert _run_sweep(tmp_path, _sweep_base(tmp_path, with_pair=False)) == 3


def test_sweep_check_missing_guard_post(tmp_path):
    res = _sweep_base(tmp_path)
    (tmp_path / "guard-post.json").unlink()
    assert _run_sweep(tmp_path, res) == 3


def test_sweep_check_write(tmp_path):
    dest = tmp_path / "f5-sweep-check.md"
    assert _run_sweep(tmp_path, _sweep_base(tmp_path), "--note", "n1", "--write", str(dest)) == 0
    lines = dest.read_text().splitlines()
    assert [l for l in lines if l.startswith("Verdict:")][0].startswith("Verdict: PASS")
    (tmp_path / "verdict.json").write_text(json.dumps({"rc": 0}))
    assert db_check.check_measurement(dest) == []
