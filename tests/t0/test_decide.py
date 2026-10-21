import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import db_check  # noqa: E402
import decide  # noqa: E402
from slotstream_spark.memory import footprint as fp  # noqa: E402
from slotstream_spark.memory.ledger import Ledger  # noqa: E402

YARD = {"prefill_tps_8k": 32.1, "prefill_cold_8k_s": 346.5, "prefill_cold_8k_tokens": 11124}
GIB = 1 << 30
CHUNKS = 40


def child(ttft=20.0, prefill=17.0, follow=6.0, reused=0, drift=0):
    pool = CHUNKS * GIB
    samples = [{"tag": "start", "vmrss": 500_000_000, "vmhwm": 500_000_000, "nvml": 0, "pinned": 0, "unified": {}},
               {"tag": "prefill", "vmrss": 4_000_000_000 + pool, "vmhwm": 4_000_000_000 + pool,
                "nvml": 6_000_000_000 + pool, "pinned": 0, "unified": {"pool": pool}}]
    flags = {"pool": [True, True]}
    pk = fp.peak([fp.sample_from_dict(s) for s in samples], fp.flags_from_dict(flags))
    pkd = fp.peak_to_dict(pk)
    pkd["once_bytes"] += drift
    return {"ok": True, "returncode": 0, "chunks": CHUNKS, "samples": samples, "flags": flags, "peak": pkd,
            "memavailable_drop_bytes": 50 * GIB,
            "cold": {"prompt_tokens": 8192, "reused_tokens": reused, "ttft_s": ttft, "queue_s": 0.01,
                     "prefill_s": prefill, "first_sample_s": 0.2, "store_s": 1.0},
            "followup": {"prompt_tokens": 9000, "reused_tokens": 6144, "ttft_s": follow}}


def capture(n=5, **kw):
    res = {"mode": "D5", "samples": n, "failures": [], "children": {f"cold8k-{i}": child(**kw) for i in range(n)}}
    res["children"]["once2k"] = {**child(), "cold": {"prompt_tokens": 2048, "ttft_s": 3.0, "prefill_s": 2.5,
                                                    "bytes_requested": 1000}}
    res["children"]["scope16k"] = {**child(), "runs": [{"prefill_pass": 4096, "read_scope": 8192, "scopes": 0,
                                                         "bytes_requested": 5, "records_read": 1, "prefill_s": 40.0,
                                                         "prompt_tokens": 16384}]}
    return res


def judge(res, stopgo="GO"):
    return decide.f5_prefill(res, YARD, stopgo)


def test_pass_records_axis_t_and_exit_zero():
    rows, decisions, info, code = judge(capture())
    assert code == 0 and decisions == [] and all(r[3] for r in rows)
    axis = [i for i in info if i.startswith("Axis T:")][0]
    assert axis == "Axis T: WIN vs llama.cpp arm B cold 8k prefill 32.1 tok/s (346.5 s on 11124 tokens, CPU)"
    assert any("2K once" in i for i in info) and any("16K pass 4096 scope 8192" in i for i in info)


def test_cold_median_over_the_bar_fails_with_the_plan_message():
    rows, decisions, info, code = judge(capture(ttft=31.0, prefill=28.0))
    assert code == 1
    assert decisions[0].startswith("F5 gate failed: cold 8K TTFT median (s)") and "(plan §2.2), design decision required" in decisions[0]


def test_median_not_mean():
    res = capture()
    for i, v in enumerate((100.0, 100.0, 20.0, 20.0, 20.0)):
        res["children"][f"cold8k-{i}"]["cold"]["ttft_s"] = v
    assert judge(res)[3] == 0                              # median 20 even though the mean is 52


def test_followup_over_the_bar_fails():
    assert judge(capture(follow=12.5))[3] == 1


def test_reused_tokens_in_a_cold_sample_invalidates_it():
    res = capture()
    res["children"]["cold8k-3"]["cold"]["reused_tokens"] = 4096
    rows, decisions, info, code = judge(res)
    assert code == 1 and "cold8k-3" in decisions[0] and "reused" in decisions[0]


def test_peak_over_ledger_fails():
    res = capture()
    c = res["children"]["cold8k-0"]
    c["memavailable_drop_bytes"] = Ledger("D5").expected_peak_bytes(CHUNKS) + 1
    assert judge(res)[3] == 1


def test_recorded_peak_must_match_the_samples():
    with pytest.raises(decide.BadInput):
        judge(capture(drift=1))


def test_go_on_ttft_with_axis_lose_stops_and_asks_go_does_not():
    res = capture(ttft=40.0, prefill=70.0)
    _, dec_ttft, info, code = judge(res, "GO_ON_TTFT")
    assert code == 1 and any(d.startswith("DECISION: engine loses on both axes") for d in dec_ttft)
    assert any(i.startswith("Axis T: LOSE") for i in info)
    _, dec_go, _, code_go = judge(res, "GO")
    assert code_go == 1 and not any(d.startswith("DECISION:") for d in dec_go)


def test_cli_writes_a_record_db_check_accepts(tmp_path, capsys):
    d = tmp_path / "20261013T000000-f5-ttft"
    d.mkdir()
    (d / "result.json").write_text(json.dumps(capture()))
    (d / "verdict.json").write_text("{}")
    rec = tmp_path / "f5-prefill.md"
    code = decide.main(["f5-prefill", str(d), "--write", str(rec)])
    out = capsys.readouterr().out
    assert code in (0, 1)
    text = rec.read_text()
    assert "Verdict:" in text and "Axis T:" in text and "| cold 8K TTFT median (s) |" in text
    assert db_check.check_measurement(rec, tmp_path) == []


def test_missing_capture_is_exit_three(tmp_path):
    assert decide.main(["f5-prefill", str(tmp_path / "nope")]) == 3


# ------------------------------------------------------------------------------------------------ F6 (steps 77-78)
import shutil  # noqa: E402

NOISE = 0.004216


def snap():
    return {"running": [], "nvme_temp_mc": 50000}


def ab_capture(tmp, ratios, label="f6-ab-lookahead", names=("off", "attention-corrected"), a_tps=20.0, same=True,
               with_snapshot=True, eligible=None, mtp=None):
    d = tmp / "runs" / label
    (d / "arms").mkdir(parents=True)
    rows = []
    for i, r in enumerate(ratios):
        ok = True if eligible is None else eligible[i]
        row = {"pair": i, "order": "AB" if i % 2 == 0 else "BA", "a": a_tps, "b": None if r is None else a_tps * r,
               "eligible": ok, "reasons": [] if ok else ["nvme_hot"], "ratio": r, "a_ident": "h", "b_ident": "h" if same else "x"}
        if with_snapshot:
            row["snapshot"] = snap()
        rows.append(row)
        for arm in "AB":
            arm_out = {"decode_tps": a_tps if arm == "A" else a_tps * (r or 1), "tokens_sha256": "h"}
            if mtp and arm == "B":
                arm_out["mtp"] = mtp
            else:
                arm_out["lookahead"] = {"pool": {"issued": 10, "adopted": 8, "discarded": 2, "failed": 0}}
            (d / "arms" / f"p{i:02d}-{arm}.json").write_text(json.dumps(arm_out))
    res = {"label": label, "metric": "decode_tps", "arms": {"A": names[0], "B": names[1]}, "pairs_planned": 16,
           "pairs_run": len(rows), "pairs": rows}
    (d / "result.json").write_text(json.dumps(res))
    (d / "verdict.json").write_text(json.dumps({"pass": True}))
    return d


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    for rel in ("slotstream_spark/config/platform.toml", "slotstream_spark/config/policy.toml"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / rel, root / rel)
    h3 = root / "db/records/measurements/h3.md"
    h3.parent.mkdir(parents=True, exist_ok=True)
    h3.write_text(f"# h3\n\n```toml\n[aa]\nmedian_abs_dev = {NOISE}\n```\n")
    (root / "db/records/decisions").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(decide, "ROOT", root)
    monkeypatch.setattr(decide, "H3", str(h3))
    monkeypatch.setattr(decide.yardstick, "yardstick", lambda path=None: {"decode_tps_median": 32.9})
    return root


def toml_value(root, rel, section, key):
    import tomllib
    return tomllib.loads((root / rel).read_text())[section][key]


def test_f6_lookahead_adopts_and_writes_record_and_default(repo, tmp_path, capsys):
    d = ab_capture(tmp_path, [1.05] * 16)
    assert decide.main(["f6-lookahead", str(d), "--write"]) == 0
    rec = (repo / decide.LA_RECORD).read_text()
    assert "Decision: ADOPT_ATTENTION_CORRECTED" in rec and "paired median 1.0500" in rec and "clean 16" not in rec
    assert "0.004216" in rec and "positive fraction 1.000" in rec and "issued 160, adopted 128" in rec
    assert toml_value(repo, "slotstream_spark/config/platform.toml", "lookahead", "mode") == "attention-corrected"
    assert all((k + ':') in rec for k in ('Decision', 'Evidence', 'Reversal', 'Capture', 'Verdict'))


def test_f6_lookahead_below_the_bar_keeps_off(repo, tmp_path):
    d = ab_capture(tmp_path, [1.01] * 16)
    assert decide.main(["f6-lookahead", str(d), "--write"]) == 0
    assert "Decision: KEEP_OFF" in (repo / decide.LA_RECORD).read_text()
    assert toml_value(repo, "slotstream_spark/config/platform.toml", "lookahead", "mode") == "off"


def test_f6_positive_fraction_must_reach_80_percent(repo, tmp_path):
    d = ab_capture(tmp_path, [1.5] * 12 + [0.9] * 4)                 # median 1.5, 75% positive
    assert decide.main(["f6-lookahead", str(d), "--write"]) == 0
    assert "Decision: KEEP_OFF" in (repo / decide.LA_RECORD).read_text()


def test_f6_needs_five_clean_pairs_and_writes_nothing(repo, tmp_path, capsys):
    d = ab_capture(tmp_path, [1.2] * 16, eligible=[i < 4 for i in range(16)])
    assert decide.main(["f6-lookahead", str(d), "--write"]) == 1
    assert "4 clean pairs < 5" in capsys.readouterr().out and not (repo / decide.LA_RECORD).exists()
    assert toml_value(repo, "slotstream_spark/config/platform.toml", "lookahead", "mode") == "off"


def test_f6_futility_stop_decides_keep_off_with_fewer_than_five_pairs(repo, tmp_path, capsys):
    d = ab_capture(tmp_path, [0.8] * 4)
    res = json.loads((d / "result.json").read_text())
    res["stopped"] = "futility"
    (d / "result.json").write_text(json.dumps(res))
    assert decide.main(["f6-lookahead", str(d), "--write"]) == 0
    rec = (repo / decide.LA_RECORD).read_text()
    assert "Decision: KEEP_OFF" in rec and "stopped for futility after 4 clean pairs" in rec
    assert toml_value(repo, "slotstream_spark/config/platform.toml", "lookahead", "mode") == "off"


def test_f6_refuses_when_the_margin_is_inside_the_noise(repo, tmp_path, capsys, monkeypatch):
    d = ab_capture(tmp_path, [1.2] * 16)
    monkeypatch.setattr(decide, "aa_noise", lambda path=None: 0.03)
    assert decide.main(["f6-lookahead", str(d), "--write"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("DECISION: F3 A/A noise 0.030000 ≥ adoption margin 0.02 — the A/B cannot decide at 16 pairs; stop and ask")
    assert not (repo / decide.LA_RECORD).exists()


def test_f6_mtp_margin_is_checked_against_the_noise_too(repo, tmp_path, capsys, monkeypatch):
    d = ab_capture(tmp_path, [1.2] * 16, label="f6-ab-mtp", names=("mtp-off/lookahead-off", "mtp-resident/lookahead-off"))
    monkeypatch.setattr(decide, "aa_noise", lambda path=None: 0.06)
    assert decide.main(["f6-mtp", str(d), "--mode", "D5", "--write"]) == 1
    assert "adoption margin 0.05" in capsys.readouterr().out


def test_f6_different_outputs_are_a_correctness_failure(repo, tmp_path, capsys):
    d = ab_capture(tmp_path, [1.3] * 16, same=False)
    assert decide.main(["f6-lookahead", str(d), "--write"]) == 2
    assert "correctness failure" in capsys.readouterr().out and not (repo / decide.LA_RECORD).exists()


def test_f6_rows_without_snapshot_exit_3(repo, tmp_path, capsys):
    d = ab_capture(tmp_path, [1.3] * 16, with_snapshot=False)
    for argv in (["f6-lookahead", str(d)], ["f6-mtp", str(d), "--mode", "D5"], ["f6-target", str(d)]):
        assert decide.main(argv) == 3
    assert "decide: pairs without snapshot" in capsys.readouterr().err


MTP_OUT = {"accepted_by_depth": [60, 30], "attempts_by_depth": [100, 60], "rounds": 100, "emitted_by_round": [2] * 100}


def test_f6_mtp_adopted_writes_acceptance_and_policy(repo, tmp_path):
    d = ab_capture(tmp_path, [1.12] * 16, label="f6-ab-mtp", names=("mtp-off/lookahead-attention", "mtp-resident/lookahead-attention"),
                   mtp=MTP_OUT)
    assert decide.main(["f6-mtp", str(d), "--mode", "D5", "--write"]) == 0
    rec = (repo / decide.MTP_RECORD).read_text()
    assert "Decision: ADOPT_RESIDENT" in rec and "depth 1 960/1600 = 0.600" in rec and "depth 2 480/960 = 0.500" in rec
    assert "lookahead in both arms: attention" in rec
    pol = "slotstream_spark/config/policy.toml"
    assert [toml_value(repo, pol, "mtp", k) for k in ("D1", "D2", "D5")] == ["off", "off", "resident"]
    assert "off (not measured)" in (repo / pol).read_text()


def test_f6_mtp_below_1_05_is_not_enabled(repo, tmp_path):
    d = ab_capture(tmp_path, [1.04] * 16, label="f6-ab-mtp", names=("mtp-off/lookahead-off", "mtp-resident/lookahead-off"), mtp=MTP_OUT)
    assert decide.main(["f6-mtp", str(d), "--mode", "D5", "--write"]) == 0
    assert "Decision: KEEP_OFF" in (repo / decide.MTP_RECORD).read_text()
    assert toml_value(repo, "slotstream_spark/config/policy.toml", "mtp", "D5") == "off"


def test_f6_mtp_only_knows_d5(repo, tmp_path):
    d = ab_capture(tmp_path, [1.2] * 16)
    assert decide.main(["f6-mtp", str(d), "--mode", "D2"]) == 3


def test_f6_target_pass_and_axis_d_line(repo, tmp_path, capsys):
    (repo / decide.LA_RECORD).write_text("Decision: ADOPT_ATTENTION_CORRECTED\n")
    (repo / "db/records/decisions/stopgo-f3.md").write_text("Median ratio 0.6150, positive fraction 0.00\n")
    d = ab_capture(tmp_path, [1.25] * 16, a_tps=20.0)                  # B arm 25 tok/s
    assert decide.main(["f6-target", str(d), "--min", "23.0"]) == 0
    rec = (repo / decide.TARGET_RECORD).read_text()
    assert "D5 decode median 25.000 tok/s over 16 clean pairs" in rec
    assert "vs llama.cpp arm B (32.9 tok/s, full 43,520 MiB cache, LB2): 76.0% — Axis D: LOSE" in rec
    assert "same-day paired ratio was 0.6150" in rec and rec.rstrip().endswith("Verdict: PASS — D5 decode median at or above the F6 target")


def test_f6_target_on_the_mtp_capture_reads_arm_a(repo, tmp_path, capsys):
    (repo / decide.LA_RECORD).write_text("Decision: ADOPT_ATTENTION\n")        # would select arm B on a lookahead capture
    d = ab_capture(tmp_path, [1.3] * 16, label="f6-ab-mtp", a_tps=24.0,
                   names=("mtp-off/lookahead-attention", "mtp-resident/lookahead-attention"), mtp=MTP_OUT)
    assert decide.main(["f6-target", str(d), "--min", "23.0"]) == 0
    out = capsys.readouterr().out
    assert "D5 decode median 24.000 tok/s over 16 clean pairs (MTP A/B arm A: lookahead ADOPT_ATTENTION, MTP off, cohort B0)" in out
    assert "- arm in force: MTP A/B arm A" in (repo / decide.TARGET_RECORD).read_text()


def test_f6_target_writes_notes(repo, tmp_path):
    (repo / decide.LA_RECORD).write_text("Decision: KEEP_OFF\n")
    d = ab_capture(tmp_path, [1.0] * 16, a_tps=25.0)
    assert decide.main(["f6-target", str(d), "--note", "first note", "--note", "second note"]) == 0
    rec = (repo / decide.TARGET_RECORD).read_text()
    assert "- note: first note" in rec and "- note: second note" in rec
    assert rec.index("- note: second note") < rec.index("Verdict:")


def test_f6_target_below_23_stops_with_the_plan_line(repo, tmp_path, capsys):
    (repo / decide.LA_RECORD).write_text("Decision: KEEP_OFF\n")
    d = ab_capture(tmp_path, [1.5] * 16, a_tps=19.7)                   # lookahead kept off: the A arm (19.7) is in force
    assert decide.main(["f6-target", str(d), "--min", "23.0"]) == 1
    out = capsys.readouterr().out
    assert "D5 decode 19.70 < 23 tok/s at end of F6: design decision required" in out
    assert "Verdict: FAIL" in (repo / decide.TARGET_RECORD).read_text()


def test_f6_target_axis_d_win_at_or_above_the_yardstick(repo, tmp_path):
    (repo / decide.LA_RECORD).write_text("Decision: ADOPT_ATTENTION\n")
    d = ab_capture(tmp_path, [1.7] * 16, a_tps=20.0)                   # 34 tok/s
    assert decide.main(["f6-target", str(d), "--no-write"]) == 0
    assert not (repo / decide.TARGET_RECORD).exists()


def test_frozen_cohorts_match_their_manifest():
    import hashlib
    d = ROOT / "bench" / "cohorts"
    man = json.loads((d / "manifest.json").read_text())
    want = {"B0": ("test", 33), "B1": ("validation", 32)}
    for name, (split, n) in want.items():
        c = man["cohorts"][name]
        data = (d / c["file"]).read_bytes()
        rows = [json.loads(l) for l in data.decode().splitlines()]
        assert hashlib.sha256(data).hexdigest() == c["sha256"] and c["verified_sha"] == c["requests"] == len(rows) == n
        assert all(r["split"] == split for r in rows) and len(c["pair_ids"]) == 16
        ids = [r["id"] for r in rows]
        assert all(i in ids for i in c["pair_ids"]) and c["pair_ids"] == sorted(c["pair_ids"])
        est = {r["id"]: r["prompt_tokens_estimate"] for r in rows}
        assert all(est[i] <= man["pair_max_token_estimate"] for i in c["pair_ids"])
        firsts = [r["id"] for r in rows if r["prompt_tokens_estimate"] <= man["pair_max_token_estimate"]][:16]
        assert firsts == c["pair_ids"]
    assert man["seed"] == 1729
