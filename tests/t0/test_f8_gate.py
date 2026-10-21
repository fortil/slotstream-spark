"""Step 90: the F8 entry decision (decide.py f8-entry) and the skip/enter handling of tools/f8_gate.py."""
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import decide  # noqa: E402
import f8_gate  # noqa: E402

PHASES = ("sample", "ngram_hash", "embed_read", "ngram_wait", "launch", "gpu_wait", "ensure_hits",
          "expert_miss_read", "expert_copy_tail")


def entry(host_share, io_share=0.10, tokens=64, wall=1_000_000_000):
    """One prompt of a profile result whose host phases are `host_share` of the wall (spread over the host phases)."""
    host = int(wall * host_share)
    io = int(wall * io_share)
    ph = {p: 0 for p in PHASES}
    ph["launch"] = host // 2
    ph["ensure_hits"] = host - host // 2 - 1000
    ph["expert_miss_read"] = io
    ph["gpu_wait"] = wall - host - io
    other = wall - sum(ph.values())          # 1000 ns, counted with the host
    return {"phases_ns": ph, "wall_ns": wall, "other_ns": other, "tokens_profiled": tokens, "tok_s": 20.0}


def result(*shares):
    return {"profile": True, "prompts": {f"p{i}": entry(s) for i, s in enumerate(shares)}}


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A scratch repo root with a platform.toml, the g2 record and a capture dir; decide.py reads it through ROOT."""
    (tmp_path / "slotstream_spark/config").mkdir(parents=True)
    shutil.copy(ROOT / "slotstream_spark/config/platform.toml", tmp_path / "slotstream_spark/config/platform.toml")
    (tmp_path / "db/records/measurements").mkdir(parents=True)
    (tmp_path / "db/records/decisions").mkdir(parents=True)
    shutil.copy(ROOT / decide.F8_G2, tmp_path / decide.F8_G2)
    monkeypatch.setattr(decide, "ROOT", tmp_path)
    monkeypatch.setattr(f8_gate, "ROOT", tmp_path)
    return tmp_path


def capture(tree, res, name="20261014T170000-f8-partition"):
    d = tree / "db/sources/runs/2026/10" / name
    d.mkdir(parents=True)
    (d / "result.json").write_text(json.dumps(res))
    return d


def run(tree, res, write=True):
    d = capture(tree, res)
    return decide.main(["f8-entry", str(d)] + (["--write"] if write else []))


def test_fraction_counts_host_phases_and_other_not_gpu_wait_or_io():
    frac, io, per = decide.f8_fraction(result(0.06, 0.06))
    assert frac == pytest.approx(0.06, abs=1e-6)
    assert io == pytest.approx(0.10, abs=1e-6)
    assert set(per) == {"p0", "p1"}


def test_fraction_is_pooled_over_prompts_by_wall_time():
    a, b = entry(0.30, wall=1_000_000_000), entry(0.00, wall=3_000_000_000)
    frac, _, _ = decide.f8_fraction({"profile": True, "prompts": {"a": a, "b": b}})
    assert frac == pytest.approx(0.30 * 1 / 4, abs=1e-6)


def test_skip_below_ten_percent_with_g2_pass(tree, capsys):
    assert run(tree, result(0.056, 0.084)) == 0
    text = (tree / decide.F8_RECORD).read_text()
    assert "Decision: skip" in text and "Verdict: PASS" in text
    assert "g2 verdict" in text and ": PASS" in text
    assert "token_graph = false" in (tree / "slotstream_spark/config/platform.toml").read_text()
    assert "F8 skip" in capsys.readouterr().out


def test_enter_at_ten_percent_with_g2_pass_keeps_the_flag_false(tree):
    assert run(tree, result(0.12, 0.12)) == 0
    assert "Decision: enter" in (tree / decide.F8_RECORD).read_text()
    assert "token_graph = false" in (tree / "slotstream_spark/config/platform.toml").read_text()


def test_g2_failed_forces_skip_even_above_ten_percent(tree):
    g2 = tree / decide.F8_G2
    g2.write_text(g2.read_text().replace("Verdict: PASS", "Verdict: FAIL"))
    assert run(tree, result(0.30)) == 0
    assert "Decision: skip" in (tree / decide.F8_RECORD).read_text()


def test_a_partition_that_does_not_add_up_is_exit_3(tree):
    res = result(0.05)
    res["prompts"]["p0"]["other_ns"] += 5
    assert run(tree, res) == 3
    assert not (tree / decide.F8_RECORD).exists()


def test_fewer_than_16_profiled_tokens_is_exit_3(tree):
    res = result(0.05)
    res["prompts"]["p0"]["tokens_profiled"] = 8
    assert run(tree, res) == 3


def test_not_a_profile_result_is_exit_3(tree):
    assert run(tree, {"profile": False, "prompts": {}}) == 3


def test_the_real_g2_record_passes_and_the_real_flag_is_false():
    assert decide.f8_g2() == "PASS"
    assert f8_gate.flag(ROOT) is False


# ---- f8_gate.py -------------------------------------------------------------------------------------------

def test_gate_on_skip_verifies_and_prints(tree, capsys):
    run(tree, result(0.056))
    capsys.readouterr()
    assert f8_gate.step("token-graph", tree) == 0
    assert "F8 skipped (host+waits = 5.6% < 10%)" in capsys.readouterr().out
    assert f8_gate.step("verdict", tree) == 0


def test_gate_fails_when_a_skip_has_the_flag_on(tree):
    run(tree, result(0.056))
    p = tree / "slotstream_spark/config/platform.toml"
    p.write_text(p.read_text().replace("token_graph = false", "token_graph = true"))
    assert f8_gate.step("token-graph", tree) == 1


def test_gate_fails_when_the_record_contradicts_its_numbers(tree):
    run(tree, result(0.056))
    p = tree / decide.F8_RECORD
    p.write_text(p.read_text().replace("Decision: skip", "Decision: enter"))
    assert f8_gate.step("token-graph", tree) == 1


def test_gate_without_a_record_is_exit_3(tree):
    assert f8_gate.step("token-graph", tree) == 3


def test_gate_enter_runs_the_commands_before_1315_and_refuses_after(tree):
    run(tree, result(0.12))
    seen = []
    ok = lambda cmds, root: seen.append(cmds) or 0                       # noqa: E731
    assert f8_gate.step("token-graph", tree, now=datetime(2026, 10, 14, 13, 15), runner=ok) == 0
    flat = [" ".join(c) for c in seen[0]]
    assert any("pytest" in c and "test_token_graph.py" in c and "run.sh --heavy token-graph" in c for c in flat)   # on the Spark
    assert any("jobs.sh start f8-ab-soak" in c and "36000" in c for c in flat)
    assert f8_gate.step("token-graph", tree, now=datetime(2026, 10, 14, 13, 16), runner=ok, pause_probe=lambda: False) == 75
    assert f8_gate.step("token-graph", tree, now=datetime(2026, 10, 14, 13, 16), runner=ok, pause_probe=lambda: True) == 0
    assert f8_gate.step("verdict", tree, runner=ok) == 0
    assert "--timeout 500" in " ".join(seen[-1][0])
    assert "decide.py f8-token-graph" in " ".join(seen[-1][1]) and "--write" in " ".join(seen[-1][1])


# ---- step 92: decide.py f8-token-graph --------------------------------------------------------------------
def ab_result(ratios, ident_b="x"):
    rows = [{"pair": i, "order": "AB", "a": 20.0, "b": 20.0 * r, "eligible": True, "reasons": [], "ratio": r,
             "snapshot": {"nvme": 1}, "a_ident": "x", "b_ident": ident_b} for i, r in enumerate(ratios)]
    return {"label": "f8-ab", "arms": {"A": "f3-segmented", "B": "token-graph"}, "pairs_planned": len(rows),
            "pairs_run": len(rows), "pairs": rows}


def soak(**kw):
    s = {"ab_rc": 0, "soak_seconds": 28800, "soak_seconds_wanted": 28800, "soak_arms": 100, "soak_arms_failed": 0,
         "spot_checks": 16, "spot_checks_differing": 0, "cuda_context_losses": 0, "token_graph_arms": 116,
         "token_graph_tokens": 100000, "token_graph_segmented_tokens": 5, "token_graph_fallbacks": 5, "early_stop": None}
    s.update(kw)
    return s


NOISE = 0.004216
GOOD = [1.08] * 16


def test_token_graph_adopted_when_every_bar_is_met():
    j = decide.f8_tg_judge(ab_result(GOOD), soak(), NOISE)
    assert j["code"] == 0 and j["adopted"] and j["reasons"] == []


@pytest.mark.parametrize("ratios,kw,fragment", [
    ([1.03] * 16, {}, "A/B"),                                           # median under 1.05
    (GOOD, {"soak_seconds": 28000}, "soak 28000 s < 28800"),            # a soak shorter than 8 h
    (GOOD, {"spot_checks_differing": 1}, "spot bit-checks differ"),
    (GOOD, {"cuda_context_losses": 1}, "CUDA context losses"),
    (GOOD, {"soak_arms_failed": 2}, "soak arms failed"),
    (GOOD, {"token_graph_tokens": 0}, "never ran on the token graph"),
    (GOOD, {"early_stop": "kernel log"}, "stopped early"),
])
def test_token_graph_kept_off_when_a_bar_fails(ratios, kw, fragment):
    j = decide.f8_tg_judge(ab_result(ratios), soak(**kw), NOISE)
    assert j["code"] == 0 and not j["adopted"]
    assert fragment in "; ".join(j["reasons"])


def test_token_graph_needs_80_percent_positive_pairs():
    j = decide.f8_tg_judge(ab_result([1.2] * 12 + [0.9] * 4), soak(), NOISE)
    assert not j["adopted"] and any("A/B" in r for r in j["reasons"])


def test_token_graph_outputs_that_differ_are_exit_2_and_noise_is_a_decision_line():
    assert decide.f8_tg_judge(ab_result(GOOD, ident_b="y"), soak(), NOISE)["code"] == 2
    j = decide.f8_tg_judge(ab_result(GOOD), soak(), 0.06)
    assert j["code"] == 1 and j["lines"][0].startswith("DECISION:")


def test_token_graph_record_and_flag(tree):
    cap = capture(tree, ab_result(GOOD), name="20261015T100000-f8-ab")
    (cap / "result.json").write_text(json.dumps(ab_result(GOOD)))
    sd = tree / "db/sources/runs/2026/10/20261015T100001-f8-ab-soak"
    sd.mkdir(parents=True)
    (sd / "f8_soak.json").write_text(json.dumps(soak()))
    shutil.copy(ROOT / decide.H3, tree / decide.H3)
    assert decide.main(["f8-token-graph", str(cap), "--soak", str(sd), "--write"]) == 0
    text = (tree / decide.F8_TG_RECORD).read_text()
    assert "Decision: ADOPT" in text and text.rstrip().splitlines()[-1].startswith("Verdict: PASS")
    assert f8_gate.flag(tree) is True
    (sd / "f8_soak.json").write_text(json.dumps(soak(soak_seconds=100)))
    assert decide.main(["f8-token-graph", str(cap), "--soak", str(sd), "--write"]) == 0
    assert "Decision: KEEP_OFF" in (tree / decide.F8_TG_RECORD).read_text()
    assert f8_gate.flag(tree) is False
