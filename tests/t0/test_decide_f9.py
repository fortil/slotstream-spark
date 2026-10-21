"""tools/decide.py f9-quality and f9-campaign on synthetic captures (plan step 95)."""
import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("decide_mod", ROOT / "tools/decide.py")
dm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dm)
ct = dm._contract()
C = ct.load()


# ------------------------------------------------------------------------------------------------ quality
def ppl_doc(value=5.39, windows=145, tokens=2047):
    per = [{"window": i, "start": i * 2048, "nll": tokens * math.log(value), "tokens": tokens, "seconds": 5.0} for i in range(windows)]
    p = math.exp(sum(r["nll"] for r in per) / sum(r["tokens"] for r in per))
    return {"ppl": p, "per_window": per, "rows_sha256": "r" * 64, "ids_sha256": "i" * 64}


def prompt_doc(top1=23 / 24, kl_mean=0.01, kl_p99=0.05, kl_max=0.1, pair=None, ngram=True, routing=0.01):
    pair = pair or {"top1": 0.95, "kl_mean": 0.01, "kl_p99": 0.05, "kl_max": 0.08}
    layers = [{"layer": k, "rel_l2": 0.001, "rel_l2_pair": 0.001, "limit": 1.0, "limit_src": "calibrated", "limit_cal": 0.02,
               "separable": True, "ok": True, "routing": routing, "routing_pair": 0.01, "routing_limit": 1.0,
               "routing_limit_src": "calibrated", "routing_limit_cal": 0.2, "routing_separable": True, "routing_ok": True}
              for k in range(48)]
    return {"tokens": 24, "ngram_ok": ngram, "layers": layers, "ok": True, "failing_layers": [],
            "logits": {"top1": top1, "kl_mean": kl_mean, "kl_p99": kl_p99, "kl_max": kl_max, "pair": pair, "ok": True,
                       "judgeable": True, "fixed_limits_ok": True, "limits_seq": {}, "limit_src": {}}}


def oracle_doc(ppl, factor=1.0, windows=32, **over):
    per = [{"window": r["window"], "start": r["start"], "nll": r["nll"] - r["tokens"] * math.log(factor), "tokens": r["tokens"], "seconds": 1.0}
           for r in ppl["per_window"][:windows]]
    doc = {"per_window": per, "windows": windows, "ids_sha256": ppl["ids_sha256"], "rows_sha256": ppl["rows_sha256"], "dtype": "torch.bfloat16"}
    doc.update(over)
    return doc


def make_quality(tmp_path, ppl=None, prompts=None, probe="quality probe: passed 15, failed 0\n", oracle=None):
    cap = tmp_path / "cap"
    q = cap / "quality"
    q.mkdir(parents=True)
    (q / "ppl.json").write_text(json.dumps(ppl or ppl_doc()))
    (q / "teacher_forced.json").write_text(json.dumps({"prompts": prompts or {f"res{k}": prompt_doc() for k in range(8)}, "ok": True}))
    (q / "probe.txt").write_text(probe)
    o = tmp_path / "ocap" / "quality"
    o.mkdir(parents=True)
    (o / "oracle_ppl.json").write_text(json.dumps(oracle if oracle is not None else oracle_doc(ppl or ppl_doc())))
    return cap


def run_quality(cap):
    return dm.f9_quality(cap / "quality", cap.parent / "ocap" / "quality")[:5]


def test_quality_pass(tmp_path):
    rows, info, decisions, code, c = run_quality(make_quality(tmp_path))
    assert code == 0 and not decisions and all(r[3] is not False for r in rows)
    assert any("PPL" in r[0] for r in rows) and any(r[0] == "quality probe" for r in rows)


def test_ppl_is_judged_against_the_oracle_not_the_reference(tmp_path):
    # factor = engine PPL / oracle PPL on every window
    for k, (f, code) in enumerate(((1.015, 0), (0.985, 0), (1.025, 1))):
        d = tmp_path / str(k)
        d.mkdir()
        pd = ppl_doc(5.9)                                  # far from 5.3914 and still a pass when the oracle agrees
        rows, info, decisions, c_, c = run_quality(make_quality(d, ppl=pd, oracle=oracle_doc(pd, f)))
        assert c_ == code, (f, decisions)
        if code:
            assert any("wikitext-2 PPL" in x and "needs a decision" in x for x in decisions)
    assert any("README reference 5.3914" in i and "informative" in i for i in info)
    assert not any(r[0].startswith("wikitext-2 PPL over 145") for r in rows)


def test_oracle_inputs_that_do_not_line_up_are_unreadable(tmp_path):
    pd = ppl_doc()
    cases = {"31": oracle_doc(pd, windows=31), "sha": oracle_doc(pd, ids_sha256="x" * 64), "rows": oracle_doc(pd, rows_sha256="x" * 64),
             "nodtype": oracle_doc(pd, dtype="")}
    shifted = oracle_doc(pd)
    shifted["per_window"][3]["start"] += 1
    cases["start"] = shifted
    for name, od in cases.items():
        d = tmp_path / name
        d.mkdir()
        with pytest.raises(dm.BadInput):
            run_quality(make_quality(d, ppl=pd, oracle=od))


def test_no_oracle_is_exit_3(tmp_path, capsys):
    cap = make_quality(tmp_path)
    assert dm.main(["f9-quality", str(cap)]) == 3
    assert "oracle PPL capture required" in capsys.readouterr().err
    assert dm.main(["f9-quality", str(cap), "--oracle", str(tmp_path / "ocap")]) == 0


def test_ppl_ratio_checks_alignment():
    a = [{"window": 0, "start": 0, "nll": 100.0, "tokens": 10}]
    assert abs(dm.ppl_ratio(a, a)["ratio_minus_1"]) < 1e-12
    with pytest.raises(dm.BadInput):
        dm.ppl_ratio(a, [{"window": 0, "start": 1, "nll": 100.0, "tokens": 10}])
    with pytest.raises(dm.BadInput):
        dm.ppl_ratio(a, a + a)


def test_top1_is_pooled_and_a_weak_prompt_is_informative(tmp_path):
    p = {f"res{k}": prompt_doc(top1=23 / 24) for k in range(8)}
    p["res0"] = prompt_doc(top1=19 / 24)
    rows, info, decisions, code, c = run_quality(make_quality(tmp_path, prompts=p))
    assert code == 0 and not decisions
    row0 = [r for r in rows if r[0] == "res0 top-1"][0]
    assert row0[3] is None and "19/24" in row0[1]
    pooled = [r for r in rows if r[0].startswith("top-1 pooled")][0]
    assert pooled[3] is True and "180/192" in pooled[1]
    d = tmp_path / "low"
    d.mkdir()
    p = {f"res{k}": prompt_doc(top1=20 / 24) for k in range(8)}                 # 0.8333 each
    rows, info, decisions, code, c = run_quality(make_quality(d, prompts=p))
    assert code == 1 and [x for x in decisions if "top-1 pooled" in x] and len(decisions) == 1


def test_non_integral_or_inconsistent_top1_counts_are_unreadable(tmp_path):
    p = {f"res{k}": prompt_doc() for k in range(8)}
    p["res1"] = prompt_doc(top1=0.97)                                              # 0.97 x 24 = 23.28
    with pytest.raises(dm.BadInput, match="whole number"):
        run_quality(make_quality(tmp_path, prompts=p))
    d = tmp_path / "inc"
    d.mkdir()
    p = {f"res{k}": prompt_doc() for k in range(8)}
    p["res2"]["logits"].update({"top1_agree": 20, "positions": 24})                # top1 says 23
    with pytest.raises(dm.BadInput, match="inconsistent"):
        run_quality(make_quality(d, prompts=p))


def test_wrong_window_or_token_counts_are_unreadable_input(tmp_path):
    with pytest.raises(dm.BadInput, match="windows"):
        run_quality(make_quality(tmp_path, ppl=ppl_doc(windows=144)))
    d = tmp_path / "t"
    d.mkdir()
    with pytest.raises(dm.BadInput, match="tokens"):
        run_quality(make_quality(d, ppl=ppl_doc(tokens=2048)))


def test_the_stored_ok_is_never_trusted(tmp_path):
    bad = prompt_doc(kl_mean=0.09)                                                 # beyond the 0.05 ceiling
    bad["logits"]["ok"] = True
    p = {f"res{k}": prompt_doc() for k in range(8)}
    p["res3"] = bad
    rows, info, decisions, code, c = run_quality(make_quality(tmp_path, prompts=p))
    assert code == 1 and any("res3 kl_mean" in d for d in decisions)


def test_kl_beyond_the_per_sequence_limit_fails_and_inside_a_widened_limit_passes(tmp_path):
    p = {f"res{k}": prompt_doc() for k in range(8)}
    p["res1"] = prompt_doc(kl_mean=0.06)                      # above the 0.05 ceiling
    rows, info, decisions, code, c = run_quality(make_quality(tmp_path, prompts=p))
    assert code == 1 and any("res1 kl_mean" in d for d in decisions)
    d = tmp_path / "wide"
    d.mkdir()
    pair = {"top1": 0.95, "kl_mean": 0.02, "kl_p99": 0.05, "kl_max": 0.08}                   # 3 x pair = 0.06 -> clipped to the ceiling
    p = {f"res{k}": prompt_doc() for k in range(8)}
    p["res2"] = prompt_doc(kl_mean=0.045, pair=pair)
    rows, info, decisions, code, c = run_quality(make_quality(d, prompts=p))
    assert code == 0


def test_a_pair_above_a_ceiling_is_a_design_gate(tmp_path):
    p = {f"res{k}": prompt_doc() for k in range(8)}
    p["res5"] = prompt_doc(pair={"top1": 0.9, "kl_mean": 0.01, "kl_p99": 0.05, "kl_max": 1.6})
    rows, info, decisions, code, c = run_quality(make_quality(tmp_path, prompts=p))
    assert code == 1 and any(d.startswith("DESIGN GATE: res5") for d in decisions)


def test_missing_pair_data_is_exit_3(tmp_path):
    p = {f"res{k}": prompt_doc() for k in range(8)}
    del p["res0"]["logits"]["pair"]
    with pytest.raises(dm.BadInput, match="logits.pair"):
        run_quality(make_quality(tmp_path, prompts=p))
    d = tmp_path / "r"
    d.mkdir()
    p = {f"res{k}": prompt_doc() for k in range(8)}
    del p["res4"]["layers"][7]["routing_pair"]
    with pytest.raises(dm.BadInput, match="routing_pair"):
        run_quality(make_quality(d, prompts=p))


def test_ngram_routing_probe_and_prompt_count(tmp_path):
    p = {f"res{k}": prompt_doc() for k in range(8)}
    p["res6"] = prompt_doc(ngram=False)
    assert run_quality(make_quality(tmp_path, prompts=p))[3] == 1
    d = tmp_path / "rt"
    d.mkdir()
    p = {f"res{k}": prompt_doc() for k in range(8)}
    p["res2"] = prompt_doc(routing=5.0)                     # far above any seq_rule limit
    rows, info, decisions, code, c = run_quality(make_quality(d, prompts=p))
    assert code == 1 and any("res2 routing" in x for x in decisions)
    d2 = tmp_path / "pr"
    d2.mkdir()
    assert run_quality(make_quality(d2, probe="quality probe: passed 14, failed 1\n"))[3] == 1
    d3 = tmp_path / "np"
    d3.mkdir()
    with pytest.raises(dm.BadInput, match="8 reserved"):
        run_quality(make_quality(d3, prompts={f"res{k}": prompt_doc() for k in range(7)}))


def test_record_carries_the_four_rule_lines_two_captures_and_one_verdict(tmp_path):
    cap = make_quality(tmp_path)
    rows, info, decisions, code, c, ppl = dm.f9_quality(cap / "quality", tmp_path / "ocap" / "quality")
    dest = tmp_path / "f9-quality.md"
    dm.write_f9_quality(rows, info, decisions, code, c, "cap/", "ocap/", str(dest), ppl)
    text = dest.read_text()
    assert sum(1 for x in text.splitlines() if x.startswith("Capture: ")) == 2
    assert any(x.startswith("Top-1 rule: pooled over the 8 reserved prompts") for x in text.splitlines())
    assert any(x.startswith("PPL rule: engine vs transformers oracle on the same 32 wikitext windows") for x in text.splitlines())
    assert "bench/campaign/f9_oracle_job.sh" in text and "| res0 top-1 |" in text and "| info |" in text
    assert "| NO |" not in text
    assert ("Logits rule: seq KL (factor 3.0, floor = calibrated limit, ceilings 0.05/0.5/1.5 nats; top-1 calibrated), "
            "design decision 18") in text.splitlines()
    assert "Routing rule: seq_rule (factor 2.0, ceiling c_k/2), design decision 16" in text.splitlines()
    assert text.count("\nVerdict:") == 1 and "Verdict: PASS" in text and "worst routing ratio" in text
    import db_check
    rec = tmp_path / "r.md"
    rec.write_text(text.replace("Capture: cap/", "Capture: cap").replace("Capture: ocap/", "Capture: ocap"))
    for n in ("cap", "ocap"):
        (tmp_path / n).mkdir(exist_ok=True)
        (tmp_path / n / "verdict.json").write_text("{}")
    assert db_check.check_measurement(rec, root=tmp_path) == []


def test_a_stored_pooled_count_is_used_when_present(tmp_path):
    p = {f"res{k}": prompt_doc() for k in range(8)}
    for r in p.values():
        r["logits"].update({"top1_agree": 23, "positions": 24})
    rows, info, decisions, code, c = run_quality(make_quality(tmp_path, prompts=p))
    assert code == 0 and [r for r in rows if r[0].startswith("top-1 pooled")][0][1].endswith("(184/192)")


def test_the_145_window_figure_is_info_only(tmp_path):
    pd = ppl_doc(5.39)
    pd["per_window"][100]["nll"] *= 1.5                         # outside the oracle windows: informative only
    pd["ppl"] = math.exp(sum(r["nll"] for r in pd["per_window"]) / sum(r["tokens"] for r in pd["per_window"]))
    o = oracle_doc(ppl_doc(5.39))
    rows, info, decisions, code, c = run_quality(make_quality(tmp_path, ppl=pd, oracle=o))
    assert code == 0 and any("over 145 windows / 296,815 tokens" in i for i in info)


def test_the_per_window_table_and_worst_window_are_reported(tmp_path):
    pd = ppl_doc()
    o = oracle_doc(pd)
    o["per_window"][7]["nll"] *= 0.99
    cap = make_quality(tmp_path, ppl=pd, oracle=o)
    rows, info, decisions, code, c, ppl = dm.f9_quality(cap / "quality", tmp_path / "ocap" / "quality")
    assert ppl["worst_window"] == 7 and len(ppl["per_window"]) == 32 and ppl["engine_above"] == 1
    assert any("max |ratio-1|" in i and "at window 7" in i for i in info)


def test_oracle_and_oracle_latest_are_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        dm.main(["f9-quality", "x", "--oracle", "a", "--oracle-latest", "b"])


def test_rule_lines_are_four_and_literal():
    rl = dm.rule_lines(C)
    assert len(rl) == 4 and rl[2].startswith("Top-1 rule: pooled over the 8 reserved prompts")
    assert "|PPL_engine/PPL_oracle - 1| <= 2%" in rl[3] and "5.3914 +-1% informative" in rl[3]


def test_a_prompt_ok_ignores_top1_and_follows_kl_ngram_layers():
    p = prompt_doc(top1=10 / 24)
    assert dm.f9_prompt("res0", p, C)["ok"] is True
    assert dm.f9_prompt("res0", prompt_doc(kl_mean=0.09), C)["ok"] is False
    assert dm.f9_prompt("res0", prompt_doc(ngram=False), C)["ok"] is False


def test_decide_f9_quality_has_no_literal_limit():
    src = (ROOT / "tools/decide.py").read_text()
    a, b = src.index("def f9_ppl"), src.index("def load_cells")
    block = src[a:b]
    for lit in ("0.98", "0.01", "0.88976", "0.03723", "0.02"):
        assert lit not in block
    for stored in ('lg["ok"]', 'r["ok"]', 'r.get("ok")', 'lg.get("ok")', 'ly["ok"]', 'ly["routing_ok"]'):
        assert stored not in block, stored


# ------------------------------------------------------------------------------------------------ campaign
def sample_row(flags_pool=True):
    return {"tag": "0", "vmrss": 7_000_000_000 + 51_539_607_552, "vmhwm": 8_000_000_000 + 51_539_607_552, "nvml": 7_500_000_000,
            "pinned": 0, "unified": {"pool": 51_539_607_552}}


def with_peak(cell):
    from slotstream_spark.memory import footprint as fp
    cell["flags"] = {"pool": [True, True]}
    cell["footprint_samples"] = [sample_row()]
    pk = fp.peak([fp.sample_from_dict(d) for d in cell["footprint_samples"]], fp.flags_from_dict(cell["flags"]))
    cell["peak"] = fp.peak_to_dict(pk)
    cell["chunks"] = 48
    cell["memavailable_drop_bytes"] = 59_000_000_000
    cell["mode"] = "D5"
    return cell


def decode_cell(vals):
    return with_peak({"cell": "decode-8k", "mode": "D5", "kind": "decode", "ok": True,
                      "samples": [{"decode_tps": v, "mtp_policy": "resident", "pool_prefill": {"exhausted": 0}} for v in vals]})


def ttft_cell(name, tokens, ttft, tps, reused=0, follow=1.0, prompt_tokens=None):
    samples = [{"sample": i, "cold": {"ttft_s": ttft, "prefill_s": tokens / tps, "prefill_tps": tps, "reused_tokens": reused,
                                      "prompt_tokens": prompt_tokens or tokens},
                "followup": {"ttft_s": follow, "reused_tokens": tokens - 1, "new_tokens": 60}} for i in range(5)]
    return with_peak({"cell": name, "mode": "D5", "kind": "ttft-cold", "tokens": tokens, "ok": True, "samples": samples,
                      "medians": {"ttft_s": ttft, "prefill_s": tokens / tps, "prefill_tps": tps}})


def cells(decode=(33.0,) * 5, ttft=22.0, tps=370.0, follow=1.0, reused=0):
    return {"decode-8k": decode_cell(decode), "ttft-cold-2k": ttft_cell("ttft-cold-2k", 2048, 6.0, 340.0),
            "ttft-cold-8k": ttft_cell("ttft-cold-8k", 8192, ttft, tps, reused=reused, follow=follow),
            "ttft-cold-32k": ttft_cell("ttft-cold-32k", 32000, 95.0, 340.0)}


YARD = {"decode_tps_median": 32.9, "prefill_tps_8k": 32.1}


def test_campaign_decode_and_ttft_win(tmp_path):
    rows, info, decisions, jline, code = dm.f9_d5(cells(), YARD, "GO_ON_TTFT")
    assert code == 0 and not decisions and all(r[3] for r in rows)
    assert jline.startswith("Justification: decode+ttft (decode 33.00 tok/s = 100.3% of arm B 32.9; cold 8K prefill 370.0 tok/s = 11.5x arm B 32.1)")


def test_ttft_only_passes_with_go_on_ttft_and_is_a_decision_without(tmp_path):
    rows, info, decisions, jline, code = dm.f9_d5(cells(decode=(26.0,) * 5), YARD, "GO_ON_TTFT")
    assert code == 0 and jline.startswith("Justification: ttft")
    rows, info, decisions, jline, code = dm.f9_d5(cells(decode=(26.0,) * 5), YARD, "GO")
    assert code == 1 and any(d.startswith("DECISION: engine justified on TTFT only (decode 26.00 tok/s = 79.0% of arm B 32.9") for d in decisions)


def test_decode_only_passes(tmp_path):
    rows, info, decisions, jline, code = dm.f9_d5(cells(ttft=31.0, tps=263.0), YARD, "GO")
    assert jline.startswith("Justification: decode (") and any("cold 8K TTFT median" in d for d in decisions) and code == 1     # the absolute bar fails


def test_neither_axis_is_a_decision(tmp_path):
    rows, info, decisions, jline, code = dm.f9_d5(cells(decode=(26.0,) * 5, ttft=31.0, tps=263.0), YARD, "GO_ON_TTFT")
    assert code == 1 and jline.startswith("Justification: none")
    assert "F9: engine beats llama.cpp arm B on neither axis — needs a decision" in decisions


def test_absolute_bars_still_fail(tmp_path):
    assert dm.f9_d5(cells(decode=(22.0,) * 5), YARD, "GO_ON_TTFT")[4] == 1
    assert dm.f9_d5(cells(follow=12.5), YARD, "GO_ON_TTFT")[4] == 1


def test_a_cold_sample_with_reused_tokens_invalidates_the_campaign(tmp_path):
    rows, info, decisions, jline, code = dm.f9_d5(cells(reused=16), YARD, "GO_ON_TTFT")
    assert code == 1 and jline is None and "reused tokens" in decisions[0]


def test_bad_inputs_are_exit_3(tmp_path):
    c = cells()
    del c["ttft-cold-32k"]
    with pytest.raises(dm.BadInput, match="ttft-cold-32k"):
        dm.f9_d5(c, YARD, "GO")
    c = cells()
    c["decode-8k"]["ok"] = False
    c["decode-8k"]["failures"] = ["prompt 2"]
    with pytest.raises(dm.BadInput, match="did not finish ok"):
        dm.f9_d5(c, YARD, "GO")
    c = cells()
    c["ttft-cold-8k"]["peak"]["once_bytes"] += 1
    with pytest.raises(dm.BadInput, match="differs from the samples"):
        dm.f9_d5(c, YARD, "GO")
    with pytest.raises(dm.BadInput, match="8000 prompt tokens"):
        dm.f9_d5({**cells(), "ttft-cold-8k": ttft_cell("ttft-cold-8k", 8192, 22.0, 370.0, prompt_tokens=8000)}, YARD, "GO")


def test_the_axes_use_the_yardstick_not_literals():
    src = (ROOT / "tools/decide.py").read_text()
    a, b = src.index("def f9_d5"), src.index("def f9_d1")
    assert "32.9" not in src[a:b].replace("arm B 32.9", "") and "32.1" not in src[a:b]
    assert 'yard["decode_tps_median"]' in src[a:b] and 'yard["prefill_tps_8k"]' in src[a:b]


def test_campaign_main_writes_both_records(tmp_path, monkeypatch):
    cap = tmp_path / "cap"
    for name, c in cells().items():
        d = cap / f"d5-{name}"
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps(c))
    d1 = cap / "d1-decode-8k"
    d1.mkdir()
    (d1 / "result.json").write_text(json.dumps(decode_cell((20.0,) * 3) | {"decode_tps_median": 20.0, "mode": "D1"}))
    monkeypatch.setattr(dm, "ROOT", tmp_path)
    monkeypatch.setattr(dm, "yardstick", type("V", (), {"yardstick": staticmethod(lambda: YARD)}))
    monkeypatch.setattr(dm, "stopgo_decision", lambda path=None: "GO_ON_TTFT")
    assert dm.main(["f9-campaign", str(cap), "--write"]) == 0
    d5 = (tmp_path / dm.F9_D5).read_text()
    assert "Justification: decode+ttft" in d5 and d5.count("\nVerdict:") == 1 and "Verdict: PASS" in d5
    assert "Verdict: INFO" in (tmp_path / dm.F9_D1).read_text()
