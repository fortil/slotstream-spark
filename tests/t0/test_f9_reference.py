"""tools/quality/llamacpp_ppl.py helpers and tools/decide.py f9-diff on synthetic results (plan steps 96-97). No GPU, no network."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tools" / "quality")]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lp = load("llamacpp_ppl_mod", "tools/quality/llamacpp_ppl.py")
dm = load("decide_mod_f9ref", "tools/decide.py")
ct = dm._contract()
C = ct.load()
EXC = json.loads((ROOT / dm.Q41_EXCEPTIONS).read_text())
TOK = 2047


def win(k, nll=3000.0):
    return {"window": k, "start": k * 2048, "nll": nll, "tokens": TOK, "seconds": 90.0}


def engine_doc(n=145, nll=3000.0):
    return {"per_window": [win(k, nll) for k in range(n)], "ids_sha256": "i" * 64, "rows_sha256": "r" * 64}


def ref_doc(factor=1.0, **over):
    per = [win(k, 3000.0 - TOK * math.log(factor)) for k in range(32)]
    doc = {"smoke_run": False, "ppl_rc": 0,
           "sweep": {"tensors": 3215, "bad": [], "tolerated": [{"name": n, "count": 5, "max_abs_err": 3e-8} for n in EXC["tensors"]]},
           "ppl": {"per_window": per, "ids_sha256": "i" * 64, "rows_sha256": "r" * 64, "prefill_device": "cpu", "op_offload": False,
                   "memfree_stop": False, "memorymax_mib": 47616, "memorymax_table_mib": 47616, "memfree_at_start_mib": 98000,
                   "memfree_min_mib": 50000, "memcg_peak_mib": 47000, "llama_tok_s": 22.6},
           "manifest": {"tensors": 3000, "file_bytes": 1, "exceptions": [], "type_overrides": ["blk.1.ple_conv1d.weight"],
                        "dequantized_bf16": [f"blk.{i}.hc_attn_up.weight" for i in range(194)]},
           "below_099": [{"name": "blk.0.hc_attn_up.weight", "corr": 0.97, "family": "hyper_connection"},
                         {"name": "blk.1.attn_gate.weight", "corr": 0.9897, "family": "other"}],
           "corr_floor": 0.95, "failures": [], "check_tensors": 1224}
    doc.update(over)
    return doc


def judge(ref, engine=None):
    return dm.f9_diff(ref, engine or engine_doc(), {"ppl": 5.7}, C, EXC)


def test_a_matching_result_passes_and_prints_the_three_way_line():
    rows, info, decisions, code, r = judge(ref_doc(1.0))
    assert code == 0 and not decisions and all(x[3] for x in rows)
    assert any(i.startswith("PPL(32): engine ") and "transformers oracle 5.7000 (f9-quality)" in i and "llama.cpp exact-Q4_1" in i for i in info)
    assert any(i.startswith("hyper-connection matrices written as dequantized BF16") and i.endswith(": 194 tensors") for i in info)
    assert any("blk.1.ple_conv1d.weight" in i for i in info) and any(i.startswith("[0.95, 0.99) blk.1.attn_gate.weight 0.9897") for i in info) \
        and any(i.startswith("[0.95, 0.99) blk.0.hc_attn_up.weight 0.9700") for i in info) \
        and any(x[0] == "mapping check" and x[3] for x in rows)


def test_two_percent_is_the_edge():
    for f, code in ((1.015, 0), (0.985, 0), (1.03, 1), (0.97, 1)):
        rows, info, decisions, c_, r = judge(ref_doc(f))
        assert c_ == code, (f, decisions)
        if code:
            assert any("disagree by" in d and "numerics investigation needed" in d for d in decisions)


def test_ids_or_window_mismatch_is_unreadable():
    bad = ref_doc()
    bad["ppl"]["ids_sha256"] = "x" * 64
    with pytest.raises(dm.BadInput):
        judge(bad)
    bad = ref_doc()
    bad["ppl"]["per_window"][3]["start"] += 1
    with pytest.raises(dm.BadInput):
        judge(bad)
    bad = ref_doc()
    bad["ppl"]["per_window"] = bad["ppl"]["per_window"][:31]
    with pytest.raises(dm.BadInput):
        judge(bad)
    with pytest.raises(dm.BadInput):
        judge(ref_doc(smoke_run=True))


def test_soft_stop_sweep_and_device_failures():
    r = ref_doc()
    r["ppl"]["memfree_stop"] = True
    assert judge(r)[3] == 1
    r = ref_doc()
    r["sweep"]["bad"] = ["x: not exact"]
    assert judge(r)[3] == 1
    r = ref_doc()
    r["sweep"]["tolerated"] = r["sweep"]["tolerated"][:-1]
    assert judge(r)[3] == 1
    r = ref_doc()
    r["sweep"]["tolerated"][0]["max_abs_err"] = 1e-6
    assert judge(r)[3] == 1
    r = ref_doc()
    r["ppl"]["prefill_device"] = "gpu"
    assert judge(r)[3] == 1


def test_a_floor_other_than_095_or_a_failure_fails_the_mapping_row():
    for over in ({"corr_floor": 0.99}, {"failures": [{"name": "blk.2.x.weight", "corr": 0.9}]}, {"check_tensors": 1223}):
        rows, info, decisions, code, r = judge(ref_doc(**over))
        assert code == 1 and any("mapping check" in d for d in decisions), over
        assert any(x[0] == "mapping check" and not x[3] for x in rows)


def test_llamacpp_ppl_helpers(tmp_path):
    rows = [win(k) for k in range(4)]
    assert [r["window"] for r in lp.check_windows(rows[::-1], 4)] == [0, 1, 2, 3]
    with pytest.raises(ValueError):
        lp.check_windows(rows[:3], 4)
    with pytest.raises(ValueError):
        lp.check_windows(rows + [win(1)], 4)
    s = lp.summarize(rows)
    assert s["windows"] == 4 and s["tokens_scored"] == 4 * TOK and abs(s["ppl"] - math.exp(3000.0 / TOK)) < 1e-9
    d = lp.parse_footprint("FOOTPRINT_MIB=57344 MEMORYMAX_MIB=47616 MEMFREE_MIB=98000 TABLE_MIB=47616")
    assert d["MEMORYMAX_MIB"] == 47616 and d["TABLE_MIB"] == 47616
    with pytest.raises(ValueError):
        lp.parse_footprint("FOOTPRINT_MIB=57344")
    p = tmp_path / "m.csv"
    p.write_text("epoch,memfree_mib,cached_mib\n1,500,1\n2,300,1\n3,900,1\n")
    assert lp.memfree_min(p) == 300 and lp.memfree_min(tmp_path / "none.csv") is None


def test_fixed_flags_keep_the_lb_yardstick_set():
    for f in ("-fit", "-np", "--no-repack", "-lm", "mmap", "-cmoe", "-b", "2048", "--no-op-offload"):
        assert f in lp.FIXED_FLAGS


def test_f9_diff_cli_without_a_capture_is_exit_3(tmp_path, capsys):
    assert dm.main(["f9-diff", str(tmp_path / "none"), "--engine", str(tmp_path)]) == 3
