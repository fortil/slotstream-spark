"""tools/quality/oracle_ppl.py: window math, resume state and result file (plan step 95b). No GPU, no network."""
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


op = load("oracle_ppl_mod", "tools/quality/oracle_ppl.py")
score = load("score_mod2", "slotstream_spark/engine/score.py")
SHA, DT = "a" * 64, "torch.bfloat16"


def line(k, sha=SHA, dtype=DT, nll=3000.0):
    return {"window": k, "start": k * 2048, "nll": nll, "tokens": 2047, "seconds": 1.0, "ids_sha256": sha, "dtype": dtype}


def test_window_starts_equal_the_engine_slices():
    n = 300_000
    assert op.window_starts(n, 32) == [a for a, _ in score.window_slices(n, 2048, 32)]
    assert op.window_starts(n, 32)[:3] == [0, 2048, 4096] and len(op.window_starts(n, 32)) == 32


def test_the_contract_asks_for_32_windows():
    assert op.contract_oracle() == (32, 2048)


def test_plan_windows_resumes_a_partial_state():
    done, missing = op.plan_windows([line(0), line(1), line(5)], 32, SHA, DT)
    assert sorted(done) == [0, 1, 5] and missing == [k for k in range(32) if k not in (0, 1, 5)]
    assert op.plan_windows([], 32, SHA, DT)[1] == list(range(32))


def test_a_foreign_state_exits_with_code_2():
    with pytest.raises(SystemExit) as e:
        op.plan_windows([line(0, sha="b" * 64)], 32, SHA, DT)
    assert e.value.code == 2


def test_plan_windows_refuses_a_foreign_sha_or_dtype():
    with pytest.raises(SystemExit):
        op.plan_windows([line(0, sha="b" * 64)], 32, SHA, DT)
    with pytest.raises(SystemExit):
        op.plan_windows([line(0, dtype="torch.float32")], 32, SHA, DT)


def test_a_duplicate_line_counts_once():
    done, missing = op.plan_windows([line(2, nll=1.0), line(2, nll=2.0)], 4, SHA, DT)
    assert len(done) == 1 and done[2]["nll"] == 1.0 and missing == [0, 1, 3]


def test_the_final_json_of_32_windows_is_full():
    rows = [line(k) for k in range(32)]
    r = op.build_result(rows, 32, 32, SHA, "r" * 64, DT, "state.jsonl", "cap/quality/ppl.json", {"torch": "x"})
    assert r["tokens_scored"] == 32 * 2047 == 65_504 and r["full"] is True and r["windows"] == 32
    assert abs(r["ppl"] - math.exp(3000.0 / 2047)) < 1e-9 and "ids_sha256" not in r["per_window"][0]
    assert r["ids_sha256"] == SHA and r["dtype"] == DT and r["oracle"].startswith("tools/ref_layerwise.py")


def test_31_windows_are_not_full():
    r = op.build_result([line(k) for k in range(31)], 31, 32, SHA, "r" * 64, DT, "s", None, {})
    assert r["full"] is False


def test_nll_from_logits_matches_a_hand_computed_cross_entropy():
    torch = pytest.importorskip("torch")
    ids = [2, 0, 1, 2, 1, 0]
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(6, 3, generator=g)
    got, scored = op.nll_from_logits(logits, ids, tile=2)
    want = sum(-math.log(math.exp(float(logits[i, ids[i + 1]])) / sum(math.exp(float(v)) for v in logits[i])) for i in range(5))
    assert scored == 5 and abs(got - want) < 1e-9


def test_the_job_script_pins_the_engine_capture_and_the_snapshot():
    src = (ROOT / "bench/campaign/f9_oracle_job.sh").read_text()
    for needle in ("--engine-ppl", "LAST-f9-engine", "snap.sh", "capture.sh f9-oracle", "oracle_ppl.py"):
        assert needle in src, needle


def test_the_tool_makes_no_full_fp32_copy_and_uses_the_checkpoint_dtype():
    src = (ROOT / "tools/quality/oracle_ppl.py").read_text()
    assert "ORACLE_DTYPE" not in src.split('"""')[2] and ".float()" not in src and "float32" not in src.split('"""')[2]
