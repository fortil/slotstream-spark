"""Per-sequence limits of design decision 16 (tests/gates/contract.py): numpy only, no torch."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tests" / "gates"), str(ROOT / "tools")]

import contract as ct  # noqa: E402

C = ct.load()
T, K = 1000, 10


def _ids():
    return np.tile(np.arange(K, dtype=np.int32), (T, 1))


def _with_disagreement(base, changes):
    """Replace `changes` entries (one per token first, then a second per token, ...) with ids outside the set."""
    out = base.copy()
    for n in range(changes):
        out[n % T, (n // T) % K] = 1000 + n
    return out


def test_seq_limit_reference_points():
    lim, src = ct.seq_limit("routing", 43, 0.2306, C)
    assert src == "ceiling" and abs(lim - C.routing["control"][43] / 2) < 1e-12
    assert ct.seq_limit("routing", 43, 0.05, C) == (0.21, "calibrated")
    lim, src = ct.seq_limit("routing", 43, 0.12, C)
    assert src == "sequence" and abs(lim - 0.25) < 1e-12


@pytest.mark.parametrize("table", ["layer", "routing"])
def test_seq_limit_never_tighter(table):
    t = getattr(C, table)
    for k, sep in enumerate(t["separable"]):
        if sep:
            assert ct.seq_limit(table, k, 0.0, C)[0] == t["limits"][k]
            assert t["limits"][k] <= t["control"][k] / 2


def test_routing_check_seq_dev7_case_ok_and_scrambled_not():
    o = _ids()
    e = _with_disagreement(o, 2376)
    o32 = _with_disagreement(o, 2306)
    r = ct.routing_check_seq(43, e, o, o32, C)
    assert abs(r["value"] - 0.2376) < 1e-9 and abs(r["pair"] - 0.2306) < 1e-9
    assert r["ok"] and r["limit_src"] == "ceiling" and r["separable"] and r["limit_cal"] == 0.21
    bad = _with_disagreement(o, 8000)
    r = ct.routing_check_seq(43, bad, o, o32, C)
    assert r["value"] >= 0.78 and not r["ok"]


def test_layer_check_seq_same_shape():
    rng = np.random.default_rng(0)
    o = rng.standard_normal((50, 64))
    o32 = o + 0.15 * rng.standard_normal(o.shape) * np.linalg.norm(o) / np.linalg.norm(rng.standard_normal(o.shape))
    pair = ct.rel_l2(o, o32)
    near = o32 + 0.0
    r = ct.layer_check_seq(43, near, o, o32, C)
    assert r["ok"] and abs(r["pair"] - pair) < 1e-12
    wrong = rng.standard_normal(o.shape)
    r = ct.layer_check_seq(43, wrong, o, o32, C)
    assert not r["ok"] and r["value"] > 0.9


def test_not_separable_layer_is_informational():
    o = _ids()
    r = ct.routing_check_seq(0, _with_disagreement(o, 9000), o, o, C)
    assert not r["separable"] and r["ok"]


# ---- design decision 18: per-sequence KL limits for the logits (step 70a) ----
PAIR = {"top1": 0.994140625, "kl_mean": 0.004408653354052665, "kl_p99": 0.06475244648703134,
        "kl_max": 1.1491823956944724}
ENGINE = {"top1": 0.99267578125, "kl_mean": 0.004348508799017182, "kl_p99": 0.04749969207287939,
          "kl_max": 1.4930166431466012}


def test_logits_seq_limits_sweep_check_pair():
    lim = ct.logits_seq_limits(PAIR, C)
    assert lim["top1_min"] == C.logits["top1_min"]
    for k in ("kl_mean_max_nats", "kl_p99_max_nats"):
        assert lim[k] == C.logits[k] and lim["src"][k] == "calibrated"
    assert lim["kl_max_nats"] == 1.5 and lim["src"]["kl_max_nats"] == "ceiling"
    assert lim["pair_over_ceiling"] == []


def test_logits_judge_seq_engine_passes_f2_does_not():
    j = ct.logits_judge_seq(ENGINE, PAIR, C)
    assert j["ok"] is True and j["fixed_limits_ok"] is False and j["limit_src"]["kl_max_nats"] == "ceiling"
    assert ct.logits_judge_seq({**ENGINE, "kl_max": 1.537}, PAIR, C)["ok"] is False


def test_logits_seq_pair_over_ceiling_not_judgeable():
    pair = {**PAIR, "kl_max": 1.6}
    j = ct.logits_judge_seq({**ENGINE, "kl_max": 0.1}, pair, C)
    assert j["judgeable"] is False and j["ok"] is False
    assert ct.logits_seq_limits(pair, C)["pair_over_ceiling"] == ["kl_max"]


def test_logits_seq_never_tighter():
    lim = ct.logits_seq_limits({"top1": 1.0, "kl_mean": 0.0, "kl_p99": 0.0, "kl_max": 0.0}, C)
    for _, cal, ceil in ct.LOGITS_SEQ:
        assert lim[cal] == C.logits[cal] and lim["src"][cal] == "calibrated" and C.logits[cal] <= C.logits[ceil]


def test_logits_seq_sequence_binding():
    lim = ct.logits_seq_limits({**PAIR, "kl_max": 0.4}, C)
    assert lim["kl_max_nats"] == pytest.approx(1.2) and lim["src"]["kl_max_nats"] == "sequence"


def test_load_requires_logits_seq_rule(tmp_path):
    lines = [l for l in ct.PATH.read_text().splitlines() if not l.startswith("seq_rule = \"kl_X_seq")]
    p = tmp_path / "c.toml"
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ct.ContractNotCalibrated):
        ct.load(p)


def test_top1_pooled_arithmetic():
    items = [("a", 20, 24), ("b", 24, 25), ("c", 23, 23), ("d", 27, 28), ("e", 37, 38), ("f", 33, 34), ("g", 25, 26), ("h", 21, 22)]
    r = ct.top1_pooled(items, C)
    assert (r["agree"], r["positions"]) == (210, 220) and abs(r["top1"] - 210 / 220) < 1e-15 and r["ok"]
    assert r["per_prompt"][0] == {"id": "a", "agree": 20, "positions": 24, "top1": 20 / 24}
    assert not ct.top1_pooled([("x", 85, 100)], C)["ok"]


def test_top1_pooled_refuses_bad_counts():
    for items in ([("a", 25, 24)], [("a", 1, 0)], [("a", 1.0, 2)], [("a", 1, 2.0)], [("a", True, 2)], [("a", -1, 2)], []):
        with pytest.raises(ValueError):
            ct.top1_pooled(items, C)


def test_logits_kl_ok_true_false_and_not_judgeable():
    m = {"top1": 0.5, "kl_mean": 0.01, "kl_p99": 0.05, "kl_max": 0.1}
    pair = {"top1": 0.95, "kl_mean": 0.01, "kl_p99": 0.05, "kl_max": 0.08}
    assert ct.logits_kl_ok(ct.logits_judge_seq(m, pair, C))                       # top-1 is not part of it
    assert not ct.logits_kl_ok(ct.logits_judge_seq({**m, "kl_mean": 0.09}, pair, C))
    assert not ct.logits_kl_ok(ct.logits_judge_seq(m, {**pair, "kl_max": 1.6}, C))


def test_logits_check_carries_the_counts():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((9, 50)).astype(np.float32)
    r = ct.logits_check(a, a, C)
    assert r["positions"] == r["top1_agree"] and isinstance(r["positions"], int) and r["positions"] > 0
    assert ct.logits_judge_seq(r, r, C)["positions"] == r["positions"]


@pytest.mark.parametrize("drop", ["top1_pooled_rule", "oracle_rel_tol"])
def test_load_requires_the_pooled_and_oracle_keys(tmp_path, drop):
    lines = [l for l in ct.PATH.read_text().splitlines() if not l.startswith(drop + " =")]
    p = tmp_path / "c.toml"
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ct.ContractNotCalibrated):
        ct.load(p)
