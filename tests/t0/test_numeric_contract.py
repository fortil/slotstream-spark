import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / "slotstream_spark" / "config" / "numeric_contract.toml"
SEQ_RULE = ("L_seq = min(max(L_k, seq_factor*d_k + margin), c_k/seq_ceiling_ratio); d_k = metric(oracle bf16, "
            "oracle fp32) on the judged sequence; asserted iff separable[k]")
RULE = "L_k = 2*s_k + margin; separable iff c_k >= 4*s_k and c_k >= 2*L_k"


def load():
    return tomllib.loads(PATH.read_text())


def test_layer_and_routing_keys():
    c = load()
    L, R = c["layer"], c["routing"]
    assert L["metric"] == "rel_l2" and L["rule"] == RULE and L["margin"] == 0.01
    assert L["control_ratio_min"] == 4.0 and isinstance(L["min_separable_layers"], int) and L["min_separable_layers"] == 4
    assert R["metric"] == "top10_set_disagreement" and R["rule"] == RULE and R["margin"] == 0.01
    assert R["control_ratio_min"] == 4.0 and R["ngram_ids"] == "exact"
    for t in (L, R):
        assert t["seq_rule"] == SEQ_RULE and t["seq_factor"] == 2.0 and t["seq_ceiling_ratio"] == 2.0
        for a in ("spread", "control", "limits", "separable"):
            assert isinstance(t[a], list) and len(t[a]) in (0, 48), a
        assert len(t["spread"]) == len(t["control"]) == len(t["limits"]) == len(t["separable"])
        assert all(isinstance(v, bool) for v in t["separable"])
        assert all(isinstance(v, float) for a in ("spread", "control", "limits") for v in t[a])


def test_rule_parameters_and_fixed_tables():
    c = load()
    L = c["logits"]
    assert L["metric"] == "top1_agreement_and_kl_nats"
    assert L["rule"] == ("top1_min = 1 - top1_factor*(1 - o_top1); kl_X_max = kl_factor*o_kl_X; "
                         "valid iff top1_min >= top1_floor and every kl_X_max <= kl_X_ceiling")
    assert L["seq_rule"] == ("kl_X_seq = min(max(kl_X_max_nats, kl_factor*p_X), kl_X_ceiling_nats) for X in {mean, p99, max}; "
                             "p_X = statistic X of KL(oracle bf16 -> oracle fp32) on the judged sequence, same token ids as the "
                             "engine dump; top1_min stays calibrated; judgeable iff every p_X <= kl_X_ceiling_nats")
    assert (L["top1_factor"], L["kl_factor"], L["top1_floor"]) == (2.0, 3.0, 0.85)
    assert (L["kl_mean_ceiling_nats"], L["kl_p99_ceiling_nats"], L["kl_max_ceiling_nats"]) == (0.05, 0.5, 1.5)
    for k in ("o_top1", "o_kl_mean_nats", "o_kl_p99_nats", "o_kl_max_nats", "top1_min",
              "kl_mean_max_nats", "kl_p99_max_nats", "kl_max_nats"):
        assert isinstance(L[k], float), k
    assert isinstance(L["o_positions"], int) and not isinstance(L["o_positions"], bool)
    F = c["fixture"]
    assert F["rule"] == "kl_max_nats = [logits].kl_max_nats" and isinstance(F["kl_max_nats"], float)
    assert c["ops"] == {"rel_l2_max": 2e-2, "hc_rel_l2_max": 5e-3, "fused_moe_rel_l2_max": 1e-2,
                        "attn_oracle_rel_l2_max": 5e-3, "forecast_tie_rel": 0.01, "control_min_ratio": 10.0}
    P = dict(c["perplexity"])
    rule = P.pop("oracle_rule")
    assert isinstance(rule, str) and rule
    assert P == {"wikitext2_ref": 5.3914, "rel_tol": 0.01, "windows": 145, "window_tokens": 2048, "oracle_windows": 32,
                 "oracle_rel_tol": 0.02, "reference_informative": True}
    assert isinstance(c["logits"]["top1_pooled_rule"], str) and c["logits"]["top1_pooled_rule"]
    assert c["ablation"] == {"ppl_correct_max": 20.0, "swap_min_ratio": 5.0, "noexp_min_ratio": 5.0}
    assert c["engine_self"]["bit_exact"] == ["pool_size", "resize", "readfault_retry", "mtp_vs_plain",
                                             "lookahead_on_off", "prefix_resume", "sweep_cold_warm"]


def test_logits_limits_follow_the_rule_when_calibrated():
    c = load()
    L = c["logits"]
    if L["top1_min"] == -1.0:
        return
    assert abs(L["top1_min"] - (1 - L["top1_factor"] * (1 - L["o_top1"]))) < 1e-12
    for k in ("mean", "p99", "max"):
        key = "kl_max_nats" if k == "max" else f"kl_{k}_max_nats"
        assert abs(L[key] - L["kl_factor"] * L[f"o_kl_{k}_nats"]) < 1e-12
        assert L[key] <= L[f"kl_{k}_ceiling_nats"]
    assert L["top1_min"] >= L["top1_floor"]
    assert c["fixture"]["kl_max_nats"] == L["kl_max_nats"]
    assert L["o_positions"] > 0


def test_calibrated_arrays_follow_the_rule_when_filled():
    L = load()["layer"]
    if not L["limits"]:
        return
    assert all(abs(l - (2 * s + L["margin"])) < 1e-12 for l, s in zip(L["limits"], L["spread"]))
    assert sum(L["separable"]) >= L["min_separable_layers"]
