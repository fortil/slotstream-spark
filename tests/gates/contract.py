"""The numeric contract as code: every engine-vs-oracle gate imports these checks instead of hard-coding limits.

Reads slotstream_spark/config/numeric_contract.toml. Fails loudly when it is not calibrated (empty [layer] or
[routing] arrays, or a -1.0 in [logits] / [fixture]).
"""
import json
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from calibrate_contract import logits_metrics, rel_l2, topk_disagreement  # noqa: E402

PATH = ROOT / "slotstream_spark" / "config" / "numeric_contract.toml"
LOGITS_KEYS = ("top1_min", "kl_mean_max_nats", "kl_p99_max_nats", "kl_max_nats")
LOGITS_SEQ = (("kl_mean", "kl_mean_max_nats", "kl_mean_ceiling_nats"), ("kl_p99", "kl_p99_max_nats", "kl_p99_ceiling_nats"),
              ("kl_max", "kl_max_nats", "kl_max_ceiling_nats"))


class ContractNotCalibrated(RuntimeError):
    pass


@dataclass
class Contract:
    layer: dict
    routing: dict
    logits: dict
    fixture: dict
    raw: dict


def load(path=PATH):
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    for sec in ("layer", "routing"):
        for key in ("spread", "control", "limits", "separable"):
            if not raw[sec].get(key):
                raise ContractNotCalibrated(f"contract not calibrated: [{sec}] {key} is empty ({path})")
        for key in ("seq_rule", "seq_factor", "seq_ceiling_ratio"):
            if key not in raw[sec]:
                raise ContractNotCalibrated(f"contract not calibrated: [{sec}] {key} is missing ({path})")
    if "seq_rule" not in raw["logits"]:
        raise ContractNotCalibrated(f"contract not calibrated: [logits] seq_rule is missing ({path})")
    if "top1_pooled_rule" not in raw["logits"]:
        raise ContractNotCalibrated(f"contract not calibrated: [logits] top1_pooled_rule is missing ({path})")
    for key in ("oracle_windows", "oracle_rel_tol"):
        if key not in raw.get("perplexity", {}):
            raise ContractNotCalibrated(f"contract not calibrated: [perplexity] {key} is missing ({path})")
    for key in LOGITS_KEYS:
        if float(raw["logits"].get(key, -1.0)) < 0:
            raise ContractNotCalibrated(f"contract not calibrated: [logits] {key} = {raw['logits'].get(key)} ({path})")
    if float(raw["fixture"].get("kl_max_nats", -1.0)) < 0:
        raise ContractNotCalibrated(f"contract not calibrated: [fixture] kl_max_nats ({path})")
    return Contract(raw["layer"], raw["routing"], raw["logits"], raw["fixture"], raw)


def layer_check(k, engine, oracle, c=None):
    """-> (rel_l2, limit, separable, ok); `ok` is True for a layer that is not separable (informational)."""
    c = c or load()
    v = rel_l2(engine, oracle)
    lim, sep = float(c.layer["limits"][k]), bool(c.layer["separable"][k])
    return v, lim, sep, (v <= lim) if sep else True


def routing_check(k, ids_engine, ids_oracle, c=None):
    """-> (disagreement, limit, separable, ok); layer 0 is informational by construction (not separable)."""
    c = c or load()
    v = topk_disagreement(ids_engine, ids_oracle)
    lim, sep = float(c.routing["limits"][k]), bool(c.routing["separable"][k])
    return v, lim, sep, (v <= lim) if sep else True


def seq_limit(table, k, pair, c=None):
    """Per-sequence limit of design decision 16 -> (L_seq, src).

    L_seq = min(max(L_k, seq_factor*pair + margin), c_k/seq_ceiling_ratio); `pair` is the oracle bf16-vs-fp32 metric
    on the judged sequence. `src` names the binding term: "calibrated" (L_k), "sequence" or "ceiling".
    """
    c = c or load()
    t = {"layer": c.layer, "routing": c.routing}[table]
    cal = float(t["limits"][k])
    seq = float(t["seq_factor"]) * float(pair) + float(t["margin"])
    ceil = float(t["control"][k]) / float(t["seq_ceiling_ratio"])
    lim = min(max(cal, seq), ceil)
    if lim == ceil and ceil < max(cal, seq):
        return lim, "ceiling"
    return (cal, "calibrated") if cal >= seq else (seq, "sequence")


def _check_seq(table, k, v, pair, c):
    lim, src = seq_limit(table, k, pair, c)
    sep = bool(getattr(c, table)["separable"][k])
    return {"value": float(v), "pair": float(pair), "limit_cal": float(getattr(c, table)["limits"][k]),
            "limit_seq": float(lim), "limit_src": src, "separable": sep, "ok": bool(v <= lim) if sep else True}


def routing_check_seq(k, ids_engine, ids_oracle, ids_oracle32, c=None):
    """Routing disagreement engine-vs-oracle judged against the limit widened by the oracle's own bf16-vs-fp32 pair."""
    c = c or load()
    return _check_seq("routing", k, topk_disagreement(ids_engine, ids_oracle),
                      topk_disagreement(ids_oracle, ids_oracle32), c)


def layer_check_seq(k, engine, oracle, oracle32, c=None):
    """rel_l2 engine-vs-oracle judged against the limit widened by rel_l2(oracle, oracle fp32)."""
    c = c or load()
    return _check_seq("layer", k, rel_l2(engine, oracle), rel_l2(oracle, oracle32), c)


def logits_check(engine, oracle, c=None):
    """Full vocabulary, shift-first, KL(oracle -> engine) per position; ok only when all four limits hold."""
    import numpy as np
    c = c or load()
    kl, agree = logits_metrics(oracle, engine)
    out = {"top1": float(np.mean(agree)), "kl_mean": float(kl.mean()), "kl_p99": float(np.percentile(kl, 99)),
           "kl_max": float(kl.max()), "top1_agree": int(np.asarray(agree).sum()), "positions": int(np.asarray(agree).size)}
    L = c.logits
    out["ok"] = bool(out["top1"] >= L["top1_min"] and out["kl_mean"] <= L["kl_mean_max_nats"]
                     and out["kl_p99"] <= L["kl_p99_max_nats"] and out["kl_max"] <= L["kl_max_nats"])
    return out


def logits_seq_limits(pair, c=None):
    """Per-sequence KL limits of design decision 18: min(max(calibrated, kl_factor*pair), ceiling) per statistic.

    `pair` holds the oracle bf16-vs-fp32 statistics on the judged sequence. -> dict with `top1_min` (calibrated), the three
    limit keys, `src` (limit key -> "calibrated" | "sequence" | "ceiling") and `pair_over_ceiling` (statistics whose pair
    value is above its ceiling, which makes the sequence not judgeable).
    """
    c = c or load()
    L = c.logits
    out = {"top1_min": float(L["top1_min"]), "src": {}, "pair_over_ceiling": []}
    for x, cal_k, ceil_k in LOGITS_SEQ:
        cal, ceil, seq = float(L[cal_k]), float(L[ceil_k]), float(L["kl_factor"]) * float(pair[x])
        out[cal_k] = min(max(cal, seq), ceil)
        out["src"][cal_k] = "ceiling" if ceil < max(cal, seq) else ("calibrated" if cal >= seq else "sequence")
        if float(pair[x]) > ceil:
            out["pair_over_ceiling"].append(x)
    return out


def logits_judge_seq(metrics, pair, c=None):
    """Judge `metrics` (a logits_check dict) against the per-sequence limits derived from `pair` (another one)."""
    c = c or load()
    lim = logits_seq_limits(pair, c)
    L = c.logits
    out = {k: metrics[k] for k in ("top1", "kl_mean", "kl_p99", "kl_max")}
    for k in ("top1_agree", "positions"):
        if k in metrics:
            out[k] = metrics[k]
    out["fixed_limits_ok"] = bool(metrics["top1"] >= L["top1_min"] and metrics["kl_mean"] <= L["kl_mean_max_nats"]
                                  and metrics["kl_p99"] <= L["kl_p99_max_nats"] and metrics["kl_max"] <= L["kl_max_nats"])
    out["pair"] = {k: pair[k] for k in ("top1", "kl_mean", "kl_p99", "kl_max")}
    out["limits_seq"] = {"top1_min": lim["top1_min"], **{cal_k: lim[cal_k] for _, cal_k, _ in LOGITS_SEQ}}
    out["limit_src"] = dict(lim["src"])
    out["judgeable"] = not lim["pair_over_ceiling"]
    out["ok"] = bool(out["judgeable"] and metrics["top1"] >= lim["top1_min"]
                     and all(metrics[x] <= lim[cal_k] for x, cal_k, _ in LOGITS_SEQ))
    return out


def logits_kl_ok(judged):
    """The three KL statistics hold their per-sequence limits and the sequence is judgeable (top-1 is judged pooled)."""
    lim = judged["limits_seq"]
    return bool(judged["judgeable"] and all(judged[x] <= lim[cal_k] for x, cal_k, _ in LOGITS_SEQ))


def top1_pooled(items, c=None):
    """Top-1 over a hold-out judged once: items = [(id, agree, positions)] -> sum(agree)/sum(positions) >= top1_min."""
    c = c or load()
    agree = positions = 0
    per = []
    for pid, a, n in items:
        if isinstance(a, bool) or isinstance(n, bool) or not isinstance(a, int) or not isinstance(n, int):
            raise ValueError(f"{pid}: agree and positions must be ints ({a!r}, {n!r})")
        if n <= 0 or a < 0 or a > n:
            raise ValueError(f"{pid}: agree {a} / positions {n} is not a valid count")
        agree, positions = agree + a, positions + n
        per.append({"id": pid, "agree": a, "positions": n, "top1": a / n})
    if positions == 0:
        raise ValueError("top1_pooled: no positions")
    top1, floor = agree / positions, float(c.logits["top1_min"])
    return {"top1": top1, "agree": agree, "positions": positions, "top1_min": floor, "ok": bool(top1 >= floor), "per_prompt": per}


def logits_check_seq(engine, oracle, oracle32, c=None):
    """logits_judge_seq(engine vs oracle, oracle32 vs oracle); the three arrays must come from one ids list (callers assert)."""
    c = c or load()
    return logits_judge_seq(logits_check(engine, oracle, c), logits_check(oracle32, oracle, c), c)


def ngram_check(a, b):
    """The n-gram ids are exact."""
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    return bool(a.shape == b.shape and (a == b).all())


def summary(results):
    """JSON-ready dict for $CAP/result.json: the results plus the contract values they were judged against."""
    c = load()
    return {**results, "contract": {"path": str(PATH.relative_to(ROOT)), "layer_metric": c.layer["metric"],
                                    "logits": {k: c.logits[k] for k in LOGITS_KEYS},
                                    "fixture_kl_max_nats": c.fixture["kl_max_nats"],
                                    "logits_seq_rule": c.logits["seq_rule"]}}


def dumps(results):
    return json.dumps(summary(results), indent=1, sort_keys=True)
