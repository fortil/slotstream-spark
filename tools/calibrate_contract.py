#!/usr/bin/env python3
"""Calibrate the per-layer and routing limits of numeric_contract.toml from the oracle's own noise.

  calibrate_contract.py measure --golden DIR --fp32 DIR --swap DIR --noexp DIR --out JSON    (Spark, numpy only)
  calibrate_contract.py apply CALIB_JSON --toml PATH --record PATH                              (Mac, stdlib + tomllib)

Rule (one shared function, `rule`): L_k = 2*s_k + margin; separable iff c_k >= 4*s_k and c_k >= 2*L_k.
Dump dirs follow tools/make_goldens.py: DIR/dev/<id>/{layer_LL_out.f32, router_LL_ids.i32, logits.f32,
tokens.json}. The goldens are the bf16 oracle; --fp32 is the same oracle in float32; --swap and --noexp are the
bf16 oracle with a wrong expert unpacking.
"""
import argparse
import json
import math
import re
import sys
import tomllib
from pathlib import Path

CONTROL_RATIO_MIN = 4.0
ARRAYS = ("spread", "control", "limits", "separable")
TABLES = ("layer", "routing")
SCALARS = {"logits": ("o_top1", "o_kl_mean_nats", "o_kl_p99_nats", "o_kl_max_nats", "o_positions",
                      "top1_min", "kl_mean_max_nats", "kl_p99_max_nats", "kl_max_nats"),
           "fixture": ("kl_max_nats",)}


def rule(spread, control, margin=0.01, ratio=CONTROL_RATIO_MIN):
    limit = 2 * spread + margin
    return limit, bool(control >= ratio * spread and control >= 2 * limit)


# ---------------------------------------------------------------- measure (numpy)

def _np():
    import numpy as np
    return np


def rel_l2(a, b):
    np = _np()
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def topk_disagreement(a, b):
    """1 - |A inter B| / k averaged over tokens; a, b int [T,k]."""
    np = _np()
    a, b = np.asarray(a), np.asarray(b)
    k = a.shape[1]
    inter = np.array([len(set(x.tolist()) & set(y.tolist())) for x, y in zip(a, b)])
    return float(np.mean(1.0 - inter / k))


def logits_metrics(ref, other):
    """Per-position KL(ref -> other) over the full vocabulary (shift-first) and argmax agreement."""
    np = _np()
    r = np.asarray(ref, dtype=np.float64)
    o = np.asarray(other, dtype=np.float64)
    r = r - r.max(axis=1, keepdims=True)
    o = o - o.max(axis=1, keepdims=True)
    rz = np.log(np.exp(r).sum(axis=1, keepdims=True))
    oz = np.log(np.exp(o).sum(axis=1, keepdims=True))
    lp, lq = r - rz, o - oz
    kl = (np.exp(lp) * (lp - lq)).sum(axis=1)
    return np.maximum(kl, 0.0), (r.argmax(axis=1) == o.argmax(axis=1))


def _load(d, stem, ext, dtype, T, last):
    np = _np()
    return np.fromfile(Path(d) / f"{stem}{ext}", dtype=dtype).reshape(T, last)


def measure(golden, fp32, swap, noexp, nlayers=48, hidden=10240, topk=10, vocab=248320):
    np = _np()
    ids = sorted(p.name for p in (Path(golden) / "dev").iterdir() if p.is_dir())
    if not ids:
        raise SystemExit(f"no dev prompts under {golden}/dev")
    per = {pid: {} for pid in ids}
    all_kl, all_top1 = [], []
    for pid in ids:
        dirs = {n: Path(d) / "dev" / pid for n, d in (("g", golden), ("f", fp32), ("s", swap), ("n", noexp))}
        for n, d in dirs.items():
            if not (d / "DONE").exists() and not (d / "tokens.json").exists():
                raise SystemExit(f"{n}: missing dump {d}")
        T = len(json.load(open(dirs["g"] / "tokens.json"))["ids"])
        layers = []
        for L in range(nlayers):
            lay, rt = f"layer_{L:02d}_out", f"router_{L:02d}_ids"
            g = _load(dirs["g"], lay, ".f32", np.float32, T, hidden)
            gi = _load(dirs["g"], rt, ".i32", np.int32, T, topk)
            si = _load(dirs["s"], rt, ".i32", np.int32, T, topk)
            fi = _load(dirs["f"], rt, ".i32", np.int32, T, topk)
            layers.append({
                "layer": L,
                "spread": rel_l2(g, _load(dirs["f"], lay, ".f32", np.float32, T, hidden)),
                "control_swap": rel_l2(_load(dirs["s"], lay, ".f32", np.float32, T, hidden), g),
                "control_noexp": rel_l2(_load(dirs["n"], lay, ".f32", np.float32, T, hidden), g),
                "routing_spread": topk_disagreement(fi, gi),
                "routing_control": topk_disagreement(si, gi)})
        kl, top1 = logits_metrics(_load(dirs["g"], "logits", ".f32", np.float32, T, vocab),
                                  _load(dirs["f"], "logits", ".f32", np.float32, T, vocab))
        all_kl.append(kl)
        all_top1.append(top1)
        per[pid] = layers
    out_layers = []
    for L in range(nlayers):
        rows = [per[pid][L] for pid in ids]
        out_layers.append({
            "layer": L,
            "spread_max": max(r["spread"] for r in rows),
            "control_min": min(min(r["control_swap"], r["control_noexp"]) for r in rows),
            "routing_spread_max": max(r["routing_spread"] for r in rows),
            "routing_control_min": min(r["routing_control"] for r in rows),
            "per_prompt": {pid: {k: v for k, v in per[pid][L].items() if k != "layer"} for pid in ids}})
    kl = np.concatenate(all_kl)
    return {"prompts": ids, "layers": out_layers,
            "logits": {"top1": float(np.concatenate(all_top1).mean()), "kl_mean": float(kl.mean()),
                       "kl_p99": float(np.percentile(kl, 99)), "kl_max": float(kl.max()), "positions": int(kl.size)}}


# ---------------------------------------------------------------- apply (stdlib)

def derive(calib, cfg):
    """-> {'layer': arrays, 'routing': arrays} computed with `rule`."""
    res = {}
    for table, sk, ck in (("layer", "spread_max", "control_min"), ("routing", "routing_spread_max", "routing_control_min")):
        margin, ratio = cfg[table]["margin"], cfg[table]["control_ratio_min"]
        arr = {a: [] for a in ARRAYS}
        for x in sorted(calib["layers"], key=lambda r: r["layer"]):
            s, c = float(x[sk]), float(x[ck])
            lim, sep = rule(s, c, margin, ratio)
            if table == "routing" and x["layer"] == 0:
                sep = False  # informational by construction
            arr["spread"].append(s)
            arr["control"].append(c)
            arr["limits"].append(lim)
            arr["separable"].append(sep)
        res[table] = arr
    return res


def rule_logits(o, cfg_logits):
    """Logits limits from the oracle's own bf16-vs-fp32 pair; never clamps.

    o: {"top1","kl_mean","kl_p99","kl_max"}. Returns (limits, violations of the validity range).
    """
    c = cfg_logits
    lim = {"top1_min": 1 - c["top1_factor"] * (1 - o["top1"]),
           "kl_mean_max_nats": c["kl_factor"] * o["kl_mean"],
           "kl_p99_max_nats": c["kl_factor"] * o["kl_p99"],
           "kl_max_nats": c["kl_factor"] * o["kl_max"]}
    bad = []
    if lim["top1_min"] < c["top1_floor"]:
        bad.append(f"top1_min {lim['top1_min']:.4f} < top1_floor {c['top1_floor']}")
    for key, ceil in (("kl_mean_max_nats", "kl_mean_ceiling_nats"), ("kl_p99_max_nats", "kl_p99_ceiling_nats"),
                      ("kl_max_nats", "kl_max_ceiling_nats")):
        if lim[key] > c[ceil]:
            bad.append(f"{key} {lim[key]:.5f} > {ceil} {c[ceil]}")
    return lim, bad


def _fmt(v):
    return ("true" if v else "false") if isinstance(v, bool) else repr(float(v))


def rewrite(text, arrays, scalars=None):
    """Replace the single-line arrays under [layer]/[routing] and the SCALARS keys of [logits]/[fixture]."""
    scalars = scalars or {}
    out, table, seen, seen_s = [], None, set(), set()
    for line in text.splitlines():
        m = re.match(r"^\[(\w+)\]", line)
        if m:
            table = m.group(1)
        k = re.match(r"^(spread|control|limits|separable)\s*=\s*\[.*\]\s*$", line)
        if k and table in arrays:
            name = k.group(1)
            line = f"{name} = [" + ", ".join(_fmt(v) for v in arrays[table][name]) + "]"
            seen.add((table, name))
        s = re.match(r"^(\w+)(\s*=\s*)(-?[0-9][0-9.eE+-]*)(\s*#.*)?$", line)
        if s and table in scalars and s.group(1) in scalars[table]:
            v = scalars[table][s.group(1)]
            line = f"{s.group(1)}{s.group(2)}{int(v) if s.group(1) == 'o_positions' else repr(float(v))}{s.group(4) or ''}"
            seen_s.add((table, s.group(1)))
        out.append(line)
    missing = {(t, a) for t in arrays for a in ARRAYS} - seen
    if missing:
        raise SystemExit(f"toml has no single-line array for {sorted(missing)}")
    missing_s = {(t, k) for t in scalars for k in scalars[t]} - seen_s
    if missing_s:
        raise SystemExit(f"toml has no single-line scalar for {sorted(missing_s)}")
    return "\n".join(out) + "\n"


def _strip(cfg):
    return {t: {k: v for k, v in body.items()
                if not ((t in TABLES and k in ARRAYS) or k in SCALARS.get(t, ()))} if isinstance(body, dict) else body
            for t, body in cfg.items()}


def apply(calib_path, toml_path, record_path):
    calib = json.loads(Path(calib_path).read_text())
    text = Path(toml_path).read_text()
    cfg = tomllib.loads(text)
    arrays = derive(calib, cfg)
    nsep = sum(arrays["layer"]["separable"])
    nsep_r = sum(arrays["routing"]["separable"])
    lg = calib["logits"]
    limits, bad = rule_logits(lg, cfg["logits"])
    problems = []
    if nsep < cfg["layer"]["min_separable_layers"]:
        problems.append(f"only {nsep} layers are separable (need {cfg['layer']['min_separable_layers']})")
    problems += [f"oracle bf16-vs-fp32 logits pair: {b}" for b in bad]
    rows = ["", "## Calibrated arrays (oracle-only dumps)", "",
            "| layer | spread s | control c | limit L | separable | routing s | routing c | routing L | routing separable |",
            "|---|---|---|---|---|---|---|---|---|"]
    for k in range(len(arrays["layer"]["limits"])):
        a, r = arrays["layer"], arrays["routing"]
        rows.append(f"| {k} | {a['spread'][k]:.4f} | {a['control'][k]:.4f} | {a['limits'][k]:.4f} | "
                    f"{'yes' if a['separable'][k] else 'no'} | {r['spread'][k]:.4f} | {r['control'][k]:.4f} | "
                    f"{r['limits'][k]:.4f} | {'yes' if r['separable'][k] else 'no'} |")
    c = cfg["logits"]
    inside = {"top1_min": limits["top1_min"] >= c["top1_floor"],
              "kl_mean_max_nats": limits["kl_mean_max_nats"] <= c["kl_mean_ceiling_nats"],
              "kl_p99_max_nats": limits["kl_p99_max_nats"] <= c["kl_p99_ceiling_nats"],
              "kl_max_nats": limits["kl_max_nats"] <= c["kl_max_ceiling_nats"]}
    yn = lambda v: "yes" if v else "no"
    rows += ["", f"Oracle bf16 vs fp32 logits ({lg.get('positions', '?')} positions): top-1 {lg['top1']:.4f}, "
             f"KL mean {lg['kl_mean']:.5f}, p99 {lg['kl_p99']:.5f}, max {lg['kl_max']:.5f} nats.",
             f"Separable layers: {nsep}/{len(arrays['layer']['limits'])} (layer), {nsep_r}/{len(arrays['routing']['limits'])} (routing).", "",
             "| statistic | oracle pair | rule | limit | validity | inside |", "|---|---|---|---|---|---|",
             f"| top-1 | {lg['top1']:.4f} | 1 - {c['top1_factor']}*(1 - o) | {limits['top1_min']:.5f} | >= {c['top1_floor']} | {yn(inside['top1_min'])} |",
             f"| KL mean | {lg['kl_mean']:.5f} | {c['kl_factor']}*o | {limits['kl_mean_max_nats']:.5f} | <= {c['kl_mean_ceiling_nats']} | {yn(inside['kl_mean_max_nats'])} |",
             f"| KL p99 | {lg['kl_p99']:.5f} | {c['kl_factor']}*o | {limits['kl_p99_max_nats']:.5f} | <= {c['kl_p99_ceiling_nats']} | {yn(inside['kl_p99_max_nats'])} |",
             f"| KL max | {lg['kl_max']:.5f} | {c['kl_factor']}*o | {limits['kl_max_nats']:.5f} | <= {c['kl_max_ceiling_nats']} | {yn(inside['kl_max_nats'])} |",
             "", f"Fixture: kl_max_nats = {limits['kl_max_nats']:.5f}", ""]
    if problems:
        rows.append("Calibration status: FAILED — " + "; ".join(problems))
        _replace_section(record_path, rows)
        print("DESIGN GATE FAILED: " + "; ".join(problems) + " — stop and ask the user", file=sys.stderr)
        return 1
    scalars = {"logits": {"o_top1": lg["top1"], "o_kl_mean_nats": lg["kl_mean"], "o_kl_p99_nats": lg["kl_p99"],
                          "o_kl_max_nats": lg["kl_max"], "o_positions": lg.get("positions", 0), **limits},
               "fixture": {"kl_max_nats": limits["kl_max_nats"]}}
    new = rewrite(text, arrays, scalars)
    new_cfg = tomllib.loads(new)
    if _strip(new_cfg) != _strip(cfg):
        raise SystemExit("apply would change a non-calibrated key; refusing to write")
    for t in TABLES:
        for a in ARRAYS:
            if len(new_cfg[t][a]) != len(arrays[t]["limits"]):
                raise SystemExit(f"{t}.{a} has the wrong length after rewrite")
    Path(toml_path).write_text(new)
    rows.append("Calibration status: OK")
    _replace_section(record_path, rows)
    print(json.dumps({"separable_layers": nsep, "separable_routing": nsep_r, "logits": lg, "logits_limits": limits}))
    return 0


def _replace_section(record_path, rows):
    """Replace from '## Calibrated arrays' to the end of the file; append when absent."""
    p = Path(record_path)
    text = p.read_text() if p.exists() else ""
    i = text.find("\n## Calibrated arrays")
    if i < 0 and text.startswith("## Calibrated arrays"):
        i = 0
    head = text[:i] if i >= 0 else text.rstrip("\n") + "\n"
    p.write_text(head + "\n".join(rows) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    for n in ("golden", "fp32", "swap", "noexp", "out"):
        m.add_argument(f"--{n}", required=True)
    p = sub.add_parser("apply")
    p.add_argument("calib")
    p.add_argument("--toml", required=True)
    p.add_argument("--record", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "measure":
        res = measure(a.golden, a.fp32, a.swap, a.noexp)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(json.dumps({"prompts": res["prompts"], "logits": res["logits"]}))
        return 0
    return apply(a.calib, a.toml, a.record)


if __name__ == "__main__":
    sys.exit(main())
