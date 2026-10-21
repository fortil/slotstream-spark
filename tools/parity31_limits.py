#!/usr/bin/env python3
"""Per-layer parity31 limits from the oracle's own bf16-vs-fp32 spread (pre-registered rule, stdlib only).

  parity31_limits.py CALIB_RESULT_JSON --write LIMITS_JSON --record RECORD_MD --capture CAPTURE_DIR

Rule: s_k = spread_bf16_fp32, c_k = min(control_noexp, control_swap), L_k = 2*s_k + MARGIN.
A layer is separable iff c_k >= 4*s_k and c_k >= 2*L_k. Exit 1 unless 4 layers are all separable.
"""
import argparse
import json
import sys
from pathlib import Path

MARGIN = 0.01
RULE = "L_k = 2*s_k + 0.01; separable iff c_k >= 4*s_k and c_k >= 2*L_k"
ROOT = Path(__file__).resolve().parent.parent


def derive(result):
    out = []
    for x in result["layers"]:
        s = float(x["spread_bf16_fp32"])
        c = min(float(x["control_noexp"]), float(x["control_swap"]))
        lim = 2 * s + MARGIN
        out.append({"layer": x["layer"], "spread": s, "control": c, "limit": lim,
                    "separable": bool(c >= 4 * s and c >= 2 * lim),
                    "bf16_vs_golden": x.get("bf16_vs_golden"), "fp32_vs_golden": x.get("fp32_vs_golden"),
                    "control_noexp": x.get("control_noexp"), "control_swap": x.get("control_swap")})
    return out


def rel(p, root=ROOT):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(root))
    except ValueError:
        return p.name


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--write", required=True)
    ap.add_argument("--record", required=True)
    ap.add_argument("--capture", required=True)
    a = ap.parse_args(argv)
    result = json.loads(Path(a.result).read_text())
    layers = derive(result)
    bad = [x["layer"] for x in layers if not x["separable"]]
    good = len(layers) == 4 and not bad
    lim = {"rule": RULE, "margin": MARGIN,
           "layers": [{k: x[k] for k in ("layer", "spread", "control", "limit", "separable")} for x in layers],
           "source_capture": rel(a.capture)}
    Path(a.write).write_text(json.dumps(lim, indent=2) + "\n")
    rows = ["| layer | spread s | noexp | swap | bf16 vs golden | fp32 vs golden | limit L | c/s | c/L | separable |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "n/a" if v is None else f"{v:.4f}"
    for x in layers:
        cs = x["control"] / x["spread"] if x["spread"] else float("inf")
        rows.append(f"| {x['layer']} | {x['spread']:.4f} | {f(x['control_noexp'])} | {f(x['control_swap'])} | "
                    f"{f(x['bf16_vs_golden'])} | {f(x['fp32_vs_golden'])} | {x['limit']:.4f} | {cs:.1f} | "
                    f"{x['control'] / x['limit']:.1f} | {'yes' if x['separable'] else 'no'} |")
    if good:
        verdict = "Verdict: INFO — limits written, 4/4 layers separable"
    else:
        k = bad[0] if bad else len(layers)
        verdict = (f"Verdict: FAIL — DESIGN GATE: per-layer rel_l2 cannot separate noise from a wrong expert at "
                   f"layer {k} — stop and ask the user")
    rec = [f"# oracle-parity31-calibration: parity31 limits from the oracle's own bf16 vs fp32 spread",
           f"Capture: {rel(a.capture)}/",
           f"Rule: {RULE} (s = rel_l2 of the bf16 oracle vs the fp32 oracle, c = min of the noexp and swap controls vs the golden)",
           "", *rows, "", verdict, ""]
    Path(a.record).write_text("\n".join(rec))
    if not good:
        print("DESIGN GATE FAILED: per-layer rel_l2 cannot separate noise from a wrong expert at layer "
              f"{bad[0] if bad else 'n/a'} — stop and ask the user", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
