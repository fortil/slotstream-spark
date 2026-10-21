#!/usr/bin/env python3
"""Decode performance model of plan section 3.8 (stdlib only).

  t_comp = 2.87/(eta*243) + 1.327/(eta*Bpool)                 seconds
  t_io   = 48*[(1-h^10)*0.61 + (10(1-h)-(1-h^10))*0.25]       milliseconds
  t_copy = 48*(1-h^10)*t_memcpy
  t_host = 48*tau
  t      = t_comp + t_host + (1-phi)*(t_io + t_copy)

Usage:
  perf_model.py --plan        ranges from the plan assumptions
  perf_model.py --measured    point values from slotstream_spark/config/platform.toml
"""
import argparse
import sys
import tomllib
from pathlib import Path

LAYERS = 48
B_DENSE = 2.87
B_EXP = 1.327
BW_DENSE = 243.0
MODES = {"D1": (0.64, 0.70), "D2": (0.74, 0.78), "D5": (0.96, 0.98)}
PLATFORM = Path(__file__).resolve().parent.parent / "slotstream_spark" / "config" / "platform.toml"


def t_comp_ms(eta, bpool):
    return 1000.0 * (B_DENSE / (eta * BW_DENSE) + B_EXP / (eta * bpool))


def t_io_ms(h):
    miss10 = 1.0 - h ** 10
    return LAYERS * (miss10 * 0.61 + (10.0 * (1.0 - h) - miss10) * 0.25)


def t_copy_ms(h, t_memcpy_ms):
    return LAYERS * (1.0 - h ** 10) * t_memcpy_ms


def t_host_ms(tau_us):
    return LAYERS * tau_us / 1000.0


def t_token_ms(h, eta, tau_us, t_memcpy_ms, bpool, phi):
    return (t_comp_ms(eta, bpool) + t_host_ms(tau_us)
            + (1.0 - phi) * (t_io_ms(h) + t_copy_ms(h, t_memcpy_ms)))


def tok_s(*a, **k):
    return 1000.0 / t_token_ms(*a, **k)


def plan_ranges():
    """{mode: ((lo, hi) without lookahead, (lo, hi) with lookahead)} in tok/s."""
    out = {}
    for mode, (h_lo, h_hi) in MODES.items():
        lo = dict(eta=0.5, tau_us=120.0, t_memcpy_ms=0.15, bpool=222.0)
        hi = dict(eta=0.7, tau_us=60.0, t_memcpy_ms=0.10, bpool=222.0)
        out[mode] = (
            (tok_s(h_lo, phi=0.0, **lo), tok_s(h_hi, phi=0.0, **hi)),
            (tok_s(h_lo, phi=0.3, **lo), tok_s(h_hi, phi=0.5, **hi)),
        )
    return out


def measured_table(platform=PLATFORM):
    m = tomllib.load(open(platform, "rb"))["measured"]
    eta = m["gemv_2560_gbs"] / BW_DENSE
    rows = {}
    for mode, (h_lo, h_hi) in MODES.items():
        kw = dict(eta=eta, tau_us=m["tau_us"], t_memcpy_ms=m["t_memcpy_ms"],
                  bpool=m["pool_gather_gbs"])
        rows[mode] = {
            "no_lookahead": (tok_s(h_lo, phi=0.0, **kw), tok_s(h_hi, phi=0.0, **kw)),
            "lookahead": (tok_s(h_lo, phi=0.3, **kw), tok_s(h_hi, phi=0.5, **kw)),
        }
    return eta, m, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true")
    g.add_argument("--measured", action="store_true")
    ap.add_argument("--platform", default=str(PLATFORM))
    a = ap.parse_args(argv)
    if a.plan:
        print("| mode | h | no lookahead tok/s | lookahead tok/s |")
        print("|---|---|---|---|")
        for mode, (nl, la) in plan_ranges().items():
            h = MODES[mode]
            print(f"| {mode} | {h[0]:.2f}-{h[1]:.2f} | {nl[0]:.1f}-{nl[1]:.1f} | {la[0]:.1f}-{la[1]:.1f} |")
    else:
        eta, m, rows = measured_table(a.platform)
        print(f"eta={eta:.3f} tau_us={m['tau_us']} t_memcpy_ms={m['t_memcpy_ms']} "
              f"Bpool={m['pool_gather_gbs']} t_comp_ms={t_comp_ms(eta, m['pool_gather_gbs']):.1f}")
        print("| mode | h | no lookahead tok/s | lookahead tok/s |")
        print("|---|---|---|---|")
        for mode, r in rows.items():
            h = MODES[mode]
            nl, la = r["no_lookahead"], r["lookahead"]
            print(f"| {mode} | {h[0]:.2f}-{h[1]:.2f} | {nl[0]:.1f}-{nl[1]:.1f} | {la[0]:.1f}-{la[1]:.1f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
