#!/usr/bin/env python3
"""Design-gate checker.  check_gates.py gemv|host|yardstick RESULT_JSON

Exit 0 = gate passed, 2 = gate triggered (stop and ask the user), 3 = unreadable input.
"""
import json
import statistics
import sys

GEMV_MIN_GBS = 121.5          # 50 % of 243 GB/s
GEMV_PEAK_GBS = 243.0
HOST_MAX_TAU_US = 150.0
HOST_MAX_FOOTPRINT = 2.5e9
YARDSTICK_MIN_TPS = 23.0
YARDSTICK_MAX_TTFT_S = 30.0


def check_gemv(res):
    # Gate value: call time inside a captured CUDA graph with L2 flushed between replays.
    # The do_bench figure ("gbs") is informational only.
    gbs = float(res["shapes"]["2560x2560"]["gbs_cold_graph"])
    if gbs >= GEMV_MIN_GBS:
        return 0, f"gemv gate PASS: affine Triton GEMV (cold-L2 graph) {gbs:.1f} GB/s = {100 * gbs / GEMV_PEAK_GBS:.1f}% of 243"
    return 2, (f"DESIGN GATE FAILED (Decision §7 row 1): affine Triton GEMV (cold-L2 graph) {gbs:.1f} GB/s = "
               f"{100 * gbs / GEMV_PEAK_GBS:.1f}% of 243 < 50%. Stop and ask the user: plan B (port MLX qmv.cu) "
               "or plan C (llama.cpp MMVQ Q4_1).")


def check_host(res):
    tau = float(res["tau_us"])
    foot = float(res["footprint_bytes"])
    if tau <= HOST_MAX_TAU_US and foot <= HOST_MAX_FOOTPRINT:
        return 0, f"host gate PASS: tau {tau:.1f} us, footprint {foot / 1e9:.2f} GB"
    return 2, (f"DESIGN GATE FAILED (Decision §7 rows 2-3): tau_us = {tau:.1f} (max {HOST_MAX_TAU_US:.0f}), "
               f"footprint = {foot / 1e9:.2f} GB (max {HOST_MAX_FOOTPRINT / 1e9:.1f}). Stop and ask the user.")


def check_yardstick(res):
    cfg = res.get("q41-a-d5")
    if cfg is None:
        return 2, "DESIGN GATE (Decision §7 row 6): q41-a-d5 is missing from summary.json; stop and ask."
    if cfg.get("envelope_exceeded"):
        return 2, "DESIGN GATE (Decision §7 row 6): q41-a-d5 envelope_exceeded; stop and ask."
    if cfg.get("memfree_stop"):
        return 2, "DESIGN GATE (Decision §7 row 6): q41-a-d5 was stopped by the MemFree soft floor; stop and ask."
    prompts = cfg.get("prompts", {})
    meds = [p["decode_tps"]["median"] for p in prompts.values() if p.get("decode_tps", {}).get("median") is not None]
    if not meds:
        return 2, "DESIGN GATE (Decision §7 row 6): q41-a-d5 has no decode measurements; stop and ask."
    tps = statistics.median(meds)
    ttft8 = (prompts.get("8k") or {}).get("ttft_s", {}).get("median")
    cpu = (cfg.get("server") or {}).get("op_offload") is False
    if cpu:
        if tps >= YARDSTICK_MIN_TPS:
            return 2, ("DESIGN GATE (Decision §7 row 6, D5 translation, CPU prefill): unmodified llama.cpp already "
                       f"reaches the engine's D5 decode objective ({tps:.1f} tok/s; 8k TTFT {ttft8} s is CPU prefill, "
                       "informative); stop and ask whether to continue")
    elif tps >= YARDSTICK_MIN_TPS and ttft8 is not None and ttft8 <= YARDSTICK_MAX_TTFT_S:
        return 2, ("DESIGN GATE (Decision §7 row 6, D5 translation): unmodified llama.cpp already meets the "
                   f"engine's D5 F6 objective ({tps:.1f} tok/s, 8k TTFT {ttft8:.1f} s); stop and ask whether to continue")
    return 0, f"yardstick gate PASS: q41-a-d5 {tps:.1f} tok/s, 8k TTFT {ttft8} s (llama.cpp does not meet the objective) prefill={'cpu' if cpu else 'gpu'}"


GATES = {"gemv": check_gemv, "host": check_host, "yardstick": check_yardstick}


def main(argv):
    if len(argv) != 2 or argv[0] not in GATES:
        print(__doc__)
        return 3
    try:
        with open(argv[1]) as f:
            res = json.load(f)
        rc, msg = GATES[argv[0]](res)
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"check_gates: cannot evaluate {argv[1]}: {e!r}", file=sys.stderr)
        return 3
    print(msg)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
