#!/usr/bin/env python3
"""One arm of the F6 A/Bs: a fresh engine process decoding one cohort prompt in D5 (plan step 77).

  f6_arm.py --cohort B1 --pair 3 --pool cold|table --out FILE [--warmup-tokens 128] [--tokens 512] [--seed 42]
  env: SLOTSTREAM_LOOKAHEAD=off|attention|attention-corrected   SLOTSTREAM_MTP=off|resident   (both read by the engine's own
       policy readers: engine.lookahead.LookaheadConfig.load and engine.mtp.policy)

The prompt is the `--pair`-th pair prompt of the cohort in bench/cohorts/manifest.json, rendered as its chat messages with
thinking off. The engine starts, compiles its kernels on an unrelated prompt (bench.ttft.warm_up), then
(--pool cold) empties the pool or (--pool table, step 77f) pre-fills it from expert_freq_gb10.json right after the engine is
built, before the first timestamp, and keeps it through the warm-up prompt; then it prefills the prompt through the sweep driver and decodes `warmup + tokens` greedy tokens (seed 42, no eos stop, so every arm
produces the same count). The metric is over the last `tokens` tokens: (N - 1) / (t_last - t_first) on the emission
timestamps, which is (N-1)/sum(inter-token). `tokens_sha256` hashes all emitted ids, so the A/B can check that the two arms
produced the same output.
"""
import argparse
import array
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

COHORTS = ROOT / "bench" / "cohorts"
WARMUP_CTX = 3072                      # bench.ttft.warm_up's prompt needs this much room


def load_request(cohort, pair):
    man = json.loads((COHORTS / "manifest.json").read_text())
    c = man["cohorts"][cohort]
    rid = c["pair_ids"][pair]
    for line in (COHORTS / c["file"]).read_text().splitlines():
        r = json.loads(line)
        if r["id"] == rid:
            return r
    raise SystemExit(f"f6_arm: {rid} not in {c['file']}")


def ids_sha256(ids):
    return hashlib.sha256(array.array("i", [int(t) for t in ids]).tobytes()).hexdigest()


def decode_tps(stamps, warmup, tokens):
    """(N-1)/(t_last - t_first) over the `tokens` emissions after the first `warmup` ones."""
    window = stamps[warmup:warmup + tokens]
    if len(window) < 2 or window[-1] <= window[0]:
        return None
    return (len(window) - 1) / (window[-1] - window[0])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cohort", required=True, choices=("B0", "B1"))
    ap.add_argument("--pair", type=int, required=True)
    ap.add_argument("--pool", required=True, choices=("cold", "table"))
    ap.add_argument("--mode", default="D5")
    ap.add_argument("--warmup-tokens", type=int, default=128)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from bench import ttft
    from slotstream_spark.engine import lookahead as la
    from slotstream_spark.engine import mtp as mtp_mod
    from slotstream_spark.engine.generate import generate
    from slotstream_spark.engine.prefill import PrefillPlan
    from slotstream_spark.engine.tokenizer import Tokenizer
    from slotstream_spark.kernels import forecast as fc
    from slotstream_spark.runtime.graphs import GraphSet
    from slotstream_spark.runtime.readback import Readback
    import torch

    req = load_request(a.cohort, a.pair)
    tok = Tokenizer.from_checkpoint(os.environ["SLOTSTREAM_MODEL"])
    ids = tok.encode(tok.render_chat(req["messages"], enable_thinking=False))
    total = a.warmup_tokens + a.tokens
    lcfg = la.LookaheadConfig.load()
    mtp_mode = mtp_mod.policy(a.mode)
    max_ctx = -(-(max(len(ids) + total + 64, WARMUP_CTX + 64)) // 64) * 64
    eng = ttft.bring_up(a.mode, max_ctx, prefill=(a.pool == "table"))
    model, pool, cfg = eng["model"], eng["pool"], eng["cfg"]
    model.st.ensure_act(torch.bfloat16)

    corr = None
    tap = None
    if lcfg.on:
        if lcfg.corrected:
            root = os.environ.get("SLOTSTREAM_R")
            st = la.tap_status(Path(root) / "model" if root else Path(os.environ["SLOTSTREAM_MODEL"]).parent)
            if not st["present"]:
                raise SystemExit(f"f6_arm: lookahead mode attention-corrected but the correction file is absent ({st['reason']})")
            corr = fc.TapCorrection.load(Path(os.environ["SLOTSTREAM_MODEL"]) / fc.SHIPPED_REL, hidden=cfg["hidden_size"],
                                         experts=cfg["num_experts"], targets=47)
        tap = la.ForecastTap(corr)
    rb = Readback(model.vocab, forecast=lcfg.on)
    g1 = GraphSet(model, 1, readback=rb, forecast=tap)
    head = g3 = None
    if mtp_mode == "resident":
        packed = Path(os.environ["SLOTSTREAM_PACKED"])
        head = mtp_mod.MTPHead(cfg, packed / "mtp_trunk.safetensors", packed / "mtp_experts.bin", embed=model.seg.embed,
                               lm_head=model.W("lm_head.weight"), device="cuda", max_ctx=max_ctx)
        g3 = GraphSet(model, 3, readback=rb, pool_handle=g1.pool_handle, forecast=tap, record=True, multi=True)
    if lcfg.on:
        pool.enable_prefetch(lcfg.slot_cap, lcfg.min_evictable)
        g1.lookahead = la.Scheduler(pool, lcfg)
    ttft.warm_up(eng, clear=(a.pool == "cold"))
    model.st.reset_inplace()
    plan = PrefillPlan.from_envelope(eng["env"], prefill_pass=4096, read_scope=8192)

    stamps = []
    t0 = time.perf_counter()
    res = generate(model, ids, max_tokens=total, greedy=True, seed=a.seed, eos=None, driver=eng["driver"], plan=plan,
                   on_token=lambda _t: stamps.append(time.perf_counter()), graphs=g1, mtp=head, graphs3=g3)
    torch.cuda.synchronize()
    tps = decode_tps(stamps, a.warmup_tokens, a.tokens)
    la_stats = res["lookahead"]
    out = {"cohort": a.cohort, "pair": a.pair, "prompt_id": req["id"], "prompt_sha256": req["prompt_sha256"],
           "prompt_tokens": len(ids), "warmup_tokens": a.warmup_tokens, "tokens_measured": a.tokens,
           "n_tokens": len(res["tokens"]), "decode_tps": tps, "tokens_sha256": ids_sha256(res["tokens"]),
           "tokens": res["tokens"], "nan": bool(res["nan"]), "prefill_s": round(res["timings"]["prefill_s"], 3),
           "lookahead_mode": lcfg.mode, "mtp_policy": mtp_mode, "seed": a.seed, "mode": a.mode, "slots": eng["slots"],
           "first_token_s": round(stamps[0] - t0, 3) if stamps else None,
           "pool_policy": a.pool, "pool_prefill": eng["pool_prefill"], "pool_cleared_after_warmup": a.pool == "cold",
           "lookahead": la_stats, "mtp": res["mtp"], "pool": {k: v for k, v in pool.stats().items() if isinstance(v, (int, float))},
           "versions": {"torch": torch.__version__, "cuda": torch.version.cuda}}
    ok = (not res["nan"]) and len(res["tokens"]) == total and tps is not None
    if la_stats is not None and la_stats.get("leaked_at_end"):
        ok = False
    if res["mtp"] is not None and res["mtp"].get("leaked_records"):
        ok = False
    if a.pool == "table" and (eng["pool_prefill"] is None or eng["pool_prefill"]["exhausted"] != 0):
        ok = False
    out["ok"] = bool(ok)
    Path(a.out).write_text(json.dumps(out, sort_keys=True) + "\n")
    print(f"{a.cohort}[{a.pair}] {req['id']}: {len(ids)} prompt tokens, {len(res['tokens'])} new, decode {tps} tok/s, "
          f"lookahead {lcfg.mode}, mtp {mtp_mode}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
