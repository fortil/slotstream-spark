#!/usr/bin/env python3
"""Probe: the pool pins left by a long sweep prefill when the decode graphs were attached before the request.

  tools/spark/run.sh --heavy pins-probe auto 1500 -- tools/spark/py.sh tools/probes/pins_after_prefill.py
      [--window 131072] [--sizes 8192,32768,49152,65536,78000] [--arm attached|detached|both]

Builds the engine the way `cli serve` does (pool prefill from the table, sweep driver, then GraphSet(model, 1), whose
capture leaves the graphs attached) and, per prompt size, runs one `generate()` of two tokens. The `attached` arm calls
`graphs.attach()` before the prefill (the state a fresh server is in for its first request); the `detached` arm calls
`graphs.detach()` (the state every later request starts from). Before the first decode step it prints the readback
base, the sequence word, `seg.gen`, the host progress word and the pinned slot count; the outcome is `ok` or the
`slotio.Exhausted` message. One JSON line per (arm, size) and a summary to $CAP/result.json when CAP is set.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def parse_sizes(s):
    return [int(x) for x in s.split(",") if x]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=131072)
    ap.add_argument("--sizes", default="8192,32768,49152,65536,78000")
    ap.add_argument("--arm", choices=("attached", "detached", "both"), default="attached")
    ap.add_argument("--prompts", default=str(ROOT / "bench" / "prompts" / "ttft_long.json"))
    a = ap.parse_args(argv)
    import slotio
    from bench import ttft
    from bench.campaign.run_campaign import cut_at_most
    from slotstream_spark.engine.generate import generate
    from slotstream_spark.engine.prefill import PrefillPlan
    from slotstream_spark.engine.tokenizer import Tokenizer
    from slotstream_spark.runtime.graphs import GraphSet
    cap = Path(os.environ["CAP"]) if os.environ.get("CAP") else None
    tok = Tokenizer.from_checkpoint(os.environ["SLOTSTREAM_MODEL"])
    prompts = ttft.load_prompts(a.prompts)
    text_ids = tok.encode(prompts["text"])
    eng = ttft.bring_up("D5", a.window, prefill=True, window=a.window)
    model, pool, driver, env = eng["model"], eng["pool"], eng["driver"], eng["env"]
    print(f"pool prefill: {eng['pool_prefill']}", flush=True)
    graphs = GraphSet(model, 1)
    seg, rb = model.seg, graphs.rb
    plan = PrefillPlan.from_envelope(env, prefill_pass=4096, read_scope=8192)
    rows = []

    def state(tag):
        return {tag: {"rb_base": int(rb.base), "attached": bool(rb.attached), "seq": int(rb.seq[0]),
                      "seg_gen": int(seg.gen), "progress": int(seg.progress[0]),
                      "pinned": int(pool.stats()["pinned"]), "slots": int(pool.pool.slots)}}

    orig_run = graphs._run

    def run_logged(toks, eager=False, timings=None):
        if not rows[-1].get("before_decode"):
            rows[-1].update(state("before_decode"))
            print(json.dumps(rows[-1]["before_decode"]), flush=True)
        return orig_run(toks, eager=eager, timings=timings)

    graphs._run = run_logged
    arms = ("attached", "detached") if a.arm == "both" else (a.arm,)
    for arm in arms:
        for n in parse_sizes(a.sizes):
            ids, _body = cut_at_most(tok, text_ids, n)
            model.st.reset_inplace()
            if arm == "attached":
                graphs.attach()
            else:
                graphs.detach()
            row = {"arm": arm, "tokens": len(ids)}
            row.update(state("before_prefill"))
            rows.append(row)
            t0 = time.perf_counter()
            try:
                res = generate(model, ids, max_tokens=2, greedy=True, eos=None, driver=driver, plan=plan, graphs=graphs)
                row["outcome"] = "ok"
                row["tokens_out"] = len(res["tokens"])
            except slotio.Exhausted as e:
                row["outcome"] = f"Exhausted: {e}"
            except Exception as e:  # noqa: BLE001 - the finding is the error
                row["outcome"] = f"{type(e).__name__}: {e}"
            row["seconds"] = round(time.perf_counter() - t0, 1)
            row.update(state("after"))
            graphs.detach()
            print(json.dumps(row), flush=True)
    if cap is not None:
        (cap / "result.json").write_text(json.dumps({"tool": "tools/probes/pins_after_prefill.py", "window": a.window,
                                                     "rows": rows}, indent=1, sort_keys=True) + "\n")
        print("wrote", cap / "result.json", flush=True)
    return 0 if all(r["outcome"] == "ok" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
