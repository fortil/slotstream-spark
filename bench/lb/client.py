#!/usr/bin/env python3
"""Streaming client for llama-server /v1/chat/completions (stdlib only).

  client.py --port 8091 --ids short,2k,8k --reps 5 --warmup 1 --out client.jsonl [--model NAME]

Each request is temperature 0, seed 1234, max_tokens 256, stream true. One JSON line per
measured rep: {prompt_id, rep, ttft_s, n, decode_tps, prompt_tokens, timings, cache_n, prompt_n, prefill_tps, cold}.
decode_tps = (n - 1) / (t_last - t_first) over the content chunks.
--no-cache-prompt sends "cache_prompt": false; a cold prefill is valid only when cache_n == 0 (cold == true).
prefill_tps = prompt_n / prompt_ms * 1000 from the server timings (null without them).
"""
import argparse
import http.client
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
try:
    from slotstream_spark.bench.prompts import build_prompt  # noqa: F401
except ImportError:  # run as a script without the package installed
    sys.path.insert(0, str(HERE.parents[1]))
    from slotstream_spark.bench.prompts import build_prompt  # noqa: F401
def run_once(host, port, prompt, model="local", max_tokens=256, timeout=1800, cache_prompt=True):
    req = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True,
           "temperature": 0, "seed": 1234, "max_tokens": max_tokens,
           "stream_options": {"include_usage": True}}
    if not cache_prompt:
        req["cache_prompt"] = False
    body = json.dumps(req).encode()
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    t0 = time.perf_counter()
    conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:200]!r}")
    stamps, usage, timings = [], None, None
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            break
        obj = json.loads(data)
        if obj.get("usage"):
            usage = obj["usage"]
        if obj.get("timings"):
            timings = obj["timings"]
        for ch in obj.get("choices", []):
            d = ch.get("delta", {})
            if d.get("content") or d.get("reasoning_content"):
                stamps.append(time.perf_counter())
    conn.close()
    n = len(stamps)
    rec = {"ttft_s": (stamps[0] - t0) if stamps else None, "n": n,
           "decode_tps": ((n - 1) / (stamps[-1] - stamps[0])) if n > 1 and stamps[-1] > stamps[0] else None,
           "prompt_tokens": (usage or {}).get("prompt_tokens") or (timings or {}).get("prompt_n"),
           "timings": timings}
    t = timings or {}
    rec["cache_n"] = t.get("cache_n")
    rec["prompt_n"] = t.get("prompt_n")
    pms = t.get("prompt_ms")
    rec["prefill_tps"] = (t["prompt_n"] / pms * 1000) if t.get("prompt_n") is not None and pms else None
    rec["cold"] = (rec["cache_n"] == 0)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--prompts", default=str(HERE / "prompts.json"))
    ap.add_argument("--ids", default="short,2k,8k")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--model", default="local")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--no-cache-prompt", action="store_true", help='send "cache_prompt": false (cold prefill)')
    ap.add_argument("--out", default="-")
    a = ap.parse_args(argv)
    prompts = json.loads(Path(a.prompts).read_text())
    out = sys.stdout if a.out == "-" else open(a.out, "a")
    for pid in a.ids.split(","):
        text = build_prompt(prompts[pid])
        for w in range(a.warmup):
            run_once(a.host, a.port, text, a.model, a.max_tokens, cache_prompt=not a.no_cache_prompt)
        for rep in range(a.reps):
            rec = run_once(a.host, a.port, text, a.model, a.max_tokens, cache_prompt=not a.no_cache_prompt)
            rec = {"prompt_id": pid, "rep": rep, **rec}
            out.write(json.dumps(rec) + "\n")
            out.flush()
    if out is not sys.stdout:
        out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
