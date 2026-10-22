#!/usr/bin/env python3
"""Probe: one long conversation through a running server or router, without streaming, as the desktop app sends it.

  tools/spark/py.sh tools/probes/router_request.py --url http://127.0.0.1:14000 --model supervisor-reasoning
      [--tokens 78000] [--turns 2] [--max-tokens 8] [--timeout 1500]

Cuts the prompt from bench/prompts/ttft_long.json with the checkpoint's tokenizer, posts it to <url>/v1/chat/completions
with stream=false, then the follow-up turn of bench/ttft.py on the same conversation; prints the status, the elapsed
seconds, the usage (prompt_tokens, cached_tokens or reused_prefix_tokens) and the first characters of the reply, one
JSON line per turn. ROUTER_API_KEY in the environment goes out as a bearer token and is never printed. Exit 0 when every
turn answered 200. Nothing is started or stopped: the engine behind the URL must be up or reachable through its front.
"""
import argparse
import http.client
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def post_json(url, path, body, timeout):
    u = urlsplit(url)
    conn = (http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection)(u.hostname, u.port, timeout=timeout)
    headers = {"content-type": "application/json"}
    key = os.environ.get("ROUTER_API_KEY")
    if key:
        headers["authorization"] = f"Bearer {key}"
    t0 = time.perf_counter()
    conn.request("POST", (u.path.rstrip("/") + path), json.dumps(body), headers)
    r = conn.getresponse()
    raw = r.read()
    conn.close()
    try:
        doc = json.loads(raw)
    except ValueError:
        doc = {"raw": raw[:2000].decode("utf-8", "replace")}
    return r.status, doc, round(time.perf_counter() - t0, 2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=78000)
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--prompts", default=str(ROOT / "bench" / "prompts" / "ttft_long.json"))
    a = ap.parse_args(argv)
    from bench import ttft
    from bench.campaign.run_campaign import cut_at_most
    from slotstream_spark.engine.tokenizer import Tokenizer
    tok = Tokenizer.from_checkpoint(os.environ["SLOTSTREAM_MODEL"])
    prompts = ttft.load_prompts(a.prompts)
    ids, body = cut_at_most(tok, tok.encode(prompts["text"]), a.tokens)
    msgs = [{"role": "user", "content": body}]
    ok = True
    for turn in range(a.turns):
        if turn:
            msgs = msgs + [{"role": "assistant", "content": prompts["reply"]}, {"role": "user", "content": prompts["followup"]}]
        st, doc, s = post_json(a.url, "/v1/chat/completions",
                               {"model": a.model, "messages": msgs, "stream": False, "max_tokens": a.max_tokens, "temperature": 0}, a.timeout)
        usage = doc.get("usage") or {}
        row = {"turn": turn + 1, "status": st, "seconds": s, "prompt_tokens": usage.get("prompt_tokens"),
               "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", usage.get("reused_prefix_tokens")),
               "error": doc.get("error") or doc.get("raw"),
               "text": ((doc.get("choices") or [{}])[0].get("message") or {}).get("content", "")[:80] if st == 200 else None}
        print(json.dumps(row), flush=True)
        ok = ok and st == 200
        if st != 200:
            break
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
