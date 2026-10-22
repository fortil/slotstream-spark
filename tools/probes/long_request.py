#!/usr/bin/env python3
"""Probe: one long conversation through the real server, without streaming, so an engine error comes back as a body.

  tools/spark/run.sh --heavy long78k auto 1800 -- tools/spark/py.sh tools/probes/long_request.py [--tokens 78000]
      [--window 131072] [--turns 2] [--max-tokens 8] [--port 18439]

Starts `cli serve` at the window (a fresh prefix directory, like a campaign sample), cuts the prompt from
bench/prompts/ttft_long.json, sends it to /v1/chat/completions with stream=false and prints the status, the body (an
error body in full) and the elapsed seconds; then the follow-up turn of bench/ttft.py on the same server, whose
`usage.reused_prefix_tokens` says what the prefix cache resumed. The server log's tail is printed at the end and the
whole thing goes to $CAP/result.json when CAP is set. Exit 0 when every turn answered 200.
"""
import argparse
import http.client
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def post_json(port, path, body, timeout):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    t0 = time.perf_counter()
    c.request("POST", path, json.dumps(body), {"content-type": "application/json"})
    r = c.getresponse()
    raw = r.read()
    c.close()
    try:
        doc = json.loads(raw)
    except ValueError:
        doc = {"raw": raw[:4000].decode("utf-8", "replace")}
    return r.status, doc, round(time.perf_counter() - t0, 2)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=78000)
    ap.add_argument("--window", type=int, default=131072)
    ap.add_argument("--turns", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--port", type=int, default=18439)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--prompts", default=str(ROOT / "bench" / "prompts" / "ttft_long.json"))
    a = ap.parse_args(argv)
    from bench import ttft
    from bench.campaign.run_campaign import Server, cut_at_most
    from slotstream_spark.engine.tokenizer import Tokenizer
    cap = Path(os.environ["CAP"]) if os.environ.get("CAP") else Path(tempfile.mkdtemp(prefix="long-request-"))
    tok = Tokenizer.from_checkpoint(os.environ["SLOTSTREAM_MODEL"])
    prompts = ttft.load_prompts(a.prompts)
    ids, body = cut_at_most(tok, tok.encode(prompts["text"]), a.tokens)
    prefix_dir = Path(tempfile.mkdtemp(prefix="long-prefix-", dir=os.environ.get("SLOTSTREAM_R") and str(Path(os.environ["SLOTSTREAM_R"]) / "tmp")))
    srv = Server("D5", a.port, cap, prefix_dir, a.window)
    res = {"tool": "tools/probes/long_request.py", "tokens": len(ids), "window": a.window, "turns": [], "ok": False}
    try:
        health = srv.start()
        res["chunks"] = int(health.get("chunks", 0))
        model = srv.model_id()
        msgs = [{"role": "user", "content": body}]
        for turn in range(a.turns):
            if turn:
                msgs = msgs + [{"role": "assistant", "content": prompts["reply"]}, {"role": "user", "content": prompts["followup"]}]
            st, doc, s = post_json(a.port, "/v1/chat/completions",
                                   {"model": model, "messages": msgs, "stream": False, "max_tokens": a.max_tokens, "temperature": 0}, a.timeout)
            row = {"turn": turn + 1, "status": st, "seconds": s, "usage": doc.get("usage"),
                   "error": doc.get("error") or doc.get("raw"),
                   "text": (doc.get("choices") or [{}])[0].get("message", {}).get("content") if st == 200 else None}
            res["turns"].append(row)
            print(json.dumps(row)[:3000], flush=True)
            if st != 200:
                break
        res["ok"] = all(t["status"] == 200 for t in res["turns"]) and len(res["turns"]) == a.turns
    except Exception as e:  # noqa: BLE001 - the finding is the error
        res["error"] = f"{type(e).__name__}: {e}"
        print(res["error"], flush=True)
    finally:
        srv.stop()
    log = cap / "serve.log"
    if log.exists():
        tail = log.read_text(errors="replace").splitlines()[-40:]
        res["serve_log_tail"] = tail
        print("--- serve.log tail ---", flush=True)
        for line in tail:
            print(line[:300], flush=True)
    (cap / "result.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print("wrote", cap / "result.json", flush=True)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
