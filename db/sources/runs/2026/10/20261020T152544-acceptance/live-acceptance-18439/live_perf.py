import json, sys, time, urllib.request
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "bench"))
url, out = sys.argv[1], Path(sys.argv[2])
TTFT_MAX_S, DECODE_MIN_TPS, PROMPT_TOKENS = 30.0, 17.0, 8192


def get(path):
    with urllib.request.urlopen(url + path, timeout=60) as r:
        return json.loads(r.read())


def stream_chat(model, content, max_tokens):
    """-> (ttft s, completion tokens, seconds from first to last content chunk, prompt tokens)."""
    body = json.dumps({"model": model, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": max_tokens,
                       "temperature": 0, "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = last = None
    chunks, usage = 0, {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                delta = ch.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    now = time.monotonic()
                    first = first if first is not None else now
                    last = now
                    chunks += 1
    if first is None:
        raise RuntimeError("no content in the stream")
    toks = int(usage.get("completion_tokens") or chunks)
    return first - t0, toks, last - first, int(usage.get("prompt_tokens") or 0)


model = get("/v1/models")["data"][0]["id"]
stream_chat(model, "Say HI", 8)                                    # warm-up on the fresh server
from slotstream_spark.engine.tokenizer import Tokenizer              # noqa: E402
import os                                                           # noqa: E402
import ttft as bt                                                   # noqa: E402
tok = Tokenizer.from_checkpoint(os.environ["SLOTSTREAM_MODEL"])
prompts = bt.load_prompts()
_, body = bt.build_ids(tok, prompts, PROMPT_TOKENS)
ttft, _, _, ptok = stream_chat(model, body, 4)
ok_ttft = ttft <= TTFT_MAX_S and ptok >= PROMPT_TOKENS - 64
print("cold 8K TTFT %.2f s (limit %.0f s), prompt tokens %d" % (ttft, TTFT_MAX_S, ptok))
_, toks, secs, _ = stream_chat(model, "Write a short story about a lighthouse keeper, at least 300 words.", 192)
tps = (toks - 1) / secs if secs > 0 and toks > 1 else 0.0
ok_dec = tps >= DECODE_MIN_TPS
print("D5 decode %.2f tok/s over %d tokens (floor %.0f tok/s)" % (tps, toks, DECODE_MIN_TPS))
out.write_text(json.dumps({"ttft_s": ttft, "prompt_tokens": ptok, "decode_tps": tps, "decode_tokens": toks,
                           "ok_ttft": ok_ttft, "ok_decode": ok_dec}, indent=1) + "\n")
sys.exit(0 if ok_ttft and ok_dec else 1)
