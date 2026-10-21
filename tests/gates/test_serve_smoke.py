"""Step 84 gate: the real engine behind the four dialects, started by `tools/serve_gates.sh <port> smoke`.

One answer per dialect; a cold ~31K-token Anthropic stream gets `ping` events at most every 10.5 s until the first
content; turn 2 of a ~9K-token conversation reports cache_read_input_tokens > 0 (a 31K one cannot be retained in memory); count_tokens answers in under a second
while that stream is still reading its prompt. Stdlib HTTP only (the Spark environment has no test client).
"""
import http.client
import json
import os
import threading
import time

import pytest

pytestmark = pytest.mark.gate

PORT = int(os.environ.get("SERVE_PORT", "18436"))
PING_GAP_MAX_S = 10.5
COLD_TOKENS = 31_000
CACHE_TOKENS = 9_000          # the memory tier holds ~36K tokens; a 31K conversation belongs to the disk tier (step 86)
STATE = {}


def post(path, body, timeout=900):
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    c.request("POST", path, json.dumps(body), {"content-type": "application/json"})
    r = c.getresponse()
    raw = r.read()
    c.close()
    return r.status, json.loads(raw)


def model_id():
    if "model" not in STATE:
        c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
        c.request("GET", "/v1/models")
        STATE["model"] = json.loads(c.getresponse().read())["data"][0]["id"]
    return STATE["model"]


def test_version_answers():
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
    c.request("GET", "/api/version")
    assert c.getresponse().status == 200


def test_one_answer_per_dialect():
    q = "Say hello in one short sentence."
    s, b = post("/v1/messages", {"model": model_id(), "max_tokens": 24, "messages": [{"role": "user", "content": q}]})
    assert s == 200 and b["type"] == "message" and b["usage"]["output_tokens"] > 0, b
    s, b = post("/v1/chat/completions", {"model": model_id(), "max_tokens": 24, "messages": [{"role": "user", "content": q}]})
    assert s == 200 and b["choices"][0]["finish_reason"] in ("stop", "length") and b["usage"]["completion_tokens"] > 0, b
    s, b = post("/v1/responses", {"model": model_id(), "max_output_tokens": 24, "input": q})
    assert s == 200 and b["status"] in ("completed", "incomplete") and b["usage"]["output_tokens"] > 0, b
    s, b = post("/api/chat", {"model": model_id(), "stream": False, "options": {"num_predict": 24},
                              "messages": [{"role": "user", "content": q}]})
    assert s == 200 and b["done"] is True and b["eval_count"] > 0, b


def prompt_body(tokens):
    from bench import ttft
    from slotstream_spark.engine.tokenizer import Tokenizer
    tok = Tokenizer.from_checkpoint(os.environ["SLOTSTREAM_MODEL"])
    _ids, body = ttft.build_ids(tok, ttft.load_prompts(), tokens)
    return body


def cold_prompt():
    return prompt_body(COLD_TOKENS)


def test_cold_stream_pings_and_count_tokens_stay_responsive():
    body = cold_prompt()
    STATE["body"] = body
    req = {"model": model_id(), "max_tokens": 16, "stream": True, "messages": [{"role": "user", "content": body}]}
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=900)
    t0 = time.monotonic()
    conn.request("POST", "/v1/messages", json.dumps(req), {"content-type": "application/json"})
    resp = conn.getresponse()
    assert resp.status == 200
    counted = {}

    def count_during():
        time.sleep(3.0)
        t = time.monotonic()
        s, b = post("/v1/messages/count_tokens", {"model": model_id(), "messages": [{"role": "user", "content": "hello there"}]}, 30)
        counted.update(status=s, seconds=time.monotonic() - t, body=b)

    th = threading.Thread(target=count_during)
    th.start()
    stamps, text, first_content, last, gaps = [], "", None, t0, []
    event = None
    usage = {}
    while True:
        line = resp.readline()
        if not line:
            break
        now = time.monotonic()
        line = line.decode().rstrip("\n")
        if line.startswith("event: "):
            event = line[7:]
            if first_content is None:
                gaps.append(now - last)
                last = now
                if event == "content_block_delta":
                    first_content = now - t0
        elif line.startswith("data: ") and event == "message_delta":
            usage = json.loads(line[6:]).get("usage", {})
    conn.close()
    th.join()
    pings = sum(1 for g in gaps if g is not None)
    assert first_content is not None, "no content arrived"
    assert max(gaps) <= PING_GAP_MAX_S, f"gap between events before the first content: {max(gaps):.2f} s ({gaps})"
    assert counted.get("status") == 200 and counted["seconds"] < 1.0, counted
    STATE["first_content_s"] = first_content
    print(f"cold stream: first content {first_content:.1f} s, {pings} events before it, max gap {max(gaps):.2f} s; "
          f"count_tokens {counted['seconds']:.3f} s")


def test_turn_two_reads_the_cache():
    body = prompt_body(CACHE_TOKENS)
    first = [{"role": "user", "content": body}]
    s, b = post("/v1/messages", {"model": model_id(), "max_tokens": 16, "messages": first})
    assert s == 200, b
    reply = "".join(c.get("text", "") for c in b["content"] if c.get("type") == "text") or "Understood."
    msgs = first + [{"role": "assistant", "content": reply},
                    {"role": "user", "content": "Summarize the text above in one sentence."}]
    t = time.monotonic()
    s, b = post("/v1/messages", {"model": model_id(), "max_tokens": 16, "messages": msgs})
    assert s == 200, b
    print(f"turn 2: {time.monotonic() - t:.1f} s, usage {b['usage']}")
    assert b["usage"]["cache_read_input_tokens"] > 0, b["usage"]
