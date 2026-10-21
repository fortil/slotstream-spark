#!/usr/bin/env python3
"""Profile of a batch-style workload against the front (SPK-3, step A3).

  python bench/workload_profile.py --base-url http://127.0.0.1:18434 [--out FILE] [--cold-timeout 600]   Spark

Replays the shape of the calls a nightly batch client sends to an OpenAI-compatible chat endpoint, one at a time:
  cold     one 1-token request on a cold socket (engine stopped); retried on 503 + Retry-After until 200:
           the time to engine ready
  json16k  a ~16,000-token prompt, response_format json_object, max_tokens 6000
  json25k  a ~25,500-token prompt, response_format json_object, max_tokens 6000
  text8k   a ~8,000-token prompt, plain text, max_tokens 6000
Every call is non-streaming, like the client; no temperature. Each prompt starts with a fresh nonce line, so the
prefix cache cannot match an earlier call (a cold prompt, never a pre-warmed one). The prompt text is the frozen
span in bench/prompts/ttft.json; the chars-per-token ratio is calibrated from `usage.prompt_tokens` of a short
probe, and the real prompt_tokens of each call is recorded.

Per call: prompt_tokens, completion_tokens, total seconds, a decode tok/s floor
(completion_tokens / total_s, prefill included), finish_reason, and whether the
call went over the attempt timeout (300 s). Pass rule, fixed before measuring: every warm call total <= 270 s at
max_tokens 6000. The cold start is reported, not gated.

The result goes to stdout and to --out as json; `verdict.json` for the capture comes from tools/capture.sh.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "bench" / "prompts" / "ttft.json"
MODEL = os.environ.get("WORKLOAD_MODEL", "qwen3.8-flash-next")
ATTEMPT_TIMEOUT_S = 300
PASS_TOTAL_S = 270
CALLS = (
    # name, target prompt tokens, json mode, max_tokens
    ("json16k", 16000, True, 6000),
    ("json25k", 25500, True, 6000),
    ("text8k", 8000, False, 6000),
)
PROBE_CHARS = 4000
INSTRUCTION = "Read the material and answer about it."


def post(base_url, body, timeout, key=None):
    """POST /v1/chat/completions. Returns (status, retry_after or None, parsed json or None)."""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(base_url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(), headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, None, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            doc = json.loads(raw)
        except ValueError:
            doc = None
        ra = e.headers.get("Retry-After")
        return e.code, (float(ra) if ra else None), doc


def make_body(content, json_mode, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


def make_prompt(text, chars):
    """A prompt of `chars` characters of the frozen text behind a unique nonce line (no prefix-cache hit)."""
    nonce = uuid.uuid4().hex
    return f"[{nonce}]\n{INSTRUCTION}\n\n{text[:chars]}"


def calibrate(base_url, text, key=None, timeout=ATTEMPT_TIMEOUT_S, _post=post):
    """chars per token from a short probe (max_tokens 1)."""
    content = make_prompt(text, PROBE_CHARS)
    st, _, doc = _post(base_url, make_body(content, False, 1), timeout, key)
    if st != 200 or not doc:
        raise RuntimeError(f"calibration probe failed: status {st}")
    return len(content) / doc["usage"]["prompt_tokens"]


def cold_start(base_url, key=None, timeout_s=600, _post=post, _sleep=time.sleep, _clock=time.monotonic):
    """Seconds from the first request to the first 200, retrying on 503 with the Retry-After the front sends."""
    t0 = _clock()
    attempts = 0
    while True:
        attempts += 1
        try:
            st, ra, _ = _post(base_url, make_body("ping", False, 1), ATTEMPT_TIMEOUT_S, key)
        except (urllib.error.URLError, OSError):
            st, ra = 0, None
        if st == 200:
            return {"seconds": round(_clock() - t0, 3), "attempts": attempts}
        if _clock() - t0 > timeout_s:
            raise RuntimeError(f"engine not ready after {timeout_s} s (last status {st})")
        _sleep(ra if ra else 5.0)


def run_call(base_url, name, content, json_mode, max_tokens, key=None, _post=post, _clock=time.monotonic):
    t0 = _clock()
    st, _, doc = _post(base_url, make_body(content, json_mode, max_tokens), ATTEMPT_TIMEOUT_S + 60, key)
    total = _clock() - t0
    if st != 200 or not doc:
        return {"name": name, "status": st, "total_s": round(total, 3), "ok": False}
    u = doc["usage"]
    out = {"name": name, "status": st, "ok": True, "json_mode": json_mode, "max_tokens": max_tokens,
           "prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"],
           "total_s": round(total, 3), "finish_reason": doc["choices"][0]["finish_reason"],
           "over_attempt_timeout": total > ATTEMPT_TIMEOUT_S}
    out["tok_s_floor"] = round(u["completion_tokens"] / total, 2) if total > 0 else 0.0
    return out


def verdict(cold, calls):
    """(PASS|FAIL, one line). Pass rule: every call completed and total <= PASS_TOTAL_S; the cold start is reported."""
    bad = [c for c in calls if not c["ok"]]
    if bad:
        return "FAIL", f"call(s) failed: {', '.join(c['name'] for c in bad)}"
    slow = [c for c in calls if c["total_s"] > PASS_TOTAL_S]
    over = [c for c in calls if c["over_attempt_timeout"]]
    parts = ", ".join(f"{c['name']} {c['total_s']:.1f} s" for c in calls)
    if slow:
        extra = f"; {len(over)} over {ATTEMPT_TIMEOUT_S} s" if over else ""
        return "FAIL", f"over {PASS_TOTAL_S} s: {', '.join(c['name'] for c in slow)} ({parts}){extra}"
    return "PASS", f"cold start {cold['seconds']:.1f} s, all calls <= {PASS_TOTAL_S} s ({parts})"


def profile(base_url, text, key=None, cold_timeout=600, _post=post, _sleep=time.sleep, _clock=time.monotonic):
    cold = cold_start(base_url, key, cold_timeout, _post, _sleep, _clock)
    cpt = calibrate(base_url, text, key, _post=_post)
    calls = []
    for name, target, json_mode, max_tokens in CALLS:
        chars = min(int(target * cpt), len(text))
        content = make_prompt(text, chars)
        calls.append(run_call(base_url, name, content, json_mode, max_tokens, key, _post, _clock))
    v, line = verdict(cold, calls)
    return {"cold_start": cold, "chars_per_token": round(cpt, 3), "calls": calls, "calls_over_300s":
            sum(1 for c in calls if c.get("over_attempt_timeout")), "verdict": v, "verdict_line": line}


def load_text(path=PROMPTS):
    return json.loads(Path(path).read_text())["text"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18434")
    ap.add_argument("--out")
    ap.add_argument("--cold-timeout", type=float, default=600)
    ap.add_argument("--key-env", default="SLOTSTREAM_API_KEY", help="name of the env var holding the bearer key")
    a = ap.parse_args(argv)
    res = profile(a.base_url, load_text(), os.environ.get(a.key_env) or None, a.cold_timeout)
    s = json.dumps(res, indent=1)
    print(s)
    if a.out:
        Path(a.out).write_text(s + "\n")
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
