#!/bin/bash
# SPDX-License-Identifier: MIT
# Port of Tools/api_robustness.sh from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
# Robustness gates for the serving layer (port of third_party/slotstream/Tools/api_robustness.sh for Linux).
# Every input here either crashed the server or produced silently wrong output upstream. Each one runs against a
# LIVE server (nothing is spawned here) and asserts the process still answers afterwards.
#
# Usage: tools/api_robustness.sh --url http://127.0.0.1:18436
#   SLOTSTREAM_URL                 default for --url
#   SLOTSTREAM_MODEL               model directory (looks for mtp.safetensors next to the weights)
#   SLOTSTREAM_SERVER_PYTHON       command that can import slotstream_spark (default tools/spark/py.sh)
#   TMPDIR                         logs go to ${TMPDIR:-$HOME/slotstream-spark/tmp}
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
URL=${SLOTSTREAM_URL:-}
while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL=${2:?--url needs a value}; shift 2 ;;
    *) echo "api_robustness.sh: unexpected argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$URL" ] || { echo "usage: api_robustness.sh --url http://127.0.0.1:PORT" >&2; exit 2; }
URL=${URL%/}
HOSTPORT=${URL#http://}
PORT=${HOSTPORT##*:}
PASS=0; FAIL=0
LOGDIR=${TMPDIR:-$HOME/slotstream-spark/tmp}
mkdir -p "$LOGDIR"
LOG="$LOGDIR/api_robustness-$PORT.log"
: >"$LOG"
read -r -a SERVER_PYTHON <<<"${SLOTSTREAM_SERVER_PYTHON:-tools/spark/py.sh}"
export PORT

# Cases whose result differs from the macOS server (status, wording or measured quantity), and why. Every cases
# below still asserts something; the entry says what is different. An entry without a reason fails the run.
LINUX_DEVIATIONS="tags-size version vision-wording vision-metric header-case head-length overlength-body model-name"
linux_deviation() {
  case "$1" in
    tags-size) printf %s "/api/tags size is the resident trunk the engine holds (ledger.TRUNK), not the 103.8 GB of the download; the pinned manifest total is accepted too" ;;
    version) printf %s "no binary exists; /api/version is compared with the version the server build configures (ServerConfig)" ;;
    vision-wording) printf %s "text-only server: every image part is a 400 'images are not accepted' before the tokenizer; the macOS wordings 'not fetched', 'usable url', 'decode' and 'raw generation cannot carry images' do not exist, the status (400) is the same" ;;
    vision-metric) printf %s "images_rejected_total counts refused image parts (9 of the 10 image requests here); images:[1,2,3] is a plain validation error and is not counted" ;;
    header-case) printf %s "h11 lowercases response header names, so the CORS header is matched case-insensitively" ;;
    head-length) printf %s "HEAD carries the GET Content-Length (RFC 9110) instead of 0; the check counts body bytes on the wire, which must be 0" ;;
    overlength-body) printf %s "the Ollama error body is {error: text} with no machine code field, so the check asserts status 400, a text error that names the window, and no done:true" ;;
    model-name) printf %s "the served id is the checkpoint directory name; the macOS tag 'qwen3.8-flash-next:4bit' is replaced by the id read from /v1/models, and the untagged alias 'qwen3.8-flash-next' still resolves" ;;
  esac
}
for k in $LINUX_DEVIATIONS; do
  r=$(linux_deviation "$k")
  [ -n "${r// /}" ] || { echo "FAIL  LINUX_DEVIATIONS[$k] has no reason"; exit 1; }
  echo "NOTE  deviation $k: $r"
done

# TOTAL_WEIGHT_BYTES from model.lock.json: the required files; the optional draft head counts when it is present,
# since the macOS /api/tags reported what is on disk.
TOTAL_WEIGHT_BYTES=$(python3 -c 'import json;print(json.load(open("model.lock.json"))["total_bytes"])')
MTPFILE="${SLOTSTREAM_MODEL:-$HOME/slotstream-spark/models/qwen38-flash-next-mlx-4bit}/mtp.safetensors"
[ -f "$MTPFILE" ] && TOTAL_WEIGHT_BYTES=$((TOTAL_WEIGHT_BYTES + $(stat -c%s "$MTPFILE")))
TRUNK_BYTES=$(cd "$ROOT" && "${SERVER_PYTHON[@]}" -c 'from slotstream_spark.memory.ledger import TRUNK; print(TRUNK)' 2>/dev/null || echo 0)

say() { printf '%s\n' "$*"; }
# Liveness is the server answering, not a pid we own: /api/version is probed after every case (ok and bad both).
alive() { curl -fsS --max-time 5 "$URL/api/version" >/dev/null 2>&1; }
ok()  { say "PASS  $1"; PASS=$((PASS+1)); alive || { say "FAIL  the server stopped answering /api/version after: $1"; FAIL=$((FAIL+1)); }; }
bad() { say "FAIL  $1${2:+  ($2)}"; FAIL=$((FAIL+1)); alive || { say "FAIL  the server stopped answering /api/version after: $1"; FAIL=$((FAIL+1)); }; }
for _ in $(seq 1 90); do alive && break; sleep 1; done
alive || { say "FAIL  server at $URL never answered /api/version"; exit 1; }

MODEL=$(curl -s --max-time 20 "$URL/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
[ -n "$MODEL" ] || { say "FAIL  /v1/models did not name a model"; exit 1; }
export MODEL
say "NOTE  serving model id: $MODEL"
metric() { curl -s --max-time 10 "$URL/metrics" | awk -v n="slotstream_$1" '$1==n {print $2; exit}'; }

post() { curl -s --max-time 300 -X POST "$URL$1" -d "$2"; }
post_success() { curl -fsS --max-time 300 -X POST "$URL$1" -d "$2"; }
completed_content() (
  set -o pipefail
  post_success "$1" "$2" | python3 tools/api_generation.py "${3:-ollama-chat}"
)

# Browser origins are loopback-only; arbitrary websites must not be able to
# drive a costly localhost model through CORS/private-network preflight.
C=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 \
    -H 'Origin: https://attacker.example' "$URL/api/version")
[ "$C" = 403 ] && ok "non-loopback browser origin is refused" || bad "foreign Origin returned $C"
O=$(curl -si --max-time 20 -H 'Origin: http://localhost:3000' \
    "$URL/api/version" | tr -d '\r' | awk -F': ' 'tolower($1)=="access-control-allow-origin"{print $2}')
[ "$O" = 'http://localhost:3000' ] && ok "loopback browser origin is allowed exactly" \
    || bad "loopback CORS response wrong" "$O"

C=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -X POST \
    "$URL/api/chat" -d '{"model":"some-other-model","messages":[{"role":"user","content":"hi"}]}')
[ "$C" = 400 ] && ok "wrong model is rejected instead of silently relabeled" || bad "wrong model returned $C"
C=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -X POST \
    "$URL/api/chat" -d '{"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function"}]}')
[ "$C" = 400 ] && ok "unsupported Ollama tools are rejected explicitly" || bad "tools returned $C"
C=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -X POST \
    "$URL/v1/chat/completions" -d '{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"}}}}')
[ "$C" = 400 ] && ok "unsupported OpenAI response_format json_schema is rejected explicitly" || bad "response_format json_schema returned $C"
C=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -X POST \
    "$URL/api/chat" -d '{"stream":1,"messages":[{"role":"user","content":"hi"}]}')
[ "$C" = 400 ] && ok "numeric stream is not mistaken for a JSON boolean" || bad "numeric stream returned $C"
C=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -X POST \
    "$URL/api/chat" -d '{"stream":false,"messages":[{"role":"user","content":"hi"}],"options":{"temperature":"cold"}}')
[ "$C" = 400 ] && ok "wrongly typed sampling options are rejected" || bad "string temperature returned $C"
C=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -X POST \
    "$URL/api/chat" -d '{"stream":false,"messages":[{"role":"user","content":"hi"}],"options":{"temperature":1e300}}')
[ "$C" = 400 ] && ok "numbers that overflow the sampler are rejected" || bad "overflowing temperature returned $C"
C=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -X POST \
    "$URL/v1/chat/completions" -d '{"stream":false,"messages":[{"role":"user","content":"hi","name":"silently-dropped"}]}')
[ "$C" = 400 ] && ok "unsupported message semantics are not silently dropped" || bad "message name returned $C"
C=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -X POST \
    "$URL/v1/chat/completions" -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":0}')
[ "$C" = 400 ] && ok "OpenAI max_tokens 0 cannot become an unbounded generation" || bad "max_tokens 0 returned $C"

# --- crashes: the process must survive each of these ---
R=$(post /api/chat '{"stream":false,"messages":[{"role":"user","content":"Say OK"}],"options":{"seed":-1,"num_predict":4}}')
if alive && [ -n "$R" ]; then ok "seed -1 (Ollama's random default) does not kill the server"
else bad "seed -1 killed the server"; fi

R=$(post /api/chat '{"stream":false,"messages":[{"role":"user","content":"Say OK"}],"options":{"num_predict":-1}}')
N=$(printf '%s' "$R" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("eval_count",-1))' 2>/dev/null)
if alive && [ "${N:-0}" -gt 0 ]; then ok "num_predict -1 (until EOS) generates instead of trapping"
else bad "num_predict -1 killed the server or produced nothing"; fi

python3 - "$PORT" <<'PY' >/dev/null 2>&1
import socket, sys, time
s = socket.create_connection(("127.0.0.1", int(sys.argv[1])))
b = b'{"messages":[{"role":"user","content":"Write a long poem about the sea"}],"options":{"num_predict":200,"temperature":0}}'
s.sendall(b"POST /api/chat HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n" % len(b) + b)
s.recv(200); time.sleep(2); s.close()
PY
sleep 6
if alive; then ok "client disconnecting mid-stream does not kill the server (SIGPIPE)"
else bad "mid-stream disconnect killed the server"; fi

# --- silently wrong output ---
# The streaming contract: concatenating every delta must reproduce the
# non-streamed text exactly, including multi-token characters, scalars that
# merge into an already-sent grapheme (U+FE0F), and stop sequences that
# straddle a token boundary.
if python3 - "$PORT" <<'PYEOF'
import json, sys, http.client
from tools.api_generation import ollama_text
P = int(sys.argv[1])
def call(body, stream):
    b = dict(body); b["stream"] = stream
    c = http.client.HTTPConnection("127.0.0.1", P, timeout=300)
    c.request("POST", "/api/chat", json.dumps(b), {"Content-Type": "application/json"})
    response = c.getresponse()
    assert response.status == 200, "generation HTTP failure"
    d = response.read().decode(); c.close()
    if not stream:
        return ollama_text(json.loads(d))
    return ollama_text([json.loads(l) for l in d.splitlines() if l.strip()], stream=True)
cases = [
 ("plain",           "Say exactly: hello world", {}),
 ("emoji only",      "Reply with exactly these five emoji and nothing else: rocket, fire, star, heart, tree", {}),
 ("emoji + text",    "Write one short sentence about space with exactly one rocket emoji.", {}),
 ("CJK",             "Write the word for cat in Japanese, Chinese and Russian. Just the three words.", {}),
 ("mixed scripts",   "Reply with exactly: caf\u00e9 na\u00efve \u4e2d\u6587 \U0001f600", {}),
 ("1-char stop",     "Count from 1 to 9, digits only, one per line.", {"stop": ["5"]}),
 ("multi-char stop", "Count from 1 to 20 as words, one per line.", {"stop": ["five"]}),
 ("newline stop",    "Write three short lines of a poem.", {"stop": ["\n\n"]}),
 ("stop never hit",  "Say exactly: alpha beta gamma", {"stop": ["ZZZQQQ"]}),
 ("longer output",   "List the first 12 prime numbers, comma separated.", {}),
]
bad = []
for name, prompt, extra in cases:
    o = {"num_predict": 60, "temperature": 0}; o.update(extra)
    body = {"messages": [{"role": "user", "content": prompt}], "options": o}
    ns, st = call(body, False), call(body, True)
    if ns != st:
        bad.append(f"{name}: non-stream={ns!r} streamed={st!r}")
for b in bad:
    print(b, file=sys.stderr)
sys.exit(1 if bad else 0)
PYEOF
then ok "streamed deltas reassemble to the non-streamed text (10 cases)"
else bad "streaming does not reassemble to the non-streamed text"; fi

for BADP in '"top_p":0' '"top_p":-1' '"min_p":1.5'; do
  R_STATUS=0
  R=$(completed_content /api/chat "{\"stream\":false,\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"options\":{\"num_predict\":6,\"temperature\":1,$BADP}}") || R_STATUS=$?
  if [ "$R_STATUS" -ne 0 ]; then bad "out-of-range $BADP did not complete generation"
  else case "$R" in
    *'!!!'*|'') bad "out-of-range $BADP produces garbage" "got '$R'" ;;
    *) ok "out-of-range $BADP falls back sanely (got '$R')" ;;
  esac; fi
done

# An empty prompt is Ollama's documented "load" request (the CLI sends one when
# an interactive session opens). It must be acknowledged with no text and
# without touching the engine: generating from it would read an uninitialized
# tensor, which is what this gate originally guarded against.
R=$(post /api/generate '{"raw":true,"prompt":"","stream":false,"options":{"num_predict":8}}')
case "$R" in
  *'"done_reason":"load"'*) case "$R" in *'"response":""'*) ok "empty prompt is the load request: acknowledged, never answered from an uninitialized tensor" ;;
    *) bad "load acknowledgment carried text" "$R" ;; esac ;;
  *) bad "empty prompt was answered or refused instead of acknowledged" "$R" ;; esac

R_STATUS=0
R=$(completed_content /v1/chat/completions '{"messages":[{"role":"user","content":[{"type":"text","text":"Reply with exactly: ARRAYOK"}]}],"max_tokens":8,"temperature":0}' openai) || R_STATUS=$?
if [ "$R_STATUS" -ne 0 ]; then bad "array-form content did not complete generation"
else case "$R" in *ARRAYOK*) ok "OpenAI array-form content is read, not dropped" ;;
  *) bad "array-form content dropped" "got '$R'" ;; esac; fi

R_STATUS=0
R=$(completed_content /api/chat '{"stream":false,"messages":[{"role":"user","content":"Count from 1 to 9, digits only, one per line."}],"options":{"num_predict":40,"temperature":0,"stop":["4"]}}') || R_STATUS=$?
if [ "$R_STATUS" -ne 0 ]; then bad "stop sequence request did not complete generation"
else case "$R" in *4*) bad "stop sequence ignored" "got '$(printf %s "$R" | tr '\n' ' ')'" ;;
  *) ok "stop sequence honored (got '$(printf %s "$R" | tr '\n' ' ')')" ;; esac; fi

# --- limits and protocol ---
R=$(python3 - "$PORT" <<'PY'
import json,sys,urllib.request
port,big=sys.argv[1],"word "*40000
req=urllib.request.Request(f"http://127.0.0.1:{port}/api/chat",
    data=json.dumps({"stream":False,"messages":[{"role":"user","content":big}]}).encode(),
    headers={"Content-Type":"application/json"})
try:
    with urllib.request.urlopen(req,timeout=120) as response:
        status, body = response.status, response.read().decode()
except urllib.error.HTTPError as e:
    status, body = e.code, e.read().decode()
print(json.dumps({"status": status, "body": json.loads(body)}))
PY
)
if printf '%s' "$R" | python3 -c 'import json,sys; r=json.load(sys.stdin); b=r["body"]; assert r["status"]==400 and isinstance(b.get("error"),str) and "window" in b["error"] and b.get("done") is not True'; then
  ok "over-length prompt is refused with a typed 400, not a silent stall"
else
  bad "over-length prompt lacks the typed HTTP 400 refusal" "$(printf %.180s "$R")"
fi

V=$(curl -s --max-time 20 "$URL/api/version" | python3 -c 'import json,sys;print(json.load(sys.stdin)["version"])')
B_STATUS=0
B=$(cd "$ROOT" && "${SERVER_PYTHON[@]}" -c 'from slotstream_spark.server.app import ServerConfig; print(ServerConfig().version)' 2>/dev/null) || B_STATUS=$?
[ "$B_STATUS" -eq 0 ] && [ "$V" = "$B" ] && ok "/api/version ($V) matches the server build" || bad "/api/version stale or build version check failed" "api=$V build=$B exit=$B_STATUS"

S=$(curl -s --max-time 20 "$URL/api/tags" | python3 -c 'import json,sys;print(json.load(sys.stdin)["models"][0]["size"])')
if [ "$S" = "$TOTAL_WEIGHT_BYTES" ]; then ok "/api/tags size matches the pinned manifest"
elif [ "$S" = "$TRUNK_BYTES" ]; then
  ok "/api/tags size matches the pinned manifest (the resident trunk, see LINUX_DEVIATIONS tags-size)"
else bad "/api/tags size wrong" "$S != $TOTAL_WEIGHT_BYTES (manifest) and != $TRUNK_BYTES (trunk)"; fi

# --- the Ollama CLI's wire format ---
# Its ShowRequest serializes every field, so `ollama run` opens with empty
# name/system/template/options, and its chat may carry keep_alive and a null
# options. Rejecting those broke the CLI in 0.1.8 without any gate noticing.
R=$(post /api/show '{"model":"'"$MODEL"'","name":"","system":"","template":"","options":{},"verbose":false}')
case "$R" in *'"capabilities"'*) ok "/api/show accepts the Ollama CLI request shape and advertises capabilities" ;;
  *) bad "/api/show rejects the Ollama CLI request shape" "$(printf %.90s "$R")" ;; esac
R=$(post /api/show '{"name":"'"$MODEL"'"}')
case "$R" in *'"capabilities"'*) ok "/api/show accepts the deprecated name alias" ;;
  *) bad "/api/show rejects the name alias" "$(printf %.90s "$R")" ;; esac
R=$(post /api/show '{"model":"'"$MODEL"'","system":"You are a pirate"}')
case "$R" in *"not supported"*) ok "/api/show refuses a non-empty system override instead of ignoring it" ;;
  *) bad "/api/show silently accepted a system override" "$(printf %.90s "$R")" ;; esac
R=$(post /api/show '{"model":"'"$MODEL"'","foo":1}')
case "$R" in *"unsupported request field"*) ok "/api/show still rejects unknown fields" ;;
  *) bad "/api/show accepted an unknown field" "$(printf %.90s "$R")" ;; esac
R_STATUS=0
R=$(completed_content /api/chat '{"model":"'"$MODEL"'","stream":false,"keep_alive":"5m","options":null,"messages":[{"role":"user","content":"Reply with exactly: pong"}]}') || R_STATUS=$?
if [ "$R_STATUS" -ne 0 ]; then bad "/api/chat rejects keep_alive or null options without a completed generation"
else case "$R" in ""|*"unsupported"*|*"must be"*) bad "/api/chat rejects keep_alive or null options" "$(printf %.90s "$R")" ;;
  *) ok "/api/chat accepts keep_alive and null options (the CLI's defaults)" ;; esac; fi
# One-shot `ollama run model "prompt"` uses /api/generate with empty suffix/system/template.
R_STATUS=0
R=$(completed_content /api/generate '{"model":"'"$MODEL"'","prompt":"Reply with exactly: pong","suffix":"","system":"","template":"","options":{},"stream":false}' ollama-generate) || R_STATUS=$?
if [ "$R_STATUS" -ne 0 ]; then bad "/api/generate rejects the Ollama CLI one-shot shape without a completed generation"
else case "$R" in ""|*"unsupported"*|*"must be"*) bad "/api/generate rejects the Ollama CLI one-shot shape" "$(printf %.90s "$R")" ;;
  *) ok "/api/generate accepts the Ollama CLI one-shot shape (empty suffix/system/template)" ;; esac; fi
R=$(post /api/generate '{"model":"'"$MODEL"'","prompt":"def f(","suffix":"return 1","stream":false}')
case "$R" in *"not supported"*) ok "/api/generate refuses a non-empty suffix instead of ignoring it" ;;
  *) bad "/api/generate silently accepted a suffix" "$(printf %.90s "$R")" ;; esac
# Interactive `ollama run` opens with Ollama's documented "load" request: an empty prompt.
R=$(post /api/generate '{"model":"'"$MODEL"'","prompt":"","suffix":"","system":"","template":"","options":{}}')
case "$R" in *'"done_reason":"load"'*) ok "/api/generate with an empty prompt is the Ollama load request, acknowledged" ;;
  *) bad "/api/generate rejects the empty-prompt load request" "$(printf %.90s "$R")" ;; esac
R=$(post /api/chat '{"model":"'"$MODEL"'","messages":[]}')
case "$R" in *'"done_reason":"load"'*) ok "/api/chat with no messages is the Ollama load request, acknowledged" ;;
  *) bad "/api/chat rejects the empty-messages load request" "$(printf %.90s "$R")" ;; esac

L=$(python3 - "$PORT" <<'PYEOF'
import socket, sys
s = socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=10)
s.sendall(b"HEAD /api/tags HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
out = b""
while True:
    d = s.recv(65536)
    if not d:
        break
    out += d
head, _, body = out.partition(b"\r\n\r\n")
print(len(body) if head.startswith(b"HTTP/1.1 200") else "bad:" + head[:40].decode(errors="replace"))
PYEOF
)
[ "${L:-x}" = "0" ] && ok "HEAD returns no body" || bad "HEAD returned a body" "body bytes=$L"

C=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -X POST "$URL/api/chat" -d '{not json')
[ "$C" = "400" ] && ok "malformed JSON returns 400" || bad "malformed JSON returned $C"

# --- metadata must not block behind a running generation --------------------
# /api/tags and /api/ps read pool numbers. Taking the generation lock to do it
# made them hang for the length of a request, and because the accept loop also
# waited on the connection semaphore, enough blocked metadata calls stopped the
# server answering anything at all. A polling GUI saw a working server as dead.
if python3 - "$PORT" <<'PYEOF'
import http.client, json, os, socket, sys, threading, time
from tools.api_generation import ollama_text
P = int(sys.argv[1]); M = os.environ["MODEL"]
observed = threading.Event()
content_times = []; generation_errors = []; generation_complete = []
def gen():
    c = http.client.HTTPConnection("127.0.0.1", P, timeout=600)
    try:
        c.request("POST", "/api/chat", json.dumps({"model": M, "stream": True,
            "messages": [{"role": "user", "content": "Write a long poem about the sea."}],
            "options": {"num_predict": 60, "temperature": 0}}), {"Content-Type": "application/json"})
        response = c.getresponse()
        assert response.status == 200, "generation HTTP failure"
        frames = []
        for raw in response:
            if not raw.strip(): continue
            frame = json.loads(raw); frames.append(frame)
            if (frame.get("done") is False and "error" not in frame
                    and isinstance(frame.get("message"), dict)
                    and frame["message"].get("role") == "assistant"
                    and isinstance(frame["message"].get("content"), str)
                    and frame["message"]["content"]):
                content_times.append(time.monotonic()); observed.set()
        ollama_text(frames, stream=True)
        generation_complete.append(True)
    except Exception as e:
        generation_errors.append("generation failed: %s: %s" % (type(e).__name__, e))
    finally:
        c.close(); observed.set()
t = threading.Thread(target=gen, daemon=True); t.start()
problems = []; probe_intervals = []
def timed(method, path, body=None):
    t0 = time.monotonic()
    try:
        c = http.client.HTTPConnection("127.0.0.1", P, timeout=8)
        c.request(method, path, json.dumps(body) if body else None,
                  {"Content-Type": "application/json"})
        r = c.getresponse(); r.read(); c.close()
        status = r.status
    except Exception as e:
        status = type(e).__name__
    end = time.monotonic(); probe_intervals.append((t0, end))
    return status, end - t0
if not observed.wait(60) or not content_times:
    problems.append("generation did not produce content before metadata probes")
else:
    for method, path, body in [("GET", "/api/version", None), ("GET", "/api/tags", None),
                               ("GET", "/api/ps", None), ("GET", "/v1/models", None),
                               ("POST", "/api/show", {"model": M})]:
        st, el = timed(method, path, body)
        if st != 200 or el > 2.0:
            problems.append("%s %s -> %s in %.1fs" % (method, path, st, el))
    hold = []
    try:
        for _ in range(34):   # more than maxConcurrentConnections
            try:
                k = socket.create_connection(("127.0.0.1", P), timeout=5)
                hold.append(k)
                k.sendall(b"GET /api/tags HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            except Exception:
                pass
        st, el = timed("GET", "/api/version")
        # past limit_concurrency uvicorn answers 503 at once; a block would show as a timeout or a slow answer
        if st not in (200, 503) or el > 2.0:
            problems.append("/api/version under connection load -> %s in %.1fs" % (st, el))
    finally:
        for k in hold: k.close()
t.join(timeout=120)
problems.extend(generation_errors)
if t.is_alive() or generation_complete != [True]:
    problems.append("concurrent generation did not finish successfully")
if (not probe_intervals or not content_times
        or content_times[0] > probe_intervals[0][0]
        or content_times[-1] < probe_intervals[-1][1]):
    problems.append("generated content did not span all metadata probes")
for line in problems:
    print(line, file=sys.stderr)
sys.exit(1 if problems else 0)
PYEOF
then ok "metadata endpoints answer during a generation, and the accept loop keeps accepting"
else bad "metadata endpoints block behind generation"; fi

# --- `ollama show` sends the name in `name` and an EMPTY `model` ------------
R=$(post /api/show '{"model":"","system":"","template":"","verbose":false,"options":null,"name":"'"$MODEL"'"}')
case "$R" in *'"capabilities"'*) ok "/api/show accepts an empty model with the name in the alias (ollama show)" ;;
  *) bad "/api/show rejects the ollama show shape" "$(printf %.90s "$R")" ;; esac
R=$(post /api/chat '{"model":"qwen3.8-flash-next","stream":false,"messages":[{"role":"user","content":"Say OK"}],"options":{"num_predict":4}}')
case "$R" in *'"message"'*) ok "an untagged model name resolves to the only model" ;;
  *) bad "untagged model name rejected" "$(printf %.90s "$R")" ;; esac
R=$(post /api/chat '{"model":"'"$MODEL"'","stream":false,"messages":[{"role":"user","content":"hi"}],"options":{"num_ctx":4096}}')
case "$R" in *"unsupported options field"*) ok "a semantic Ollama knob (num_ctx) is still refused, never silently dropped" ;;
  *) bad "num_ctx was silently accepted" "$(printf %.90s "$R")" ;; esac

# --- OpenAI clients send null for "unset", and defaults on every call -------
for F in '"max_tokens":null' '"stop":null' '"temperature":null' '"seed":null' '"stream_options":null'; do
  C=$(curl -s -o /dev/null -w '%{http_code}' --max-time 120 -X POST "$URL/v1/chat/completions" \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":4,$F}")
  [ "$C" = 200 ] && ok "/v1 treats $F as unset" || bad "/v1 rejected $F" "$C"
done
for F in '"n":1' '"frequency_penalty":0' '"user":"u1"' '"logprobs":false' '"logit_bias":{}' '"tools":[]' '"response_format":{"type":"text"}' '"response_format":{"type":"json_object"}'; do
  C=$(curl -s -o /dev/null -w '%{http_code}' --max-time 120 -X POST "$URL/v1/chat/completions" \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":4,$F}")
  [ "$C" = 200 ] && ok "/v1 accepts the no-op default $F" || bad "/v1 rejected the no-op default $F" "$C"
done
for F in '"n":2' '"frequency_penalty":0.5' '"logprobs":true' '"tools":[{"type":"function"}]' '"response_format":{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"}}}'; do
  C=$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 -X POST "$URL/v1/chat/completions" \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],$F}")
  [ "$C" = 400 ] && ok "/v1 still refuses the real feature $F" || bad "/v1 accepted $F" "$C"
done

# --- think: reasoning belongs in `thinking`, not in the answer --------------
R_STATUS=0
R=$(post_success /api/chat '{"model":"'"$MODEL"'","stream":false,"think":true,"messages":[{"role":"user","content":"What is 2+2?"}],"options":{"num_predict":80,"temperature":0}}') || R_STATUS=$?
if [ "$R_STATUS" -eq 0 ] && printf '%s' "$R" | python3 -c '
import json, sys
from tools.api_generation import ollama_text
response = json.load(sys.stdin)
ollama_text(response)
m = response["message"]
c, t = m.get("content", ""), m.get("thinking", "")
sys.exit(0 if t.strip() and c.strip() and "</think>" not in c and "</think>" not in t else 1)'; then
  ok "think:true splits reasoning into message.thinking and leaves the answer clean"
else bad "think:true leaked reasoning into content" "$(printf %.120s "$R")"; fi

# --- a short reply streams token by token -----------------------------------
if python3 - "$PORT" <<'PYEOF'
import http.client, json, os, sys
from tools.api_generation import ollama_text
P = int(sys.argv[1])
c = http.client.HTTPConnection("127.0.0.1", P, timeout=600)
c.request("POST", "/api/chat", json.dumps({"model": os.environ["MODEL"], "stream": True,
    "messages": [{"role": "user", "content": "Count from 1 to 8, digits only, comma separated."}],
    "options": {"num_predict": 16, "temperature": 0}}), {"Content-Type": "application/json"})
response = c.getresponse()
assert response.status == 200, "generation HTTP failure"
objs = [json.loads(l) for l in response.read().decode().splitlines() if l.strip()]
c.close()
ollama_text(objs, stream=True)
deltas = [o for o in objs if not o["done"] and o["message"]["content"]]
evals = objs[-1]["eval_count"]
print("%d content deltas for %d tokens" % (len(deltas), evals), file=sys.stderr)
sys.exit(0 if len(deltas) >= max(3, evals // 2) else 1)
PYEOF
then ok "a short reply arrives as per-token deltas, not one batched chunk"
else bad "streaming is still batched into multi-token bursts"; fi

# --- an unseeded request is not one fixed stream ----------------------------
# 40 tokens, not 12: the model opens this prompt with the same confident phrase
# every time ("Here is a fun fact for you:") and only diverges once it reaches
# the fact itself, so a 12-token window reported a working sampler as a stuck
# one about as often as not.
FUN='{"model":"'"$MODEL"'","stream":false,"messages":[{"role":"user","content":"Tell me a fun fact."}],"options":{"num_predict":40,"temperature":1.0}}'
A_STATUS=0; A=$(completed_content /api/chat "$FUN") || A_STATUS=$?
B_STATUS=0; B=$(completed_content /api/chat "$FUN") || B_STATUS=$?
D_STATUS=0; D=$(completed_content /api/chat "$FUN") || D_STATUS=$?
if [ "$A_STATUS" -ne 0 ] || [ "$B_STATUS" -ne 0 ] || [ "$D_STATUS" -ne 0 ]; then bad "unseeded requests did not complete generation"
elif [ "$A" = "$B" ] && [ "$B" = "$D" ]; then bad "unseeded requests replay one fixed stream" "$(printf %.60s "$A")"
else ok "unseeded requests vary, as the API documents"; fi
SEEDED='{"model":"'"$MODEL"'","stream":false,"messages":[{"role":"user","content":"Tell me a fun fact."}],"options":{"num_predict":12,"temperature":1.0,"seed":7}}'
S1_STATUS=0; S1=$(completed_content /api/chat "$SEEDED") || S1_STATUS=$?
S2_STATUS=0; S2=$(completed_content /api/chat "$SEEDED") || S2_STATUS=$?
[ "$S1_STATUS" -eq 0 ] && [ "$S2_STATUS" -eq 0 ] && [ "$S1" = "$S2" ] && ok "an explicit seed still reproduces exactly" || bad "seeded requests are not reproducible completed generations"

# --- HTTP: routing, framing, and honest status codes ------------------------
C=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 "$URL/api/tags?x=1")
[ "$C" = 200 ] && ok "a query string does not 404 the route" || bad "query string returned $C"
C=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -I "$URL/api/version")
[ "$C" = 200 ] && ok "HEAD on a real path is 200" || bad "HEAD /api/version returned $C"
C=$(curl -s -o /dev/null -w '%{http_code}' --max-time 20 -I "$URL/nope")
[ "$C" = 404 ] && ok "HEAD on an unknown path is 404, not a blanket 200" || bad "HEAD /nope returned $C"
R=$(python3 - "$PORT" <<'PYEOF'
import os, socket, sys
P = int(sys.argv[1])
def raw(payload):
    s = socket.create_connection(("127.0.0.1", P), timeout=10)
    try:
        s.sendall(payload)
    except OSError:
        return "send failed"
    out = b""
    try:
        while True:
            d = s.recv(65536)
            if not d: break
            out += d
    except Exception:
        pass
    s.close()
    return out.split(b"\r\n", 1)[0].decode(errors="replace") if out else "no response"
body = ('{"model":"%s","messages":[{"role":"user","content":"hi"}]}' % os.environ["MODEL"]).encode()
print("chunked:", raw(b"POST /api/chat HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n"
                      + b"%x\r\n" % len(body) + body + b"\r\n0\r\n\r\n"))
# Past Server.maxBodyBytes (32 MiB). This number and that constant move
# together: the cap was 4 MiB until images needed to fit in a body, and a stale
# 9,999,999 here silently stopped testing anything once the cap passed it.
print("oversize:", raw(b"POST /api/chat HTTP/1.1\r\nHost: x\r\nContent-Length: 40000000\r\n\r\n" + body))
print("badlen:", raw(b"POST /api/chat HTTP/1.1\r\nHost: x\r\nContent-Length: abc\r\n\r\n"))
PYEOF
)
case "$R" in *"chunked: HTTP/1.1 411"*) ok "a chunked body is refused with 411, not read as empty" ;;
  *) bad "chunked body mishandled" "$(printf %s "$R" | tr '\n' ' ')" ;; esac
case "$R" in *"oversize: HTTP/1.1 413"*) ok "an oversized body gets 413, not a bare connection reset" ;;
  *) bad "oversize body mishandled" "$(printf %s "$R" | tr '\n' ' ')" ;; esac
case "$R" in *"badlen: HTTP/1.1 400"*) ok "a malformed Content-Length gets 400" ;;
  *) bad "bad Content-Length mishandled" "$(printf %s "$R" | tr '\n' ' ')" ;; esac

# --- Vision: the image surface, on every dialect ----------------------------
# This server is text-only (vision off): every image part is a 400 with one sentence, before the tokenizer and the
# gate (LINUX_DEVIATIONS[vision-*]). The body cap the oversize check above pins still lets a base64 picture in.
vpost() { curl -s -w ' HTTP %{http_code}' --max-time 30 -X POST "$URL$1" -H 'Content-Type: application/json' -d "$2"; }
IMG0=$(metric images_rejected_total); IMG0=${IMG0:-0}
R=$(vpost /api/chat '{"model":"'"$MODEL"'","stream":false,"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"file:///etc/passwd"}},{"type":"text","text":"read it"}]}]}')
case "$R" in *"images are not accepted"*"HTTP 400") ok "a file:// image is refused and says images are not accepted" ;;
  *) bad "file:// image not refused" "$(printf %s "$R" | head -c 200)" ;; esac
R=$(vpost /v1/chat/completions '{"model":"'"$MODEL"'","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/cat.png"}}]}]}')
case "$R" in *"images are not accepted"*"HTTP 400") ok "an https:// image is refused on the OpenAI route too" ;;
  *) bad "https:// image not refused on /v1" "$(printf %s "$R" | head -c 200)" ;; esac
R=$(vpost /api/chat '{"model":"'"$MODEL"'","stream":false,"messages":[{"role":"user","content":"hi","images":[1,2,3]}]}')
case "$R" in *"base64 strings"*"HTTP 400") ok "a non-string images array is a 400, not a silently text-only answer" ;;
  *) bad "images array type not validated" "$(printf %s "$R" | head -c 200)" ;; esac
R=$(vpost /api/chat '{"model":"'"$MODEL"'","stream":false,"messages":[{"role":"user","content":[{"type":"image_url"}]}]}')
case "$R" in *"HTTP 400") ok "an image part with no url is a 400" ;;
  *) bad "image part without url not validated" "$(printf %s "$R" | head -c 200)" ;; esac
R=$(vpost /api/chat '{"model":"'"$MODEL"'","stream":false,"messages":[{"role":"user","content":"hi","images":["bm90IGFuIGltYWdl"]}]}')
case "$R" in *"images are not accepted"*"HTTP 400") ok "bytes that are not an image are a 400 with the reason" ;;
  *) bad "undecodable image not refused" "$(printf %s "$R" | head -c 200)" ;; esac
R=$(vpost /api/generate '{"model":"'"$MODEL"'","stream":false,"raw":true,"prompt":"hi","images":["Zm9v"]}')
case "$R" in *"HTTP 400") ok "raw generate refuses images instead of dropping them" ;;
  *) bad "raw + images not refused" "$(printf %s "$R" | head -c 200)" ;; esac
# one image case per dialect: 400, the sentence, and the rejection counter moves
B64=iVBORw0KGgo=
R=$(vpost /v1/messages '{"model":"'"$MODEL"'","max_tokens":10,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"'$B64'"}}]}]}')
case "$R" in *"images are not accepted"*"HTTP 400") ok "image: Anthropic messages refuses an image block with a 400" ;;
  *) bad "Anthropic image block not refused" "$(printf %s "$R" | head -c 200)" ;; esac
R=$(vpost /v1/chat/completions '{"model":"'"$MODEL"'","messages":[{"role":"user","content":[{"type":"text","text":"what"},{"type":"image_url","image_url":{"url":"data:image/png;base64,'$B64'"}}]}]}')
case "$R" in *"images are not accepted"*"HTTP 400"*|*"invalid_image"*"HTTP 400") ok "image: OpenAI chat refuses a data-URL image with a 400" ;;
  *) bad "OpenAI chat image not refused" "$(printf %s "$R" | head -c 200)" ;; esac
R=$(vpost /v1/responses '{"model":"'"$MODEL"'","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,'$B64'"}]}]}')
case "$R" in *"images are not accepted"*"HTTP 400") ok "image: Responses refuses an input_image with a 400" ;;
  *) bad "Responses input_image not refused" "$(printf %s "$R" | head -c 200)" ;; esac
R=$(vpost /api/chat '{"model":"'"$MODEL"'","stream":false,"messages":[{"role":"user","content":"x","images":["'$B64'"]}]}')
case "$R" in *"images are not accepted"*"HTTP 400") ok "image: Ollama chat refuses an images array with a 400" ;;
  *) bad "Ollama chat images not refused" "$(printf %s "$R" | head -c 200)" ;; esac
IMG1=$(metric images_rejected_total); IMG1=${IMG1:-0}
[ $((IMG1 - IMG0)) -eq 9 ] \
  && ok "every refused image part moved slotstream_images_rejected_total ($IMG0 -> $IMG1)" \
  || bad "images_rejected_total did not count the 9 refused image parts" "$IMG0 -> $IMG1"
curl -s --max-time 20 "$URL/v1/models" | grep -q '"created"' \
  && ok "/v1/models carries created" || bad "/v1/models has no created field"
R=$(curl -s --max-time 120 -X POST "$URL/v1/chat/completions" \
    -d '{"model":"'"$MODEL"'","stream":true,"max_tokens":6,"temperature":0,"messages":[{"role":"user","content":"Say HI"}]}' \
    | sed -n 's/^data: //p' | head -1)
printf '%s' "$R" | grep -q '"role":"assistant"' && ok "the first SSE delta announces the role" \
  || bad "first SSE delta has no role" "$(printf %.90s "$R")"

alive && ok "server still up after every probe" || bad "server died during the run"
say ""
say "robustness: passed $PASS, failed $FAIL"
say "$PASS passed, $FAIL failed"
[ $FAIL -eq 0 ]
