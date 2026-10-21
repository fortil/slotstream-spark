#!/bin/bash
# serve_gates.sh <port> <suite>
# Starts the D5 server under the current guard run (127.0.0.1 only), waits up to 300 s for /api/version, runs the
# suite and stops the server. Suites: smoke (step 84); api (step 87: api_robustness.sh and openai_tools_gate.py);
# live-acceptance (step 99): cold 8K TTFT <= 30 s, D5 decode >= 17 tok/s, the api suite, images refused in four dialects,
# and the 15-question quality probe. The perf part runs first, on the freshly started server, so the 8K prompt is cold.
# Runs on the Spark from $R/src, inside `tools/spark/run.sh --heavy ... -- bash tools/serve_gates.sh PORT SUITE`.
set -u
PORT=${1:?usage: serve_gates.sh <port> <suite>}
SUITE=${2:?usage: serve_gates.sh <port> <suite>}
cd "$(dirname "$0")/.."
case "$SUITE" in
  smoke) ;;
  api) ;;
  live-acceptance) ;;
  *) echo "serve_gates: unknown suite '$SUITE'" >&2; exit 2 ;;
esac

DIR=${CAP:-${TMPDIR:-$HOME/slotstream-spark/tmp}}
mkdir -p "$DIR"
LOG="$DIR/serve-$PORT.log"
URL="http://127.0.0.1:$PORT"

tools/spark/py.sh -m slotstream_spark.cli serve --mode "${SLOTSTREAM_MODE:-D5}" --port "$PORT" >"$LOG" 2>&1 &
SERVER=$!

stop_server() {
  kill -TERM "$SERVER" 2>/dev/null || return 0
  for _ in $(seq 1 60); do
    kill -0 "$SERVER" 2>/dev/null || return 0
    sleep 0.5
  done
  kill -KILL "$SERVER" 2>/dev/null
}
trap stop_server EXIT

up=0
for _ in $(seq 1 300); do
  if ! kill -0 "$SERVER" 2>/dev/null; then
    echo "serve_gates: the server exited before it was ready; tail of $LOG:" >&2
    tail -n 30 "$LOG" >&2
    exit 1
  fi
  if curl -fsS --max-time 2 "$URL/api/version" >/dev/null 2>&1; then up=1; break; fi
  sleep 1
done
if [ "$up" != 1 ]; then
  echo "serve_gates: /api/version did not answer in 300 s; tail of $LOG:" >&2
  tail -n 30 "$LOG" >&2
  exit 1
fi

# live-acceptance, part 1: cold 8K TTFT and decode floor, measured over HTTP on the server just started.
live_perf() {
  local out=$1 script=$2
  cat > "$script" <<'PYPERF'
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
PYPERF
  tools/spark/py.sh "$script" "$URL" "$out"
}

# live-acceptance, part 2: one image request per dialect, each must be a 400 that says images are not accepted.
live_images() {
  local model b64=iVBORw0KGgo= r bad=0 n=0 post
  model=$(curl -s --max-time 20 "$URL/v1/models" | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
  post() { curl -s -w ' HTTP %{http_code}' --max-time 30 -X POST "$URL$1" -H 'Content-Type: application/json' -d "$2"; }
  chk() { n=$((n + 1)); case "$2" in *"images are not accepted"*"HTTP 400"*|*"invalid_image"*"HTTP 400"*) echo "PASS  images $1 -> 400" ;; *) echo "FAIL  images $1: $(printf %s "$2" | head -c 200)"; bad=$((bad + 1)) ;; esac; }
  r=$(post /v1/chat/completions '{"model":"'"$model"'","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,'$b64'"}}]}]}'); chk openai-chat "$r"
  r=$(post /api/chat '{"model":"'"$model"'","stream":false,"messages":[{"role":"user","content":"x","images":["'$b64'"]}]}'); chk ollama-chat "$r"
  r=$(post /v1/messages '{"model":"'"$model"'","max_tokens":10,"messages":[{"role":"user","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"'$b64'"}}]}]}'); chk anthropic-messages "$r"
  r=$(post /v1/responses '{"model":"'"$model"'","input":[{"type":"message","role":"user","content":[{"type":"input_image","image_url":"data:image/png;base64,'$b64'"}]}]}'); chk openai-responses "$r"
  echo "images: $((n - bad)) of $n refused with a 400"
  [ "$bad" = 0 ]
}

rc=0
case "$SUITE" in
  smoke)
    SERVE_PORT=$PORT tools/spark/py.sh -m pytest -q -s -m gate tests/gates/test_serve_smoke.py
    rc=$?
    ;;
  api)
    # SERVE_API_PARTS narrows a re-run while a gate is being fixed; the step-87 run uses both parts.
    PARTS=${SERVE_API_PARTS:-"robustness tools"}
    OUT="$DIR/api-gates-$PORT"
    mkdir -p "$OUT"
    r1=0; r2=0
    case " $PARTS " in *" robustness "*)
      TMPDIR=$DIR SLOTSTREAM_URL=$URL bash tools/api_robustness.sh --url "$URL" 2>&1 | tee "$OUT/robustness.txt"
      r1=${PIPESTATUS[0]} ;;
    esac
    case " $PARTS " in *" tools "*)
      tools/spark/py.sh tools/openai_tools_gate.py --url "$URL" --output "$OUT/openai-tools.jsonl" 2>&1 | tee "$OUT/openai-tools.txt"
      r2=${PIPESTATUS[0]} ;;
    esac
    echo "serve_gates: api_robustness rc=$r1, openai_tools_gate rc=$r2 (raw tools-gate log: $OUT/openai-tools.jsonl)"
    [ "$r1" = 0 ] && [ "$r2" = 0 ] || rc=1
    ;;
  live-acceptance)
    OUT="$DIR/live-acceptance-$PORT"
    mkdir -p "$OUT"
    rp=0; ra=0; ri=0; rq=0
    live_perf "$OUT/perf.json" "$OUT/live_perf.py" 2>&1 | tee "$OUT/perf.txt"
    rp=${PIPESTATUS[0]}
    TMPDIR=$DIR SLOTSTREAM_URL=$URL bash tools/api_robustness.sh --url "$URL" 2>&1 | tee "$OUT/robustness.txt"
    r1=${PIPESTATUS[0]}
    tools/spark/py.sh tools/openai_tools_gate.py --url "$URL" --output "$OUT/openai-tools.jsonl" 2>&1 | tee "$OUT/openai-tools.txt"
    r2=${PIPESTATUS[0]}
    [ "$r1" = 0 ] && [ "$r2" = 0 ] || ra=1
    live_images 2>&1 | tee "$OUT/images.txt"
    ri=${PIPESTATUS[0]}
    bash tools/quality/quality_probe.sh --url "$URL" 2>&1 | tee "$OUT/probe.txt"
    rq=${PIPESTATUS[0]}
    echo "serve_gates: perf rc=$rp, api+tools rc=$ra, images rc=$ri, probe rc=$rq"
    [ "$rp" = 0 ] && [ "$ra" = 0 ] && [ "$ri" = 0 ] && [ "$rq" = 0 ] || rc=1
    ;;
esac
echo "serve_gates: $SUITE rc=$rc (server log: $LOG)"
exit $rc
