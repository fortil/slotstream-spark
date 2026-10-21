#!/bin/bash
# SPDX-License-Identifier: MIT
# Port of Tools/quality_probe.sh from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
# Behavioural quality probe: 15 short questions against a live server (port of third_party/slotstream/Tools/quality_probe.sh).
#   quality_probe.sh --url http://127.0.0.1:PORT
# Each question goes to /api/chat (temperature 0, 60 new tokens, thinking off by the template) and the answer must match
# an extended regex. It catches gross quantization or kernel damage; it does not compare against another model.
# Prints one PASS/FAIL line per item and `quality probe: passed N, failed M`; exit 0 only with 15/15.
set -uo pipefail
URL=${SLOTSTREAM_URL:-}
while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL=${2:?--url needs a value}; shift 2 ;;
    *) echo "quality_probe.sh: unexpected argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$URL" ] || { echo "usage: quality_probe.sh --url http://127.0.0.1:PORT" >&2; exit 2; }
URL=${URL%/}
PASS=0; FAIL=0
MODEL=$(curl -s --max-time 30 "$URL/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null)
[ -n "$MODEL" ] || { echo "quality_probe.sh: no model id from $URL/v1/models" >&2; exit 2; }

ask() { # ask <prompt> <extended-regex the answer must match> <label>
  local out
  out=$(curl -s --max-time 600 "$URL/api/chat" -d "$(python3 -c '
import json,sys; print(json.dumps({"model":sys.argv[2],
 "messages":[{"role":"user","content":sys.argv[1]}],"stream":False,
 "options":{"temperature":0,"num_predict":60}}))' "$1" "$MODEL")" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["message"]["content"])' 2>/dev/null)
  if printf '%s' "$out" | grep -Eqi "$2"; then
    echo "PASS  $3"; PASS=$((PASS+1))
  else
    echo "FAIL  $3 -> $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-90)"; FAIL=$((FAIL+1))
  fi
}

echo "== factual recall =="
ask "What is the capital of France? One word." "Paris" "capital of France"
ask "Who wrote the play Hamlet? Just the name." "Shakespeare" "author of Hamlet"
ask "What is the chemical symbol for gold? Just the symbol." "\bAu\b" "symbol for gold"
ask "How many continents are there? Just the number." "\b7\b|seven" "continent count"

echo "== arithmetic and reasoning =="
ask "What is 17 times 23? Just the number." "391" "17x23"
ask "If a train leaves at 2pm and arrives 3 hours 30 minutes later, what time does it arrive? Just the time." "5:?30|5\.30|half past five" "elapsed time"
ask "A shop sells pens for 3 dollars each. How much do 7 pens cost? Just the number." "\b21\b" "multi-step arithmetic"
ask "Sort these numbers ascending, comma separated, nothing else: 12, 3, 47, 8" "3,? *8,? *12,? *47" "sorting"
ask "Which is larger, 9.11 or 9.9? Answer with just the number." "9\.9\b" "decimal comparison"

echo "== instruction following =="
ask "Reply with exactly the word BANANA and nothing else." "^BANANA\.?$" "exact-word obedience"
ask "List three primary colors, comma separated, no other words." "red.*blue|blue.*red" "list format"
ask "Answer only YES or NO: is the sky blue on a clear day?" "^YES\.?$" "yes/no obedience"

echo "== language and code =="
ask "Translate to Spanish, nothing else: The house is big." "casa.*grande|grande.*casa" "translation"
ask "Write a Python one-liner that returns the sum of a list named xs. Code only." "sum\(xs\)" "python one-liner"
ask "Complete: The mitochondria is the ___ of the cell. One word." "powerhouse" "cloze completion"

echo
echo "quality probe: passed $PASS, failed $FAIL"
[ "$FAIL" -eq 0 ] && [ "$PASS" -eq 15 ]
