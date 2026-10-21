#!/bin/bash
# Spark: the context-window campaign (the engine at a 131,072-token window). Start it with
#   tools/spark/jobs.sh start w131k-engine --heavy w131k-engine auto 10800 -- bash bench/campaign/w131k_engine_job.sh
# Under the guard, from a frozen copy of the sources (snap.sh) and inside tools/capture.sh, like f9_engine_job.sh, so
# the capture holds the guard record, the 1 s MemAvailable series and the production window for the impact report.
#
# Phases (W131K_PHASES, default "envelope,d5"; a failing bar is a result, tools/decide.py judges it):
#   envelope  tools/prefill_envelope.py --contexts 65536,131072, never --write        -> $CAP/envelope/result.json
#   d5        run_campaign.py --mode D5 --samples 5 --window 131072 with the cells decode-8k, ttft-cold-8k and
#             ttft-cold-78k (bench/prompts/ttft_long.json)                              -> $CAP/d5-<cell>/result.json
# $CAP/w131k_job.json keeps the return code of every phase. Judges: tools/decide.py w131k-campaign and w131k-envelope,
# tools/impact_report.py --label w131k-engine. W131K_WINDOW, W131K_SAMPLES and W131K_CONTEXTS override the defaults for
# a re-run of the tooling; the record is the run with the defaults.
if [ -z "${SLOTSTREAM_SRC:-}" ]; then
  exec bash tools/spark/snap.sh w131k-engine bash bench/campaign/w131k_engine_job.sh "$@"
fi
if [ -z "${W131K_CAPTURED:-}" ]; then
  export W131K_CAPTURED=1
  exec bash tools/capture.sh w131k-engine -- bash bench/campaign/w131k_engine_job.sh "$@"
fi
set -u
PY=tools/spark/py.sh
: "${CAP:?CAP is not set (the guard exports it)}"
WINDOW=${W131K_WINDOW:-131072}
SAMPLES=${W131K_SAMPLES:-5}
CONTEXTS=${W131K_CONTEXTS:-65536,131072}
RC_FILE=$CAP/w131k_job.rc
: > "$RC_FILE"
rec() { echo "$1=$2" >> "$RC_FILE"; echo "w131k-engine: $1 rc=$2"; }

PHASES=${W131K_PHASES:-envelope,d5}
want() { case ",$PHASES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

if want envelope; then
  mkdir -p "$CAP/envelope"
  CAP="$CAP/envelope" $PY tools/prefill_envelope.py --contexts "$CONTEXTS" 2>&1 | tee "$CAP/envelope/envelope.log"
  rec envelope "${PIPESTATUS[0]}"
fi

if want d5; then
  $PY bench/campaign/run_campaign.py --mode D5 --samples "$SAMPLES" --window "$WINDOW" \
    --cells decode-8k,ttft-cold-8k,ttft-cold-78k 2>&1 | tee "$CAP/campaign-d5.log"
  rec campaign_d5 "${PIPESTATUS[0]}"
fi

python3 - "$RC_FILE" "$CAP/w131k_job.json" <<'PY'
import json, sys
rc = dict(l.strip().split("=", 1) for l in open(sys.argv[1]) if "=" in l)
json.dump({k: int(v) for k, v in rc.items()}, open(sys.argv[2], "w"), indent=1, sort_keys=True)
PY
missing=0
if want envelope; then
  [ -s "$CAP/envelope/result.json" ] || { echo "w131k-engine: missing output $CAP/envelope/result.json"; missing=1; }
fi
if want d5; then
  for f in "$CAP/d5-decode-8k/result.json" "$CAP/d5-ttft-cold-8k/result.json" "$CAP/d5-ttft-cold-78k/result.json"; do
    [ -s "$f" ] || { echo "w131k-engine: missing output $f"; missing=1; }
  done
fi
exit $missing
