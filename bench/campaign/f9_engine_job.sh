#!/bin/bash
# Spark: the F9 engine qualification job (plan step 93). Start it with
#   tools/spark/jobs.sh start f9-engine --heavy f9-engine auto 16200 -- bash bench/campaign/f9_engine_job.sh
# Under the guard, from a frozen copy of the sources (snap.sh) and inside tools/capture.sh, so the capture holds the guard
# record, the 1 s MemAvailable series, the per-container pressure series and the production window for the impact report.
#
# Phases, one model process at a time (a failing bar is a result, not a failure of the job; tools/decide.py judges it):
#   quality   ppl_wikitext.py (145 windows), teacher_forced.py (8 reserved prompts, fp32 oracle pair), quality_probe.sh (15/15,
#             a server of its own)                                                       -> $CAP/quality/
#   campaign  run_campaign.py D5 with N=5 (decode-8k, cold 2K/8K/32K TTFT, follow-up)    -> $CAP/d5-<cell>/result.json
#   campaign  run_campaign.py D1 with N=3, informative                                    -> $CAP/d1-<cell>/result.json
# $CAP/f9_job.json keeps the return code of every phase. The job exits 1 only when a phase left no output at all.
# F9_PHASES (default "ppl,tf,probe,d5,d1") selects phases, for re-runs after a harness fix; a run that skips one is not a
# complete qualification capture. The probe server runs in its own session and is stopped as a group, so nothing it started
# outlives the job (the guard's postcheck refuses a job whose process group is still alive).
if [ -z "${SLOTSTREAM_SRC:-}" ]; then
  exec bash tools/spark/snap.sh f9-engine bash bench/campaign/f9_engine_job.sh "$@"
fi
if [ -z "${F9_CAPTURED:-}" ]; then
  export F9_CAPTURED=1
  exec bash tools/capture.sh f9-engine -- bash bench/campaign/f9_engine_job.sh "$@"
fi
set -u
PY=tools/spark/py.sh
: "${CAP:?CAP is not set (the guard exports it)}"
Q=$CAP/quality
mkdir -p "$Q"
RC_FILE=$CAP/f9_job.rc
: > "$RC_FILE"
rec() { echo "$1=$2" >> "$RC_FILE"; echo "f9-engine: $1 rc=$2"; }

PHASES=${F9_PHASES:-ppl,tf,probe,d5,d1}
want() { case ",$PHASES," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }

bash tools/quality/fetch_wikitext.sh || { rec fetch_wikitext $?; exit 1; }
rec fetch_wikitext 0

if want ppl; then
  $PY tools/quality/ppl_wikitext.py --mode D5 --out "$Q" 2>&1 | tee "$Q/ppl.log"
  rec ppl "${PIPESTATUS[0]}"
fi

if want tf; then
  $PY tools/quality/teacher_forced.py all 2>&1 | tee "$Q/teacher_forced.log"
  rec teacher_forced "${PIPESTATUS[0]}"
fi

if want probe; then
PROBE_PORT=18438
setsid $PY -m slotstream_spark.cli serve --mode D5 --port $PROBE_PORT > "$Q/probe-serve.log" 2>&1 &
SERVER=$!
up=0
for _ in $(seq 1 900); do
  kill -0 $SERVER 2>/dev/null || break
  if curl -fsS --max-time 2 "http://127.0.0.1:$PROBE_PORT/health" >/dev/null 2>&1; then up=1; break; fi
  sleep 1
done
if [ $up = 1 ]; then
  bash tools/quality/quality_probe.sh --url "http://127.0.0.1:$PROBE_PORT" 2>&1 | tee "$Q/probe.txt"
  rec probe "${PIPESTATUS[0]}"
else
  echo "f9-engine: the probe server did not come up" | tee "$Q/probe.txt"
  rec probe 3
fi
kill -TERM -- "-$SERVER" 2>/dev/null
for _ in $(seq 1 90); do kill -0 $SERVER 2>/dev/null || break; sleep 1; done
kill -KILL -- "-$SERVER" 2>/dev/null
wait $SERVER 2>/dev/null
fi

if want d5; then
  $PY bench/campaign/run_campaign.py --mode D5 --samples 5 2>&1 | tee "$CAP/campaign-d5.log"
  rec campaign_d5 "${PIPESTATUS[0]}"
fi
if want d1; then
  $PY bench/campaign/run_campaign.py --mode D1 --samples 3 2>&1 | tee "$CAP/campaign-d1.log"
  rec campaign_d1 "${PIPESTATUS[0]}"
fi

python3 - "$RC_FILE" "$CAP/f9_job.json" <<'PY'
import json, sys
rc = dict(l.strip().split("=", 1) for l in open(sys.argv[1]) if "=" in l)
json.dump({k: int(v) for k, v in rc.items()}, open(sys.argv[2], "w"), indent=1, sort_keys=True)
PY
missing=0
for f in "$Q/ppl.json" "$Q/teacher_forced.json" "$Q/probe.txt" "$CAP/d5-decode-8k/result.json" "$CAP/d5-ttft-cold-8k/result.json"; do
  [ -s "$f" ] || { echo "f9-engine: missing output $f"; missing=1; }
done
exit $missing
