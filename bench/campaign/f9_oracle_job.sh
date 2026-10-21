#!/bin/bash
# Spark: the F9 oracle perplexity job (plan step 95c). Start it with
#   tools/spark/jobs.sh start f9-oracle --heavy f9-oracle 16384 9000 -- bash bench/campaign/f9_oracle_job.sh
# Same wrapping as f9_engine_job.sh: a frozen copy of the sources (snap.sh) inside tools/capture.sh, under the guard.
# Scores the first 32 wikitext windows of the engine run with the transformers oracle in the checkpoint dtype. The state
# file under $R/state/f9-oracle/ makes it resumable: a cut job is restarted with the same command and continues.
# F9_ENGINE_CAP (default: the capture named by $R/runs/LAST-f9-engine) must hold quality/ppl.json; its shas are checked.
if [ -z "${SLOTSTREAM_SRC:-}" ]; then
  exec bash tools/spark/snap.sh f9-oracle bash bench/campaign/f9_oracle_job.sh "$@"
fi
if [ -z "${F9_CAPTURED:-}" ]; then
  export F9_CAPTURED=1
  exec bash tools/capture.sh f9-oracle -- bash bench/campaign/f9_oracle_job.sh "$@"
fi
set -u
PY=tools/spark/py.sh
: "${CAP:?CAP is not set (the guard exports it)}"
R=${SLOTSTREAM_R:?SLOTSTREAM_R is not set}
Q=$CAP/quality
mkdir -p "$Q"
ENGINE_CAP=${F9_ENGINE_CAP:-$(cat "$R/runs/LAST-f9-engine")}
python3 - "$CAP/f9_job.json" "$ENGINE_CAP" <<'PY'
import json, sys
json.dump({"engine_capture": sys.argv[2]}, open(sys.argv[1], "w"), indent=1, sort_keys=True)
PY
bash tools/quality/fetch_wikitext.sh || exit 1
$PY tools/quality/oracle_ppl.py --windows "${F9_ORACLE_WINDOWS:-32}" --out "$Q" --engine-ppl "$ENGINE_CAP/quality/ppl.json" 2>&1 | tee "$Q/oracle_ppl.log"
exit "${PIPESTATUS[0]}"
