#!/bin/bash
# Spark: F9 diagnostic. The same llamacpp_ppl.py protocol as f9_reference_job.sh phase 4, on the ORIGINAL Q4_1 GGUF
# (--weights q41) instead of our repack. Start it with
#   tools/spark/jobs.sh start f9-diag --heavy f9-diag 57344 5400 --memcg slotstream-spark-f9-ref -- bash bench/campaign/f9_diag_job.sh
# Same engine ids/windows check as the reference job; writes $CAP/f9-diag/{windows.jsonl,result.json}.
if [ -z "${SLOTSTREAM_SRC:-}" ]; then
  exec bash tools/spark/snap.sh f9-diag bash bench/campaign/f9_diag_job.sh "$@"
fi
if [ -z "${F9_CAPTURED:-}" ]; then
  export F9_CAPTURED=1
  exec bash tools/capture.sh f9-diag -- bash bench/campaign/f9_diag_job.sh "$@"
fi
set -u
. tools/spark/env.sh
PY=tools/spark/py.sh
: "${CAP:?CAP is not set (the guard exports it)}"
R=${SLOTSTREAM_R:-$R}
ENGINE_CAP=${F9_ENGINE_CAP:-$(cat "$R/runs/LAST-f9-engine")}
mkdir -p "$CAP/f9-diag"
$PY tools/quality/llamacpp_ppl.py --engine-ppl "$ENGINE_CAP/quality/ppl.json" --windows 32 --weights q41 \
  --out "$CAP/f9-diag" --memfree-log "$CAP/ppl-memfree.csv" --unit slotstream-spark-f9-ref 2>&1 | tee "$CAP/llamacpp_ppl.log"
exit "${PIPESTATUS[0]}"
