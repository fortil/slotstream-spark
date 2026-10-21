#!/bin/bash
# Spark: F9c diagnostic. Build BASE with the tensors matching TAKE replaced by DONOR's (tools/gguf_overlay.py), then score the
# first WINDOWS windows of the engine's ids with llama.cpp on it. Start it with
#   tools/spark/jobs.sh start NAME --heavy f9c 57344 5400 --memcg slotstream-spark-f9-ref -- bash bench/campaign/f9c_overlay_job.sh NAME WINDOWS TAKE_REGEX [BASE DONOR [BF16_REGEX]]
# The 6th argument (or F9C_BF16) is a regex: the matching donor tensors are written as BF16 (dequantized) instead of Q4_1.
# BASE defaults to the original Q4_1 GGUF, DONOR to $R/model/q4_1-exact. Writes $CAP/NAME/{windows.jsonl,result.json}; the
# overlay file lives in $R/tmp/f9c/NAME and is removed at the end unless F9C_KEEP=1.
if [ -z "${SLOTSTREAM_SRC:-}" ]; then
  exec bash tools/spark/snap.sh "f9c-$1" bash bench/campaign/f9c_overlay_job.sh "$@"
fi
if [ -z "${F9_CAPTURED:-}" ]; then
  export F9_CAPTURED=1
  exec bash tools/capture.sh "f9c-$1" -- bash bench/campaign/f9c_overlay_job.sh "$@"
fi
set -u
. tools/spark/env.sh
PY=tools/spark/py.sh
: "${CAP:?CAP is not set (the guard exports it)}"
R=${SLOTSTREAM_R:-$R}
LB=${LB/#\~/$HOME}
NAME=$1; WIN=$2; TAKE=$3
BASE=${4:-$LB/gguf/Qwen3.8-Flash-Next-Q4_1}
DONOR=${5:-$R/model/q4_1-exact}
F9C_BF16=${6:-${F9C_BF16:-}}
ENGINE_CAP=${F9_ENGINE_CAP:-$(cat "$R/runs/LAST-f9-engine")}
OUTD=$R/tmp/f9c/$NAME
mkdir -p "$OUTD" "$CAP/$NAME"
avail=$(df -BG --output=avail "$R" | tail -1 | tr -dc 0-9)
[ "${avail:-0}" -ge 150 ] || { echo "f9c: ${avail:-?} GB free at $R, 150 GB needed"; exit 1; }
BF=(); [ -n "${F9C_BF16:-}" ] && BF=(--bf16 "$F9C_BF16")
$PY tools/gguf_overlay.py --base "$BASE" --donor "$DONOR" --take "$TAKE" "${BF[@]}" --out "$OUTD/model.gguf" 2>&1 | tee "$CAP/overlay.log"
[ "${PIPESTATUS[0]}" = 0 ] || exit 1
$PY tools/quality/llamacpp_ppl.py --engine-ppl "$ENGINE_CAP/quality/ppl.json" --windows "$WIN" --gguf "$OUTD/model.gguf" \
  --out "$CAP/$NAME" --memfree-log "$CAP/ppl-memfree.csv" --unit slotstream-spark-f9-ref 2>&1 | tee "$CAP/llamacpp_ppl.log"
rc=${PIPESTATUS[0]}
if [ "${F9C_KEEP:-0}" != 1 ]; then
  $PY - "$OUTD/model.gguf" <<'PY'
import os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
os.close(fd)
os.unlink(sys.argv[1])
PY
  rmdir "$OUTD" 2>/dev/null
fi
exit "$rc"
