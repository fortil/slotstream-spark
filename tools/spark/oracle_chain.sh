#!/bin/bash
# Spark: the oracle chain job. Resumable (state in $R/state/oracle/; a phase marker is written after the
# phase finishes; a restart skips finished phases). Runs under tools/capture.sh (CAP set).
#   (1) 16-token generation          (2) bf16 goldens, both sets      (3) second bf16 run (B)
#   (4) reproducibility manifest     (5) calibration dumps (dev only) (6) contract measure
#   (7) perplexity ablation          (8) copy the results into $CAP
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$HERE/tools/spark/env.sh"
: "${CAP:?oracle_chain.sh must run under capture.sh}"
cd "$HERE"
SRC=$HERE
PY=tools/spark/py.sh
STATE=$R/state/oracle
mkdir -p "$STATE" "$R/tmp"
DATA=$SRC/tests/goldens/data
PROMPTS=$SRC/tests/goldens/prompts.json

phase() { [ -f "$STATE/phase$1.done" ]; }
mark() { date -Is > "$STATE/phase$1.done"; }

# (1) cacheless greedy generation; ref_layerwise writes result.json to its own CAP
if ! phase 1; then
  mkdir -p "$STATE/gen16.cap"
  CAP="$STATE/gen16.cap" $PY tools/ref_layerwise.py --generate 16 \
    --prompt "Explain in three sentences why the sky is blue." --state "$STATE/gen16"
  cp "$STATE/gen16.cap/result.json" "$STATE/gen16.json"
  mark 1
fi
# (2) goldens, bf16, correct experts, dev and reserved
if ! phase 2; then
  $PY tools/make_goldens.py --prompts "$PROMPTS" --out "$DATA" --state "$R/tmp/goldens-state"
  mark 2
fi
# (3) second run for the reproducibility check
if ! phase 3; then
  $PY tools/make_goldens.py --prompts "$PROMPTS" --out "$R/tmp/goldens-B" --state "$R/tmp/goldens-state-B"
  mark 3
fi
# (4) manifest (exit 1 if a file differs between the two runs), then drop run B
if ! phase 4; then
  $PY tools/make_goldens.py --prompts "$PROMPTS" --compare "$DATA" "$R/tmp/goldens-B" \
    --manifest "$SRC/tests/goldens/manifest.json"
  rm -rf "$R/tmp/goldens-B" "$R/tmp/goldens-state-B"
  mark 4
fi
# (5) calibration dumps, dev set only. The fp32 pass sets ORACLE_DTYPE; the two controls run WITHOUT it
# (bf16 oracle variants), as the [layer] rule of numeric_contract.toml defines c_k.
if ! phase 5; then
  ORACLE_DTYPE=float32 $PY tools/make_goldens.py --prompts "$PROMPTS" --sets dev \
    --out "$R/tmp/calib/fp32" --state "$R/tmp/calib-state"
  $PY tools/make_goldens.py --prompts "$PROMPTS" --sets dev --experts-mode swap \
    --out "$R/tmp/calib/swap" --state "$R/tmp/calib-state"
  $PY tools/make_goldens.py --prompts "$PROMPTS" --sets dev --experts-mode noexp \
    --out "$R/tmp/calib/noexp" --state "$R/tmp/calib-state"
  mark 5
fi
# (6) measure (tools/calibrate_contract.py, step 33), then drop the dumps
if ! phase 6; then
  $PY tools/calibrate_contract.py measure --golden "$DATA" --fp32 "$R/tmp/calib/fp32" \
    --swap "$R/tmp/calib/swap" --noexp "$R/tmp/calib/noexp" --out "$STATE/calibration.json"
  rm -rf "$R/tmp/calib" "$R/tmp/calib-state"
  mark 6
fi
# (7) perplexity ablation
if ! phase 7; then
  $PY tools/ppl_ablate.py --texts "$SRC/tests/goldens/ppl_texts.json" --modes correct,swap,noexp \
    --state "$STATE/ppl" --out "$STATE/ppl.json"
  mark 7
fi
# (8) results into this run's capture (a restarted job has a new $CAP)
cp "$STATE/gen16.json" "$CAP/gen16.json"
cp "$STATE/calibration.json" "$CAP/calibration.json"
cp "$STATE/ppl.json" "$CAP/ppl.json"
cp "$SRC/tests/goldens/manifest.json" "$CAP/goldens_manifest.json"
$PY - "$DATA" "$CAP/goldens_summary.json" <<'PYE'
import json, os, sys
root, out = sys.argv[1:3]
n = sum(len(f) for _, _, f in os.walk(root))
dirs = sum(len(os.listdir(os.path.join(root, d))) for d in os.listdir(root))
json.dump({"prompt_dirs": dirs, "files": n}, open(out, "w"))
PYE
echo "oracle chain done"
