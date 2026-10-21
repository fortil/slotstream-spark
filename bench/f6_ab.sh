#!/bin/bash
# Spark: the F6 A/B job (plan step 77). Start it with
#   tools/spark/jobs.sh start f6-ab --heavy f6-ab auto 14400 -- bash bench/f6_ab.sh
# (the guard refuses it after ~19:45 and between 00:00 and 04:00). Two paired A/Bs, 16 pairs each, a fresh engine process per
# arm and pair (bench/f6_arm.py), D5, greedy seed 42, thinking off, warmup 128 + 512 measured tokens, ctx <= 8K:
#   1  lookahead on B1: SLOTSTREAM_LOOKAHEAD=off vs the variant (attention-corrected when the correction file is present, else
#      attention), MTP off in both, label f6-ab-lookahead
#   2  MTP on B0: SLOTSTREAM_MTP=off vs resident, the lookahead variant in both arms when the first A/B adopted it
#      (bench/stats.py adopt, margin 1.02, >= 5 clean pairs, >= 80% positive, identical outputs), else off, label f6-ab-mtp
# The A/Bs exit 1 when they do not pass their own bar; that is a result, not a failure of the job: tools/decide.py judges the
# captures on the Mac after `tools/spark/fetch.sh`.
# Added 2026-10-14 (step 77f): `--mtp-only` skips the lookahead A/B (its KEEP_OFF decision stands) and the MTP A/B runs with
# lookahead off; every arm runs `--pool table` (pool pre-filled from expert_freq_gb10.json at engine start and kept through
# the warm-up prompt), so both arms of an A/B share the pool policy.
if [ -z "${SLOTSTREAM_SRC:-}" ]; then
  exec bash tools/spark/snap.sh f6-ab bash bench/f6_ab.sh "$@"
fi
MTP_ONLY=0
[ "${1:-}" = "--mtp-only" ] && MTP_ONLY=1
set -u
PY=tools/spark/py.sh
PAIRS=${F6_PAIRS:-16}

VARIANT=$($PY -c 'import os
from pathlib import Path
from slotstream_spark.engine import lookahead as la
print("attention-corrected" if la.tap_status(Path(os.environ["SLOTSTREAM_R"]) / "model")["present"] else "attention")') || exit 2
echo "f6-ab: lookahead variant $VARIANT, $PAIRS pairs"

arm() {  # arm COHORT LOOKAHEAD MTP -> the command ab.py runs (fresh process, {pair} and {out} filled by ab.py)
  echo "env SLOTSTREAM_LOOKAHEAD=$2 SLOTSTREAM_MTP=$3 $PY bench/f6_arm.py --cohort $1 --pair {pair} --pool table --out {out} --warmup-tokens 128 --tokens 512 --seed 42"
}

if [ "$MTP_ONLY" = 1 ]; then
  LA=off
  echo "f6-ab: lookahead A/B skipped (KEEP_OFF, db/records/decisions/f6-lookahead-d5.md); lookahead in the MTP A/B: off"
else
  $PY bench/ab.py --label f6-ab-lookahead --pairs "$PAIRS" --metric decode_tps --ident tokens_sha256 \
    --min-median 1.02 --min-positive 0.80 --keep-arms --a-name off --b-name "$VARIANT" \
    --a-cmd "$(arm B1 off off)" --b-cmd "$(arm B1 "$VARIANT" off)"
  echo "f6-ab: lookahead A/B finished rc=$?"

  LAST=$(cat "$SLOTSTREAM_R/runs/LAST-f6-ab-lookahead" 2>/dev/null || true)
  if [ -n "$LAST" ] && $PY bench/stats.py adopt --result "$LAST/result.json" --margin 1.02; then
    LA=$VARIANT
  else
    LA=off
  fi
  echo "f6-ab: lookahead in the MTP A/B: $LA"

fi

$PY bench/ab.py --label f6-ab-mtp --pairs "$PAIRS" --metric decode_tps --ident tokens_sha256 \
  --min-median 1.05 --min-positive 0.80 --keep-arms --a-name "mtp-off/lookahead-$LA" --b-name "mtp-resident/lookahead-$LA" \
  --a-cmd "$(arm B0 "$LA" off)" --b-cmd "$(arm B0 "$LA" resident)"
echo "f6-ab: MTP A/B finished rc=$?"
exit 0
