#!/bin/bash
# Spark: the F8 token-graph A/B and soak job (plan step 91). Start it with
#   tools/spark/jobs.sh start f8-ab-soak --heavy f8-ab-soak auto 36000 -- bash tools/spark/snap.sh f8-ab-soak bash bench/f8_ab_soak.sh
# Phase 1, the A/B: 16 fresh-process pairs in D5 on the B1 cohort (bench/f6_arm.py, --pool table, greedy seed 42, warmup 128 +
# 512 measured tokens), A = the F3 segmented graphs (SLOTSTREAM_TOKEN_GRAPH=off), B = the single per-token graph (on), lookahead
# and MTP off in both; label f8-ab, bench/ab.py rows carry the eligibility snapshot (device NVMe limit). Identical outputs
# (tokens_sha256) are checked per pair by tools/decide.py. The arms append their token-graph counters to
# $SLOTSTREAM_R/runs/f8-tg-stats.jsonl (SLOTSTREAM_TOKEN_GRAPH_STATS), so "on" can be shown to have run on the graph.
# Phase 2, the soak, only when the A/B passed its own bar (median >= 1.05, >= 80% positive pairs, 16 clean): a loop of
# fresh-process token-graph arms of 1,024 measured tokens for SOAK_SECONDS (8 h = 28,800 s); every 30 minutes a spot bit-check
# runs the same pair on the F3 path and on the graph and compares tokens_sha256. A failed arm, a mismatch or a kernel-log line
# with NVRM Xid / NV_ERR / "context" loss is counted; the soak ends early on the first CUDA error and says so.
# Result: $CAP/f8_soak.json (also printed as the last line "f8-ab-soak: ..."); tools/decide.py judges it on the Mac.
if [ -z "${SLOTSTREAM_SRC:-}" ]; then
  exec bash tools/spark/snap.sh f8-ab-soak bash bench/f8_ab_soak.sh "$@"
fi
set -u
PY=tools/spark/py.sh
PAIRS=${F8_PAIRS:-16}
SOAK_SECONDS=${F8_SOAK_SECONDS:-28800}
SPOT_EVERY=${F8_SPOT_EVERY:-1800}
STATS="$SLOTSTREAM_R/runs/f8-tg-stats.jsonl"
SOAK_OUT=${CAP:-$SLOTSTREAM_R/runs}/f8_soak.json
mkdir -p "$(dirname "$SOAK_OUT")"
rm -f "$STATS"

arm() {  # arm TOKEN_GRAPH COHORT -> the command ab.py runs ({pair} and {out} filled by ab.py)
  echo "env SLOTSTREAM_TOKEN_GRAPH=$1 SLOTSTREAM_TOKEN_GRAPH_STATS=$STATS SLOTSTREAM_LOOKAHEAD=off SLOTSTREAM_MTP=off $PY bench/f6_arm.py --cohort $2 --pair {pair} --pool table --out {out} --warmup-tokens 128 --tokens 512 --seed 42"
}

T0=$(date +%s)
KLOG0=$(date '+%Y-%m-%d %H:%M:%S')
echo "f8-ab-soak: A/B $PAIRS pairs, soak ${SOAK_SECONDS}s, spot-check every ${SPOT_EVERY}s"

$PY bench/ab.py --label f8-ab --pairs "$PAIRS" --metric decode_tps --ident tokens_sha256 \
  --min-median 1.05 --min-positive 0.80 --keep-arms --a-name f3-segmented --b-name token-graph \
  --a-cmd "$(arm off B1)" --b-cmd "$(arm on B1)"
AB_RC=$?
echo "f8-ab-soak: A/B finished rc=$AB_RC after $(( $(date +%s) - T0 )) s"
ON_LINES=$(wc -l < "$STATS" 2>/dev/null || echo 0)
echo "f8-ab-soak: token-graph counters appended by $ON_LINES arms"

klog_bad() {  # kernel-log lines since KLOG0 that mean a lost CUDA context or a GPU memory error
  journalctl -k --since "$KLOG0" 2>/dev/null | grep -Ec 'NVRM: Xid|NV_ERR|Out of memory|oom-kill|Killed process' || true
}

SOAK_RUN=0 SOAK_FAILED=0 SPOT_RUN=0 SPOT_BAD=0 CTX_LOST=0 EARLY=""
if [ "$AB_RC" = 0 ]; then
  S0=$(date +%s)
  LAST_SPOT=$S0
  I=0
  while [ $(( $(date +%s) - S0 )) -lt "$SOAK_SECONDS" ]; do
    P=$(( I % 16 ))
    O="$SLOTSTREAM_R/tmp/f8-soak-arm.json"
    rm -f "$O"
    if ! env SLOTSTREAM_TOKEN_GRAPH=on SLOTSTREAM_TOKEN_GRAPH_STATS="$STATS" SLOTSTREAM_LOOKAHEAD=off SLOTSTREAM_MTP=off \
        $PY bench/f6_arm.py --cohort B1 --pair "$P" --pool table --out "$O" --warmup-tokens 128 --tokens 1024 --seed 42 >/dev/null 2>&1; then
      SOAK_FAILED=$((SOAK_FAILED + 1))
      echo "f8-ab-soak: soak arm $I (pair $P) failed"
    fi
    SOAK_RUN=$((SOAK_RUN + 1))
    if [ "$(klog_bad)" != 0 ]; then
      CTX_LOST=1
      EARLY="kernel log reported a CUDA/GPU error after soak arm $I"
      echo "f8-ab-soak: $EARLY"
      break
    fi
    if [ $(( $(date +%s) - LAST_SPOT )) -ge "$SPOT_EVERY" ]; then
      LAST_SPOT=$(date +%s)
      OA="$SLOTSTREAM_R/tmp/f8-spot-a.json" OB="$SLOTSTREAM_R/tmp/f8-spot-b.json"
      rm -f "$OA" "$OB"
      env SLOTSTREAM_TOKEN_GRAPH=off SLOTSTREAM_LOOKAHEAD=off SLOTSTREAM_MTP=off \
        $PY bench/f6_arm.py --cohort B1 --pair "$P" --pool table --out "$OA" --warmup-tokens 128 --tokens 512 --seed 42 >/dev/null 2>&1
      env SLOTSTREAM_TOKEN_GRAPH=on SLOTSTREAM_TOKEN_GRAPH_STATS="$STATS" SLOTSTREAM_LOOKAHEAD=off SLOTSTREAM_MTP=off \
        $PY bench/f6_arm.py --cohort B1 --pair "$P" --pool table --out "$OB" --warmup-tokens 128 --tokens 512 --seed 42 >/dev/null 2>&1
      SPOT_RUN=$((SPOT_RUN + 1))
      if ! $PY -c 'import json,sys
a,b=(json.load(open(p)) for p in sys.argv[1:3])
sys.exit(0 if a["tokens_sha256"]==b["tokens_sha256"] and a["ok"] and b["ok"] else 1)' "$OA" "$OB" 2>/dev/null; then
        SPOT_BAD=$((SPOT_BAD + 1))
        echo "f8-ab-soak: spot bit-check $SPOT_RUN (pair $P) differs"
      else
        echo "f8-ab-soak: spot bit-check $SPOT_RUN (pair $P) identical"
      fi
    fi
    I=$((I + 1))
  done
  SOAK_ELAPSED=$(( $(date +%s) - S0 ))
else
  SOAK_ELAPSED=0
  echo "f8-ab-soak: the A/B did not pass its bar; the soak is skipped (a result, not a failure of the job)"
fi

KBAD=$(klog_bad)
LAST=$(cat "$SLOTSTREAM_R/runs/LAST-f8-ab" 2>/dev/null || true)
$PY - "$SOAK_OUT" "$AB_RC" "$SOAK_ELAPSED" "$SOAK_SECONDS" "$SOAK_RUN" "$SOAK_FAILED" "$SPOT_RUN" "$SPOT_BAD" "$CTX_LOST" "$KBAD" "$STATS" "$LAST" "$EARLY" <<'PYEOF'
import json, sys
out, ab_rc, el, want, run, failed, spot, spot_bad, ctx, kbad, stats, last, early = sys.argv[1:14]
lines = []
try:
    lines = [json.loads(x) for x in open(stats).read().splitlines() if x.strip()]
except OSError:
    pass
res = {"ab_rc": int(ab_rc), "ab_capture": last, "soak_seconds": int(el), "soak_seconds_wanted": int(want),
       "soak_arms": int(run), "soak_arms_failed": int(failed), "spot_checks": int(spot), "spot_checks_differing": int(spot_bad),
       "cuda_context_losses": int(ctx) + int(kbad), "token_graph_arms": len(lines),
       "token_graph_tokens": sum(x["graph_tokens"] for x in lines), "token_graph_segmented_tokens": sum(x["segmented_tokens"] for x in lines),
       "token_graph_fallbacks": sum(x["fallbacks"] for x in lines), "early_stop": early or None}
open(out, "w").write(json.dumps(res, indent=1, sort_keys=True) + "\n")
print("f8-ab-soak: " + json.dumps(res, sort_keys=True))
PYEOF
exit 0
