#!/bin/bash
# Mac: run CMD on the Spark in $SRC.
# run.sh [--timeout S] [--heavy LABEL FOOTPRINT_MIB DUR_S [--memcg SCOPE] [--container]
#         [--nolock] [--exclusive] [--allow-truncate]] -- CMD...
# Exit: CMD's rc; 75 precheck refused; 76 cut; 77 postcheck failed; 255 ssh failure.
set -u
cd "$(dirname "$0")/../.."
. tools/spark/env.sh

TIMEOUT=300
HEAVY=0
LABEL="" FOOTPRINT="" DUR=""
FLAGS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --timeout) TIMEOUT=$2; shift 2 ;;
    --heavy) HEAVY=1; LABEL=$2; FOOTPRINT=$3; DUR=$4; shift 4 ;;
    --memcg) FLAGS="$FLAGS --memcg $2"; shift 2 ;;
    --container|--nolock|--exclusive|--allow-truncate) FLAGS="$FLAGS $1"; shift ;;
    --) shift; break ;;
    *) echo "run.sh: unexpected argument: $1" >&2; exit 2 ;;
  esac
done
[ $# -gt 0 ] || { echo "run.sh: missing command after --" >&2; exit 2; }
[ "$TIMEOUT" -le 3600 ] 2>/dev/null || TIMEOUT=3600

CMDQ=$(printf '%q ' "$@")
PREFIX="cd $SRC && export SLOTSTREAM_MODE=\${SLOTSTREAM_MODE:-${SLOTSTREAM_MODE:-D5}} XDG_RUNTIME_DIR=/run/user/\$(id -u) &&"

if [ "$HEAVY" = 0 ]; then
  GATE="echo 1000 > /proc/self/oom_score_adj;"
  exempt=0
  if [ "${1:-}" = "tools/spark/jobs.sh" ]; then
    case "${2:-}" in status|wait|log) exempt=1 ;; esac
  fi
  if [ "$exempt" = 0 ]; then
    # PAUSED (state/PAUSED, NO_WINDOWS=1) lifts the 00:00-04:00 refusal; a malformed file refuses.
    GATE="$GATE . tools/spark/env.sh; paused_load || { echo 'REFUSED: malformed PAUSED'; exit 75; };"
    GATE="$GATE h=\$((10#\$(date +%H))); [ \$h -ge 4 ] || [ \$PAUSED_NO_WINDOWS = 1 ] || { echo 'REFUSED: 00:00-04:00'; exit 75; };"
  fi
  spark_ssh "$PREFIX $GATE exec timeout --kill-after=10 $TIMEOUT $CMDQ"
  exit $?
fi

# Heavy: the container environment adds --container automatically.
if spark_ssh "grep -q '^ENV_KIND=container' $R/ENV" >/dev/null 2>&1; then
  case "$FLAGS" in *--container*) ;; *) FLAGS="$FLAGS --container" ;; esac
fi

spark_ssh "$PREFIX deploy/guard.sh precheck$FLAGS $FOOTPRINT $DUR"
rc=$?
if [ $rc -ne 0 ]; then
  [ $rc -eq 255 ] && exit 255
  exit 75
fi
spark_ssh "$PREFIX exec deploy/guard.sh run$FLAGS $LABEL $FOOTPRINT $DUR -- $CMDQ"
exit $?
