#!/bin/bash
# 23:59 backstop: stop every slotstream-spark process, container and unit. Always exits 0.
#   kill-all.sh            the cron/timer call: does nothing while state/PAUSED has NO_WINDOWS=1 (the 23:59 stop is a window)
#   kill-all.sh --force    manual call: always kills (the STOP-file path and the user use this)
# Host cron/timers are installed by the user, not by agents (RUNBOOK: with production paused the cron line may stay).
export XDG_RUNTIME_DIR=/run/user/$(id -u)
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../tools/spark/env.sh"
GUARD_ROOT=${GUARD_ROOT:-$R}
GUARD_DOCKER=${GUARD_DOCKER:-docker}
KILLALL_WAIT=${KILLALL_WAIT:-30}
KILLALL_SYSTEMCTL=${KILLALL_SYSTEMCTL:-systemctl}

log() { echo "$(date -Is) kill-all: $*"; }
log "start"

FORCE=0; [ "${1:-}" = "--force" ] && FORCE=1
if [ -e "$GUARD_ROOT/run/STOP" ]; then FORCE=1; fi
if ! paused_load "$GUARD_ROOT/state/PAUSED"; then log "malformed pause file, behaving as unpaused"; PAUSED_NO_WINDOWS=0; fi
if [ "$FORCE" = 0 ] && [ "$PAUSED_NO_WINDOWS" = 1 ]; then
  log "skipped: pause mode (NO_WINDOWS=1); use --force or the STOP file to kill"
  exit 0
fi

for f in "$GUARD_ROOT"/run/*.run; do
  [ -f "$f" ] || continue
  pgid=$(sed -n 's/^pgid=//p' "$f")
  ctr=$(sed -n 's/^ctr=//p' "$f")
  [ -n "$ctr" ] && $GUARD_DOCKER stop -t 10 "$ctr" >/dev/null 2>&1
  [ -n "$pgid" ] && kill -TERM -- "-$pgid" 2>/dev/null
  i=0
  while [ $i -lt "$KILLALL_WAIT" ] && [ -n "$pgid" ] && [ -n "$(pgrep -g "$pgid" 2>/dev/null)" ]; do
    sleep 1; i=$((i+1))
  done
  [ -n "$ctr" ] && $GUARD_DOCKER kill "$ctr" >/dev/null 2>&1
  [ -n "$pgid" ] && kill -KILL -- "-$pgid" 2>/dev/null
  sleep 0.5
  if [ -z "$pgid" ] || [ -z "$(pgrep -g "$pgid" 2>/dev/null)" ]; then
    rm -f "$f"
    log "stopped $(basename "$f")"
  else
    log "still alive after KILL: $(basename "$f")"
  fi
done

for c in $($GUARD_DOCKER ps -q --filter name=slotstream-spark- 2>/dev/null); do
  $GUARD_DOCKER kill "$c" >/dev/null 2>&1
  log "docker kill $c"
done

$KILLALL_SYSTEMCTL --user stop 'slotstream-spark-job-*.service' 'slotstream-spark-lb*.scope' \
  slotstream-spark.socket slotstream-spark-front.service slotstream-spark.service >/dev/null 2>&1
log "done"
exit 0
