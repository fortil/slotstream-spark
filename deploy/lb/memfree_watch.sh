#!/bin/bash
# Spark: 1 Hz MemFree watcher with a soft floor.
#   memfree_watch.sh PID FLOOR_MIB LOG [MEMINFO]
# Appends epoch,memfree_mib,cached_mib,root_file_mib,children_file_mib to LOG while PID lives.
# If FLOOR_MIB > 0 and MemFree < FLOOR_MIB: TERM PID, write "memfree_stop <epoch> <mib> <floor>" to
# LOG.stop, KILL after 10 s if still alive, exit 5. Exit 0 when PID is gone without a stop.
PID=${1:?pid} FLOOR=${2:?floor} LOG=${3:?log} MI=${4:-/proc/meminfo}
[ "$LOG" = /dev/null ] || [ -s "$LOG" ] || echo "epoch,memfree_mib,cached_mib,root_file_mib,children_file_mib" >> "$LOG"
while kill -0 "$PID" 2>/dev/null; do
  now=$(date +%s)
  read -r mf ca < <(awk '/^MemFree:/{f=int($2/1024)} /^Cached:/{c=int($2/1024)} END{print f+0, c+0}' "$MI" 2>/dev/null)
  rf=$(awk '$1=="file"{print int($2/1048576)}' /sys/fs/cgroup/memory.stat 2>/dev/null)
  cf=$(cat /sys/fs/cgroup/*/memory.stat 2>/dev/null | awk '$1=="file"{s+=$2} END{if (NR) print int(s/1048576)}')
  echo "$now,$mf,$ca,$rf,$cf" >> "$LOG"
  if [ "$FLOOR" -gt 0 ] && [ "${mf:-0}" -lt "$FLOOR" ]; then
    echo "memfree_stop $now $mf $FLOOR" > "$LOG.stop"
    kill -TERM "$PID" 2>/dev/null
    for ((i = 0; i < 10; i++)); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
    kill -0 "$PID" 2>/dev/null && kill -KILL "$PID" 2>/dev/null
    exit 5
  fi
  sleep 1
done
exit 0
