#!/bin/bash
# Spark: capture package around a command.
#   capture.sh LABEL -- CMD...
# Uses CAP from the guard; without it creates $R/runs/YYYY/MM/<ts>-LABEL and LAST-LABEL.
# Exit: CMD's rc.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$HERE/tools/spark/env.sh"

LABEL=${1:-}
[ -n "$LABEL" ] && [ "${2:-}" = "--" ] || { echo "usage: capture.sh LABEL -- CMD..." >&2; exit 2; }
shift 2
[ $# -gt 0 ] || { echo "capture.sh: empty command" >&2; exit 2; }

if [ -z "${CAP:-}" ]; then
  CAP="$R/runs/$(date +%Y/%m)/$(date +%Y%m%dT%H%M%S)-$LABEL"
  mkdir -p "$CAP"
  echo "$CAP" > "$R/runs/LAST-$LABEL"
fi
mkdir -p "$CAP"
export CAP SLOTSTREAM_CAPTURE_DIR="$CAP"

START=$(date +%s)
START_ISO=$(date -Iseconds)
SYNCED=$(cat "$SRC/.synced" 2>/dev/null || echo unknown)
{
  printf '%q ' "$@"; echo
  echo "synced=$SYNCED"
  # Host paths are written as $R so a capture carries no machine-specific path
  env | grep '^SLOTSTREAM_' | sort | sed "s#$R#\$R#g"
  echo "CAP=${CAP/#$R/\$R}"
  echo "start=$START_ISO"
} > "$CAP/command.txt"

python3 "$HERE/tools/hostinfo.py" > "$CAP/host-before.json" 2>/dev/null

SAMPLERS=()
iostat -dxm -t "${SLOTSTREAM_NVME_DEVICE:-nvme0n1}" 1 > "$CAP/iostat.txt" 2>/dev/null & SAMPLERS+=($!)
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_event_reasons.active \
  --format=csv -lms 500 > "$CAP/nvsmi-gpu.csv" 2>/dev/null & SAMPLERS+=($!)
nvidia-smi --query-compute-apps=timestamp,pid,process_name,used_memory --format=csv -l 1 > "$CAP/nvsmi-apps.csv" 2>/dev/null & SAMPLERS+=($!)
# Some drivers reject the last gpu field: retry without it if the file stayed empty.
sleep 1
if [ ! -s "$CAP/nvsmi-gpu.csv" ]; then
  nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu \
    --format=csv -lms 500 > "$CAP/nvsmi-gpu.csv" 2>/dev/null & SAMPLERS+=($!)
fi

(
  echo "epoch,MemAvailable_kB,MemFree_kB,Cached_kB" > "$CAP/meminfo.csv"
  while :; do
    awk -v e="$(date +%s)" '$1=="MemAvailable:"{a=$2} $1=="MemFree:"{f=$2} $1=="Cached:"{c=$2} END{print e","a","f","c}' /proc/meminfo >> "$CAP/meminfo.csv"
    sleep 1
  done
) & SAMPLERS+=($!)

PSI_CONTAINERS=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -v '^slotstream-spark' | tr '\n' ' ')
(
  echo "epoch,container,io_some_avg10,mem_some_avg10" > "$CAP/prod-psi.csv"
  while :; do
    for c in $PSI_CONTAINERS; do
      id=$(docker inspect --format '{{.Id}}' "$c" 2>/dev/null) || continue
      d=/sys/fs/cgroup/system.slice/docker-$id.scope
      io=$(sed -n 's/^some .*avg10=\([0-9.]*\).*/\1/p' "$d/io.pressure" 2>/dev/null)
      mem=$(sed -n 's/^some .*avg10=\([0-9.]*\).*/\1/p' "$d/memory.pressure" 2>/dev/null)
      echo "$(date +%s),$c,${io:-},${mem:-}" >> "$CAP/prod-psi.csv"
    done
    sleep 1
  done
) & SAMPLERS+=($!)

stop_samplers() {
  local p
  for p in "${SAMPLERS[@]}"; do pkill -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; done
  for p in "${SAMPLERS[@]}"; do wait "$p" 2>/dev/null; done
}
trap stop_samplers EXIT

( "$@" > >(tee "$CAP/stdout.txt") 2> >(tee "$CAP/stderr.txt" >&2) ) &
PID=$!
pidstat -t -u -d -r -p "$PID" 1 > "$CAP/pidstat.txt" 2>/dev/null & SAMPLERS+=($!)
(
  while kill -0 "$PID" 2>/dev/null; do
    { date +%s; cat "/proc/$PID/io" 2>/dev/null; } >> "$CAP/proc-io.txt"
    sleep 1
  done
) & SAMPLERS+=($!)

wait "$PID"; RC=$?
END=$(date +%s)
END_ISO=$(date -Iseconds)
sleep 1
stop_samplers
trap - EXIT
# Let process-substitution tees finish so no process of our group outlives us.
wait 2>/dev/null

python3 "$HERE/tools/hostinfo.py" > "$CAP/host-after.json" 2>/dev/null

if [ -n "${SWAP_PATTERN:-}" ] && [ -f "$HERE/tools/spark/prod_reads.sh" ]; then
  bash "$HERE/tools/spark/prod_reads.sh" --window "$START" "$END" --out "$CAP/prometheus-window.json" >/dev/null 2>&1 || true
fi

REASON=""; [ "$RC" -eq 0 ] || REASON="rc=$RC"
ELIG=false; [ "$RC" -eq 0 ] && ELIG=true
printf '{"label":"%s","rc":%s,"eligible":%s,"reason":"%s","start":"%s","end":"%s","seconds":%s,"synced":"%s"}\n' \
  "$LABEL" "$RC" "$ELIG" "$REASON" "$START_ISO" "$END_ISO" "$((END-START))" "$SYNCED" > "$CAP/verdict.json"

echo "CAPTURE $CAP rc=$RC"
exit $RC
