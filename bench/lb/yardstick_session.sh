#!/bin/bash
# Spark: llama.cpp yardstick session (LB2). Run under the guard (needs CAP).
#   yardstick_session.sh --configs q41-a-d5,iq3-r-d5 [--arm-b] [--port 8091]
# Config names are <weights>-<arm>-<mode>, e.g. q41-a-d5. --arm-b appends q41-b-d5.
# Per config: two server processes (A/A); for q41-a-d5 also one extra process per -t in 8, 12, 20
# (3 reps of "short"), skipped if SLOTSTREAM_DEADLINE leaves less than 30 minutes.
# Output: $CAP/<config>/{p1,p2,t8,t12,t20}/ and $CAP/summary.json.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$HERE/tools/spark/env.sh"

CONFIGS="" ARMB=0 PORT=8091
while [ $# -gt 0 ]; do
  case "$1" in
    --configs) CONFIGS=$2; shift 2 ;;
    --arm-b) ARMB=1; shift ;;
    --port) PORT=$2; shift 2 ;;
    *) echo "session.sh: unexpected argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$CONFIGS" ] || { echo "session.sh: --configs is required" >&2; exit 2; }
[ "$ARMB" = 1 ] && CONFIGS="$CONFIGS,q41-b-d5"
CAP=${CAP:?session.sh needs CAP (run it under the guard or capture.sh)}
UNIT=slotstream-spark-lb
DEADLINE=${SLOTSTREAM_DEADLINE:-0}
LPID="" SPID=""

stop_server() {
  [ -n "$LPID" ] && { kill -TERM "$LPID" 2>/dev/null; wait "$LPID" 2>/dev/null; }
  [ -n "$SPID" ] && { kill -TERM "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; }
  LPID="" SPID=""
}
trap stop_server EXIT

# soft_stop DIR -> 0 and writes DIR/memfree_stop.json when the MemFree soft-floor watcher stopped the server
soft_stop() {
  local dir=$1
  [ -f "$dir/memfree.csv.stop" ] || return 1
  python3 - "$dir" <<'PY'
import json, os, sys
d = sys.argv[1]
_, t, mf, floor = open(os.path.join(d, "memfree.csv.stop")).read().split()[:4]
json.dump({"memfree_stop": True, "memfree_mib": int(mf), "floor_mib": int(floor), "t": int(t)},
          open(os.path.join(d, "memfree_stop.json"), "w"))
PY
  return 0
}

# run_process CONFIG LABEL THREADS REPS IDS  -> 0 ok, 3 soft-floor stop, 2 envelope exceeded, 1 server failed
run_process() {
  local cfg=$1 label=$2 threads=$3 reps=$4 ids=$5 w a m dir hu footprint maxm i code pid nv mf table fl opo
  IFS=- read -r w a m <<< "$cfg"
  a=$(echo "$a" | tr a-z A-Z); m=$(echo "$m" | tr a-z A-Z)
  dir=$CAP/$cfg/$label
  mkdir -p "$dir"
  # MemFree is read once, right before launch, and passed to both calls so the record is what was used.
  mf=$(awk '/^MemFree:/{print int($2/1024)}' /proc/meminfo)
  hu=$(bash "$HERE/deploy/lb/llamacpp.sh" --print-footprint --weights "$w" --arm "$a" --mode "$m" --memfree-mib "$mf") || return 1
  table=$(echo "$hu" | sed -n 's/.*TABLE_MIB=\([0-9]*\).*/\1/p')
  footprint=$(echo "$hu" | sed -n 's/.*FOOTPRINT_MIB=\([0-9]*\).*/\1/p')
  maxm=$(echo "$hu" | sed -n 's/.*MEMORYMAX_MIB=\([0-9]*\).*/\1/p')
  fl=$(bash "$HERE/deploy/lb/llamacpp.sh" --print-flags --weights "$w" --arm "$a" --mode "$m") || return 1
  opo=$(echo "$fl" | sed -n 's/.*OP_OFFLOAD=\([a-z]*\).*/\1/p')

  python3 "$HERE/bench/lb/sample20ms.py" --scope "$UNIT" --out "$dir/sample20ms.csv" & SPID=$!
  bash "$HERE/deploy/lb/llamacpp.sh" --weights "$w" --arm "$a" --mode "$m" --threads "$threads" \
    --memfree-mib "$mf" --memfree-stop 14336 --memfree-log "$dir/memfree.csv" \
    --port "$PORT" --unit "$UNIT" > "$dir/server.log" 2>&1 & LPID=$!

  code=000
  for ((i = 0; i < 900; i++)); do
    kill -0 "$LPID" 2>/dev/null || { echo "session.sh: server exited during load ($cfg/$label)" >&2; stop_server; soft_stop "$dir" && return 3; return 1; }
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
    [ "$code" = 200 ] && break
    sleep 1
  done
  if [ "$code" != 200 ]; then echo "session.sh: /health not 200 in 900 s ($cfg/$label)" >&2; stop_server; soft_stop "$dir" && return 3; return 1; fi
  sleep 2
  [ -n "$SPID" ] && { kill -TERM "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; SPID=""; }

  # m-lb4: what the server really holds, against the envelope.
  pid=$(pgrep -f "llama-server.*--port $PORT" | head -1)
  nv=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
       | awk -F', *' -v p="$pid" '$1==p{s+=$2} END{print s+0}')
  python3 - "$dir" "$nv" "$footprint" "$maxm" "$UNIT" "$mf" "$table" "$opo" 14336 <<'PY'
import glob, json, os, re, sys
d, nv, footprint, maxm, unit = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
memfree, table = int(sys.argv[6]), int(sys.argv[7])
op_offload, stop_mib = sys.argv[8] == "true", int(sys.argv[9])


def root_direct():
    # root memcg 'file' minus the sum of the top-level children's 'file' (page cache charged to root itself)
    try:
        def f(p):
            for l in open(p):
                k, v = l.split()
                if k == "file":
                    return int(v)
        tot = f("/sys/fs/cgroup/memory.stat")
        kids = sum(f(p) or 0 for p in glob.glob("/sys/fs/cgroup/*/memory.stat"))
        return (tot - kids) // (1 << 20)
    except (OSError, ValueError, TypeError):
        return None


rd = root_direct()
cg = (glob.glob(f"/sys/fs/cgroup/**/{unit}.scope", recursive=True) or [None])[0]
cur, stat = None, {}
if cg:
    try:
        cur = int(open(cg + "/memory.current").read()) // (1 << 20)
        stat = {k: int(v) // (1 << 20) for k, v in (l.split() for l in open(cg + "/memory.stat")) if k in ("anon", "file", "shmem")}
    except OSError:
        pass
bufs = [l.strip() for l in open(os.path.join(d, "server.log"), errors="replace") if "buffer size" in l][:20]
json.dump({"nvml_mib": nv, "memory_current_mib": cur, "memory_stat_mib": stat, "buffer_lines": bufs,
           "footprint_mib": footprint, "memorymax_mib": maxm, "memorymax_table_mib": table,
           "memfree_at_start_mib": memfree, "memfree_source": "/proc/meminfo before launch",
           "op_offload": op_offload, "memfree_stop_mib": stop_mib, "root_direct_file_mib": rd}, open(os.path.join(d, "mlb4.json"), "w"), indent=1)
json.dump({"envelope_exceeded": nv + maxm > footprint - 256, "nvml_mib": nv, "memorymax_mib": maxm, "memorymax_table_mib": table,
           "memfree_at_start_mib": memfree, "footprint_mib": footprint,
           "op_offload": op_offload, "memfree_stop_mib": stop_mib, "root_direct_file_mib": rd},
          open(os.path.join(d, "envelope.json"), "w"))
PY
  if grep -q '"envelope_exceeded": true' "$dir/envelope.json"; then
    echo "session.sh: envelope exceeded for $cfg ($nv + $maxm > $footprint - 256)" >&2
    stop_server
    return 2
  fi

  python3 "$HERE/bench/lb/client.py" --port "$PORT" --ids "$ids" --reps "$reps" --warmup 1 --out "$dir/client.jsonl"
  local rc=$?
  stop_server
  soft_stop "$dir" && return 3
  return $rc
}

IFS=, read -ra CFGS <<< "$CONFIGS"
for cfg in "${CFGS[@]}"; do
  skip=0
  for p in p1 p2; do
    run_process "$cfg" "$p" 16 5 short,2k,8k
    rc=$?
    { [ $rc = 2 ] || [ $rc = 3 ]; } && { skip=1; break; }
    [ $rc = 0 ] || echo "session.sh: $cfg/$p failed (rc=$rc)" >&2
  done
  if [ "$cfg" = q41-a-d5 ] && [ $skip = 0 ]; then
    for t in 8 12 20; do
      if [ "$DEADLINE" -gt 0 ] && [ $((DEADLINE - $(date +%s))) -lt 1800 ]; then
        echo "session.sh: skipping -t $t (less than 30 min to the deadline)" >&2
        continue
      fi
      run_process "$cfg" "t$t" "$t" 3 short || true
    done
  fi
done

python3 "$HERE/bench/lb/summarize.py" "$CAP" > "$CAP/summary.json"
exit 0
