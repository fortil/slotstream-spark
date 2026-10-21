#!/bin/bash
# Spark: guard drill. Runs six cases through the real deploy/guard.sh and kill-all.sh.
#   a deadline | b llama-swap flip | c MemAvailable | d container | e memcg | k kill-all
# Writes $R/runs/YYYY/MM/<ts>-guard-drill with drill.json {cases, all_pass} and verdict.json.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$HERE/tools/spark/env.sh"
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
IMG=vllm/vllm-openai:v0.28.0
HOG="$HERE/tools/drills/alloc_hog.py"
GUARD="$HERE/deploy/guard.sh"

DRILL="$R/runs/$(date +%Y/%m)/$(date +%Y%m%dT%H%M%S)-guard-drill"
mkdir -p "$DRILL" "$R/tmp" "$R/run"
echo "$DRILL" > "$R/runs/LAST-guard-drill"
START=$(date +%s)
TSV="$DRILL/cases.tsv"; : > "$TSV"

# Precondition: nothing of ours is running (kill-all in case k stops everything of ours).
if docker ps --format '{{.Names}}' | grep -q '^slotstream-spark-'; then
  echo "REFUSED: own containers are running" >&2; exit 75
fi
for f in "$R"/run/*.run; do
  [ -f "$f" ] || continue
  pg=$(sed -n 's/^pgid=//p' "$f")
  if [ -n "$pg" ] && [ -n "$(pgrep -g "$pg" 2>/dev/null)" ]; then
    echo "REFUSED: live run $(basename "$f")" >&2; exit 75
  fi
done

record() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$TSV"; echo "case $1: $([ "$2" = true ] && echo PASS || echo FAIL) - $3"; }
ma_mib() { awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo; }

cleanup() {
  docker rm -f slotstream-spark-drill-d slotstream-spark-drill-k >/dev/null 2>&1
  systemctl --user stop slotstream-spark-drill-e.scope >/dev/null 2>&1
}
trap cleanup EXIT

# collect NAME LABEL REASON: common checks after a guard run; sets C_OK C_DETAIL
collect() {
  local name=$1 label=$2 reason=$3 cap pgid cut ok=true detail="" i gone=-1
  mkdir -p "$DRILL/$name"
  cap=$(cat "$R/runs/LAST-$label")
  for f in guard.json guard-events.jsonl guard-post.json cut; do
    [ -f "$cap/$f" ] && cp "$cap/$f" "$DRILL/$name/"
  done
  cut=$(cat "$cap/cut" 2>/dev/null)
  [ "$GRC" = 76 ] || { ok=false; detail="$detail rc=$GRC;"; }
  [ "$cut" = "$reason" ] || { ok=false; detail="$detail cut=${cut:-none} expected=$reason;"; }
  grep -q "\"reason\":\"$reason\"" "$cap/guard-events.jsonl" 2>/dev/null || { ok=false; detail="$detail no event $reason;"; }
  pgid=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["pgid"])' "$cap/guard.json" 2>/dev/null)
  gone=-1
  for i in $(seq 1 45); do
    if [ -z "$(pgrep -g "$pgid" 2>/dev/null)" ] && ! docker ps --format '{{.Names}}' | grep -q '^slotstream-spark-drill-'; then gone=$i; break; fi
    sleep 1
  done
  [ $gone -ge 0 ] || { ok=false; detail="$detail group or container alive after 45 s;"; }
  C_OK=$ok; C_DETAIL="rc=$GRC cut=${cut:-none} gone<=${gone}s;$detail"
  C_CAP=$cap
}

# ---- a: deadline
echo '{"running":[]}' > "$R/tmp/drill-running.json"
export SLOTSTREAM_RUNNING_URL="file://$R/tmp/drill-running.json"
mkdir -p "$DRILL/a"
"$GUARD" run drill-a 2048 60 -- python3 "$HOG" --mib 1024 --ignore-term --sleep 600 > "$DRILL/a/out.txt" 2>&1; GRC=$?
collect a drill-a deadline
record a "$C_OK" "deadline: $C_DETAIL"

# ---- b: llama-swap flips to non-empty after 8 s
echo '{"running":[]}' > "$R/tmp/drill-running.json"
mkdir -p "$DRILL/b"
( sleep 8; echo '{"running":[{"model":"drill"}]}' > "$R/tmp/drill-running.json" ) &
FLIP=$!
T0=$(date +%s)
"$GUARD" run drill-b 2048 60 -- python3 "$HOG" --mib 256 --ignore-term --sleep 600 > "$DRILL/b/out.txt" 2>&1; GRC=$?
T1=$(date +%s); wait $FLIP 2>/dev/null
collect b drill-b running
B_OK=$C_OK
[ $((T1 - T0)) -le 40 ] || B_OK=false
record b "$B_OK" "running flip: cut after $((T1 - T0)) s; $C_DETAIL"
echo '{"running":[]}' > "$R/tmp/drill-running.json"

# ---- c: MemAvailable below the threshold
mkdir -p "$DRILL/c"
UM=$(( $(ma_mib) + 1024 ))
MEMAVAILABLE_MIN_MIB=$UM "$GUARD" run drill-c 2048 60 -- python3 "$HOG" --mib 256 --ignore-term --sleep 600 > "$DRILL/c/out.txt" 2>&1; GRC=$?
collect c drill-c memavailable
record c "$C_OK" "memavailable (threshold $UM): $C_DETAIL"

# ---- d: container
mkdir -p "$DRILL/d"
(
  for i in $(seq 1 90); do
    a=$(docker inspect -f '{{.HostConfig.OomScoreAdj}}' slotstream-spark-drill-d 2>/dev/null)
    cp=$(docker inspect -f '{{.State.Pid}}' slotstream-spark-drill-d 2>/dev/null)
    if [ -n "$a" ] && [ -n "$cp" ] && [ "$cp" != 0 ] && [ -r "/proc/$cp/oom_score_adj" ]; then
      echo "hostconfig=$a proc=$(cat /proc/$cp/oom_score_adj)" > "$DRILL/d/oomadj.txt"; break
    fi
    sleep 1
  done
) &
OOMW=$!
"$GUARD" run --container drill-d 2048 60 -- docker run --rm --name slotstream-spark-drill-d --network none --oom-score-adj 1000 \
  --user "$(id -u):$(id -g)" -v "$HERE:/w/src:ro" --entrypoint python3 "$IMG" /w/src/tools/drills/alloc_hog.py --mib 1024 --ignore-term --sleep 600 \
  > "$DRILL/d/out.txt" 2>&1; GRC=$?
kill $OOMW 2>/dev/null; wait $OOMW 2>/dev/null
collect d drill-d deadline
D_OK=$C_OK
grep -q 'hostconfig=1000 proc=1000' "$DRILL/d/oomadj.txt" 2>/dev/null || { D_OK=false; C_DETAIL="$C_DETAIL oom_score_adj not 1000;"; }
record d "$D_OK" "container: $C_DETAIL $(cat "$DRILL/d/oomadj.txt" 2>/dev/null)"

# ---- e: memcg
mkdir -p "$DRILL/e"
head -c 67108864 /dev/zero > "$R/tmp/drill-dontneed.bin"; cat "$R/tmp/drill-dontneed.bin" > /dev/null
(
  cap=""
  for i in $(seq 1 60); do
    cap=$(cat "$R/runs/LAST-drill-e" 2>/dev/null)
    [ -f "$cap/cut" ] && break
    sleep 0.2
  done
  pg=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["pgid"])' "$cap/guard.json" 2>/dev/null)
  for i in $(seq 1 400); do
    if [ -z "$(pgrep -g "$pg" 2>/dev/null)" ]; then date +%s.%N > "$DRILL/e/t_gone.txt"; break; fi
    sleep 0.05
  done
) &
EW=$!
rm -f "$R/runs/LAST-drill-e"
GUARD_DONTNEED="$R/tmp/drill-dontneed.bin" "$GUARD" run --memcg slotstream-spark-drill-e drill-e 1536 120 -- \
  systemd-run --user --scope --unit=slotstream-spark-drill-e -p MemoryMax=4G python3 "$HOG" --mib 2048 --ignore-term --sleep 600 \
  > "$DRILL/e/out.txt" 2>&1; GRC=$?
wait $EW 2>/dev/null
collect e drill-e memcg_over
E_OK=$C_OK
TT=$(sed -n 's/^TERM //p' "$DRILL/e/out.txt" | head -1)
TG=$(cat "$DRILL/e/t_gone.txt" 2>/dev/null)
DELTA=$(python3 -c 'import sys
t,g=sys.argv[1:3]
print(round(float(g)-float(t),2) if t and g else -1)' "$TT" "$TG")
python3 -c 'import sys;d=float(sys.argv[1]);sys.exit(0 if 0<=d<=2.5 else 1)' "$DELTA" || { E_OK=false; C_DETAIL="$C_DETAIL TERM->gone ${DELTA}s;"; }
# dontneed event logged and written before guard-post.json
python3 - "$DRILL/e/guard-events.jsonl" "$C_CAP/guard-post.json" <<'PY' || { E_OK=false; C_DETAIL="$C_DETAIL dontneed missing or after guard-post;"; }
import json, os, sys
ev = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
d = [e for e in ev if e.get("reason") == "dontneed"]
sys.exit(0 if d and d[0]["t"] <= int(os.path.getmtime(sys.argv[2])) + 1 else 1)
PY
[ "$(systemctl --user is-active slotstream-spark-drill-e.scope 2>/dev/null)" != active ] || { E_OK=false; C_DETAIL="$C_DETAIL scope still active;"; }
record e "$E_OK" "memcg: TERM->gone ${DELTA}s (limit 2 s + 0.5 s teardown); $C_DETAIL"

# ---- k: kill-all
mkdir -p "$DRILL/k"
setsid python3 "$HOG" --mib 128 --ignore-term --sleep 600 > "$DRILL/k/hog.txt" 2>&1 &
KP=$!
sleep 1
KPG=$(ps -o pgid= -p "$KP" | tr -d ' ')
docker run -d --rm --name slotstream-spark-drill-k --network none --oom-score-adj 1000 --user "$(id -u):$(id -g)" \
  -v "$HERE:/w/src:ro" --entrypoint python3 "$IMG" /w/src/tools/drills/alloc_hog.py --mib 128 --ignore-term --sleep 600 > /dev/null 2>&1
printf 'pgid=%s\nctr=slotstream-spark-drill-k\nlabel=drill-k\nstart=%s\n' "$KPG" "$(date +%s)" > "$R/run/drill-k.run"
sleep 2
KOK=true; KD=""
docker ps --format '{{.Names}}' | grep -qx slotstream-spark-drill-k || { KOK=false; KD="$KD container did not start;"; }
KT=$(date +%s)
# --force: under PAUSED (NO_WINDOWS=1) kill-all without it skips by design, so case k failed on 2026-10-20.
KILLALL_SYSTEMCTL=true "$HERE/deploy/kill-all.sh" --force > "$DRILL/k/kill-all.txt" 2>&1; KRC=$?
[ $KRC = 0 ] || { KOK=false; KD="$KD kill-all rc=$KRC;"; }
for i in $(seq 1 45); do
  if [ -z "$(pgrep -g "$KPG" 2>/dev/null)" ] && ! docker ps --format '{{.Names}}' | grep -q '^slotstream-spark-drill-'; then break; fi
  sleep 1
done
[ -z "$(pgrep -g "$KPG" 2>/dev/null)" ] || { KOK=false; KD="$KD group alive;"; }
docker ps --format '{{.Names}}' | grep -q '^slotstream-spark-drill-' && { KOK=false; KD="$KD container alive;"; }
[ ! -f "$R/run/drill-k.run" ] || { KOK=false; KD="$KD run file left;"; }
record k "$KOK" "kill-all rc=$KRC in $(( $(date +%s) - KT )) s; $KD"

# ---- summary
END=$(date +%s)
python3 - "$TSV" "$DRILL" "$START" "$END" <<'PY'
import json, sys
tsv, drill, start, end = sys.argv[1:5]
cases = {}
for line in open(tsv):
    n, p, d = line.rstrip("\n").split("\t", 2)
    cases[n] = {"pass": p == "true", "detail": d}
all_pass = len(cases) == 6 and all(c["pass"] for c in cases.values())
json.dump({"cases": cases, "all_pass": all_pass}, open(drill + "/drill.json", "w"), indent=1)
json.dump({"label": "guard-drill", "rc": 0 if all_pass else 1, "eligible": all_pass,
           "reason": "" if all_pass else "case failed", "start": int(start), "end": int(end),
           "seconds": int(end) - int(start)}, open(drill + "/verdict.json", "w"))
print("all_pass" if all_pass else "FAILED CASES")
PY
echo "CAPTURE $DRILL"
python3 -c 'import json,sys;sys.exit(0 if json.load(open(sys.argv[1]))["all_pass"] else 1)' "$DRILL/drill.json"
