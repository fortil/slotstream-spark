#!/bin/bash
# Spark self-test for deploy/guard.sh. Prints "guard_t0: <n> passed, <m> failed".
cd "$(dirname "$0")/../.." || exit 1
GUARD=$PWD/deploy/guard.sh
R=${SLOTSTREAM_R_REMOTE:-$HOME/slotstream-spark}
mkdir -p "$R/tmp"
T=$(mktemp -d "$R/tmp/guard_t0.XXXXXX")
MAIN=$$
trap '[ "$BASHPID" = "$MAIN" ] && { pkill -f "$T" 2>/dev/null; rm -rf "$T"; }' EXIT

PASS=0 FAIL=0
ok() { PASS=$((PASS+1)); echo "ok   $1"; }
bad() { FAIL=$((FAIL+1)); echo "FAIL $1"; }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }

# Host independent: ignore private/env.local.sh and any inherited container count (cases that need a count set it).
export SLOTSTREAM_SKIP_LOCAL=1
unset EXPECTED_CONTAINERS
mkdir -p "$T/run" "$T/bin"
NOON=$(date -d "$(date +%Y-%m-%d) 12:00" +%s)
export GUARD_ROOT=$T GUARD_MEMINFO=$T/meminfo SLOTSTREAM_RUNNING_URL=file://$T/running \
       GUARD_DOCKER=$T/bin/docker GUARD_KLOG=true GUARD_DONTNEED=/nonexistent
cat > "$T/bin/docker" <<'DK'
#!/bin/bash
T=$(dirname "$(dirname "$0")")
case "$1" in
  ps) cat "$T/containers" ;;
  *) echo "$*" >> "$T/docker.log" ;;
esac
exit 0
DK
chmod +x "$T/bin/docker"

setmem() { printf 'MemTotal: 131072000 kB\nMemAvailable: %s kB\nMemFree: %s kB\nCached: 1000 kB\n' "$(($1*1024))" "$(($1*1024))" > "$T/meminfo"; }
setctr() { : > "$T/containers"; local i; for i in $(seq 1 "$1"); do echo "prod-$i" >> "$T/containers"; done; }
idle() { echo '{"running":[]}' > "$T/running"; }
g() { GUARD_NOW=${NOW:-$NOON} "$GUARD" "$@"; }
gr() { env -u GUARD_NOW "$GUARD" "$@"; }
reset() { setmem 82000; setctr 14; idle; rm -rf "$T/run" "$T/runs" "$T/STOP" "$T/docker.log" "$T/state"; mkdir -p "$T/run"; }
setpause() { mkdir -p "$T/state"; printf "%s" "$1" > "$T/state/PAUSED"; }

# 1
reset; g precheck 2048 60 >/dev/null; check "1 D5 passes" "[ $? = 0 ]"
# 2
reset; echo '{"running":[{"model":"a"}]}' > "$T/running"; g precheck 2048 60 >/dev/null; check "2 /running not empty refused" "[ $? = 75 ]"
# 3
reset; rm -f "$T/running"; g precheck 2048 60 >/dev/null; check "3 curl target missing refused" "[ $? = 75 ]"
# 4
reset; setmem 18000; g precheck 2048 60 >/dev/null; check "4 D5 refused at Mem 18000" "[ $? = 75 ]"
# 5
reset; SLOTSTREAM_MODE=D1 g precheck 19000 60 >/dev/null; r1=$?
SLOTSTREAM_MODE=D1 g precheck 19500 60 >/dev/null; r2=$?
check "5 D1 19000 passes, 19500 refused" "[ $r1 = 0 ] && [ $r2 = 75 ]"
# 6
reset; g precheck 70000 60 >/dev/null; check "6 footprint above ceiling refused" "[ $? = 75 ]"
# 7
reset; NOW=$(date -d "$(date +%Y-%m-%d) 23:30" +%s) g precheck 2048 1200 >/dev/null; r1=$?
NOW=$(date -d "$(date +%Y-%m-%d) 23:30" +%s) g precheck --allow-truncate 2048 1200 >/dev/null; r2=$?
check "7 23:30 DUR 1200 refused, truncate passes" "[ $r1 = 75 ] && [ $r2 = 0 ]"
# 8
reset; NOW=$(date -d "$(date +%Y-%m-%d) 02:00" +%s) g precheck 2048 60 >/dev/null; check "8 02:00 refused" "[ $? = 75 ]"
# 2a
reset; rm -f "$T/running"; SLOTSTREAM_RUNNING_URL= QUIET= g precheck 2048 60 > "$T/p2a.out"; r=$?
check "2a no URL: precheck passes and reports running off" "[ $r = 0 ] && grep -q '\"running\":\"off\"' \"$T/p2a.out\""
# 2b
reset; rm -f "$T/running"; g precheck 2048 60 >/dev/null; check "2b URL set but unreadable: refused" "[ $? = 75 ]"
# 9a
reset; setctr 3; g precheck 2048 60 >/dev/null; check "9a no container count configured: 3 containers pass" "[ $? = 0 ]"
# 9
reset; setctr 13; EXPECTED_CONTAINERS=14 g precheck 2048 60 >/dev/null; check "9 13 containers refused with EXPECTED_CONTAINERS=14" "[ $? = 75 ]"
# 10
reset; echo slotstream-spark-x >> "$T/containers"; g precheck 2048 60 >/dev/null; check "10 own container refused" "[ $? = 75 ]"
# 11
reset; flock -n "$T/.model.lock" sleep 20 & holder=$!; sleep 1
g precheck 2048 60 >/dev/null; r1=$?; g precheck --nolock 2048 60 >/dev/null; r2=$?
kill $holder 2>/dev/null; pkill -f 'sleep 20' 2>/dev/null; wait $holder 2>/dev/null; sleep 0.3
check "11 lock held refused, --nolock passes" "[ $r1 = 75 ] && [ $r2 = 0 ]"
# 12
reset; setsid sleep 30 & sp=$!; sleep 0.5
printf 'pgid=%s\nctr=\nlabel=x\nstart=0\n' "$sp" > "$T/run/x.run"
g precheck --exclusive 2048 60 >/dev/null; r1=$?; g precheck 2048 60 >/dev/null; r2=$?
kill $sp 2>/dev/null
check "12 --exclusive refused with live .run" "[ $r1 = 75 ] && [ $r2 = 0 ]"
# 13
reset; SLOTSTREAM_MODE=X g precheck 2048 60 >/dev/null; check "13 mode X refused" "[ $? = 75 ]"
# 14
reset; gr run t14 2048 60 -- bash -c 'exit 3' >/dev/null; rc=$?
check "14 rc 3 passes through, guard-post and LAST written" \
  "[ $rc = 3 ] && [ -s \"\$(cat $T/runs/LAST-t14)/guard-post.json\" ]"
# 15
reset; ( sleep 3; echo '{"running":[{"model":"a"}]}' > "$T/running" ) &
s=$(date +%s); gr run t15 2048 120 -- sleep 60 >/dev/null; rc=$?; e=$(( $(date +%s) - s ))
d=$(cat "$T/runs/LAST-t15")
check "15 /running flip cuts 76 within 20 s, group gone, event logged" \
  "[ $rc = 76 ] && [ $e -le 20 ] && grep -q '\"reason\":\"running\"' \"$d/guard-events.jsonl\" && ! pgrep -f 'sleep 60' >/dev/null"
# 16
reset; ( sleep 3; touch "$T/run/STOP" ) &
s=$(date +%s); gr run t16 2048 120 -- sleep 60 >/dev/null; rc=$?; e=$(( $(date +%s) - s ))
check "16 STOP cuts 76" "[ $rc = 76 ] && [ $e -le 20 ]"
rm -f "$T/run/STOP"
# 17
reset
s=$(date +%s); gr run t17 2048 5 -- bash -c "trap '' TERM; sleep 61" >/dev/null; rc=$?; e=$(( $(date +%s) - s ))
check "17 deadline cut 76 within 20 s with a TERM-ignoring command" \
  "[ $rc = 76 ] && [ $e -le 20 ] && ! pgrep -f 'sleep 61' >/dev/null"
# 18
reset; out=$(gr run t18 auto 60 -- bash -c 'echo FOOTPRINT=$SLOTSTREAM_FOOTPRINT_MIB'); rc=$?
check "18 auto resolves to 64848 and is exported" "[ $rc = 0 ] && echo \"\$out\" | grep -qx FOOTPRINT=64848"

# ---- pause mode: $GUARD_ROOT/state/PAUSED (NO_WINDOWS=1, EXPECTED_CONTAINERS=<n>)
TWO=$(date -d "$(date +%Y-%m-%d) 02:00" +%s)
LATE=$(date -d "$(date +%Y-%m-%d) 23:30" +%s)
P13=$'NO_WINDOWS=1\nEXPECTED_CONTAINERS=13\n'
# 19
reset; setctr 13; NOW=$TWO g precheck 2048 60 >/dev/null; r1=$?
setpause "$P13"; NOW=$TWO g precheck 2048 60 >/dev/null; r2=$?
check "19 02:00 + 13 containers: refused without PAUSED, passes with it" "[ $r1 = 75 ] && [ $r2 = 0 ]"
# 20
reset; setpause "$P13"; setctr 14; NOW=$TWO g precheck 2048 60 >/dev/null
check "20 PAUSED expecting 13: 14 containers refused" "[ $? = 75 ]"
# 21
reset; NOW=$LATE g precheck 2048 1200 >/dev/null; r1=$?
setpause $'NO_WINDOWS=1\n'; NOW=$LATE g precheck 2048 1200 >/dev/null; r2=$?
check "21 23:30 DUR 1200 refused, passes with NO_WINDOWS=1 (14 containers kept)" "[ $r1 = 75 ] && [ $r2 = 0 ]"
# 22
reset; setctr 13; setpause "$P13"; EXPECTED_CONTAINERS=14 NOW=$TWO g precheck 2048 60 >/dev/null
check "22 explicit EXPECTED_CONTAINERS=14 wins over the file" "[ $? = 75 ]"
# 23
reset; setpause $'NO_WINDOWS=maybe\n'; g precheck 2048 60 >/dev/null 2>&1; r1=$?
setpause ""; g precheck 2048 60 >/dev/null 2>&1; r2=$?
check "23 malformed PAUSED is an error (rc 2), never a silent default" "[ $r1 = 2 ] && [ $r2 = 2 ]"
# 24
reset; setctr 13; setpause "$P13"; NOW=$TWO g precheck --allow-truncate 2048 60 >/dev/null; r1=$?
check "24 --allow-truncate 23:40 rule skipped too (02:00, paused)" "[ $r1 = 0 ]"
# 25 (real clock: the run is not cut by the 23:45 segment, post file carries expected_containers and paused)
reset; setctr 13; setpause "$P13"; gr run t25 2048 60 -- bash -c 'exit 0' >/dev/null; rc=$?
d=$(cat "$T/runs/LAST-t25")
check "25 run under PAUSED: rc 0, guard-post expected_containers 13 and paused true" \
  "[ $rc = 0 ] && grep -q '\"expected_containers\":13' \"$d/guard-post.json\" && grep -q '\"paused\":true' \"$d/guard-post.json\""
# 26
reset; EXPECTED_CONTAINERS=14 gr run t26 2048 60 -- bash -c 'exit 0' >/dev/null; d=$(cat "$T/runs/LAST-t26")
check "26 run without PAUSED, EXPECTED_CONTAINERS=14: expected_containers 14 and paused false" \
  "grep -q '\"expected_containers\":14' \"$d/guard-post.json\" && grep -q '\"paused\":false' \"$d/guard-post.json\""
# 26a
reset; setctr 3; gr run t26a 2048 60 -- bash -c 'exit 0' >/dev/null; rc=$?; d=$(cat "$T/runs/LAST-t26a")
check "26a run with no count configured: rc 0 and guard-post expected_containers null" \
  "[ $rc = 0 ] && grep -q '\"expected_containers\":null' \"$d/guard-post.json\""
# 27 kill-all: skipped under NO_WINDOWS, kills with --force or the STOP file, kills without PAUSED
KA=$PWD/deploy/kill-all.sh
ka() { GUARD_ROOT=$T GUARD_DOCKER=$T/bin/docker KILLALL_WAIT=2 KILLALL_SYSTEMCTL=true bash "$KA" "$@" 2>&1; }
mkrun() { setsid sleep 47 & kp=$!; sleep 0.4; printf 'pgid=%s\nctr=\nlabel=k\nstart=0\n' "$kp" > "$T/run/k.run"; }
reset; setpause "$P13"; mkrun; out=$(ka); alive1=0; kill -0 $kp 2>/dev/null && alive1=1
out2=$(ka --force); sleep 0.5; alive2=0; kill -0 $kp 2>/dev/null && alive2=1
check "27a kill-all skips under NO_WINDOWS=1 (logged) and --force kills" \
  "[ $alive1 = 1 ] && echo \"\$out\" | grep -q 'skipped' && [ $alive2 = 0 ]"
reset; setpause "$P13"; mkrun; touch "$T/run/STOP"; ka >/dev/null; sleep 0.5; alive3=0; kill -0 $kp 2>/dev/null && alive3=1; kill $kp 2>/dev/null
check "27b kill-all kills under PAUSED when the STOP file exists" "[ $alive3 = 0 ]"
reset; mkrun; ka >/dev/null; sleep 0.5; alive4=0; kill -0 $kp 2>/dev/null && alive4=1; kill $kp 2>/dev/null
check "27c kill-all without PAUSED kills as before" "[ $alive4 = 0 ]"

echo "guard_t0: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ]
