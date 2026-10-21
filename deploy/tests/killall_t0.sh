#!/bin/bash
# Spark self-test for deploy/kill-all.sh.
cd "$(dirname "$0")/../.." || exit 1
R=${SLOTSTREAM_R_REMOTE:-$HOME/slotstream-spark}
mkdir -p "$R/tmp"
T=$(mktemp -d "$R/tmp/killall_t0.XXXXXX")
MAIN=$$
trap '[ "$BASHPID" = "$MAIN" ] && rm -rf "$T"' EXIT
mkdir -p "$T/run" "$T/bin"
cat > "$T/bin/docker" <<DK
#!/bin/bash
echo "\$*" >> "$T/docker.log"
exit 0
DK
chmod +x "$T/bin/docker"

setsid bash -c "trap '' TERM; sleep 600" &
sleep 1
pgid=$(pgrep -f 'sleep 600' | head -1 | xargs -I{} ps -o pgid= -p {} | tr -d ' ')
printf 'pgid=%s\nctr=slotstream-spark-t0\nlabel=t0\nstart=0\n' "$pgid" > "$T/run/t0.run"

GUARD_ROOT=$T GUARD_DOCKER=$T/bin/docker KILLALL_WAIT=2 KILLALL_SYSTEMCTL=true \
  bash deploy/kill-all.sh >/dev/null
rc=$?

fail=0
[ $rc = 0 ] || { echo "FAIL exit $rc"; fail=1; }
[ -z "$(pgrep -g "$pgid" 2>/dev/null)" ] || { echo "FAIL group alive"; fail=1; }
[ ! -e "$T/run/t0.run" ] || { echo "FAIL .run not removed"; fail=1; }
grep -qx 'stop -t 10 slotstream-spark-t0' "$T/docker.log" || { echo "FAIL no docker stop"; fail=1; }
grep -qx 'kill slotstream-spark-t0' "$T/docker.log" || { echo "FAIL no docker kill"; fail=1; }
# Drill case k under pause mode: run kill-all with the arguments tools/spark/drill_guard.sh passes it.
# Without --force, kill-all skips under NO_WINDOWS=1 and the drill's case k fails (2026-10-20).
mkdir -p "$T/state"
printf 'NO_WINDOWS=1\nEXPECTED_CONTAINERS=12\n' > "$T/state/PAUSED"
dargs=$(sed -n 's|.*"$HERE/deploy/kill-all.sh"\(.*\) > "$DRILL/k/kill-all.txt".*|\1|p' tools/spark/drill_guard.sh)
grep -q '"$HERE/deploy/kill-all.sh".* > "$DRILL/k/kill-all.txt"' tools/spark/drill_guard.sh ||
  { echo "FAIL drill: kill-all call not found in tools/spark/drill_guard.sh"; fail=1; }
setsid bash -c "trap '' TERM; sleep 601" &
sleep 1
pgid2=$(pgrep -f 'sleep 601' | head -1 | xargs -I{} ps -o pgid= -p {} | tr -d ' ')
printf 'pgid=%s\nctr=\nlabel=t0-drill\nstart=0\n' "$pgid2" > "$T/run/t0-drill.run"
# shellcheck disable=SC2086  # $dargs holds the drill's literal arguments
GUARD_ROOT=$T GUARD_DOCKER=$T/bin/docker KILLALL_WAIT=2 KILLALL_SYSTEMCTL=true \
  bash deploy/kill-all.sh $dargs >/dev/null
[ -z "$(pgrep -g "$pgid2" 2>/dev/null)" ] || { echo "FAIL drill: group alive under PAUSED (args: '$dargs')"; kill -KILL -- "-$pgid2" 2>/dev/null; fail=1; }
[ ! -e "$T/run/t0-drill.run" ] || { echo "FAIL drill: .run not removed under PAUSED"; fail=1; }

[ $fail = 0 ] && echo "killall_t0: passed"
exit $fail
