#!/bin/bash
# Spark, read-only: verify the units, list the two timers with their next elapse, check the tagged cron lines and linger.
set -u
cd "$(dirname "$0")/../.."
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
rc=0
fail() { echo "FAIL: $*"; rc=1; }
D=$PWD/deploy
systemd-analyze --user verify "$D"/slotstream-spark.socket "$D"/slotstream-spark-front.service \
  "$D"/slotstream-spark.service "$D"/slotstream-spark-start.timer "$D"/slotstream-spark-start.service \
  "$D"/slotstream-spark-stop.timer "$D"/slotstream-spark-stop.service && echo "verify: ok" || fail "systemd-analyze verify"
if grep -lE '^[[:space:]]*Restart=' "$D"/slotstream-spark*; then fail "a Restart= key"; fi
t=$(systemctl --user list-timers --all --no-legend 2>/dev/null | grep slotstream-spark || true)
echo "$t"
[ "$(echo "$t" | grep -c 'slotstream-spark-.*\.timer')" = 2 ] || fail "expected exactly two slotstream timers"
echo "$t" | grep -q 'slotstream-spark-start.timer' && echo "$t" | grep -q 'slotstream-spark-stop.timer' || fail "timer names"
n=$(crontab -l 2>/dev/null | grep -c '#slotstream-spark' || true)
echo "cron lines tagged: $n"
[ "$n" -ge 2 ] || fail "crontab tagged lines < 2"
crontab -l | grep '#slotstream-spark' | grep -q 'systemctl --user stop slotstream-spark.socket' || fail "stop line missing"
l=$(loginctl show-user "$(id -un)" -p Linger)
echo "$l"
[ "$l" = "Linger=yes" ] || fail "linger is not yes"
echo "socket: $(systemctl --user is-active slotstream-spark.socket 2>&1 || true)"
[ $rc = 0 ] && echo "check_units: ok"
exit $rc
