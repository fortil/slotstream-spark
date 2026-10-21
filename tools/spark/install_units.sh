#!/bin/bash
# Spark, idempotent, light: link the slotstream-spark user units (files stay in $R/src/deploy), enable and start both
# timers, start the socket only inside 04:00-23:59, merge the tagged crontab lines. Never touches other units or lines.
# Pause mode (RUNBOOK): $R/state/PAUSED with NO_WINDOWS=1 is not read here; the user runs this script on the host and
# decides whether to keep the timers and the cron lines while production is paused. Agents do not run it.
set -eu
cd "$(dirname "$0")/../.."
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
D=$PWD/deploy
UNITS="slotstream-spark.socket slotstream-spark-front.service slotstream-spark.service
slotstream-spark-start.timer slotstream-spark-start.service slotstream-spark-stop.timer slotstream-spark-stop.service"
for u in $UNITS; do
  [ -f "$D/$u" ] || { echo "missing $D/$u" >&2; exit 1; }
  systemctl --user link "$D/$u"
done
systemctl --user daemon-reload
systemctl --user enable --now slotstream-spark-start.timer slotstream-spark-stop.timer
m=$((10#$(date +%H) * 60 + 10#$(date +%M)))
if [ "$m" -ge 240 ] && [ "$m" -lt 1439 ]; then
  systemctl --user start slotstream-spark.socket
  echo "socket started (minute $m)"
else
  echo "socket not started: outside 04:00-23:59 (minute $m)"
fi
(crontab -l 2>/dev/null | grep -v '#slotstream-spark' || true; cat deploy/crontab.backup) | crontab -
echo "units installed"
