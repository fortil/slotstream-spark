#!/bin/bash
# Spark: run CMD from a frozen copy of $R/src so a later sync.sh cannot change a long job.
#   snap.sh LABEL CMD...
# Copies $R/src/ (build/ included, goldens linked) to $R/run/jobs/LABEL/src/, links runs/ beside it, exports SLOTSTREAM_SRC and
# runs CMD there. py.sh and the Python path follow SLOTSTREAM_SRC.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$HERE/tools/spark/env.sh"
LABEL=${1:?usage: snap.sh LABEL CMD...}
shift
[ $# -gt 0 ] || { echo "snap.sh: missing command" >&2; exit 2; }
D=$R/run/jobs/$LABEL
mkdir -p "$D/src" "$R/runs"
# the goldens (many GB) are linked, not copied
rsync -a --delete --exclude=/tests/goldens/data "$R/src/" "$D/src/"
if [ -d "$R/src/tests/goldens/data" ]; then ln -sfn "$R/src/tests/goldens/data" "$D/src/tests/goldens/data"; fi
ln -sfn "$R/runs" "$D/runs"
export SLOTSTREAM_SRC=$D/src
cd "$D/src"
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true
exec "$@"
