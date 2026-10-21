#!/bin/bash
# Mac: rsync the working tree to $R/src on the Spark. Prints "synced <id>".
set -eu
cd "$(dirname "$0")/../.."
. tools/spark/env.sh

if ! spark_ssh command -v rsync >/dev/null 2>&1; then
  echo "rsync missing on the Spark" >&2
  exit 3
fi
spark_ssh mkdir -p $SRC

rsync -a --delete -e "ssh $SPARK_SSH_OPTS" \
  --exclude=.git --exclude=/build/ \
  --exclude=/bench/traces/ --exclude=/tests/goldens/data/ --exclude='*.so' \
  --exclude=__pycache__/ --exclude=/.venv/ --exclude=/tools/spark/env.local.sh \
  ./ "$SPARK_HOST:$SRC/"

id=$(sync_id)
spark_ssh "echo $id > $SRC/.synced"
echo "synced $id"
