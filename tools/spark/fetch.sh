#!/bin/bash
# Mac: bring Spark results back.
#   fetch.sh                      new capture dirs -> db/sources/runs/, and newer text files of $R/src/db/
#   fetch.sh --last LABEL         copy that capture, print its local path
#   fetch.sh REMOTE LOCAL         copy a directory (REMOTE relative to $R unless absolute)
#   fetch.sh --file REMOTE LOCAL  copy one file
set -eu
cd "$(dirname "$0")/../.."
. tools/spark/env.sh

FIND_TXT="find . -type f \\( -name '*.json' -o -name '*.jsonl' -o -name '*.txt' -o -name '*.csv' -o -name '*.md' -o -name '*.log' \\) -size -51200k -print0"

rpath() { case "$1" in /*) echo "$1" ;; *) echo "$R/$1" ;; esac; }

# fetch_dir REMOTE_DIR LOCAL_DIR [extra tar args]
fetch_dir() {
  local rd=$1 ld=$2
  shift 2
  mkdir -p "$ld"
  spark_ssh "cd $rd && $FIND_TXT | tar --null -T - -cf -" | tar -xf - -C "$ld" "$@"
}

if [ "${1:-}" = "--last" ]; then
  label=$2
  last=$(spark_ssh "cat $R/runs/LAST-$label")
  base=$(basename "$last")
  mm=$(basename "$(dirname "$last")")
  yy=$(basename "$(dirname "$(dirname "$last")")")
  dest="db/sources/runs/$yy/$mm/$base"
  fetch_dir "$last" "$dest"
  echo "$dest"
  exit 0
fi

if [ "${1:-}" = "--file" ]; then
  mkdir -p "$(dirname "$3")"
  spark_ssh "cat $(rpath "$2")" > "$3"
  exit 0
fi

if [ $# -eq 2 ]; then
  fetch_dir "$(rpath "$1")" "$2"
  exit 0
fi

# No arguments: (a) captures not yet present, (b) newer text files under $R/src/db/.
caps=$(spark_ssh "cd $R/runs 2>/dev/null && find . -mindepth 3 -maxdepth 3 -type d | sed 's|^\./||'" || true)
for c in $caps; do
  [ -d "db/sources/runs/$c" ] && continue
  fetch_dir "$R/runs/$c" "db/sources/runs/$c"
  echo "captured $c"
done
if spark_ssh "test -d $SRC/db"; then
  fetch_dir "$SRC/db" "db" --keep-newer-files
fi
