#!/bin/bash
# Spark: download and verify a pinned GGUF set. dl_gguf.sh KEY (q41|iq3xxs)
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$HERE/tools/spark/env.sh"
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
KEY=${1:?usage: dl_gguf.sh KEY}
LOCK=$HERE/deploy/lb/gguf.lock.json
GG=$LB/gguf
mkdir -p "$GG"

read -r REPO REV < <(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]))[sys.argv[2]];print(d["repo"],d["revision"])' "$LOCK" "$KEY") || exit 2
PATHS=$(python3 -c 'import json,sys;print("\n".join(f["path"] for f in json.load(open(sys.argv[1]))[sys.argv[2]]["files"]))' "$LOCK" "$KEY")
mapfile -t ARR <<< "$PATHS"

bash "$HERE/tools/spark/hf_fetch.sh" "$REPO" "$REV" "$GG" "${ARR[@]}" || exit $?

python3 -c 'import json,sys
for f in json.load(open(sys.argv[1]))[sys.argv[2]]["files"]: print(f["sha256"]+"  "+f["path"])' "$LOCK" "$KEY" > "$GG/sums-$KEY.txt"
(cd "$GG" && systemd-run --user --scope -p MemoryMax=4G env LC_ALL=C sha256sum -c "sums-$KEY.txt") > "$GG/verify-$KEY.txt" 2>&1
vrc=$?
python3 - "$GG/sums-$KEY.txt" "$GG" <<'PY'
import os, sys
for line in open(sys.argv[1]):
    p = os.path.join(sys.argv[2], line.split(None, 1)[1].strip())
    if os.path.isfile(p):
        fd = os.open(p, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
PY
n=$(wc -l < "$GG/sums-$KEY.txt")
ok=$(grep -c ': OK$' "$GG/verify-$KEY.txt")
echo "verified $ok/$n"
[ "$vrc" = 0 ] && [ "$ok" = "$n" ] && exit 0
exit 1
