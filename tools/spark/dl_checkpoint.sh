#!/bin/bash
# Spark: download the pinned checkpoint and verify its 24 files.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$HERE/tools/spark/env.sh"
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}

mkdir -p "$MODEL"
bash "$HERE/tools/spark/hf_fetch.sh" pipenetwork/Qwen3.8-Flash-Next-MLX-4bit aa7c790e804bbf9d491ddb109c3d61bc4a555f7c "$MODEL" || exit $?
cp "$SRC/SHA256SUMS.model" "$R/model/"
(cd "$MODEL" && systemd-run --user --scope -p MemoryMax=4G env LC_ALL=C sha256sum -c ../SHA256SUMS.model) > "$R/model/verify-ckpt.txt" 2>&1
vrc=$?

# Drop the page cache of the files we just read (own files only).
python3 - "$MODEL" "$R/model/SHA256SUMS.model" <<'PY'
import os, sys
d, sums = sys.argv[1:3]
for line in open(sums):
    name = line.split(None, 1)[1].strip().lstrip("*")
    p = os.path.join(d, name)
    if os.path.isfile(p):
        fd = os.open(p, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
PY

ok=$(grep -c ': OK$' "$R/model/verify-ckpt.txt")
echo "verified $ok/24 files OK"
[ "$vrc" = 0 ] && [ "$ok" = 24 ] && exit 0
exit 1
