#!/bin/bash
# Spark: IO-2 on our own checkpoint shards. Run under tools/capture.sh (CAP set).
# 19 iobench runs -> $CAP/io2.jsonl, then io2_summary.py -> $CAP/summary.json.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$HERE/tools/spark/env.sh"
: "${CAP:?io2.sh must run under capture.sh}"
cd "$HERE"

OUT=$CAP/io2.jsonl
: > "$OUT"
F=("$MODEL"/model-000{01..10}.safetensors)
for f in "${F[@]}"; do [ -f "$f" ] || { echo "missing $f" >&2; exit 2; }; done

dontneed() {
  python3 - "$@" <<'PY'
import os, sys
for p in sys.argv[1:]:
    fd = os.open(p, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
PY
}

n=0
# bench [--nodrop-before] LABEL FILES... -- iobench args
# Buffered runs (no --direct) drop the page cache of their files before and after.
bench() {
  local before=1
  if [ "$1" = "--nodrop-before" ]; then before=0; shift; fi
  local label=$1; shift
  local files=() args=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do files+=("$1"); shift; done
  shift
  args=("$@")
  local buffered=1
  for a in "${args[@]}"; do [ "$a" = "--direct" ] && buffered=0; done
  [ $buffered = 1 ] && [ $before = 1 ] && dontneed "${files[@]}"
  local line
  line=$(python3 tools/probes/iobench.py run --dur 5 --label "$label" "${args[@]}" "${files[@]}") || { echo "iobench failed: $label ($line)" >&2; exit 3; }
  echo "$line" >> "$OUT"
  echo "$line" | python3 -c 'import json,sys;d=json.loads(sys.stdin.read());print(d.get("label"),d.get("GBps"),d.get("lat_us",{}).get("p50"))'
  [ $buffered = 1 ] && dontneed "${files[@]}"
  n=$((n + 1))
}

M2=$MODEL/model-00002.safetensors M3=$MODEL/model-00003.safetensors
M4=$MODEL/model-00004.safetensors M5=$MODEL/model-00005.safetensors
M6=$MODEL/model-00006.safetensors M7=$MODEL/model-00007.safetensors

# b2: large direct reads on all shards
for qd in 1 12 32; do bench "b2-819200-qd$qd" "${F[@]}" -- --bs 819200 --direct --qd $qd; done
for qd in 1 12 32; do bench "b2-51200-qd$qd" "${F[@]}" -- --bs 51200 --direct --qd $qd; done
# c1: small direct reads (row-format decision uses the 512 and 1024 runs)
bench c1-4096-qd1 "${F[@]}" -- --bs 4096 --direct --qd 1
bench c1-4096-qd32 "${F[@]}" -- --bs 4096 --direct --qd 32
bench c1-512-qd1 "${F[@]}" -- --bs 512 --slot 4096 --direct --qd 1
bench c1-1024-qd1 "${F[@]}" -- --bs 1024 --slot 4096 --direct --qd 1
# c2 / c3: 100-byte unaligned reads, buffered, with and without FADV_RANDOM
bench c2-ng100-qd1 "$M2" -- --bs 100 --slot 4096 --jitter --fadv-random --qd 1
bench c2-ng100-qd32 "$M2" -- --bs 100 --slot 4096 --jitter --fadv-random --qd 32
bench c3-ng100-qd1 "$M3" -- --bs 100 --slot 4096 --jitter --qd 1
bench c3-ng100-qd32 "$M3" -- --bs 100 --slot 4096 --jitter --qd 32
# d1: expert-sized buffered reads over the first 8 GiB of one shard
bench d1-exp-buf-qd1 "$M4" -- --bs 2764800 --limit-bytes 8589934592 --qd 1
bench d1-exp-buf-qd12 "$M5" -- --bs 2764800 --limit-bytes 8589934592 --qd 12
bench d1-exp-buf-qd32 "$M6" -- --bs 2764800 --limit-bytes 8589934592 --qd 32
# d2: warm page cache (2 GiB read first), repeated random reads inside it
dontneed "$M7"
dd if="$M7" of=/dev/null bs=16M count=128 status=none
bench --nodrop-before d2-hot-qd1 "$M7" -- --bs 2764800 --repeat --limit-bytes 2147483648 --qd 1
bench --nodrop-before d2-hot-qd12 "$M7" -- --bs 2764800 --repeat --limit-bytes 2147483648 --qd 12

echo "io2: $n runs"
[ "$n" = 19 ] || { echo "expected 19 runs" >&2; exit 4; }
python3 tools/spark/io2_summary.py "$CAP" > "$CAP/summary.json" || exit 5
