#!/bin/bash
# Spark: compile tools/quality/ppl_ids.cpp against the pinned llama.cpp build (step 12). Idempotent: nothing happens while
# $LB/bin/ppl-ids is newer than the source. The compile runs in a small memory-limited scope at low priority.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$HERE/tools/spark/env.sh"
LB=${LB/#\~/$HOME}
L=$LB/llama.cpp
SRC=$HERE/tools/quality/ppl_ids.cpp
OUT=$LB/bin/ppl-ids
mkdir -p "$LB/bin"
if [ -x "$OUT" ] && [ "$OUT" -nt "$SRC" ]; then echo "build_ppl_ids: $OUT is up to date"; exit 0; fi
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
systemd-run --user --scope --unit=slotstream-spark-ppl-ids-build -p MemoryMax=4G -p MemorySwapMax=0 nice -n 19 \
  c++ -std=c++17 -O2 -DGGML_BACKEND_SHARED -DGGML_SHARED -DLLAMA_SHARED \
  -I"$L/common" -I"$L/vendor/nlohmann/.." -I"$L/include" -I"$L/ggml/include" \
  -o "$OUT.tmp" "$SRC" \
  -Wl,-rpath,"$L/build/bin" "$L/build/bin/libllama-common.so" "$L/build/common/libllama-common-base.a" \
  "$L/build/bin/libllama.so" "$L/build/bin/libggml.so" "$L/build/bin/libggml-base.so" -lpthread
mv "$OUT.tmp" "$OUT"
echo "build_ppl_ids: built $OUT"
