#!/bin/bash
# Build and test the native slotio module. Run from the repo root (on the Spark: tools/spark/run.sh -- bash tools/build_slotio.sh).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=$(tools/spark/py.sh -c 'import sys;print(sys.executable)')
sha256sum -c slotio/tests/doctest.h.sha256
cmake -S slotio -B build/slotio -DPython_EXECUTABLE="$PY" \
  -Dnanobind_DIR="$("$PY" -m nanobind --cmake_dir)" -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build/slotio -j8 \
  && ctest --test-dir build/slotio --output-on-failure
