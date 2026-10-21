#!/bin/bash
# Spark: build tools/probes/spark_probe.cu into $R/tmp/spark_probe.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$HERE/tools/spark/env.sh"
mkdir -p "$R/tmp"
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -arch=sm_121 -o "$R/tmp/spark_probe" "$HERE/tools/probes/spark_probe.cu" \
  -L/usr/local/cuda/lib64/stubs -lcuda -lpthread
echo "built $R/tmp/spark_probe"
