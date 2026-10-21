#!/bin/bash
# Spark: build the pinned llama.cpp (LB0). Idempotent.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$HERE/tools/spark/env.sh"
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
mkdir -p "$LB" && cd "$LB" || exit 1
[ -d llama.cpp ] || git clone https://github.com/ggml-org/llama.cpp || exit 1
cd llama.cpp && git checkout 23b0202a189c44a54625aadcb37a946dd1d6278d || exit 1
systemd-run --user --scope -p MemoryMax=8G nice -n 19 sh -c 'cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=121a-real -DGGML_CUDA_NO_VMM=ON -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build -j 4 --target llama-server llama-cli llama-bench llama-perplexity' || exit 1
build/bin/llama-server --version > "$LB/build-version.txt" 2>&1
exit 0
