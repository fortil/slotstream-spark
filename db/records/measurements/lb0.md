# lb0: llama.cpp yardstick build and GGUF sets
Date: 2026-10-11
Capture: db/sources/runs/2026/10/20261011T183501-verify-artifacts/
Command: tools/spark/run.sh --heavy verify-artifacts 1024 900 --nolock -- tools/capture.sh verify-artifacts -- bash tools/spark/verify_artifacts.sh (build by the lb-build job, deploy/lb/build_llamacpp.sh; downloads by dl-q41 and dl-iq3, tools/spark/dl_gguf.sh)

| item | value |
|---|---|
| llama.cpp commit | 23b0202a189c44a54625aadcb37a946dd1d6278d |
| llama-server --version | version: 0.6.0-dev (build 11552, commit 23b0202a1) |
| build flags | GGML_CUDA=ON, CMAKE_CUDA_ARCHITECTURES=121a-real, GGML_CUDA_NO_VMM=ON, LLAMA_OPENSSL=OFF, LLAMA_BUILD_TESTS=OFF, LLAMA_BUILD_EXAMPLES=OFF, Release |
| targets | llama-server, llama-cli, llama-bench, llama-perplexity |
| Q4_1 GGUF (3 files) | 3/3 OK |
| IQ3_XXS GGUF (2 files) | 2/2 OK |

First LB gate item. Same locale fix as the checkpoint applied to the GGUF verification.

Verdict: PASS — commit 23b0202a, Q4_1 3/3 and IQ3_XXS 2/2 verified
