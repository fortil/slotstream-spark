# 4b-mlx: MLX-CUDA preflight on the Spark
Date: 2026-10-11
Capture: db/sources/runs/2026/10/20261011T172756-4b-mlx/
Command: tools/spark/run.sh --heavy 4b-mlx 2048 300 --nolock -- tools/capture.sh 4b-mlx -- <venv-mlx python> tools/probes/mlx_probe.py

| item | value |
|---|---|
| mlx | 0.32.3 |
| default device | gpu 0, NVIDIA GB10, sm_121 |
| cuda_is_available | true |
| quantize (gs64, gs32; gate/up 2560->640, down 640->2560) | ran without ERR; max abs dequant error 0.41-0.45 on random bf16 |
| quantized_matmul gs64 gate/up | B1 7.3 us amortized (126 GB/s weights), B256 33.3 us (25.2 TFLOPs) |
| quantized_matmul gs64 down | B1 9.1 us amortized (102 GB/s), B256 35.6 us (23.6 TFLOPs) |
| gather_qmm (E=128, top10, gs64) | T1 100.6 us sorted, T8 745 us sorted, T256 5131 us sorted (1.6 TFLOPs; dense same-FLOPs 36.3 TFLOPs) |
| peak memory | 1.178 GiB |
| mlx-lm 0.32.0 | `qwen4_exp` not supported; `qwen3_next` loads |
| ERR entries | 0 |

Effect on the run: MLX-CUDA works on the Spark, so `tools/spark/mtp_build.sh` converts the MTP tensors locally first and uses the mirror only as the comparison and fallback.

Verdict: PASS — cuda_is_available is true and the quantize tests ran without ERR
