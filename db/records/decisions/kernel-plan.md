# kernel-plan
Date: 2026-10-11
Decision: A (Triton)
Evidence: db/sources/runs/2026/10/20261011T190550-m0-gemv/result.json; db/records/measurements/m0-gemv.md; db/records/measurements/m0-host.md; bench/m0/check_gates.py
Reversal: Decision §7 row 1. Plan A (Triton) holds while the 2560 x 2560 affine GEMV reaches 121.5 GB/s timed inside a captured CUDA graph with L2 flushed between replays (user rule of 2026-10-11); it measured 193.75 GB/s (79.7% of 243). The triton do_bench figure (92.86 GB/s) is informational. If a later run falls under 121.5 GB/s by that rule, go to plan B (port MLX qmv.cu) or plan C (llama.cpp MMVQ Q4_1).
