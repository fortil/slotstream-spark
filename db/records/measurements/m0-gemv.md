# m0-gemv: affine 4-bit Triton GEMV and MoE read on the GB10
Date: 2026-10-11
Capture: db/sources/runs/2026/10/20261011T190550-m0-gemv/
Command: tools/spark/run.sh --heavy m0-gemv 4096 2400 --exclusive -- tools/capture.sh m0-gemv -- tools/spark/py.sh bench/m0/gemv_bench.py --moe (torch 2.13.0+cu130, triton 3.7.1)
Earlier passes under the previous gate rule (triton do_bench only): db/sources/runs/2026/10/20261011T184001-m0-gemv/ (2560x2560 = 92.08 GB/s) and db/sources/runs/2026/10/20261011T185514-m0-gemv/ (93.84 GB/s, after one tuning pass). Kernel unchanged in this run.

Gate rule (user decision of 2026-10-11): the GEMV is timed as call time inside a captured CUDA graph, with 128 MiB written between calls to evict L2. Two graphs are replayed, one with 20 x (flush, call) and one with 20 x flush; the call time is the median difference per call (15 replays each). The five fastest do_bench configs per shape are re-timed this way and the fastest is kept. GB/s counts N*K/2 + 2*N*K/64*2 + 2K + 4N bytes over that time. Threshold unchanged: >= 121.5 GB/s (50% of 243) at 2560 x 2560. The do_bench column is informational and does not gate.

| shape (N x K) | cold-L2 graph GB/s (GATE) | fraction of 243 | cold-L2 us/call | cold-L2 config | do_bench GB/s (informational) | warm-L2 graph GB/s (informational) | rel err |
|---|---|---|---|---|---|---|---|
| 640x2560 | 147.82 | 0.6083 | 6.29 | v3, BLOCK_N 4, BLOCK_K 512, SPLIT_K 1, 2 warps | 64.87 | 130.76 | 1.7e-07 |
| 1280x2560 | 182.42 | 0.7507 | 10.16 | v3, BLOCK_N 4, BLOCK_K 256, SPLIT_K 1, 4 warps | 78.16 | 272.18 | 3.0e-07 |
| 2560x2560 | 193.75 | 0.7973 | 19.11 | v3, BLOCK_N 32, BLOCK_K 256, SPLIT_K 1, 8 warps | 92.86 | 274.23 | 4.2e-07 |
| 248320x2560 | 232.83 | 0.9581 | 1540.12 | v3, BLOCK_N 4, BLOCK_K 512, SPLIT_K 1, 2 warps | 218.39 | 227.99 | 2.5e-07 |

Correctness is checked on the first 2048 rows against x @ dequant_affine_torch(...).T in fp32 (all below 1e-3). The warm-L2 column replays 50 launches from one graph without flushing; it is informational. The do_bench numbers sit well below the graph numbers because do_bench adds a per-call launch and L2-flush gap that a plain Triton read of the same 3.7 MB also pays (102 GB/s, ad hoc diagnostic outside the capture, previous round).

The gate item is the 2560 x 2560 shape: 193.75 GB/s cold-L2 graph = 79.7% of 243 GB/s, over the 121.5 GB/s line (do_bench figure 92.86 GB/s, informational).

MoE read from a Triton gather over 10 random slots of 2,764,800 B: device memory 150.09 GB/s, pinned host memory 228.65 GB/s.

Verdict: PASS — cold-L2 graph GEMV 193.8 GB/s = 79.7% of 243 >= 50% (121.5 GB/s) at 2560 x 2560
