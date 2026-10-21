# m-lb1: H2D copy of 1,024,000-byte pieces from GGUF file pages
Date: 2026-10-11
Capture: db/sources/runs/2026/10/20261011T192510-m-lb1/
Command: tools/spark/run.sh --heavy m-lb1 2048 600 --exclusive --contenedor -- tools/capture.sh m-lb1 -- env SLOTSTREAM_ENV_KIND=container SLOTSTREAM_DOCKER_ARGS='--memory 2g' tools/spark/py.sh tools/probes/mlb1_copy.py --gguf-rel lb/gguf/Qwen3.8-Flash-Next-Q4_1/Qwen3.8-Flash-Next-Q4_1-00001-of-00003.gguf (container with --memory 2g; 300 pieces of 1,024,000 B at seeded random 4096-aligned offsets of a 39.5 GB shard; read-only copy-on-write np.memmap)

| condition | p50 ms | p99 ms | GB/s |
|---|---|---|---|
| (a) cold, after posix_fadvise(DONTNEED) | 1.7977 | 3.5924 | 0.520 |
| (b) hot, second pass | 0.0294 | 0.0319 | 34.227 |
| (c) from an anonymous numpy copy (copy to anon p50 0.0641 ms, not in the GB/s) | 0.0279 | 0.0546 | 23.412 |
| (d) from pinned memory | 0.0282 | 0.0299 | 36.193 |

GB/s is total bytes over the summed copy time. The shard's pages were dropped at the end.

Verdict: INFO — hot file pages copy to the GPU at 34.2 GB/s (>= 10), cold pages at 0.52 GB/s
