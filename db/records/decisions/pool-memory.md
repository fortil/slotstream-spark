# pool-memory
Date: 2026-10-11
Decision: vmm_host_numa
Evidence: db/sources/runs/2026/10/20261011T192313-spark-probe/result-pool.json, db/records/measurements/pool.md
Reversal: Decision §7 row 4. Rule fixed before measuring: highest gather GB/s among Triton-addressable types, within 2% prefer host_alloc. vmm_host_numa gathered 245.6 vs 226.6 GB/s for host_alloc (8%). Fall back to host_alloc if cuMemCreate on host NUMA fails on the target driver or the gain disappears in the engine (F3 A/B).
