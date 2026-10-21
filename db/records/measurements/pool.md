# pool: pool-memory types (m2 gather, m6 counters, m7 memcpy) for one 1 GiB chunk
Date: 2026-10-11
Capture: db/sources/runs/2026/10/20261011T192313-spark-probe/
Command: tools/spark/run.sh --heavy spark-probe 6144 600 --exclusive -- tools/capture.sh spark-probe -- spark_probe pool --file model-00001.safetensors --slots 388 --lanes 1,4,8,10 --types host_alloc,vmm_host_numa,anon_thp --tokens 200 (388 slots of 2,764,800 B)

m2: GPU gather and sequential read of the pool, GB/s (references: 222-226 pinned, 165 ATS).

| type | triton addressable | gather GB/s | sequential GB/s |
|---|---|---|---|
| host_alloc | yes | 226.63 | 217.72 |
| vmm_host_numa | yes | 245.57 | 241.95 |
| anon_thp | no | 161.74 | 151.65 |

m6: counters before and after allocating the chunk.

| type | MemAvailable delta (MiB) | AnonPages delta (MiB) | Shmem delta (MiB) | cgroup memory delta (MiB) | NVML own delta (MiB) |
|---|---|---|---|---|---|
| host_alloc | -1039 | +7 | +1023 | +1039 | 0 |
| vmm_host_numa | -1031 | 0 | 0 | +3 | +1024 |
| anon_thp | -1020 | +1023 | 0 | +1025 | 0 |

The pool counts against MemAvailable in every type (about 1 GiB per chunk). host_alloc shows as Shmem and in the cgroup; vmm_host_numa shows in NVML and barely in the cgroup; anon_thp shows as AnonPages and in the cgroup.

m7: per-record memcpy (2,764,800 B) into the slot, p50 ms (p99 ms), and aggregate fill GB/s, by number of pread lanes. The memcpy is done by the lane threads on the CPU; p50 is flat across types.

| type | lanes 1 | lanes 4 | lanes 8 | lanes 10 |
|---|---|---|---|---|
| host_alloc | 0.1193 (0.4154), 4.14 GB/s | 0.1295 (0.2208), 10.36 GB/s | 0.1202 (0.1711), 10.91 GB/s | 0.1206 (0.1881), 11.01 GB/s |
| vmm_host_numa | 0.1192 (0.1537), 4.29 GB/s | 0.1294 (0.2252), 10.86 GB/s | 0.1238 (0.4869), 11.06 GB/s | 0.1209 (0.1892), 11.14 GB/s |
| anon_thp | 0.1187 (0.3581), 4.31 GB/s | 0.1283 (0.2993), 10.08 GB/s | 0.1234 (0.3174), 11.03 GB/s | 0.1212 (0.4583), 10.98 GB/s |

Decision block of the probe: pool_memory=vmm_host_numa, t_memcpy_ms=0.1238 (lanes 8 row), pool_gather_gbs=245.57. Guard: rc 0, no cut, 14 containers Up after, kernel log clean.

Verdict: INFO — vmm_host_numa gathers at 245.6 GB/s, 8% above host_alloc (226.6); anon_thp is not Triton-addressable
