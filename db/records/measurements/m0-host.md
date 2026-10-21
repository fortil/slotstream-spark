# m0-host: host-side launch and wait cost, and process footprint
Date: 2026-10-11
Capture: db/sources/runs/2026/10/20261011T184058-m0-host/
Command: tools/spark/run.sh --heavy m0-host 4096 600 --exclusive -- tools/capture.sh m0-host -- tools/spark/py.sh bench/m0/host_bench.py (torch 2.13.0+cu130)

| item | value |
|---|---|
| launch_us (CPU time of replaying a captured 4-kernel graph, median of 10k) | 2.144 |
| event_sync_us (ev.synchronize after a tiny kernel) | 3.328 |
| spin_query_us (busy ev.query) | 3.264 |
| ensure_sim_us (10 dict updates and one append) | 0.336 |
| tau_us = spin_query + ensure_sim + launch | 5.744 (gate: <= 150) |
| g3 replay + spin after an idle gap | 0 us idle: 8.02 us, 200 us idle: 8.27 us, 5000 us idle: 8.90 us |
| footprint (fresh process: torch + triton, a vector add, a 2560x2560 bf16 matmul, a graph capture) | VmHWM 989,016,064 B + NVML 338,935,808 B (pynvml) = 1,327,951,872 B (gate: <= 2.5e9) |

Verdict: PASS — tau 5.7 us <= 150, footprint 1.33 GB <= 2.5 GB
