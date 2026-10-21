# m-lb2: NVMe and PSI effect of WILLNEED (-lzm on) at load
Date: 2026-10-12
Capture: db/sources/runs/2026/10/20261012T071331-lb2/
Command: part of lb2 (bash bench/lb/session.sh --configs q41-a-d5,iq3-r-d5 --arm-b); sample20ms.py runs from launch until /health is 200, p1 of each config

| config | samples | window s | peak read MB/s | mean read MB/s | max in-flight | max production-IO PSI avg10 |
|---|---|---|---|---|---|---|
| q41-a-d5 | 3,423 | 68.4 | 3,502 | 1,107 | 0 | 0.0 |
| q41-b-d5 | 6,445 | 128.9 | 2,759 | 634 | 0 | 0.0 |
| iq3-r-d5 (stopped at the soft floor) | 5,487 | 109.7 | 2,732 | 591 | 0 | 0.0 |

The q41 arm A load reads ~46 GB of mapped GGUF in 68 s with the memcg at MemoryHigh; the scope holds 46.5 GB of file cache afterwards (mlb4.json). Production IO pressure stayed at 0.0 in every window. "Max in-flight" reads 0 in all three windows.

Verdict: INFO — WILLNEED load peaks at 3.5 GB/s and 1.1 GB/s mean for arm A with production IO PSI at 0.0
