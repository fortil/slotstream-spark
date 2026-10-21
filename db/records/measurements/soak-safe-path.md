# soak-safe-path: 10^6 reads through the real pool, ensurer and GPU checksum
Date: 2026-10-12
Capture: db/sources/runs/2026/10/20261012T213232-soak/
Command: tools/spark/jobs.sh start soak --heavy soak 4096 7200 -- bash tools/spark/snap.sh soak tools/spark/py.sh tools/soak_safe_path.py --reads 1000000

| gate | value | limit | ok |
|---|---|---|---|
| reads | 1000003 | >= 1000000 | yes |
| read errors | 0 | 0 | yes |
| timeouts (batch not joined in 2 s) | 0 | 0 | yes |
| checksum mismatches | 0 | 0 | yes |
| max read (ms) | 6.491 | < 2000 | yes |
| guard cut or non-zero rc | False | False | yes |
| terminated by signal | False | False | yes |

- read latency per batch: p50 3.314 ms, p99 4.083 ms, max 6.491 ms over 50824 batches
- 2764.8 GB read in 921.61 s = 3 GB/s (limit 3); demand reads 492125, background reads 507878, ensure hits 16115
- CRC checks against an independent O_DIRECT read: 1000; GPU checksum checks: 1000 (gpu True)
- NVMe hwmon pauses (limit 56850 mC): 0

Verdict: PASS — every read exact, no error, no hang, no guard cut
