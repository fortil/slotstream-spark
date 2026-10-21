# acceptance: final acceptance on the Spark
Date: 2026-10-20
Capture: db/sources/runs/2026/10/20261020T152544-acceptance/
Command: tools/spark/run.sh --heavy acceptance auto 10800 -- bash tools/acceptance.sh final

| check | status | rc | seconds | detail |
|---|---|---|---|---|
| claims-gate | PASS | 0 | 0 | claims gate: 190 needle checks, 0 failures |
| runbook-check | PASS | 0 | 0 | runbook check: 0 problems |
| docs-check | PASS | 0 | 0 | docs check: 0 problems |
| required-records | PASS | 0 | 0 | RESULT: PASS required-records |
| oracle-parity31-calibration | PASS | 0 | 0 | RESULT: PASS oracle-parity31-calibration present with INFO |
| parity31-limits-rebind | PASS | 0 | 0 | RESULT: PASS parity31-limits-rebind |
| lb-yardstick-consistency | PASS | 0 | 0 | RESULT: PASS lb-yardstick-consistency |
| contract-rebind | PASS | 0 | 0 | RESULT: PASS contract-rebind |
| t0-suite | PASS | 0 | 55 | 1599 passed, 2 skipped, 218 deselected in 53.84s |
| t0-cuda-subset | PASS | 0 | 16 | 142 passed, 1604 deselected in 15.15s |
| gpu-fused-moe | PASS | 0 | 6 | 8 passed in 5.26s |
| gpu-attention-oracle | PASS | 0 | 3 | 7 passed in 1.95s |
| sweep-check | PASS | 0 | 379 | 6 passed in 378.58s (0:06:18) |
| prefix-exact | PASS | 0 | 224 | 7 passed in 222.55s (0:03:42) |
| decode-overlap | PASS | 0 | 477 | 3 passed in 475.83s (0:07:55) |
| mtp-check | PASS | 0 | 595 | 4 passed in 594.14s (0:09:54) |
| token-graph | SKIP | 77 | 0 | the token graph is off by decision (Decision: KEEP_OFF) |
| persistent-restart | PASS | 0 | 84 | 1 passed in 83.48s (0:01:23) |
| live-acceptance | PASS | 0 | 143 | 80 passed, 0 failed |
| ppl-16-windows | PASS | 0 | 114 | RESULT: PASS ppl-16-windows |
| self-impact | PASS | 0 | 0 | production PSI: not_applicable: production paused by the user |

- total 2096 s; skipped by decision: token-graph 
Verdict: PASS — every acceptance check passed or was skipped by decision (see the table)
