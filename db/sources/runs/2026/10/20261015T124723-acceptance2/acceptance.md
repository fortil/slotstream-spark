# acceptance: final acceptance on the Spark
Date: 2026-10-15
Capture: db/sources/runs/2026/10/20261015T124723-acceptance2/
Command: tools/spark/run.sh --heavy acceptance auto 10800 -- bash tools/acceptance.sh final

| check | status | rc | seconds | detail |
|---|---|---|---|---|
| claims-gate | PASS | 0 | 0 | claims gate: 76 needle checks, 0 failures |
| runbook-check | PASS | 0 | 0 | runbook check: 0 problems |
| required-records | PASS | 0 | 0 | RESULT: PASS required-records |
| oracle-parity31-calibration | PASS | 0 | 0 | RESULT: PASS oracle-parity31-calibration present with INFO |
| parity31-limits-rebind | PASS | 0 | 0 | RESULT: PASS parity31-limits-rebind |
| lb-yardstick-consistency | PASS | 0 | 0 | RESULT: PASS lb-yardstick-consistency |
| contract-rebind | PASS | 0 | 0 | RESULT: PASS contract-rebind |
| t0-suite | PASS | 0 | 48 | 1508 passed, 6 skipped, 218 deselected in 46.94s |
| t0-cuda-subset | PASS | 0 | 16 | 142 passed, 1517 deselected in 15.04s |
| gpu-fused-moe | PASS | 0 | 6 | 8 passed in 5.66s |
| gpu-attention-oracle | PASS | 0 | 3 | 7 passed in 1.99s |
| sweep-check | PASS | 0 | 386 | 6 passed in 385.32s (0:06:25) |
| prefix-exact | PASS | 0 | 226 | 7 passed in 225.22s (0:03:45) |
| decode-overlap | PASS | 0 | 485 | 3 passed in 484.05s (0:08:04) |
| mtp-check | PASS | 0 | 604 | 4 passed in 602.58s (0:10:02) |
| token-graph | SKIP | 77 | 0 | the token graph is off by decision (Decision: KEEP_OFF) |
| persistent-restart | PASS | 0 | 83 | 1 passed in 82.29s (0:01:22) |
| live-acceptance | PASS | 0 | 142 | 79 passed, 0 failed |
| ppl-16-windows | PASS | 0 | 113 | RESULT: PASS ppl-16-windows |
| self-impact | PASS | 0 | 0 | production PSI: not_applicable: production paused by the user |

- total 2112 s; skipped by decision: token-graph 
Verdict: PASS — every acceptance check passed or was skipped by decision (see the table)
