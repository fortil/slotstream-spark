# lb-arm-b
Date: 2026-10-11
Decision: MEASURE_ARM_B
Evidence: db/sources/runs/2026/10/20261011T192510-m-lb1/result.json, db/records/measurements/m-lb1.md
Reversal: Rule fixed in LB §12: condition (b) >= 10 GB/s gives MEASURE_ARM_B, <= 1 GB/s gives SKIP_ARM_B ("discarded on GB10"), otherwise SKIP_ARM_B ("between: no session time"). Condition (b) measured 34.227 GB/s. Skip arm B only if LB2 runs out of time.
