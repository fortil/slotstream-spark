# wait-mechanism
Date: 2026-10-11
Decision: spin_flag
Evidence: db/sources/runs/2026/10/20261011T192313-spark-probe/result-wait.json, db/records/measurements/g2-g3.md
Reversal: Decision §7 row 4. Rule fixed before measuring: spin_flag if its p50 <= event-sync p50 (4.93 vs 5.33 us). Switch to event_sync if spin_flag costs host cores or loses in the F3 session. memop_in_graph=supported.
