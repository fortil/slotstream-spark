# f3-session: D5 decode session (A/A pairs, D1 runs, traces, LB arm B pairs)
Date: 2026-10-13
Capture: db/sources/runs/2026/10/20261013T092431-f3-session/
Command: tools/spark/jobs.sh start f3-session --heavy f3-session auto 14400 --memcg slotstream-spark-lb -- bash bench/session.sh
complete: True; A/A pairs clean: 16/16; A/A median |dev| 0.0039, max 0.0213; traces ok: True
LB arm q41-b-d5 (prefill_device cpu): clean pairs 5, cache_full_pairs 0
D1 runs (tok/s): run 0 short 9.983, 2k 8.364; run 1 short 10.065, 2k 8.461; run 2 short 9.936, 2k 8.393

Judgements: db/records/measurements/h3.md, db/records/measurements/h-curve.md, db/records/decisions/stopgo-f3.md.
Verdict: INFO — session record; the judgements are in h3.md, h-curve.md and stopgo-f3.md
