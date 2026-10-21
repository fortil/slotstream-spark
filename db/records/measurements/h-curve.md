# h-curve: expert-cache hit rate by pool size, from the F3 session router traces
Date: 2026-10-13
Capture: db/sources/runs/2026/10/20261013T092431-f3-session/ (traces/trace-00..15.npz on the Spark; the .npz files are not fetched)
Command: tools/spark/py.sh tools/h_curve.py "$(cat ../runs/LAST-f3-session)/traces" --out slotstream_spark/config/h_curve_gb10.json
Traces: 48, accesses: 2926080, layers: 48. Global cache keyed by (layer, expert); N per layer = N x 48 records.

| experts/layer | records | LRU h | CLOCK h |
|---|---|---|---|
| 13 | 624 | 0.3521 | 0.3625 |
| 30 | 1440 | 0.5228 | 0.5315 |
| 48 | 2304 | 0.6235 | 0.6327 |
| 81 | 3888 | 0.7481 | 0.7534 |
| 105 | 5040 | 0.8093 | 0.8085 |
| 150 | 7200 | 0.8771 | 0.8814 |
| 200 | 9600 | 0.9426 | 0.9508 |
| 300 | 14400 | 0.9870 | 0.9868 |
| 396 | 19008 | 0.9927 | 0.9925 |
| 512 | 24576 | 0.9930 | 0.9930 |

Read by the planner from slotstream_spark/config/h_curve_gb10.json.
Verdict: INFO — simulation over the F3 router traces, not a measurement of the engine
