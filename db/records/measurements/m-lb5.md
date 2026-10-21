# m-lb5: can the engine see a state-A load coming without an operator
Date: 2026-10-11
Capture: db/sources/runs/2026/10/20261011T175055-prod-reads/
Command: tools/spark/run.sh --heavy prod-reads 1024 600 --nolock --contenedor -- tools/capture.sh prod-reads -- bash tools/spark/prod_reads.sh

Rule fixed before measuring: ok_without_operator iff every lower bound >= 5 s, where lower bound = first_rise_s - scrape_interval and first_rise_s is the time from T to the first sample above baseline + 1 GiB. Scrape interval: 15 s.

| load | first_rise_s | lower bound s |
|---|---|---|
| 1 | 19 | 4 |
| 2 | 29 | 14 |
| 3 | 18 | 3 |
| 4 | 18 | 3 |
| 5 | 22 | 7 |
| 6 | 14 | -1 |

Outcome: ok_without_operator = false (four of six lower bounds are under 5 s). The governor cannot rely on the 15-s Prometheus series to anticipate a load; it keeps polling /running at 250 ms (governor.toml).

Verdict: INFO — rule outcome ok_without_operator=false
