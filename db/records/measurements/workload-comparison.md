# workload-comparison: the engine and gpt-oss-120b served by vLLM on the same stored day
Date: 2026-10-17
Capture: db/sources/runs/2026/10/20261017T052200-workload-comparison/
Command: the pipeline's report-comparison script, run on the two report directories (gpt-oss-120b served by vLLM first, the engine second), writing one JSON file
Rule: informative; structural gates on the report (citation validation, cuts, failed reading segments) and polarity concordance over escalated items

One stored day, 13 sections, 1 reading segment, 3 supervisor calls, 0 escalated items. A is gpt-oss-120b served by vLLM, B is the engine in D5.

| gate | A gpt-oss-120b | B engine |
|---|---|---|
| citation validation | 0.0% | 23.1% |
| sections without cuts | 0 | 3 |
| reading cuts | 13 | 10 |
| synthesis and summary cuts | 20 | 18 |
| reading segments, failed | 1, 0 | 1, 0 |
| warnings | 0 | 0 |
| mean reading characters | 146.2 | 274.2 |
| escalated items | 0 | 0 |

| item | value |
|---|---|
| polarity concordance on escalated items | 0 of 0, not measurable |
| supervisor tokens in/out, A | 6024 / 3978 |
| supervisor tokens in/out, B | 6564 / 1928 |

What this does not show: 13 sections and 3 supervisor calls are too few to say either model is better; both citation rates are low; no escalated item exists, so polarity is not compared; quality is judged by structural gates, not by humans.

Verdict: INFO — the engine completed the stored day with no failed segment; the comparison is inconclusive at 13 sections
