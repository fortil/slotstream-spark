---
type: claim
id: workload-vs-reference
created: 2026-10-17
summary: Engine against gpt-oss-120b served by vLLM on one stored day: inconclusive at 13 sections
basis: measured
needle: the engine completed the stored day with no failed segment; the comparison is inconclusive at 13 sections
supported_by: db/records/measurements/workload-comparison.md
surfaces: README.md, docs/measurements.md
title: Engine against gpt-oss-120b served by vLLM on one stored day: inconclusive at 13 sections
status: current
---
Comparison of 2026-10-17, mode D5, one stored day, 13 sections, 1 reading segment, 3 supervisor calls, 0 escalated items. Citation validation 0.0% for gpt-oss-120b served by vLLM and 23.1% for the engine, both low. Sections without cuts 0 against 3. Reading cuts 13 against 10. Synthesis and summary cuts 20 against 18. Reading segments failed 1, 0 for both, so 0 failed reading segments of 1. Warnings 0 and 0. Mean reading characters 146.2 against 274.2. Polarity concordance 0 of 0, not measurable. Supervisor tokens in and out 6024 and 3978 for gpt-oss-120b, 6564 and 1928 for the engine. Too few sections to say either model is better; quality is judged by structural gates, not by humans.
