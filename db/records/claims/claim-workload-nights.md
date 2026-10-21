---
type: claim
id: workload-nights
created: 2026-10-20
summary: Two production nights of the pipeline on the engine, 26 of 26 reading segments ok
basis: measured
needle: 26 of 26 reading segments ok, 90.1 s to 133.6 s, median 113.25 s
supported_by: db/records/measurements/workload-nights.md
surfaces: README.md, docs/measurements.md
title: Two production nights of the pipeline on the engine, 26 of 26 reading segments ok
status: current
---
Nights of October 19 and 20, 2026 (nights 2 and 3 of production use), mode D5, engine started and stopped by the pipeline's control script. Night 2: 1203 mentions, 11 reading segments, 26 calls to the reasoning alias. Night 3: 1488 mentions, 15 reading segments, 30 calls. Prompt tokens per segment 7534 to 14921. 0 failed calls; no oom-kill and no NV_ERR_NO_MEMORY while the engine ran. Night 1 is excluded because the control script left the engine running when the residents started. Two nights, one pipeline.
