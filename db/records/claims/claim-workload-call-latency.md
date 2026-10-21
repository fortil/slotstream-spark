---
type: claim
id: workload-call-latency
created: 2026-10-17
summary: Cold start and call latencies of the workload profile, N = 3 calls, none over 300 s
basis: measured
needle: cold start 20.1 s, all calls <= 270 s (json16k 64.8 s, json25k 62.4 s, text8k 76.4 s)
supported_by: db/records/measurements/workload-profile.md
surfaces: README.md, docs/measurements.md
title: Cold start and call latencies of the workload profile, N = 3 calls, none over 300 s
status: current
---
Profile of 2026-10-16, mode D5, engine stopped before the run, the host's other services running, one call per shape, N = 3, max_tokens 6000. The first request on a stopped engine was answered after 20.137 s in 1 attempt. Calls: json16k 15740 prompt and 562 completion tokens in 64.778 s, json25k 25906 and 22 in 62.369 s, text8k 8154 and 1347 in 76.44 s, all finish reason stop. 0 calls went over the pipeline's 300 s attempt timeout. The pass rule fixed before the run was 270 s per warm call. One run, one call per shape, page cache probably warm.
