---
type: claim
id: workload-memory-handover
created: 2026-10-17
summary: Memory handover around the nightly window: no oom-kill, 14 of 14 containers up (one night)
basis: measured
needle: no oom-kill, residents Up after the engine stopped
supported_by: db/records/measurements/workload-rehearsal.md
surfaces: README.md, docs/measurements.md
title: Memory handover around the nightly window: no oom-kill, 14 of 14 containers up (one night)
status: current
---
Rehearsal of 2026-10-17, mode D5, one stored day replayed with the same pipeline, 13 report sections. The engine was ready 17 s after its first start (pool prefill 5.9 s, 18412 records) and again 17 s after its second start. It exited on idle after 1800 s without requests. The kernel log held no oom or allocation errors during the run, and 14 of 14 containers were up after the handover. MemAvailable was 120.1 GB with residents and engine stopped and 87.1 GB with residents up and engine stopped. The engine and the residents never ran together, so no MemAvailable minimum under both exists; one night, page cache probably warm.
