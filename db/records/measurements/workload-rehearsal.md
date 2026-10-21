# workload-rehearsal: memory handover and engine life cycle around the nightly window
Date: 2026-10-17
Capture: db/sources/runs/2026/10/20261017T044200-workload-rehearsal/
Command: set up by hand on the host: control script stops the resident services, starts the engine, the pipeline runs a stored day, then the engine stops and the residents return
Rule: no oom-kill and no GPU allocation error in the kernel log during the run; the resident services are Up after the engine stops; MemAvailable stays above the 8 GiB floor

Mode D5, one stored day replayed with the same pipeline, 13 report sections.

| item | value |
|---|---|
| first engine start to ready | 17 s (pool prefill 5.9 s, 18412 records) |
| engine exit on idle | after 1800 s without requests (SLOTSTREAM_IDLE_EXIT_S default) |
| second start to ready, on the first request of the final stage | 17 s |
| engine CPU time, second run | 2 min 35 s over 2 min 12 s |
| MemAvailable with residents stopped and engine stopped | 120.1 GB |
| MemAvailable with residents up and engine stopped | 87.1 GB |
| kernel log oom or allocation errors during the run | none |
| containers up after the handover | 14 of 14 |

What this does not show: one night, the engine and the residents never ran together, the page cache was probably warm, and the engine served 3 supervisor calls in the run.

Verdict: PASS — no oom-kill, residents Up after the engine stopped
