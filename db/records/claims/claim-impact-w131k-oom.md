---
type: claim
id: impact-w131k-oom
created: 2026-10-21
summary: The w131k-engine job left no oom-kill and MemAvailable >= 8 GiB
basis: measured
needle: no oom-kill, MemAvailable >= 8 GiB
supported_by: db/records/measurements/w131k-impact.md
surfaces: README.md, docs/measurements.md
title: The w131k-engine job left no oom-kill and MemAvailable >= 8 GiB
status: current
---
Impact report of the w131k-engine job of October 21, 2026 (the engine at a 131,072-token window): kernel log clean, the 14 expected containers up, no process of ours left. Production was paused by the user, so the latency, PSI and quiet-window criteria are recorded as not applicable.
