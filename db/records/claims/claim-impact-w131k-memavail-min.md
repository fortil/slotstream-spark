---
type: claim
id: impact-w131k-memavail-min
created: 2026-10-21
summary: Minimum MemAvailable during the w131k-engine job is 25582 MiB
basis: measured
needle: 25582
supported_by: db/records/measurements/w131k-impact.md
surfaces: README.md, RUNBOOK.md, docs/measurements.md
title: Minimum MemAvailable during the w131k-engine job is 25582 MiB
status: current
---
From the 1 s MemAvailable samples of the w131k-engine job of October 21, 2026 (the engine at a 131,072-token window, 37 to 38 chunks in the pool, the host's other services paused). The f9 job at the default window had 59853 MiB; the difference is the window's fixed bytes and the pool the governor grew into the free memory.
