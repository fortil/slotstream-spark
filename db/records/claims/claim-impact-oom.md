---
type: claim
id: impact-oom
created: 2026-10-15
summary: No oom-kill and MemAvailable stayed at or above 8 GiB during the f9 job
basis: measured
needle: no oom-kill, MemAvailable >= 8 GiB
supported_by: db/records/measurements/f9-impact.md
surfaces: README.md, RUNBOOK.md, docs/measurements.md
title: No oom-kill and MemAvailable stayed at or above 8 GiB during the f9 job
status: current
---
Production was paused by the user, so the latency, PSI and quiet-window criteria are recorded as not applicable; the oom, MemAvailable and container criteria are gates and passed.
