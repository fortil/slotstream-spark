---
type: claim
id: ttft-followup-78k
created: 2026-10-21
summary: Follow-up TTFT after a 76800-token prefix at a 131,072-token window is 7.389 s (median of 5)
basis: measured
needle: 7.389
supported_by: db/records/measurements/w131k-campaign-d5.md
surfaces: README.md, RUNBOOK.md, docs/measurements.md
title: Follow-up TTFT after a 76800-token prefix at a 131,072-token window is 7.389 s (median of 5)
status: current
---
Second turn of the same 78000-token conversation on the same server: 76800 tokens reused from the in-memory prefix cache (the last pass boundary of the first prompt), 1337 new. The limit in the record is 12 s and the reuse floor 70000 tokens. Samples 7.39, 7.39, 7.37, 7.39 and 7.37 s at a 131,072-token window (SLOTSTREAM_CONTEXT_WINDOW=131072), October 21, 2026.
