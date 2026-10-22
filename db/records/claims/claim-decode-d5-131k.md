---
type: claim
id: decode-d5-131k
created: 2026-10-21
summary: D5 decode at a 131,072-token window is 26.918 tok/s (median of 5, decode-8k cell, 37 chunks)
basis: measured
needle: 26.918
supported_by: db/records/measurements/w131k-campaign-d5.md
surfaces: README.md, RUNBOOK.md, docs/measurements.md
title: D5 decode at a 131,072-token window is 26.918 tok/s (median of 5, decode-8k cell, 37 chunks)
status: current
---
Informative beside the 29.37 tok/s of the default window: the pool held 37 chunks instead of 48, because the window's fixed bytes are 25,270,020,452 B. Median of five decode-8k samples (30.79, 26.92, 27.49, 25.29 and 24.37 tok/s), MTP policy resident, pool pre-filled, SLOTSTREAM_CONTEXT_WINDOW=131072 (a 131,072-token window), October 21, 2026. The floor in the record is 23 tok/s.
