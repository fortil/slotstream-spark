---
type: claim
id: d5-pool-chunks-131k
created: 2026-10-21
summary: The D5 pool held 38 chunks during the campaign at a 131,072-token window
basis: measured
needle: 38 chunks
supported_by: db/records/measurements/w131k-campaign-d5.md
surfaces: README.md, RUNBOOK.md, docs/measurements.md
title: The D5 pool held 38 chunks during the campaign at a 131,072-token window
status: current
---
At a 131,072-token window (SLOTSTREAM_CONTEXT_WINDOW=131072) the ledger's fixed bytes are 25,270,020,452 B against 15,851,369,316 B at the default window, so the pool is smaller: 37 chunks in the decode-8k cell and 38 chunks in the two TTFT cells of the w131k-engine campaign of October 21, 2026, with the governor growing the pool as MemAvailable allowed; 40 chunks at the D5 ceiling of 65,126 MiB, calculated from the code. The worst-cell peak (ttft-cold-78k) was 56438362112 B against the ledger's 66072209764 B at 38 chunks.
