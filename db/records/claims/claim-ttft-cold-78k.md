---
type: claim
id: ttft-cold-78k
created: 2026-10-21
summary: Cold 78K TTFT at a 131,072-token window is 221.89 s (78000 tokens, median of 5)
basis: measured
needle: 221.89 s
supported_by: db/records/measurements/w131k-campaign-d5.md
surfaces: README.md, RUNBOOK.md, docs/measurements.md
title: Cold 78K TTFT at a 131,072-token window is 221.89 s (78000 tokens, median of 5)
status: current
---
Published, not gated. Prompt of 78000 tokens cut from bench/prompts/ttft_long.json, one synthetic text; a 78K conversation of the desktop app is the case this window serves. Engine started with SLOTSTREAM_CONTEXT_WINDOW=131072 (a 131,072-token window), 38 chunks of 1 GiB in the pool, a fresh prefix directory per sample, measured on October 21, 2026 with the host's other services paused. Samples 221.91, 222.41, 220.81, 221.89 and 221.24 s; prefill 351.9 tok/s.
