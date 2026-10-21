---
type: claim
id: ppl-vs-oracle
created: 2026-10-15
summary: PPL on 32 wikitext windows is within 2 % of the transformers oracle
basis: measured
needle: engine 4.7216, oracle 4.7265
supported_by: db/records/measurements/f9-quality.md
surfaces: README.md, RUNBOOK.md, docs/measurements.md
title: PPL on 32 wikitext windows is within 2 % of the transformers oracle
status: current
---
Ratio -0.10 %, oracle in bfloat16, same 32 windows and 65,504 tokens. Informative only, never a pass: over 145 windows the engine gives 5.7338 against a published figure of 5.3914 (+6.35 %), whose text and window protocol are not published (DIAG_F9.md).
