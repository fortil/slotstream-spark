---
type: claim
id: differential-llamacpp
created: 2026-10-15
summary: Engine PPL matches llama.cpp on the exact repack within 0.05 %
basis: measured
needle: engine 4.7216, llama.cpp 4.7239
supported_by: db/records/measurements/f9-reference.md
surfaces: README.md, RUNBOOK.md, docs/measurements.md
title: Engine PPL matches llama.cpp on the exact repack within 0.05 %
status: current
---
Same 32 windows, prefill on the CPU (prefill_device=cpu). The llama.cpp reference runs the repack with the 194 hyper-connection matrices in BF16 and the rest in Q4_1.
