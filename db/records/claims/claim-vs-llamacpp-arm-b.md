---
type: claim
id: vs-llamacpp-arm-b
created: 2026-10-15
summary: Against llama.cpp arm B the engine is slower in decode and much faster in cold prefill
basis: measured
needle: decode 29.37 tok/s = 89.3% of arm B 32.9; cold 8K prefill 372.1 tok/s = 11.6x arm B 32.1
supported_by: db/records/measurements/f9-campaign-d5.md
surfaces: README.md, RUNBOOK.md, docs/measurements.md
title: Against llama.cpp arm B the engine is slower in decode and much faster in cold prefill
status: current
---
Arm B is unmodified llama.cpp, Q4_1, 43,520 MiB of experts cached on the GPU. The Justification: line of the campaign record is ttft.
