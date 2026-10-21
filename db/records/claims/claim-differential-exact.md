---
type: claim
id: differential-exact
created: 2026-10-15
summary: The reference Q4_1 repack is exact except 7 named tensors
basis: measured
needle: exact except the 7 named exceptions
supported_by: db/records/measurements/f9-reference.md
surfaces: README.md, RUNBOOK.md, docs/measurements.md
title: The reference Q4_1 repack is exact except 7 named tensors
status: current
---
Exact Q4_1 repack: exact except 7 named scale/bias tensors rounded by at most 2^-24, with ple_conv1d written as F32. Also: the 194 hyper-connection matrices are stored as BF16 in the reference, not as Q4_1 (DIAG_F9c), and 88 tensors have a mapping correlation in [0.95, 0.99).
