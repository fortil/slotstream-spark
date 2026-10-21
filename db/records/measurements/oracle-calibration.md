# oracle-calibration: contract arrays from the oracle's own bf16-vs-fp32 spread
Date: 2026-10-12
Capture: db/sources/runs/2026/10/20261012T142519-oracle-chain/
Command: part of the oracle-chain job (phase 6, tools/calibrate_contract.py measure); then tools/calibrate_contract.py apply calibration.json (step 36b), which wrote the arrays and the calibrated logits limits

Rule (db/records/decisions/numeric-contract.md): L_k = 2*s_k + 0.01; layer k separable iff c_k >= 4*s_k and c_k >= 2*L_k. s_k is the max over 8 dev prompts of the oracle bf16-vs-fp32 rel_l2; c_k the min over prompts and modes of the wrong-expert control.

| set | spread s_k (max over layers) | control c_k (min over layers) | separable |
|---|---|---|---|
| layer rel_l2 | 0.0114 at layer 0 rising to 0.2333 at layer 41 | 0.328 at layer 0, 0.906-0.937 at layers 37-41 | 44/48 |
| routing top-10 disagreement | 0.0263 at layer 0 | 0.0 at layer 0 (informational by construction) | 47/48 |

Layers not separable: 37, 39, 40, 41 (spread 0.222-0.233, control 0.906-0.937; they pass c_k >= 4*s_k by 0.0036-0.0194 and miss only c_k >= 2*L_k). The criterion of at least 4 separable layers holds (44/48).

Layers 37, 39, 40, 41 satisfy c_k >= 4*s_k and miss c_k >= 2*L_k by 0.0006 / 0.0014 / 0.0007 / 0.0164 (controls 0.906-0.937 vs 2*L 0.907-0.953); they are reported, never asserted; their routing is separable (spread 0.089-0.134, control 0.786-0.848) and layers 38 and 42-47 are separable, so a wrong expert there is caught one layer later and by the logits criteria. The per-layer rule is unchanged.

Oracle bf16-vs-fp32 logits pair (254 positions) and the limits derived by rule_logits:

| statistic | oracle pair | rule | limit | validity | inside |
|---|---|---|---|---|---|
| top-1 | 0.9449 | 1 - 2.0*(1 - o) | 0.88976 | >= 0.85 | yes |
| KL mean | 0.01241 | 3.0*o | 0.03723 | <= 0.05 | yes |
| KL p99 | 0.14196 | 3.0*o | 0.42587 | <= 0.5 | yes |
| KL max | 0.33264 | 3.0*o | 0.99792 | <= 1.5 | yes |

The [logits] limits are calibrated by the rule of numeric-contract.md (design decision 12) from this pair; [fixture] kl_max_nats = 0.99792.

Verdict: PASS — 44/48 layers separable (37, 39, 40, 41 informational), routing 47/48, logits limits by the rule (top-1 >= 0.88976, KL mean/p99/max <= 0.03723/0.42587/0.99792 nats) inside the validity range
