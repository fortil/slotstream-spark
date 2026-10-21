# numeric-contract
Date: 2026-10-12
Decision: The engine is judged against the transformers oracle by a rule fixed before any engine run. Per layer, L_k = 2*s_k + 0.01 where s_k is the oracle's own bf16-vs-fp32 spread (max over the 8 dev prompts); layer k is asserted only if c_k >= 4*s_k and c_k >= 2*L_k, with c_k the smaller wrong-expert control (swap, noexp) vs the golden (min over dev prompts). Routing uses the same rule on the top-10 set disagreement (control = swap vs golden); layer 0 is informational by construction. N-gram ids are exact. Logits criteria calibrated by rule_logits from the oracle's own bf16-vs-fp32 pair on the dev prompts: top1_min = 1 - 2*(1 - o_top1), KL mean/p99/max limits = 3*o_kl, valid only if top1_min >= 0.85 and the KL limits are <= 0.05/0.5/1.5 nats (else DESIGN GATE; never clamped). Single ops: rel_l2 <= 2e-2 against fp32 on identical inputs. One-position MLX fixtures: KL <= the calibrated [logits] kl_max_nats. Perplexity: wikitext-2 reference 5.3914 within 1%; ablation: ppl_correct <= 20, swap and noexp >= 5x. At least 4 layers must be separable and the calibrated logits limits must fall inside the validity range, otherwise DESIGN GATE. On sequences other than the 8 dev prompts, each limit is `min(max(L_k, 2*d_k + 0.01), c_k/2)` with d_k the oracle's own bf16-vs-fp32 metric on that sequence (seq_rule). On sequences other than the 8 dev prompts the KL limits are `min(max(kl_X_max_nats, 3*p_X), kl_X_ceiling_nats)` with p_X the oracle's own bf16-vs-fp32 KL statistic on that sequence; top1_min stays calibrated; a pair above a ceiling is a DESIGN GATE (logits seq_rule, 2026-10-13).
Evidence: slotstream_spark/config/numeric_contract.toml, tools/calibrate_contract.py, tests/gates/parity31_limits.json, db/records/measurements/oracle-parity31-calibration.md, db/records/measurements/oracle-parity31.md
Reversal: only with the user's approval, never after looking at engine results. The arrays under [layer] and [routing] stay empty until `tools/calibrate_contract.py apply` fills them (step 36) from oracle-only dumps; the same `rule` function reproduces the four parity31 limits. Amended 2026-10-12 with the user's answer at the step 36 design gate (BRIEF.md ## User answers), before any engine result; the logits rule and its parameters are frozen the same way. The per-sequence logits rule (2026-10-13) was approved by the user after an engine result exposed the fixed limit's defect; its inputs are oracle-only and the frozen range.

Why a rule and not a number: the MLX goldens are a second bf16 implementation, so the distance to them carries two bf16 noises plus op-order differences, and a depth-independent 2e-2 cannot hold (oracle bf16 vs fp32 spread on parity31: 0.0093 / 0.0186 / 0.0269 / 0.0770).

## Why the logits limits are calibrated (2026-10-12)

The first apply refused to write anything: the oracle's own bf16 against the same oracle in fp32, over 254 positions, scores top-1 0.9449, KL mean 0.01241, p99 0.14196, max 0.33264 nats, so the fixed 0.98 / 0.01 / 0.05 / 0.2 could not be met by any bf16 engine. The limits now come from that pair: top1_min = 1 - 2*(1 - o_top1) and KL mean/p99/max <= 3*o_kl. The engine-vs-oracle pair carries two bf16 noises; on parity31 the two-implementation distance was 2.0 / 1.54 / 1.45 / 1.06 times the single-noise spread (0.0186/0.0287/0.0390/0.0817 vs 0.0093/0.0186/0.0269/0.0770), so a linear statistic (top-1 disagreement) gets factor 2 and a quadratic one (KL) factor 3.

Validity range, fixed before any engine result: top1_min >= 0.85 and KL mean/p99/max limits <= 0.05 / 0.5 / 1.5 nats; outside it apply stops with DESIGN GATE and never clamps. Wrong experts score far outside it: swap and noexp raise teacher-forced NLL by 5.07-7.76 nats per token (perplexity 565-6,466), and a model there cannot agree with the golden argmax at 85% of positions. Perplexity within 1% of 5.3914 and the ablations stay hard gates.

Layers 37, 39, 40, 41 stay informational: they pass c_k >= 4*s_k and miss only c_k >= 2*L_k, by 0.0006-0.0164.

## Per-sequence reference (2026-10-12)

Rule: `L_seq = min(max(L_k, seq_factor*d_k + margin), c_k/seq_ceiling_ratio)`, with d_k the table's metric between the oracle in bf16 and in fp32 on the judged sequence, asserted only where the calibrated `separable[k]` is true. The three constants are those of the calibration rule: `seq_factor = 2.0` and `margin = 0.01` (from `L_k = 2*s_k + margin`) and `seq_ceiling_ratio = 2.0` (the `c_k >= 2*L_k` clause). They are keys of `[layer]` and `[routing]` in `numeric_contract.toml`; the calibrated arrays are unchanged. The rule applies to engine-vs-oracle comparisons on sequences that are not the 8 dev prompts (H2, the F5 sweep prompt, the reserved prompts of F9, the acceptance re-runs); it never makes a calibrated limit tighter (`L_k <= c_k/2` on every separable layer).

Why: the arrays were calibrated on short dev prompts, and H2 judges the engine's own 85-112 token sequences. On dev7 the oracle's own bf16-vs-fp32 pair disagrees at layer 43 routing by 0.2306, above the calibrated limit 0.21 before any engine enters; the engine scores 0.2376 against bf16. With the rule, the limit there is min(max(0.21, 0.4712), 0.3931) = 0.3931 (the ceiling c_43/2, c_43 = 0.7862). A wrong expert scores at least 0.7862 at that layer and a swapped router about 0.98, so both fail at every noise level. Near-ties at rank 10/11 flip under any rounding and inflate the pair and the engine alike, so the pair is the tie detector without a new parameter.

Approved by the user on 2026-10-12 (BRIEF.md ## User answers, H2 entries) after DIAG_H2.md; inputs are oracle-only; the calibrated arrays are unchanged.

## Per-sequence logits (2026-10-13)

Rule: for X in {mean, p99, max}, `kl_X_seq = min(max(kl_X_max_nats, kl_factor*p_X), kl_X_ceiling_nats)`, with p_X the statistic X of KL(oracle bf16 -> oracle fp32) on the judged sequence, computed on the same token ids as the engine dump. The only new input is that pair; the factor (3.0), the calibrated limits (floor) and the frozen ceilings (0.05 / 0.5 / 1.5 nats, cap) are already in the toml. A sequence whose pair is above a ceiling is a DESIGN GATE, not a pass.

The fixed `kl_max_nats` (0.99792) is a maximum over the 254 calibration positions; the F5 sweep-check judges 2,048. Numbers on that sequence: the oracle pair has kl_max 1.149, the engine 1.493, the F2 row-group path 1.537. The limit is min(max(0.99792, 3 x 1.149), 1.5) = 1.5, ceiling-bound, so the engine passes with 0.007 to spare; kl_mean and kl_p99 stay at their calibrated limits. The F2 number is above the ceiling; it is recorded as informational because that path is qualified at H2 on decode positions.

top1_min stays calibrated: top-1 agreement is a mean over positions and does not grow with their count (the pair scores 0.9941 on 2,048 positions against 0.9449 on the 254 calibration positions).

Approved by the user on 2026-10-13 (BRIEF.md ## User answers, last entry) after the sweep-check engine result exposed the defect; inputs are oracle-only plus the frozen range; the calibrated values are unchanged.

## Calibrated arrays (oracle-only dumps)

| layer | spread s | control c | limit L | separable | routing s | routing c | routing L | routing separable |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.0114 | 0.3283 | 0.0329 | yes | 0.0263 | 0.0000 | 0.0626 | no |
| 1 | 0.0147 | 0.3307 | 0.0393 | yes | 0.0237 | 0.2517 | 0.0574 | yes |
| 2 | 0.0220 | 0.4445 | 0.0539 | yes | 0.0286 | 0.3783 | 0.0671 | yes |
| 3 | 0.0276 | 0.4910 | 0.0653 | yes | 0.0393 | 0.5762 | 0.0886 | yes |
| 4 | 0.0319 | 0.5695 | 0.0739 | yes | 0.0394 | 0.6172 | 0.0888 | yes |
| 5 | 0.0342 | 0.6058 | 0.0783 | yes | 0.0458 | 0.6697 | 0.1017 | yes |
| 6 | 0.0396 | 0.6452 | 0.0892 | yes | 0.0464 | 0.6167 | 0.1029 | yes |
| 7 | 0.0396 | 0.6705 | 0.0892 | yes | 0.0289 | 0.6517 | 0.0679 | yes |
| 8 | 0.0402 | 0.6846 | 0.0904 | yes | 0.0424 | 0.8104 | 0.0948 | yes |
| 9 | 0.0409 | 0.6921 | 0.0918 | yes | 0.0364 | 0.7857 | 0.0827 | yes |
| 10 | 0.0431 | 0.6998 | 0.0962 | yes | 0.0394 | 0.7833 | 0.0888 | yes |
| 11 | 0.0469 | 0.7006 | 0.1037 | yes | 0.0643 | 0.7833 | 0.1386 | yes |
| 12 | 0.0509 | 0.7237 | 0.1119 | yes | 0.0414 | 0.8250 | 0.0928 | yes |
| 13 | 0.0552 | 0.7393 | 0.1204 | yes | 0.0464 | 0.7354 | 0.1029 | yes |
| 14 | 0.0595 | 0.7531 | 0.1289 | yes | 0.0500 | 0.8241 | 0.1100 | yes |
| 15 | 0.0609 | 0.7563 | 0.1319 | yes | 0.0646 | 0.8345 | 0.1392 | yes |
| 16 | 0.0660 | 0.7707 | 0.1420 | yes | 0.0576 | 0.8310 | 0.1252 | yes |
| 17 | 0.0693 | 0.7727 | 0.1485 | yes | 0.0576 | 0.8042 | 0.1252 | yes |
| 18 | 0.0729 | 0.7981 | 0.1559 | yes | 0.0521 | 0.7938 | 0.1142 | yes |
| 19 | 0.0773 | 0.7983 | 0.1646 | yes | 0.1026 | 0.8604 | 0.2153 | yes |
| 20 | 0.0758 | 0.7858 | 0.1617 | yes | 0.0921 | 0.8862 | 0.1942 | yes |
| 21 | 0.0797 | 0.8056 | 0.1695 | yes | 0.0711 | 0.7966 | 0.1521 | yes |
| 22 | 0.0772 | 0.8184 | 0.1645 | yes | 0.0474 | 0.6483 | 0.1047 | yes |
| 23 | 0.0849 | 0.8274 | 0.1798 | yes | 0.1158 | 0.8217 | 0.2416 | yes |
| 24 | 0.0881 | 0.8646 | 0.1861 | yes | 0.0684 | 0.8212 | 0.1468 | yes |
| 25 | 0.0912 | 0.8717 | 0.1924 | yes | 0.0684 | 0.8897 | 0.1468 | yes |
| 26 | 0.0964 | 0.8702 | 0.2027 | yes | 0.0658 | 0.8500 | 0.1416 | yes |
| 27 | 0.1053 | 0.8688 | 0.2206 | yes | 0.1053 | 0.8937 | 0.2205 | yes |
| 28 | 0.1141 | 0.8803 | 0.2381 | yes | 0.1026 | 0.8667 | 0.2153 | yes |
| 29 | 0.1310 | 0.8868 | 0.2721 | yes | 0.0895 | 0.8172 | 0.1889 | yes |
| 30 | 0.1465 | 0.8853 | 0.3029 | yes | 0.1000 | 0.8655 | 0.2100 | yes |
| 31 | 0.1646 | 0.8977 | 0.3392 | yes | 0.1053 | 0.8588 | 0.2205 | yes |
| 32 | 0.1792 | 0.9049 | 0.3684 | yes | 0.1079 | 0.8138 | 0.2258 | yes |
| 33 | 0.1882 | 0.9087 | 0.3865 | yes | 0.1184 | 0.8276 | 0.2468 | yes |
| 34 | 0.1870 | 0.9188 | 0.3840 | yes | 0.1079 | 0.8208 | 0.2258 | yes |
| 35 | 0.2017 | 0.9244 | 0.4135 | yes | 0.1289 | 0.8606 | 0.2679 | yes |
| 36 | 0.2156 | 0.8949 | 0.4413 | yes | 0.1214 | 0.8500 | 0.2529 | yes |
| 37 | 0.2216 | 0.9060 | 0.4533 | no | 0.0893 | 0.8483 | 0.1886 | yes |
| 38 | 0.2221 | 0.9149 | 0.4542 | yes | 0.1000 | 0.6897 | 0.2100 | yes |
| 39 | 0.2291 | 0.9351 | 0.4682 | no | 0.1289 | 0.7862 | 0.2679 | yes |
| 40 | 0.2290 | 0.9353 | 0.4680 | no | 0.1342 | 0.7897 | 0.2784 | yes |
| 41 | 0.2333 | 0.9368 | 0.4766 | no | 0.1211 | 0.8095 | 0.2521 | yes |
| 42 | 0.2097 | 0.9368 | 0.4294 | yes | 0.0921 | 0.7364 | 0.1942 | yes |
| 43 | 0.2089 | 0.9371 | 0.4278 | yes | 0.1000 | 0.7862 | 0.2100 | yes |
| 44 | 0.1772 | 0.9304 | 0.3643 | yes | 0.0974 | 0.7828 | 0.2047 | yes |
| 45 | 0.1482 | 0.8839 | 0.3065 | yes | 0.1105 | 0.7500 | 0.2311 | yes |
| 46 | 0.1323 | 0.8753 | 0.2745 | yes | 0.0895 | 0.6688 | 0.1889 | yes |
| 47 | 0.1252 | 0.8713 | 0.2603 | yes | 0.0816 | 0.7104 | 0.1732 | yes |

Oracle bf16 vs fp32 logits (254 positions): top-1 0.9449, KL mean 0.01241, p99 0.14196, max 0.33264 nats.
Separable layers: 44/48 (layer), 47/48 (routing).

| statistic | oracle pair | rule | limit | validity | inside |
|---|---|---|---|---|---|
| top-1 | 0.9449 | 1 - 2.0*(1 - o) | 0.88976 | >= 0.85 | yes |
| KL mean | 0.01241 | 3.0*o | 0.03723 | <= 0.05 | yes |
| KL p99 | 0.14196 | 3.0*o | 0.42587 | <= 0.5 | yes |
| KL max | 0.33264 | 3.0*o | 0.99792 | <= 1.5 | yes |

Fixture: kl_max_nats = 0.99792

Calibration status: OK
