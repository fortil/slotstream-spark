# f6-mtp-d5
Date: 2026-10-14
Capture: db/sources/runs/2026/10/20261014T114220-f6-ab-mtp/
Decision: ADOPT_RESIDENT
Evidence: db/sources/runs/2026/10/20261014T114220-f6-ab-mtp/result.json, db/records/measurements/h3.md
Reversal: `[mtp] D5` in slotstream_spark/config/policy.toml is the default; SLOTSTREAM_MTP=off|resident overrides it per process. Rule: median >= 1.05, >= 80% positive, >= 5 clean pairs, identical outputs.

- A arm `mtp-off/lookahead-off`, B arm `mtp-resident/lookahead-off`; D5, 16 planned pairs, 16 run, 16 clean; excluded: none
- paired median 1.1609 (adoption bar 1.05), positive fraction 0.875 (bar 0.80), clean pairs 16 (bar 5); median 1.1609 >= 1.05, positive fraction 0.875 >= 0.8
- F3 A/A noise [aa] median_abs_dev 0.004216 read from db/records/measurements/h3.md; adoption margin 0.05 ≥ noise
- outputs identical across the two arms in every pair that ran
- acceptance per depth over the clean pairs (B arm): depth 1 3405/3803 = 0.895, depth 2 3006/3403 = 0.883; 3813 verify rounds, 2.68 tokens per round
- lookahead in both arms: off; cohort B0 (test split)

| pair | order | A (tok/s) | B (tok/s) | ratio | clean |
|---|---|---|---|---|---|
| 0 | AB | 25.243 | 29.149 | 1.1548 | yes |
| 1 | BA | 26.205 | 32.657 | 1.2462 | yes |
| 2 | AB | 24.339 | 29.187 | 1.1992 | yes |
| 3 | BA | 26.755 | 25.434 | 0.9506 | yes |
| 4 | AB | 24.421 | 29.161 | 1.1941 | yes |
| 5 | BA | 26.289 | 28.600 | 1.0879 | yes |
| 6 | AB | 24.670 | 28.792 | 1.1671 | yes |
| 7 | BA | 25.151 | 31.478 | 1.2516 | yes |
| 8 | AB | 27.628 | 26.069 | 0.9436 | yes |
| 9 | BA | 26.474 | 29.488 | 1.1139 | yes |
| 10 | AB | 26.349 | 28.879 | 1.0960 | yes |
| 11 | BA | 26.722 | 29.910 | 1.1193 | yes |
| 12 | AB | 26.768 | 29.698 | 1.1094 | yes |
| 13 | BA | 25.524 | 30.802 | 1.2068 | yes |
| 14 | AB | 25.691 | 30.540 | 1.1888 | yes |
| 15 | BA | 25.692 | 30.618 | 1.1917 | yes |

Verdict: PASS — ADOPT_RESIDENT: paired median 1.1609, 16 clean pairs
