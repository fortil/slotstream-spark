# f6-lookahead-d5
Date: 2026-10-14
Capture: db/sources/runs/2026/10/20261014T042830-f6-ab-lookahead/
Decision: KEEP_OFF
Evidence: db/sources/runs/2026/10/20261014T042830-f6-ab-lookahead/result.json, db/records/measurements/h3.md
Reversal: `[lookahead] mode` in platform.toml is the default; SLOTSTREAM_LOOKAHEAD=off turns it off per process. Rule: median >= 1.02, >= 80% positive, >= 5 clean pairs, identical outputs.

- A arm `off`, B arm `attention-corrected`; D5, 16 planned pairs, 4 run, 4 clean; excluded: none
- paired median 0.8761 (adoption bar 1.02), positive fraction 0.000 (bar 0.80), clean pairs 4 (bar 5); stopped for futility after 4 clean pairs: the 80% positive bar could not be reached; not adoptable
- F3 A/A noise [aa] median_abs_dev 0.004216 read from db/records/measurements/h3.md; adoption margin 0.02 ≥ noise
- outputs identical across the two arms in every pair that ran
- B arm tickets over the clean pairs: issued 12762, adopted 6494, discarded 6268, failed 0 (4 arm outputs)

| pair | order | A (tok/s) | B (tok/s) | ratio | clean |
|---|---|---|---|---|---|
| 0 | AB | 21.262 | 16.331 | 0.7681 | yes |
| 1 | BA | 21.694 | 19.142 | 0.8824 | yes |
| 2 | AB | 21.208 | 18.449 | 0.8699 | yes |
| 3 | BA | 20.067 | 17.809 | 0.8875 | yes |

Verdict: PASS — KEEP_OFF: paired median 0.8761, 4 clean pairs
