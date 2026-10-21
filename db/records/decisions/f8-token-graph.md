# f8-token-graph
Date: 2026-10-14
Capture: db/sources/runs/2026/10/20261014T195021-f8-ab/
Decision: KEEP_OFF
Evidence: db/sources/runs/2026/10/20261014T195021-f8-ab/result.json, db/sources/runs/2026/10/20261014T195020-f8-ab-soak/f8_soak.json, db/records/measurements/h3.md
Reversal: `[f8] token_graph` in slotstream_spark/config/platform.toml; SLOTSTREAM_TOKEN_GRAPH=on|off overrides it per process. Rule: paired median >= 1.05, >= 80% positive pairs, identical outputs, an 8 h soak with 0 CUDA context losses.

- A arm `f3-segmented`, B arm `token-graph`; D5, 16 planned pairs, 6 run, 6 clean; excluded: none
- paired median 0.9914 (adoption bar 1.05), positive fraction 0.333 (bar 0.80); F3 A/A noise [aa] median_abs_dev 0.004216 from db/records/measurements/h3.md
- soak: 0 s of 28800 s wanted, 0 arms (0 failed), 0 spot bit-checks (0 differ), 0 CUDA context losses
- token graph counters over the arms that used it: 3834 tokens on the graph, 0 on the F3 path, 0 wait fallbacks
- reasons: A/B: median 0.9914 < 1.05; soak skipped because the A/B did not pass

| pair | order | A (tok/s) | B (tok/s) | ratio | clean |
|---|---|---|---|---|---|
| 0 | AB | 25.406 | 25.479 | 1.0029 | yes |
| 1 | BA | 24.426 | 24.011 | 0.9830 | yes |
| 2 | AB | 23.785 | 23.570 | 0.9909 | yes |
| 3 | BA | 23.346 | 23.351 | 1.0002 | yes |
| 4 | AB | 22.684 | 22.498 | 0.9918 | yes |
| 5 | BA | 24.948 | 24.408 | 0.9784 | yes |

Verdict: PASS — KEEP_OFF: paired median 0.9914, 6 clean pairs, soak 0 s
