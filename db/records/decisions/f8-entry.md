# f8-entry
Date: 2026-10-14
Capture: db/sources/runs/2026/10/20261014T182944-f8-partition/
Decision: enter
Evidence: db/sources/runs/2026/10/20261014T182944-f8-partition/result.json, db/records/measurements/g2-g3.md
Reversal: re-measure the partition after a change to the per-layer host path; enter when the host-and-wait share reaches 10% with the g2 verdict pass

- capture id: 20261014T182944-f8-partition
- fraction = host-and-wait share = 0.1133 (11.33% of the token time, pooled over 2 prompts); enter iff fraction >= 0.1 and the g2 verdict is pass
- g2 verdict (memop in graph, db/records/measurements/g2-g3.md): PASS
- counted as host-and-wait: sample, ngram_hash, embed_read, ngram_wait, launch, ensure_hits and `other` (the host work left over by the partition); counted as neither: gpu_wait (the GPU is busy) and the expert miss read plus copy tail (17.07% of the token, waiting on the NVMe, which a token graph still waits for)
- `[f8] token_graph = false` in slotstream_spark/config/platform.toml until a token graph is adopted (step 92)

| prompt | profiled tokens | wall ms | host+waits ms | share | miss read ms | gpu_wait ms | tok/s |
|---|---|---|---|---|---|---|---|
| 2k | 426 | 17321.0 | 2005.6 | 11.58% | 2490.3 | 12825.0 | 24.585 |
| short | 61 | 3103.4 | 309.2 | 9.96% | 995.9 | 1798.2 | 19.65 |

Verdict: PASS — F8 enter: host+waits 11.3% of the token >= 10% and g2 PASS
