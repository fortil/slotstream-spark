# ppl-ablate: perplexity of the oracle with correct, swapped and removed experts
Date: 2026-10-12
Capture: db/sources/runs/2026/10/20261012T142519-oracle-chain/
Command: part of the oracle-chain job (tools/ppl_ablate.py); result in ppl.json

| mode | en ppl | es ppl | ratio to correct (en / es) |
|---|---|---|---|
| correct | 3.564 | 2.747 | 1 / 1 |
| swap | 2,328.3 | 6,466.2 | 653 / 2,354 |
| noexp | 565.3 | 1,204.3 | 159 / 438 |

Against [ablation]: ppl_correct_max 20 (met, 3.56 and 2.75), swap_min_ratio 5 (met, 653), noexp_min_ratio 5 (met, 159). Texts are 88 (en) and 115 (es) tokens.

Verdict: PASS — correct ppl 3.56/2.75 <= 20; swap and noexp break perplexity by 159x or more
