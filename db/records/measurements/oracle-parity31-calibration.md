# oracle-parity31-calibration: parity31 limits from the oracle's own bf16 vs fp32 spread
Capture: db/sources/runs/2026/10/20261011T201132-oracle-p31-calib/
Rule: L_k = 2*s_k + 0.01; separable iff c_k >= 4*s_k and c_k >= 2*L_k (s = rel_l2 of the bf16 oracle vs the fp32 oracle, c = min of the noexp and swap controls vs the golden)

| layer | spread s | noexp | swap | bf16 vs golden | fp32 vs golden | limit L | c/s | c/L | separable |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0093 | 0.3722 | 0.3929 | 0.0185 | 0.0167 | 0.0286 | 40.0 | 13.0 | yes |
| 1 | 0.0186 | 0.3196 | 0.3443 | 0.0287 | 0.0251 | 0.0472 | 17.2 | 6.8 | yes |
| 2 | 0.0269 | 0.3901 | 0.4199 | 0.0390 | 0.0352 | 0.0639 | 14.5 | 6.1 | yes |
| 3 | 0.0770 | 0.4237 | 0.4765 | 0.0817 | 0.0685 | 0.1640 | 5.5 | 2.6 | yes |

Verdict: INFO — limits written, 4/4 layers separable
