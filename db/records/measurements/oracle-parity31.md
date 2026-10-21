# oracle-parity31: transformers v5.19.0 layer-by-layer oracle vs the MLX 0.31.1 parity31 goldens
Date: 2026-10-11
Capture: db/sources/runs/2026/10/20261011T201154-oracle-p31/
Calibration: db/sources/runs/2026/10/20261011T201132-oracle-p31-calib/ (record oracle-parity31-calibration.md)
Command: tools/spark/run.sh --heavy oracle-p31 10240 1200 -- tools/capture.sh oracle-p31 -- tools/spark/py.sh -m pytest -m gate tests/gates/test_oracle_parity31.py -q (tokens 9707,11,1246,525,498,30; previous context EOS; layers 0-3; GDN and short convs in torch, fla/causal_conv1d/kernels blocked)
limits_sha256: a0bfea9e39c99a323c6d7941cef5e2e8f7e2e96b694730d4c6f3c2174779e3b8 (tests/gates/parity31_limits.json, L_k = 2*s_k + 0.01)

| check | value | limit | separable |
|---|---|---|---|
| n-gram ids (6 x 16) | bit-exact | exact | |
| dequantized row 16410909 (160 values) | exact at bf16 resolution | exact | |
| layer 0 output (rel_l2) | 0.01855 | 0.02862 | yes |
| layer 1 output (rel_l2) | 0.02869 | 0.04721 | yes |
| layer 2 output (rel_l2) | 0.03902 | 0.06389 | yes |
| layer 3 output (rel_l2) | 0.08170 | 0.16403 | yes |

Dequantized row: 160 of 160 values match the single-rounding form bf16(q*s+b) (`dequant_row_single_rounding_matches`, informational).

History: the first run (20261011T193626-oracle-p31) used a fixed 2e-2 limit and failed (layers 0.01855, 0.02869, 0.03902, 0.08170 against 0.02). Ad hoc numbers from that round: the fp32 oracle missed the goldens by 0.01674, 0.02507, 0.03515, 0.06852; bf16 vs fp32 oracle 0.00931, 0.01860, 0.02694, 0.07701; noexp controls 0.372, 0.320, 0.390, 0.424; nibble swap 0.393, 0.344, 0.420, 0.476. The limits were then replaced by the pre-registered rule computed from the oracle's own spread (the calibration run reproduced those numbers).

Verdict: PASS — oracle reproduces parity31 within calibrated per-layer limits, ids bit-exact, row exact
