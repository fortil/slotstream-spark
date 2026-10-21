# ngram-format
Date: 2026-10-11
Decision: row_bytes=100
Evidence: db/sources/runs/2026/10/20261011T183603-io2/summary.json; db/records/measurements/io2.md; tools/spark/io2_summary.py
Reversal: a later io2 run where p50(c1 1024 B qd1) / p50(c1 512 B qd1) satisfies 0.1875 * (ratio - 1) > 0.05, which selects 128.

Rule (fixed before measuring): 24/128 = 18.75% of 100-byte rows straddle a 512-byte sector; row_bytes = 128 if 0.1875 * (ratio - 1) > 0.05 else 100. Measured p50 at qd1 with O_DIRECT: 199.4 us for 512 B and 199.5 us for 1024 B, ratio 1.0005, cost 0.0001, far below 0.05. Rows are read with O_DIRECT.

ngram.bin size with 100-byte rows: 32,000,153,600 B (128-byte rows would be 40,960,196,608 B).
