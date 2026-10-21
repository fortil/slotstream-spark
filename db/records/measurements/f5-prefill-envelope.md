# f5-prefill-envelope: CUDA transient and launch time of one prefill pass
Date: 2026-10-13
Capture: db/sources/runs/2026/10/20261013T140455-f5-envelope/
Command: tools/spark/run.sh --heavy f5-envelope 16384 1800 -- tools/spark/py.sh tools/prefill_envelope.py --write
Record: db/records/measurements/f5-prefill-envelope.json (written by the tool, fetched back)

Each cell is one QSA layer's store + select + sparse attention for a pass of q tokens at a context of c tokens (synthetic
bf16 inputs: the ops' memory and time depend on shapes, not values) and the grouped affine MoE of the same pass
(R = 10 q rows over 32 synthetic records) with the rank-order combine. Memory is the `torch.cuda.max_memory_allocated`
delta over the inputs; time is CUDA events after one untimed warm-up call per shape (Triton compile excluded).

| pass | context | q x k | attn MB | attn ms | MoE MB | MoE ms | fits |
|---|---|---|---|---|---|---|---|
| 256 | 2048 | 589,824 | 139.7 | 6.7 | 32.5 | 3.9 | yes |
| 256 | 4096 | 1,114,112 | 139.3 | 6.4 | 32.5 | 3.9 | yes |
| 256 | 8192 | 2,162,688 | 139.4 | 6.3 | 32.5 | 3.9 | yes |
| 256 | 16384 | 4,259,840 | 139.7 | 7.1 | 32.5 | 3.9 | yes |
| 256 | 32768 | 8,454,144 | 139.7 | 7.1 | 32.5 | 3.9 | yes |
| 512 | 2048 | 1,310,720 | 144.2 | 8.1 | 62.9 | 6.3 | yes |
| 512 | 4096 | 2,359,296 | 144.2 | 8.7 | 62.9 | 6.3 | yes |
| 512 | 8192 | 4,456,448 | 145.1 | 7.9 | 62.9 | 6.3 | yes |
| 512 | 16384 | 8,650,752 | 143.3 | 7.8 | 62.9 | 6.3 | yes |
| 512 | 32768 | 17,039,360 | 144.2 | 8.0 | 62.9 | 6.3 | yes |
| 1024 | 2048 | 3,145,728 | 153.1 | 9.5 | 125.8 | 10.7 | yes |
| 1024 | 4096 | 5,242,880 | 153.1 | 9.3 | 125.8 | 10.7 | yes |
| 1024 | 8192 | 9,437,184 | 153.1 | 9.3 | 125.8 | 10.7 | yes |
| 1024 | 16384 | 17,825,792 | 152.3 | 9.6 | 125.8 | 10.7 | yes |
| 1024 | 32768 | 34,603,008 | 153.1 | 10.2 | 125.8 | 10.7 | yes |
| 2048 | 2048 | 8,388,608 | 170.9 | 12.7 | 251.7 | 20.9 | yes |
| 2048 | 4096 | 12,582,912 | 170.9 | 12.9 | 251.7 | 20.9 | yes |
| 2048 | 8192 | 20,971,520 | 170.9 | 13.4 | 251.7 | 20.9 | yes |
| 2048 | 16384 | 37,748,736 | 170.3 | 13.8 | 251.7 | 20.9 | yes |
| 2048 | 32768 | 71,303,168 | 170.9 | 14.8 | 251.7 | 20.9 | yes |
| 4096 | 2048 | 25,165,824 | 206.5 | 15.9 | 503.3 | 41.9 | yes |
| 4096 | 4096 | 33,554,432 | 206.5 | 16.5 | 503.3 | 41.9 | yes |
| 4096 | 8192 | 50,331,648 | 206.5 | 16.6 | 503.3 | 41.9 | yes |
| 4096 | 16384 | 83,886,080 | 206.5 | 22.2 | 503.3 | 41.9 | yes |
| 4096 | 32768 | 150,994,944 | 206.5 | 18.8 | 503.3 | 41.9 | yes |

Grouped-MoE launch time at larger row counts (max of 3 timed calls, ms):

| rows | gate_up | down |
|---|---|---|
| 163840 | 49.6 | 22.5 |
| 4096 | 1.4 | 0.6 |
| 40960 | 12.9 | 5.5 |
| 81920 | 25.3 | 10.9 |

Result:

- `qk_product` = 150,994,944 (the largest cell; every cell fits the D5 prefill ledger line of 5,324,800,000 B and the 200 ms launch bound).
- `bytes_per_pair` = 0.000: the attention transient does not grow with the query-by-key product in the measured grid. The select keeps a fixed 2,048-token budget, so it is a function of the pass only: 65407 B per pass token. MoE: 122892 B per pass token. `bytes_per_pass_token` = 188299.
- `max_rows_per_launch` = 81,920 (largest measured row count within 200 ms is 163,840; capped at 81,920 rows so the temporaries stay near 1 GiB).
- Longest launch measured: 49.6 ms.

The measured product bound applies to contexts up to 32,768 tokens: beyond that the schedule halves against the product, which is not covered by a measurement.
Verdict: INFO — measurement table of the CUDA transient and launch time per pass, no pass/fail gate
