# partition-d5: decode wall-time partition in D5
Date: 2026-10-12
Capture: db/sources/runs/2026/10/20261012T211252-partition/
Command: tools/spark/run.sh --heavy partition auto 3600 -- tools/spark/py.sh -m slotstream_spark.cli bench decode --mode D5 --prompts bench/lb/prompts.json --prompt-keys short,2k --tokens 256 --warmup 1 --profile (torch 2.13.0+cu130, pool 19012 slots)

| gate | value | limit | ok |
|---|---|---|---|
| profiled run (--profile, result ok) | True | true | yes |
| 2k: named phases + other = wall (100%) | 12007820052 of 12007820052 ns | equal and the recorded other matches | yes |
| 2k: phases do not overlap (other >= 0) | 310647703 | >= 0 | yes |
| 2k: other share of wall (%) | 2.587 | <= 10 | yes |
| 2k: t_copy separate from IO (miss-read + copy-tail = read wall) | 975556886 | = 975556886 | yes |
| 2k: profiled tokens | 255 | >= 16 | yes |
| 2k: decode tok/s, (N-1)/sum of inter-token intervals (informational, H3 gates it) | 21.231 | info | yes |
| 2k: pool hit rate h over the decode (informational) | 0.9917 | info | yes |
| short: named phases + other = wall (100%) | 3254498190 of 3254498190 ns | equal and the recorded other matches | yes |
| short: phases do not overlap (other >= 0) | 73768718 | >= 0 | yes |
| short: other share of wall (%) | 2.267 | <= 10 | yes |
| short: t_copy separate from IO (miss-read + copy-tail = read wall) | 705728599 | = 705728599 | yes |
| short: profiled tokens | 60 | >= 16 | yes |
| short: decode tok/s, (N-1)/sum of inter-token intervals (informational, H3 gates it) | 18.432 | info | yes |
| short: pool hit rate h over the decode (informational) | 0.9607 | info | yes |
| tau p50 (us), ensure + launch of all-hit layer steps | 21.44 | <= 150 | yes |

Partition of the decode wall time, prompt `2k` (255 tokens, 12007.8 ms of wall):

| phase | ms | share of wall |
|---|---|---|
| sample | 19.108 | 0.16 % |
| ngram_hash | 54.738 | 0.46 % |
| embed_read | 159.114 | 1.33 % |
| ngram_wait | 93.739 | 0.78 % |
| launch | 108.698 | 0.91 % |
| gpu_wait | 10026.822 | 83.50 % |
| ensure_hits | 259.396 | 2.16 % |
| expert_miss_read | 861.693 | 7.18 % |
| expert_copy_tail | 113.863 | 0.95 % |
| other | 310.648 | 2.59 % |
| wall | 12007.820 | 100.00 % |

Partition of the decode wall time, prompt `short` (60 tokens, 3254.5 ms of wall):

| phase | ms | share of wall |
|---|---|---|
| sample | 4.225 | 0.13 % |
| ngram_hash | 6.028 | 0.19 % |
| embed_read | 38.692 | 1.19 % |
| ngram_wait | 11.293 | 0.35 % |
| launch | 38.771 | 1.19 % |
| gpu_wait | 2292.892 | 70.45 % |
| ensure_hits | 83.099 | 2.55 % |
| expert_miss_read | 628.431 | 19.31 % |
| expert_copy_tail | 77.298 | 2.38 % |
| other | 73.769 | 2.27 % |
| wall | 3254.498 | 100.00 % |

- 2k: 2735 prompt tokens, prefill 59.062 s, 256 new tokens, 121389 hits / 1011 misses; NVTX ranges on
- short: 37 prompt tokens, prefill 0.821 s, 61 new tokens, 27668 hits / 1132 misses; NVTX ranges on

Verdict: PASS — the partition adds up to 100% of the wall time with other <= 10%, t_copy separate from IO, tau within its limit
