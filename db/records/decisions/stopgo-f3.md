# stopgo-f3: F3 stop/go against llama.cpp arm B
Date: 2026-10-13
Capture: db/sources/runs/2026/10/20261013T092431-f3-session/
Compared quantity: decode tok/s, (N-1)/sum; engine D5 vs unmodified llama.cpp arm q41-b-d5 (prefill_device=cpu, op_offload=False); the LB side's CPU prefill is excluded. IQ3_XXS and arm A are not compared.
Yardstick: [vara_b] decode 32.9 tok/s, cold 8k prefill 32.1 tok/s

| pair | memfree_start_mib | memavailable_start_mib | cache_mib | memfree_min_mib | ratio | nvme_temp_mc | nvme_limit_mc |
|---|---|---|---|---|---|---|---|
| 0 | 67263 | 82003 | 40318 | 14349 | 0.6176 | 54850 | 77850 |
| 1 | 67072 | 81888 | 40127 | 14461 | 0.6150 | 57850 | 77850 |
| 2 | 67075 | 81925 | 40130 | 14366 | 0.6148 | 52850 | 77850 |
| 3 | 67088 | 81941 | 40143 | 14426 | 0.6209 | 57850 | 77850 |
| 4 | 67053 | 81905 | 40108 | 14547 | 0.6114 | 52850 | 77850 |

Clean pairs: 5; excluded: none
cache_full_pairs / clean pairs: 0 / 5
Median ratio 0.6150, positive fraction 0.00; engine 19.58 tok/s vs arm B 31.77 tok/s in the pairs (medians of per-pair means over short and 2k).
Arm B cold prefill tok/s observed in the pairs: 2k 23.9, short 5.2 (corroboration of [vara_b] prefill_tps_2k 27.2).
Absolute reference: engine D5 decode median 19.7 tok/s = 60 % of [vara_b] 32.9 tok/s (LB2, full cache; informative here, the gate input at F6/F9 through tools.vara.yardstick())

Decision: GO_ON_TTFT — decode ratio 0.615 (61% of arm B); the user chose to continue on the TTFT axis (BRIEF.md#user-answers 2026-10-13 13:00 F3 stop/go); the engine must win axis T at F5 (step 72) and F9 (step 95), rule of PLAN.md design decision 11 [arm B reduced cache in 5/5 clean pairs: min 40108 MiB (MemFree at start 67053-67263 MiB); full 43,520 needs MemFree ≥ 70,465]
Evidence: db/sources/runs/2026/10/20261013T092431-f3-session/, db/records/measurements/h3.md
Reversal: the engine does not win axis T at F5 or F9 (the rule stated in the Decision line); the stop/go is then reopened.
