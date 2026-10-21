# vara-d5
Date: 2026-10-12
Decision: YARDSTICK_RECORDED. The F3/F9 yardstick for D5 is unmodified llama.cpp Q4_1 arm B (deploy/lb/llamacpp.sh --weights q41 --arm B --mode D5: --moe-cache-mib 43520, MemoryMax 4096M, --no-op-offload; GPU decode with 43,520 MiB of experts cached on the GPU, CPU prefill), inside the D5 envelope (NVML 48,918 + memory.current 3,583 = 52,501 ≤ 57,344 MiB): decode median 32.9 tok/s (33.60/32.92/32.68 short/2k/8k). The D5-translation gate of Decision §7 row 6 fires in substance for arm B (32.9 ≥ 23 tok/s); the user answered on 2026-10-12 (BRIEF.md ## User answers): continue with the engine, yardstick = arm B, the engine justifies itself on decode, TTFT or both under PLAN.md design decision 11. Arm A (10.3 tok/s) stays recorded as the CPU-expert reference and is not a gate input. Step 65's stop/go and the F9 justification compare against [vara_b].
Evidence: db/sources/runs/2026/10/20261012T071331-lb2/summary.json, db/records/measurements/lb2.md, db/records/measurements/m-lb4.md, DIAG_LB2.md
Reversal: A GPU-prefill TTFT for llama.cpp is a new decision that needs a guard floor other than MemFree (default `--op-offload` charges the page cache of the mapped experts to the root memcg, DIAG_LB2.md). The decode figure is unchanged by the flag, so reverse only if a later run shows otherwise. The yardstick returns to arm A, or arm B's numbers change, only by a user answer or a new LB2 capture recorded here.

Yardstick details:
- Binary: llama.cpp commit 23b0202a (build 11552). GGUF: Q4_1, three files (Qwen3.8-Flash-Next-Q4_1-0000N-of-00003.gguf), verified 3/3 (lb0).
- Flag line: `-m <Q4_1 shard 1> -lm mmap -lzm on -cmoe --no-repack -ngl 99 -fit off -c 32768 -np 1 -b 2048 -ub 2048 -fa on -t 16 --cache-ram 0 -ctxcp 4 --host 127.0.0.1 --port 8091 --metrics -lv 4 --no-op-offload`, in a user scope with MemorySwapMax=0.
- HUELLA 57,344 MiB. MemoryMax = min(57,344 - CUDA(m-lb4) - 1,024, MemFree_at_start - 10,240 - CUDA - 4,096): with the m-lb4 CUDA of 5,396 MiB and MemFree at start 72,505 MiB it is min(50,924, 52,773) = 50,924 MiB; LB2 ran with the default reserve of 8,704 MiB, giving 47,616 MiB, and memory.current peaked at 47,019 MiB.
- Arm B (llama.cpp expert cache, 43,520 MiB, MemoryMax 4,096 MiB, NVML 48,918 MiB): decode 33.6 / 32.9 / 32.7 tok/s (short / 2k / 8k); informational, not the gate input.
- Decode q41-a-d5 median: 10.48 short, 10.27 2k, 10.27 8k. TTFT in the client is a warm-prefix figure (0.26 s at 8k, KV prefix reused); the cold CPU prefill is 33.9 tok/s on 2,733 tokens and 37.5 tok/s on 11,124 tokens (296.9 s). memfree.min_mib 19,104.
- IQ3_XXS resident (arm R) is informational, not a yardstick; it tripped the 14,336 MiB soft floor while loading.

Arm B details: flag line as above plus `--moe-cache-mib 43520`; MemoryMax from the table 4,096 MiB (memory.current peak 3,583 MiB); NVML 48,918 MiB; memfree_at_start_mib 72,693; memfree.min_mib 16,564; root-charged growth 2,832 MiB; A/A diff 0.8 % / 1.3 % / 0.03 % (short / 2k / 8k). Cold prefills from `q41-b-d5/p1/server.log` (in the capture): 27.2 tok/s on 2,733 tokens (100.7 s) and 32.1 tok/s on 11,124 tokens (346.5 s). The LB2 client did not send `cache_prompt=false` (cache_n = prompt - 4 on every rep), so `ttft8k_s_median` 0.13 is a warm-prefix figure and not a prefill time; the F3 pairs send it. Precondition (user answer 2026-10-13, BRIEF.md ## User answers): the full 43,520 MiB cache needs MemFree >= 70,465 MiB at launch (72,693 - (16,564 - 14,336)); below it `deploy/lb/llamacpp.sh` runs arm B with `cache = MemFree_at_start - 26,945` MiB, recorded per pair (`cache_mib`, `memfree_start_mib`, `memavailable_start_mib`); engine-vs-B pairs are judged in the same day's conditions (step 65); the figures of this record (32.9 tok/s, full cache) are the absolute reference for F6/F9 and do not change by this rule.

```toml
yardstick = "vara_b"

[vara]
arm = "q41-a-d5"
script = "deploy/lb/llamacpp.sh"
weights = "q41"
mode = "D5"
huella_mib = 57344
memorymax_mib = 47616   # as run (default CUDA reserve); 50924 with the m-lb4 CUDA figure
memfree_at_start_mib = 72505
port = 8091
op_offload = false
prefill_device = "cpu"
memfree_stop_mib = 14336
decode_tps_median = 10.3
ttft8k_s_median = 0.26   # warm KV prefix, not a prefill time
prefill_tps_8k_median = 37.5   # warmup of the 11,124-token prompt, from server.log

[vara_b]
arm = "q41-b-d5"
script = "deploy/lb/llamacpp.sh"
weights = "q41"
mode = "D5"
huella_mib = 57344
memorymax_mib = 4096
moe_cache_mib = 43520
nvml_mib = 48918
memory_current_peak_mib = 3583
memfree_at_start_mib = 72693
memfree_min_mib = 16564
port = 8091
op_offload = false
prefill_device = "cpu"
memfree_stop_mib = 14336
memfree_full_cache_mib = 70465
cache_offset_mib = 26945
cache_min_mib = 21760
decode_tps_median = 32.9
decode_tps_short = 33.6
decode_tps_2k = 32.9
decode_tps_8k = 32.7
ttft8k_s_median = 0.13   # warm KV prefix, not a prefill time
prefill_tps_2k = 27.2
prefill_tps_8k = 32.1
prefill_cold_8k_s = 346.5
prefill_cold_8k_tokens = 11124
source_capture = "db/sources/runs/2026/10/20261012T071331-lb2"
```
