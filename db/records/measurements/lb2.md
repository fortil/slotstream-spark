# lb2: unmodified llama.cpp yardstick, Q4_1 arms A and B at D5, IQ3_XXS resident
Date: 2026-10-12
Capture: db/sources/runs/2026/10/20261012T071331-lb2/
Command: tools/spark/jobs.sh start lb2 57344 18000 --memcg slotstream-spark-lb -- tools/capture.sh lb2 -- bash bench/lb/session.sh --configs q41-a-d5,iq3-r-d5 --arm-b (job 07:13:31 to 08:21:11, 4,060 s, rc 0, no guard cut; synced 7078b34-dirty-4b0eecd4)

Server flags (q41 A/B): `-lm mmap -lzm on -cmoe --no-repack -ngl 99 -fit off -c 32768 -np 1 -b 2048 -ub 2048 -fa on -t 16 --cache-ram 0 -ctxcp 4 --metrics -lv 4 --no-op-offload`; arm B adds `--moe-cache-mib 43520`. Prefill of the q41 arms runs on the CPU (`--no-op-offload`); decode runs on the GPU with the experts read from the mapped GGUF.

Decode (tok/s, median of 10 = 2 processes x 5 reps; (N-1)/sum of inter-token times):

| config | short | 2k | 8k | A/A diff (short / 2k / 8k) |
|---|---|---|---|---|
| q41-a-d5 | 10.48 | 10.27 | 10.27 | 0.8 % / 0.8 % / 0.5 % |
| q41-b-d5 | 33.60 | 32.92 | 32.68 | 0.8 % / 1.3 % / 0.03 % |
| iq3-r-d5 | not measured (see below) | | | |

Thread sweep, q41-a-d5, "short" x 3: -t 8 10.12, -t 12 9.98, -t 20 10.12, against 10.48 at -t 16.

Prefill. TTFT is CPU prefill (`--no-op-offload`, 16 threads): informative, not comparable with the engine's F5 targets. The client sends the same prompt after a warmup, and llama.cpp reuses the slot's KV prefix (`timings.cache_n` = prompt tokens - 4 in every measured rep), so the measured TTFT is a warm-prefix figure and not a prefill time: q41-a-d5 0.24 s (short), 0.26 s (2k), 0.26 s (8k); q41-b-d5 0.11 / 0.11 / 0.13 s. The only full prefills are the warmups, read from server.log (`prompt eval time`):

| config / process | prompt tokens | time | tok/s |
|---|---|---|---|
| q41-a-d5 p1 | 77 (cold, first load) | 18.7 s | 4.1 |
| q41-a-d5 p1 | 2,733 | 80.5 s | 33.9 |
| q41-a-d5 p2 | 2,733 | 80.8 s | 33.8 |
| q41-a-d5 p1 | 11,124 | 296.9 s | 37.5 |
| q41-b-d5 p1 | 2,733 | 100.7 s | 27.2 |
| q41-b-d5 p1 | 11,124 | 346.5 s | 32.1 |

The "8k" prompt is 11,166 tokens. `timings.prompt_per_second` in client.jsonl (median ~16 tok/s) comes from the 4 uncached tokens and is not a prefill figure. The server.log files are in the capture since 2026-10-12 (fetch.sh copies `*.log` from step 59 on); the figures above are unchanged.

Memory (envelope and MemFree), from envelope.json, mlb4.json and memfree.csv:

| config | memory.current MiB (MemoryMax) | NVML MiB | HUELLA 57,344 | MemFree min MiB | root-charged cache growth MiB | soft stop |
|---|---|---|---|---|---|---|
| q41-a-d5 | 47,019 (47,616) | 5,396 | 52,415 (ok) | 19,104 | 637 | no |
| q41-b-d5 | 3,583 (4,096) | 48,918 | 52,501 (ok) | 16,564 | 2,832 | no |
| iq3-r-d5 | n/a | n/a | n/a | 14,317 | 5,496 | yes, p1 during load |

Soft floor 14,336 MiB. iq3-r-d5 (arm R, resident, default `--op-offload`) was stopped by memfree_watch.sh while loading 44.6 GB to the GPU, MemFree 14,317 MiB; no request ran and p2 was skipped. It is informational, not a yardstick. No `memfree` or `memcg_over` event in guard-events.jsonl (only the final `dontneed`).

IO per token: not collected (the session does not record the server's read_bytes). NVMe during the 68 s q41-a-d5 load: mean 1,107 MB/s, peak 3,502 MB/s (see m-lb2).

Impact (§2.4, tools/impact_check.py against the latest prod-reads baseline): oom PASS (kernel log clean), memavail PASS (min 29,768 MiB), PSI PASS (max avg10 0.0), latency NOT_MEASURED (0 production requests in the window); 14 containers up before and after.

History. The first three LB2 attempts (06:08, 06:14, 06:21, captures 20261012T060834-lb2, 20261012T061408-lb2, 20261012T062154-lb2) were cut by the guard's `memfree` floor (10,240 MiB; last cut at MemFree 10,103 MiB, Cached 71 GB) while memory.current stayed at 47.1 GB and MemAvailable at 77 GB. Diagnostic runs lbdiag (default flags: root-charged cache +5.7 GB at the first 2k prefill and rising, MemFree 14,487 MiB after 234 s) and lbdiag2 (`--no-op-offload`: +0.7 GB, MemFree never below 19,086 MiB) are in DIAG_LB2.md; captures 20261012T062929-lbdiag and 20261012T063853-lbdiag2 are diagnostic evidence, not records. Fix: `--no-op-offload` for every arm with CPU experts, the 14,336 MiB soft-floor watcher, and the MemFree-based MemoryMax clamp (not binding in this run). A stale run/STOP file left by the diagnostics cut the first start at 07:13:20 before any load; it was removed and the job restarted at 07:13:31.

Verdict: PASS — q41-a-d5 complete without guard cut or soft stop (MemFree min 19,104 MiB, decode 10.3-10.5 tok/s, prefill 33.9 tok/s on the CPU); iq3-r-d5 stopped by the soft floor while loading
