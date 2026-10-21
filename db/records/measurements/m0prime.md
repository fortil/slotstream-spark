# m0prime: four real layers, design triggers and the recalibrated performance table
Date: 2026-10-12
Capture: db/sources/runs/2026/10/20261012T173240-m0prime/
Command: tools/spark/run.sh --heavy m0prime 10240 2700 -- tools/spark/py.sh bench/m0prime/run.py (torch 2.13.0+cu130)
Earlier pass: db/sources/runs/2026/10/20261012T173131-m0prime/ (Spark-only capture, not synced to the Mac; same code except the footprint accounting, which then subtracted the pool once although it shows in both VmRSS and NVML: footprint 9,121,846,184 B; GEMV 2560x2560 119.6 GB/s). The runner was corrected to subtract the pool once per counter that shows its growth, then run again.
Rule amended 2026-10-12 (BRIEF.md ## User answers, M0' entry): GEMV informational at M0', hard gate at step 55 on the engine kernel with the step-23 config adopted in tuning.py; footprint ceiling 2.9e9 B (measured + 10%); the ledger carries the measured runtime (platform.toml measured.runtime_footprint_mib = 2516).

| gate | value | limit | ok |
|---|---|---|---|
| GEMV 2560x2560 4-bit, cold-L2 graph (GB/s) — informational at M0', hard gate >= 121.5 at step 55 | 118.25 | info (>= 121.5 at step 55) | yes |
| tau p50 (us) | 14.112 | <= 150 | yes |
| runtime footprint (B) | 2637857704 | <= 2.9e+09 B (M0' rule of 2026-10-12; design budget 2.5e+09 informational) | yes |
| GDN fused vs ref rel_l2, layer 0 | 0.0037519 | <= 0.032884 | yes |
| GDN fused vs ref rel_l2, layer 1 | 0.0052895 | <= 0.039328 | yes |
| GDN fused vs ref rel_l2, layer 2 | 0.0072896 | <= 0.053902 | yes |
| KL mean, truncated logits fused vs ref (nats) | 0.00017669 | <= 0.037231 | yes |
| projected D5 decode, h = 0.98, phi = 0 (tok/s) | 18.844 | >= 17.0 | yes |
| 47 x chunk free p99 + 0.25 s (s) | 1.1244 | <= 2.0 | yes |

Measured values:

| item | value |
|---|---|
| fill | 2048 records, 5662310400 B in 0.55 s; IO 10.344 GB/s over the ensure read walls, 10.226 GB/s end to end; 10 reader lanes on CPUs [0, 1, 2, 3, 4, 10, 11, 12, 13, 14] |
| t_memcpy per record | mean 0.1346 ms; p50 0.0927 ms, p99 0.5243 ms (log2 buckets of the reader histogram) |
| chunk free | 100 samples, p50 17165749 ns, p99 18604500 ns, max 18730308 ns |
| layer step (full layer, L2 flushed per replay) | GDN 916.23 us (mean of layers 0 and 2), QSA 842.46 us |
| t_embed / t_lm_head | 21.38 us / 2533.34 us |
| t_comp_48 = 36 t_GDN + 12 t_QSA + t_embed + t_lm_head | 45.648 ms |
| tau | p50 14.112 us, p99 15.568 us over 1024 layer steps after 16 warm-up |
| runtime footprint | 2637857704 B = VmRSS 8280911872 + NVML 7929053184 (pynvml) - pool 6442450944 (subtracted per counter: RSS True, NVML True) - weights 562076760 - state 125128704 |

GEMV on every shape (cold-L2 graph; all informational at M0', 2560x2560 4-bit is the step-55 gate shape)

| shape | bits | us | GB/s |
|---|---|---|---|
| 10240x2560 | 4 | 109.41 | 135.19 |
| 10240x320 | 4 | 20.96 | 89.9 |
| 12288x2560 | 4 | 134.19 | 132.27 |
| 1x2560 | 16 | 3.86 | 2.65 |
| 248320x2560 | 4 | 2513.71 | 142.65 |
| 2560x2560 | 4 | 31.31 | 118.25 |
| 2560x6144 | 4 | 70.74 | 125.39 |
| 2560x640 | 4 | 10.02 | 93.15 |
| 320x10240 | 4 | 17.57 | 106.17 |
| 48x2560 | 16 | 4.1 | 61.3 |
| 4x10240 | 16 | 2.94 | 34.88 |
| 512x2560 | 16 | 14.73 | 178.5 |
| 512x2560 | 4 | 9.23 | 80.67 |
| 6144x2560 | 4 | 64.0 | 138.69 |
| 640x2560 | 16 | 16.19 | 202.85 |
| 640x2560 | 4 | 10.14 | 91.62 |

- GEMV engine kernel 118.25 GB/s at M0' against the 121.5 GB/s gate judged at step 55 (step-23 bench kernel 193.75 GB/s, m0-gemv.md); untuned engine config at this capture
- tau p99 15.568 us; p50 parts: wait 0.352, ensure 11.472, launch 2.224 us
- runtime footprint 2.638 GB vs the design budget 2.5 GB (decision-tecnologia.md:294): accepted on 2026-10-12; the ledger uses platform.toml measured.runtime_footprint_mib
- layer 3 (QSA) rel_l2 of the same A/B, reported: 0.011106
- truncated logits fused vs ref, printed not asserted: top-1 0.9711, KL p99 0.0028553, max 0.0058181

Recalibrated section 3.8 table (measured t_comp_48, tau p50, t_memcpy mean):

| mode | h | no lookahead tok/s | lookahead tok/s (phi 0.3 at low h, 0.5 at high h) |
|---|---|---|---|
| D1 | 0.64-0.70 | 8.9-9.5 | 10.8-13.2 |
| D2 | 0.74-0.78 | 10.0-10.6 | 11.9-14.2 |
| D5 | 0.96-0.98 | 16.9-18.8 | 18.1-20.1 |

Verdict: PASS — every M0' design trigger is within its limit (GEMV informational at M0', judged at step 55; footprint <= 2.9e9 B per the 2026-10-12 rule)
