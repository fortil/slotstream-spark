# f5-prefill: prefill and TTFT in D5
Date: 2026-10-13
Capture: db/sources/runs/2026/10/20261013T215810-f5-ttft-r4/
Command: tools/spark/run.sh --heavy f5-ttft auto 3600 -- tools/capture.sh f5-ttft -- tools/spark/py.sh bench/ttft.py --mode D5 --samples 5

| gate | value | limit | ok |
|---|---|---|---|
| cold samples with reused_tokens == 0 | 5 of 5 | all | yes |
| cold 8K TTFT median (s) | 23.831 | <= 30 (target 15) | yes |
| follow-up TTFT median (s) | 1.213 | <= 12 (target 8) | yes |
| peak bytes, max(once_bytes, MemAvailable drop), worst child once32k | 60767965184 | <= 67390976868 (ledger, 48 chunks) | yes |

- cold 8K split (medians): queue 0.000 s, prefill 23.798 s, first sample 0.001 s, store 0.032 s, TTFT 23.831 s
- cold 8K TTFT samples: 23.81, 23.83, 23.93, 23.86, 23.78
- follow-up: 1.20 s (reused 8192, read 137), 1.21 s (reused 8192, read 137), 1.21 s (reused 8192, read 137), 1.22 s (reused 8192, read 137), 1.22 s (reused 8192, read 137)
- peak cold8k-0: once 58,854,010,880 B, nonpool 6,960,508,928 B, MemAvailable drop 59,079,659,520 B, ledger 67,390,976,868 B (48 chunks)
- peak cold8k-1: once 58,855,235,584 B, nonpool 6,961,733,632 B, MemAvailable drop 59,061,682,176 B, ledger 67,390,976,868 B (48 chunks)
- peak cold8k-2: once 58,854,043,648 B, nonpool 6,960,541,696 B, MemAvailable drop 59,075,006,464 B, ledger 67,390,976,868 B (48 chunks)
- peak cold8k-3: once 58,855,084,032 B, nonpool 6,961,582,080 B, MemAvailable drop 58,973,474,816 B, ledger 67,390,976,868 B (48 chunks)
- peak cold8k-4: once 58,854,117,376 B, nonpool 6,960,615,424 B, MemAvailable drop 59,001,864,192 B, ledger 67,390,976,868 B (48 chunks)
- peak once2k: once 57,943,523,328 B, nonpool 6,050,021,376 B, MemAvailable drop 57,975,963,648 B, ledger 67,390,976,868 B (48 chunks)
- peak once32k: once 60,572,065,792 B, nonpool 8,678,563,840 B, MemAvailable drop 60,767,965,184 B, ledger 67,390,976,868 B (48 chunks)
- peak scope16k: once 59,210,883,072 B, nonpool 7,317,381,120 B, MemAvailable drop 59,432,538,112 B, ledger 67,390,976,868 B (48 chunks)
- 2K once: TTFT 8.36 s, prefill 8.34 s (245.7 tok/s), bytes requested 53,509,939,200
- 32K once: TTFT 92.74 s, prefill 92.69 s (353.5 tok/s), bytes requested 499,911,782,400
- 16K pass 4096 scope 8192: scopes 4, bytes requested 246,114,201,600, records read 89017, prefill 46.96 s (348.9 tok/s)
- 16K pass 4096 scope 0: scopes 4, bytes requested 246,114,201,600, records read 89017, prefill 42.65 s (384.2 tok/s)
- 16K pass 1024 scope 8192: scopes 7, bytes requested 398,927,462,400, records read 144288, prefill 57.92 s (282.9 tok/s)
- 16K pass 1024 scope 0: scopes 16, bytes requested 841,663,180,800, records read 304421, prefill 101.47 s (161.5 tok/s)
- Axis T: WIN vs llama.cpp arm B cold 8k prefill 32.1 tok/s (346.5 s on 11124 tokens, CPU)
- engine cold 8K prefill 344.2 tok/s, ratio 10.72 to arm B (WIN needs TTFT <= 30 s and >= 4 x 32.1 = 128.4 tok/s)

Verdict: PASS — cold 8K TTFT, follow-up TTFT and peak within the F5 bars; axis T recorded
