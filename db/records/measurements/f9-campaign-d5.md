# f9-campaign-d5: engine D5 campaign (N=5)
Date: 2026-10-14
Capture: db/sources/runs/2026/10/20261014T212529-f9-engine/
Command: tools/spark/jobs.sh start f9-engine --heavy f9-engine auto 16200 -- bash bench/campaign/f9_engine_job.sh (bench/campaign/run_campaign.py --mode D5 --samples 5)
Justification: ttft (decode 29.37 tok/s = 89.3% of arm B 32.9; cold 8K prefill 372.1 tok/s = 11.6x arm B 32.1)

| gate | value | limit | ok |
|---|---|---|---|
| cold samples with reused_tokens == 0 | all | all | yes |
| D5 decode median (tok/s), decode-8k cell | 29.374 | >= 23 | yes |
| cold 8K TTFT median (s) | 22.033 | <= 30 | yes |
| follow-up TTFT median (s) | 1.051 | <= 12 | yes |

- decode-8k samples (tok/s): 32.08, 30.37, 28.63, 29.37, 27.39 (MTP policy resident, pool pre-filled)
- cold 8K TTFT samples (s): 21.60, 22.03, 22.25, 21.94, 22.18
- follow-up: 1.05 s (reused 8192, new 137), 1.06 s (reused 8192, new 137), 1.05 s (reused 8192, new 137), 1.05 s (reused 8192, new 137), 1.05 s (reused 8192, new 137)
- ttft-cold-2k: 2048 tokens, TTFT median 6.80 s, prefill 301.8 tok/s (published, not gated)
- ttft-cold-32k: 31998 tokens, TTFT median 77.47 s, prefill 413.3 tok/s (published, not gated)
- peak decode-8k: once 59,879,993,344 B, nonpool 7,986,491,392 B, MemAvailable drop 60,217,880,576 B, ledger 67,390,976,868 B (48 chunks) (informational; the hard memory criteria are the impact ones)
- peak ttft-cold-2k: once 59,323,092,992 B, nonpool 7,783,485,440 B, MemAvailable drop 59,492,057,088 B, ledger 67,390,976,868 B (48 chunks) (informational; the hard memory criteria are the impact ones)
- peak ttft-cold-8k: once 60,243,099,648 B, nonpool 8,703,492,096 B, MemAvailable drop 60,649,349,120 B, ledger 67,390,976,868 B (48 chunks) (informational; the hard memory criteria are the impact ones)
- peak ttft-cold-32k: once 61,138,452,480 B, nonpool 9,598,844,928 B, MemAvailable drop 61,369,581,568 B, ledger 67,390,976,868 B (48 chunks) (informational; the hard memory criteria are the impact ones)

Verdict: PASS — D5 decode, cold 8K TTFT and follow-up within the absolute bars; justified against llama.cpp arm B
