# w131k-campaign-d5: engine D5 campaign at a 131,072-token window (N=5)
Date: 2026-10-21
Capture: db/sources/runs/2026/10/20261021T200848-w131k-engine/
Command: tools/spark/jobs.sh start w131k-engine --heavy w131k-engine auto 10800 -- bash bench/campaign/w131k_engine_job.sh (bench/campaign/run_campaign.py --mode D5 --samples 5 --window 131072 --cells decode-8k,ttft-cold-8k,ttft-cold-78k)

| gate | value | limit | ok |
|---|---|---|---|
| window of every cell | 131072 | 131072 | yes |
| cold samples with reused_tokens == 0 | all | all | yes |
| D5 decode median (tok/s), decode-8k cell at the window | 26.918 | >= 23 | yes |
| cold 8K TTFT median (s) | 22.407 | <= 30 | yes |
| follow-up TTFT median (s), 8K | 1.187 | <= 12 | yes |
| follow-up TTFT median (s), 78K | 7.389 | <= 12 | yes |
| follow-up reused tokens, 78K, least sample | 76800 | >= 70000 | yes |
| cold 78K prompt tokens | 78000 | >= 77744 (at most 78000, cut exactly) | yes |
| peak, max(once_bytes, MemAvailable drop), worst cell ttft-cold-78k (B) | 56438362112 | <= 66072209764 (ledger at 131072 tokens, 38 chunks) | yes |

- decode-8k samples (tok/s): 30.79, 26.92, 27.49, 25.29, 24.37 (MTP policy resident, pool pre-filled, 37 chunks)
- cold 8K TTFT samples (s): 22.53, 22.49, 22.41, 22.39, 22.37
- follow-up 8K: 1.19 s (reused 8192, new 137), 1.19 s (reused 8192, new 137), 1.19 s (reused 8192, new 137), 1.19 s (reused 8192, new 137), 1.20 s (reused 8192, new 137)
- ttft-cold-78k: 78000 tokens, TTFT median 221.89 s, prefill 351.9 tok/s (published, not gated)
- cold 78K TTFT samples (s): 221.91, 222.41, 220.81, 221.89, 221.24
- follow-up 78K: 7.39 s (reused 76800, new 1337), 7.39 s (reused 76800, new 1337), 7.37 s (reused 76800, new 1337), 7.39 s (reused 76800, new 1337), 7.37 s (reused 76800, new 1337)
- peak decode-8k: once 48,055,496,704 B, nonpool 7,973,154,816 B, MemAvailable drop 48,226,791,424 B, ledger 64,998,467,940 B (37 chunks at a 131072-token window)
- peak ttft-cold-8k: once 53,967,507,456 B, nonpool 13,165,318,144 B, MemAvailable drop 54,216,871,936 B, ledger 66,072,209,764 B (38 chunks at a 131072-token window)
- peak ttft-cold-78k: once 55,901,581,312 B, nonpool 15,099,392,000 B, MemAvailable drop 56,438,362,112 B, ledger 66,072,209,764 B (38 chunks at a 131072-token window)
- partial acceptance at the same window (ACCEPTANCE_ONLY, record kept in the captures only; this line is added by hand, a `--write` of the judge drops it): self-impact PASS (capture 20261021T203826-acc131k); t0-suite 1649 passed, prefix-exact 7 passed, live-acceptance 80 passed with the over-length probe sized to the served window (capture 20261021T213812-acc131k)

Verdict: PASS — decode, cold 8K, both follow-ups, the 78K reuse floor and every peak within the bars at a 131,072-token window
