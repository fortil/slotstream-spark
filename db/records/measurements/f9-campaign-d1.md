# f9-campaign-d1: engine D1 campaign (N=3), informative
Date: 2026-10-14
Capture: db/sources/runs/2026/10/20261014T212529-f9-engine/
Command: tools/spark/jobs.sh start f9-engine --heavy f9-engine auto 16200 -- bash bench/campaign/f9_engine_job.sh (bench/campaign/run_campaign.py --mode D1 --samples 3)

- decode-8k: ok, decode median 9.37 tok/s over 3 prompts
- ttft-cold-2k: ok, TTFT median 8.42 s, prefill 243.3 tok/s
- ttft-cold-32k: FAILED ['sample 0: RuntimeError: the stream ended without a final frame', 'sample 1: RuntimeError: the stream ended without a final frame', 'sample 2: RuntimeError: the stream ended without a final frame']
- ttft-cold-8k: ok, TTFT median 24.50 s, prefill 334.6 tok/s, follow-up 14.46 s

Verdict: INFO — D1 recorded only; no gate
