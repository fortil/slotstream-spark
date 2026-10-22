# context-window
Date: 2026-10-21
Capture: db/sources/runs/2026/10/20261021T200848-w131k-engine/
Decision: PRODUCTION_WINDOW_131072 (SLOTSTREAM_CONTEXT_WINDOW=131072 in ~/slotstream-spark/serve.env, loaded by the engine and front units)
Evidence: db/records/measurements/w131k-campaign-d5.md, db/records/measurements/w131k-impact.md, db/records/measurements/w131k-envelope.md
Reversal: remove serve.env, stop slotstream-spark.service, restart slotstream-spark-front.service (back to 32,768). Rule: revert if the night pipeline fails a segment it passed at the default window, if the front refuses starts with residents present because of the 26,148 MiB floor, or if a 78K turn exceeds the desktop app's timeout.

- The desktop app sends about 78,000 tokens per request through the model router (alias spark/supervisor-reasoning); at 32,768 the engine answered context_length_exceeded.
- Campaign at the window (N=5 per cell): decode-8k 26.918 tok/s with 37 chunks (floor 23), cold 8K TTFT 22.407 s (limit 30), follow-up 8K 1.187 s, cold 78K TTFT 221.89 s (published, not gated), follow-up 78K 7.389 s with 76800 tokens reused (limit 12 s, reuse floor 70000); worst-cell peak 56438362112 B against the ledger's 66072209764 B at 38 chunks.
- Impact: no oom-kill, lowest MemAvailable 25582 MiB, 14 containers up; production paused by the user, so latency, PSI and quiet-window criteria not applicable.
- Cost accepted: the pool at the ceiling drops from 48 to 40 chunks (37 to 38 measured), the floor rises from 17,166 to 26,148 MiB, and decode falls from 29.37 to 26.918 tok/s in the decode-8k cell.
- Switched on 2026-10-21 at 21:57 (host time), after PR #4 and PR #5 were merged and main synced: serve.env written with mode 600, the engine unit stopped, the front unit restarted with both environment files loaded. Verified through the litellm router (alias supervisor-reasoning, the desktop app's path) with tools/probes/router_request.py: a 78000-token first turn answered 200 in 227.4 s including the engine start, the follow-up turn in 7.97 s with 76800 tokens reused; /v1/models advertises context_window 131072 and max_output_tokens 8192; the engine log shows fixed 25270020452 B, floor 26148 MiB, 37 chunks at start and 38 after the governor grew the pool.

Verdict: PASS - PRODUCTION_WINDOW_131072: every gate of the campaign and the impact report passed at the window
