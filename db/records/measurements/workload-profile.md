# workload-profile: the engine behind a LiteLLM proxy, call profile of a nightly text-analysis pipeline
Date: 2026-10-16
Capture: db/sources/runs/2026/10/20261016T172708-workload-profile/
Command: bash tools/capture.sh workload-profile -- tools/spark/py.sh bench/workload_profile.py --base-url http://127.0.0.1:18434 --cold-timeout 900
Rule: a warm call of about 16K prompt tokens with max_tokens 6000 finishes in 270 s or less; no call exceeds the pipeline's 300 s attempt timeout; the cold start is reported, not gated

Mode D5, engine stopped before the run, no guard `run` (the front refuses to start the engine while the guard holds the model lock), the host's other services running. One call per shape, N = 3 calls.

| call | JSON mode | prompt tokens | completion tokens | total s | finish |
|---|---|---|---|---|---|
| json16k | json_object | 15740 | 562 | 64.778 | stop |
| json25k | json_object | 25906 | 22 | 62.369 | stop |
| text8k | no | 8154 | 1347 | 76.44 | stop |

| item | value |
|---|---|
| cold start (first request on a stopped engine) | 20.137 s, 1 attempt |
| calls over 300 s | 0 |
| calls with status 200 | 3 of 3 |

What this does not show: one run, one call per shape, page cache probably warm from earlier runs (a cold cache after a reboot loads slower), and the profile does not parse the JSON of the two json_object calls, it only checks status 200 and a finished response.

Verdict: PASS — cold start 20.1 s, all calls <= 270 s (json16k 64.8 s, json25k 62.4 s, text8k 76.4 s)
