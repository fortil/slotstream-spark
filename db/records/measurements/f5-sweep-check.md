# f5-sweep-check: sweep prefill vs pool states, read scope and the oracle on 2K ids
Date: 2026-10-13
Capture: db/sources/runs/2026/10/20261013T205832-c23-sweepcheck2/
Command: tools/spark/run.sh --heavy f5-sweep-check auto 2700 -- tools/spark/py.sh -m pytest -q -m gate tests/gates/test_sweep_check.py (run as a jobs.sh job from a snap.sh copy, 21.5 min)
GEMM prefill path (step 71d): re-judged on the same 2K ids under the 70a rule; previous path 1.4930 (20261013T150212-f5-sweep-check); new capture 20261013T205832-c23-sweepcheck2

| gate | value | limit | ok |
|---|---|---|---|
| (a) failing_layers | [] | [] | yes |
| (b) layers ok, routing_ok and sweep_vs_f2_ok | 48 of 48 | 48 of 48 | yes |
| (c) summary prompts, hash and 32 ids | ['2k', '8k', '9k'] | ['2k', '8k', '9k'] | yes |
| (d) logits.top1 | 0.99512 | >= 0.88976 (calibrated) | yes |
| (e) logits.kl_mean | 0.0037139 | <= 0.037231 (calibrated) | yes |
| (f) logits.kl_p99 | 0.050724 | <= 0.42587 (calibrated) | yes |
| (g) logits.kl_max | 1.1663 | <= 1.5 (ceiling) | yes |
| (h) logits.judgeable (pair under every ceiling) | True | True | yes |
| (i) guard post: cut, containers up | None, 14 | None, 14 | yes |

- fixed_limits_ok False (engine against the calibrated limits; false is expected on 2,048 positions)
- stored bridge (not used): ok True, limits_seq.kl_max None
- pair: top1 0.99414, kl_mean 0.0044087, kl_p99 0.064752, kl_max 1.1492
- ceilings: kl_mean 0.05, kl_p99 0.5, kl_max 1.5

Logits rule: seq KL (factor 3.0, floor = calibrated limit, ceilings 0.05/0.5/1.5 nats; top-1 calibrated), design decision 18

Verdict: PASS — sweep prefill identical across pool states and scopes; layers, routing and logits within the per-sequence rule
