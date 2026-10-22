# Log

Append-only.
- 2026-10-12 step 35: tools/perf_model.py added (plan 3.8 model, --plan reproduces the plan ranges, --measured uses platform.toml); db/records/decisions/h0.md written with gen16, goldens and calibration pending step 36.
- 2026-10-12 step 36: oracle chain done; gen16, goldens and ppl-ablate PASS; contract calibration FAIL (design gate: oracle bf16-vs-fp32 logits top-1 0.9449, KL mean 0.01241, p99 0.14196, max 0.33264 against [logits]); arrays not written, waiting for the user.
- 2026-10-12 step 36b: contract arrays and calibrated logits limits written by apply (44/48 layers, routing 47/48, top-1 >= 0.88976, KL <= 0.03723/0.42587/0.99792); oracle-calibration PASS; H0 closed.
- 2026-10-21 context window: w131k-engine campaign at a 131,072-token window PASS (decode-8k 26.918 tok/s with 37 chunks, cold 78K TTFT 221.89 s, follow-up 7.389 s with 76800 reused), w131k-impact PASS (MemAvailable min 25582 MiB), w131k-envelope INFO; six claims, decisions/context-window.md; two engine fixes first (top-k row limit, pins of the first decode step).
