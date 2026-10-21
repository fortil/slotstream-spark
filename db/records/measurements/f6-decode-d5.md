# f6-decode-d5: engine D5 decode at the end of F6
Date: 2026-10-14
Capture: db/sources/runs/2026/10/20261014T114220-f6-ab-mtp/
Command: tools/spark/jobs.sh start f6-ab --heavy f6-ab auto 14400 -- bash bench/f6_ab.sh (A/B label f6-ab-mtp)

- arm in force: MTP A/B arm A (mtp-off/lookahead-off), lookahead KEEP_OFF, MTP off; metric (N-1)/sum(inter-token) per pair
- D5 decode median 25.949 tok/s over 16 clean pairs (target >= 23)
- vs llama.cpp arm B (32.9 tok/s, full 43,520 MiB cache, LB2): 78.9% — Axis D: LOSE
- axis_d_ratio = 0.7887
- informative at F6: the F3 answer in db/records/decisions/stopgo-f3.md governs and F9 decides; the F3 same-day paired ratio was 0.6150 (arm B ran with a reduced cache)

- note: pool pre-filled from expert_freq_gb10.json at engine start and kept through the warm-up prompt in both arms (step 77f); kernels: v3 dense GEMV with the 3-row program, v3 MoE body, tuning table of step 77c
- note: lookahead KEEP_OFF from capture 20261014T042830-f6-ab-lookahead, not re-run (DIAG_F6 section 5)
Verdict: PASS — D5 decode median at or above the F6 target
