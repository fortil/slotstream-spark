# f9-quality: perplexity, teacher-forced agreement and behaviour probe of the engine
Date: 2026-10-15
Capture: db/sources/runs/2026/10/20261014T212529-f9-engine/
Capture: db/sources/runs/2026/10/20261014T235730-f9-oracle/
Command: tools/spark/jobs.sh start f9-engine --heavy f9-engine auto 16200 -- bash bench/campaign/f9_engine_job.sh (quality phase: tools/quality/ppl_wikitext.py, teacher_forced.py, quality_probe.sh); tools/spark/jobs.sh start f9-oracle --heavy f9-oracle 16384 9000 -- bash bench/campaign/f9_oracle_job.sh (tools/quality/oracle_ppl.py)
Logits rule: seq KL (factor 3.0, floor = calibrated limit, ceilings 0.05/0.5/1.5 nats; top-1 calibrated), design decision 18
Routing rule: seq_rule (factor 2.0, ceiling c_k/2), design decision 16
Top-1 rule: pooled over the 8 reserved prompts (sum of agreeing positions / sum of positions >= top1_min 0.88976, calibrated); per-prompt top-1 informative; design decision 21, user decision 2026-10-14
PPL rule: engine vs transformers oracle on the same 32 wikitext windows, |PPL_engine/PPL_oracle - 1| <= 2%; README reference 5.3914 +-1% informative; design decision 21, user decision 2026-10-14

| gate | value | limit | ok |
|---|---|---|---|
| wikitext-2 PPL engine vs transformers oracle over 32 windows / 65,504 tokens | engine 4.7216, oracle 4.7265, ratio-1 -0.10% | |ratio-1| <= 2% (oracle dtype torch.bfloat16) | yes |
| top-1 pooled over 8 reserved prompts | 0.9545 (210/220) | >= 0.88976 (calibrated; pooled pair 0.9273) | yes |
| res0 top-1 | 0.8333 (20/24) | informative (pair 0.8750) | info |
| res1 top-1 | 0.9600 (24/25) | informative (pair 1.0000) | info |
| res2 top-1 | 1.0000 (23/23) | informative (pair 0.9130) | info |
| res3 top-1 | 0.9643 (27/28) | informative (pair 0.9643) | info |
| res4 top-1 | 0.9737 (37/38) | informative (pair 0.9474) | info |
| res5 top-1 | 0.9706 (33/34) | informative (pair 0.9118) | info |
| res6 top-1 | 0.9615 (25/26) | informative (pair 0.8846) | info |
| res7 top-1 | 0.9545 (21/22) | informative (pair 0.9091) | info |
| res0 kl_mean | 0.03237 | <= 0.05000 (ceiling; pair 0.03059) | yes |
| res0 kl_p99 | 0.16903 | <= 0.42587 (calibrated; pair 0.10632) | yes |
| res0 kl_max | 0.17582 | <= 0.99792 (calibrated; pair 0.11512) | yes |
| res0 n-gram ids exact | yes | exact | yes |
| res0 routing and separable layers | 0 failing | 0 failing within seq_rule | yes |
| res1 kl_mean | 0.01295 | <= 0.03723 (calibrated; pair 0.01044) | yes |
| res1 kl_p99 | 0.09806 | <= 0.42587 (calibrated; pair 0.03492) | yes |
| res1 kl_max | 0.12222 | <= 0.99792 (calibrated; pair 0.03725) | yes |
| res1 n-gram ids exact | yes | exact | yes |
| res1 routing and separable layers | 0 failing | 0 failing within seq_rule | yes |
| res2 kl_mean | 0.00402 | <= 0.03723 (calibrated; pair 0.00899) | yes |
| res2 kl_p99 | 0.01757 | <= 0.42587 (calibrated; pair 0.04560) | yes |
| res2 kl_max | 0.01830 | <= 0.99792 (calibrated; pair 0.04643) | yes |
| res2 n-gram ids exact | yes | exact | yes |
| res2 routing and separable layers | 0 failing | 0 failing within seq_rule | yes |
| res3 kl_mean | 0.00113 | <= 0.03723 (calibrated; pair 0.00145) | yes |
| res3 kl_p99 | 0.00698 | <= 0.42587 (calibrated; pair 0.01217) | yes |
| res3 kl_max | 0.00837 | <= 0.99792 (calibrated; pair 0.01492) | yes |
| res3 n-gram ids exact | yes | exact | yes |
| res3 routing and separable layers | 0 failing | 0 failing within seq_rule | yes |
| res4 kl_mean | 0.01776 | <= 0.05000 (ceiling; pair 0.01993) | yes |
| res4 kl_p99 | 0.10458 | <= 0.42587 (calibrated; pair 0.10903) | yes |
| res4 kl_max | 0.13165 | <= 0.99792 (calibrated; pair 0.11636) | yes |
| res4 n-gram ids exact | yes | exact | yes |
| res4 routing and separable layers | 0 failing | 0 failing within seq_rule | yes |
| res5 kl_mean | 0.01278 | <= 0.03723 (calibrated; pair 0.00853) | yes |
| res5 kl_p99 | 0.05250 | <= 0.42587 (calibrated; pair 0.04954) | yes |
| res5 kl_max | 0.05970 | <= 0.99792 (calibrated; pair 0.05627) | yes |
| res5 n-gram ids exact | yes | exact | yes |
| res5 routing and separable layers | 0 failing | 0 failing within seq_rule | yes |
| res6 kl_mean | 0.00904 | <= 0.04323 (sequence; pair 0.01441) | yes |
| res6 kl_p99 | 0.04187 | <= 0.42587 (calibrated; pair 0.12711) | yes |
| res6 kl_max | 0.04813 | <= 0.99792 (calibrated; pair 0.15959) | yes |
| res6 n-gram ids exact | yes | exact | yes |
| res6 routing and separable layers | 0 failing | 0 failing within seq_rule | yes |
| res7 kl_mean | 0.01093 | <= 0.03723 (calibrated; pair 0.00733) | yes |
| res7 kl_p99 | 0.02667 | <= 0.42587 (calibrated; pair 0.01767) | yes |
| res7 kl_max | 0.02802 | <= 0.99792 (calibrated; pair 0.01848) | yes |
| res7 n-gram ids exact | yes | exact | yes |
| res7 routing and separable layers | 0 failing | 0 failing within seq_rule | yes |
| quality probe | 15/15 | 15/15 | yes |

| window | start | engine NLL | oracle NLL | ratio-1 |
|---|---|---|---|---|
| 0 | 0 | 2030.04 | 2000.68 | +1.44% |
| 1 | 2048 | 1451.51 | 1432.69 | +0.92% |
| 2 | 4096 | 1712.73 | 1729.70 | -0.83% |
| 3 | 6144 | 3564.70 | 3523.07 | +2.05% |
| 4 | 8192 | 3240.87 | 3221.10 | +0.97% |
| 5 | 10240 | 1895.66 | 1895.85 | -0.01% |
| 6 | 12288 | 1720.68 | 1706.77 | +0.68% |
| 7 | 14336 | 2613.07 | 2610.72 | +0.12% |
| 8 | 16384 | 3574.53 | 3684.12 | -5.21% |
| 9 | 18432 | 3696.33 | 3679.17 | +0.84% |
| 10 | 20480 | 4084.64 | 3962.75 | +6.13% |
| 11 | 22528 | 3928.19 | 3916.23 | +0.59% |
| 12 | 24576 | 1960.97 | 2005.03 | -2.13% |
| 13 | 26624 | 4337.27 | 4388.55 | -2.47% |
| 14 | 28672 | 3769.00 | 3773.51 | -0.22% |
| 15 | 30720 | 5039.10 | 4999.74 | +1.94% |
| 16 | 32768 | 4644.36 | 4623.87 | +1.01% |
| 17 | 34816 | 3718.60 | 3728.58 | -0.49% |
| 18 | 36864 | 3812.71 | 3763.66 | +2.43% |
| 19 | 38912 | 3082.02 | 3206.33 | -5.89% |
| 20 | 40960 | 2062.22 | 2027.49 | +1.71% |
| 21 | 43008 | 2668.40 | 2689.47 | -1.02% |
| 22 | 45056 | 2054.64 | 2033.50 | +1.04% |
| 23 | 47104 | 2380.31 | 2405.50 | -1.22% |
| 24 | 49152 | 2423.54 | 2428.58 | -0.25% |
| 25 | 51200 | 2489.64 | 2557.56 | -3.26% |
| 26 | 53248 | 4737.47 | 4774.43 | -1.79% |
| 27 | 55296 | 4894.57 | 4884.09 | +0.51% |
| 28 | 57344 | 4285.49 | 4276.47 | +0.44% |
| 29 | 59392 | 3289.76 | 3296.83 | -0.34% |
| 30 | 61440 | 3587.87 | 3601.14 | -0.65% |
| 31 | 63488 | 2920.83 | 2912.35 | +0.42% |

- res0: 24 positions; worst routing ratio 0.530 at layer 14 (calibrated); bound layers [0, 15, 25, 26, 30, 31, 32, 33, 34, 35, 37, 39, 40, 41, 47]; fixed limits not held (informative)
- res1: 25 positions; worst routing ratio 0.447 at layer 16 (calibrated); bound layers [0, 1, 12, 37, 39, 40, 41]; fixed limits held (informative)
- res2: 23 positions; worst routing ratio 0.429 at layer 7 (sequence); bound layers [0, 7, 8, 15, 26, 31, 37, 39, 40, 41]; fixed limits held (informative)
- res3: 28 positions; worst routing ratio 0.239 at layer 22 (calibrated); bound layers [0, 37, 39, 40, 41]; fixed limits held (informative)
- res4: 38 positions; worst routing ratio 0.571 at layer 22 (sequence); bound layers [0, 1, 2, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 40, 41, 46, 47]; fixed limits held (informative)
- res5: 34 positions; worst routing ratio 0.540 at layer 26 (calibrated); bound layers [0, 5, 7, 9, 12, 20, 21, 22, 23, 25, 37, 39, 40, 41]; fixed limits held (informative)
- res6: 26 positions; worst routing ratio 0.458 at layer 14 (sequence); bound layers [0, 14, 37, 39, 40, 41]; fixed limits held (informative)
- res7: 22 positions; worst routing ratio 0.390 at layer 30 (calibrated); bound layers [0, 4, 17, 21, 22, 37, 39, 40, 41]; fixed limits held (informative)
- wikitext-2 PPL over 145 windows / 296,815 tokens: 5.7338; README reference 5.3914 (+6.35%), informative: the reference's text/window protocol is not published (DIAG_F9.md)
- oracle ratio per window: max |ratio-1| 6.13% at window 10; engine NLL above the oracle's on 17 of 32 windows
- wikitext rows sha256 5c8877eeabb3b8afc09f8ec35194fdac8c55acaa25de738d27000876d7c73e4c, token ids sha256 e65198947ecb06833dcd69e042e406f52945dcc88073621d52c3e8e54a4d8e55

Verdict: PASS — PPL within +-2% of the transformers oracle on 32 windows, top-1 pooled >= 0.88976, KL, routing and layers within the per-sequence rules on the 8 reserved prompts, n-gram ids exact, probe 15/15
