# f9-reference: engine against exact-Q4_1 llama.cpp, perplexity on the same wikitext windows
Date: 2026-10-15
Capture: db/sources/runs/2026/10/20261015T103857-f9-ref/
Command: tools/spark/jobs.sh start f9-ref --heavy f9-ref 57344 12600 --memcg slotstream-spark-f9-ref -- bash bench/campaign/f9_reference_job.sh (repack_q4_1.py --sweep, --check-only and build, llamacpp.sh smoke, tools/quality/llamacpp_ppl.py)
Rule: |PPL_engine(32)/PPL_llama(32) - 1| <= 2% on the first 32 windows (same ids_sha256, same starts); design decision 21 · mapping: correlation >= 0.95 for every family

| gate | value | limit | ok |
|---|---|---|---|
| exactness sweep | 3215 tensors, 7 named exceptions, 0 not exact | 3215 tensors, exactly the 7 names of q4_1_exceptions.json, abs error <= 2^-24 | yes |
| mapping check | 1224 tensors, floor 0.95, 88 in [0.95, 0.99), 0 under the floor | 1224 tensors, floor 0.95, none under it | yes |
| memfree soft stop | no | no | yes |
| prefill device | cpu, op_offload=false | cpu, op_offload=false | yes |
| |PPL_engine(32)/PPL_llama(32) - 1| | 0.05% (engine 4.7216, llama.cpp 4.7239) | <= 2% | yes |

| window | start | engine NLL | llama.cpp NLL | ratio-1 |
|---|---|---|---|---|
| 0 | 0 | 2030.04 | 1957.70 | +3.60% |
| 1 | 2048 | 1451.51 | 1483.86 | -1.57% |
| 2 | 4096 | 1712.73 | 1702.18 | +0.52% |
| 3 | 6144 | 3564.70 | 3547.42 | +0.85% |
| 4 | 8192 | 3240.87 | 3177.54 | +3.14% |
| 5 | 10240 | 1895.66 | 1930.13 | -1.67% |
| 6 | 12288 | 1720.68 | 1728.11 | -0.36% |
| 7 | 14336 | 2613.07 | 2636.39 | -1.13% |
| 8 | 16384 | 3574.53 | 3551.55 | +1.13% |
| 9 | 18432 | 3696.33 | 3683.42 | +0.63% |
| 10 | 20480 | 4084.64 | 3979.41 | +5.27% |
| 11 | 22528 | 3928.19 | 4000.19 | -3.46% |
| 12 | 24576 | 1960.97 | 2016.53 | -2.68% |
| 13 | 26624 | 4337.27 | 4427.29 | -4.30% |
| 14 | 28672 | 3769.00 | 3769.52 | -0.03% |
| 15 | 30720 | 5039.10 | 4823.02 | +11.13% |
| 16 | 32768 | 4644.36 | 4651.45 | -0.35% |
| 17 | 34816 | 3718.60 | 3751.74 | -1.61% |
| 18 | 36864 | 3812.71 | 3760.27 | +2.59% |
| 19 | 38912 | 3082.02 | 3063.61 | +0.90% |
| 20 | 40960 | 2062.22 | 2113.56 | -2.48% |
| 21 | 43008 | 2668.40 | 2700.03 | -1.53% |
| 22 | 45056 | 2054.64 | 2078.87 | -1.18% |
| 23 | 47104 | 2380.31 | 2373.00 | +0.36% |
| 24 | 49152 | 2423.54 | 2418.19 | +0.26% |
| 25 | 51200 | 2489.64 | 2537.91 | -2.33% |
| 26 | 53248 | 4737.47 | 4699.87 | +1.85% |
| 27 | 55296 | 4894.57 | 4882.42 | +0.60% |
| 28 | 57344 | 4285.49 | 4323.04 | -1.82% |
| 29 | 59392 | 3289.76 | 3296.27 | -0.32% |
| 30 | 61440 | 3587.87 | 3680.49 | -4.42% |
| 31 | 63488 | 2920.83 | 2959.10 | -1.85% |

- PPL(32): engine 4.7216, transformers oracle 4.7265 (f9-quality), llama.cpp exact-Q4_1 4.7239
- ratio per window: max |ratio-1| 11.13% at window 15; engine NLL above llama.cpp's on 14 of 32 windows
- sweep: 3215 tensors, exact except the 7 named exceptions (abs error <= 2^-24): language_model.model.layers.1.linear_attn.in_proj_qkv.biases 1 2.980e-08; language_model.model.layers.1.linear_attn.in_proj_qkv.scales 556 2.980e-08; language_model.model.layers.1.linear_attn.in_proj_z.biases 39 2.887e-08; language_model.model.layers.1.linear_attn.in_proj_z.scales 103 2.980e-08; language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_127.scales 450 1.909e-08; language_model.model.layers.2.linear_attn.in_proj_qkv.scales 120 1.909e-08; language_model.model.layers.2.linear_attn.in_proj_z.scales 120 1.909e-08
- type_overrides (written F32): blk.1.ple_conv1d.weight
- hyper-connection matrices written as dequantized BF16 instead of Q4_1 (DIAG_F9c): 194 tensors
- correlation floor 0.95 for every family (replan 2026-10-15); tensors with correlation in [0.95, 0.99): 88
- [0.95, 0.99) blk.0.hc_ffn_up.weight 0.9773 (hyper_connection)
- [0.95, 0.99) blk.1.attn_gate.weight 0.9897 (other)
- [0.95, 0.99) blk.1.hc_ffn_up.weight 0.9873 (hyper_connection)
- [0.95, 0.99) blk.10.hc_attn_up.weight 0.9885 (hyper_connection)
- [0.95, 0.99) blk.10.hc_ffn_up.weight 0.9891 (hyper_connection)
- [0.95, 0.99) blk.11.hc_attn_up.weight 0.9871 (hyper_connection)
- [0.95, 0.99) blk.11.hc_ffn_up.weight 0.9798 (hyper_connection)
- [0.95, 0.99) blk.12.hc_attn_up.weight 0.9881 (hyper_connection)
- [0.95, 0.99) blk.12.hc_ffn_up.weight 0.9867 (hyper_connection)
- [0.95, 0.99) blk.13.hc_attn_up.weight 0.9883 (hyper_connection)
- [0.95, 0.99) blk.13.hc_ffn_up.weight 0.9783 (hyper_connection)
- [0.95, 0.99) blk.14.hc_attn_up.weight 0.9889 (hyper_connection)
- [0.95, 0.99) blk.14.hc_ffn_up.weight 0.9892 (hyper_connection)
- [0.95, 0.99) blk.15.hc_attn_up.weight 0.9894 (hyper_connection)
- [0.95, 0.99) blk.15.hc_ffn_up.weight 0.9856 (hyper_connection)
- [0.95, 0.99) blk.16.hc_attn_up.weight 0.9899 (hyper_connection)
- [0.95, 0.99) blk.16.hc_ffn_up.weight 0.9896 (hyper_connection)
- [0.95, 0.99) blk.17.hc_attn_up.weight 0.9898 (hyper_connection)
- [0.95, 0.99) blk.17.hc_ffn_up.weight 0.9881 (hyper_connection)
- [0.95, 0.99) blk.18.hc_attn_up.weight 0.9888 (hyper_connection)
- [0.95, 0.99) blk.18.hc_ffn_up.weight 0.9886 (hyper_connection)
- [0.95, 0.99) blk.19.hc_attn_up.weight 0.9898 (hyper_connection)
- [0.95, 0.99) blk.19.hc_ffn_up.weight 0.9863 (hyper_connection)
- [0.95, 0.99) blk.2.hc_ffn_up.weight 0.9867 (hyper_connection)
- [0.95, 0.99) blk.20.hc_attn_up.weight 0.9892 (hyper_connection)
- [0.95, 0.99) blk.21.hc_attn_up.weight 0.9895 (hyper_connection)
- [0.95, 0.99) blk.21.hc_ffn_up.weight 0.9874 (hyper_connection)
- [0.95, 0.99) blk.22.hc_attn_up.weight 0.9887 (hyper_connection)
- [0.95, 0.99) blk.22.hc_ffn_up.weight 0.9875 (hyper_connection)
- [0.95, 0.99) blk.23.hc_attn_up.weight 0.9879 (hyper_connection)
- [0.95, 0.99) blk.23.hc_ffn_up.weight 0.9861 (hyper_connection)
- [0.95, 0.99) blk.24.hc_attn_up.weight 0.9885 (hyper_connection)
- [0.95, 0.99) blk.24.hc_ffn_up.weight 0.9766 (hyper_connection)
- [0.95, 0.99) blk.25.hc_attn_up.weight 0.9663 (hyper_connection)
- [0.95, 0.99) blk.25.hc_ffn_up.weight 0.9882 (hyper_connection)
- [0.95, 0.99) blk.26.hc_ffn_up.weight 0.9864 (hyper_connection)
- [0.95, 0.99) blk.27.hc_attn_up.weight 0.9875 (hyper_connection)
- [0.95, 0.99) blk.27.hc_ffn_up.weight 0.9787 (hyper_connection)
- [0.95, 0.99) blk.28.hc_attn_up.weight 0.9890 (hyper_connection)
- [0.95, 0.99) blk.28.hc_ffn_up.weight 0.9884 (hyper_connection)
- [0.95, 0.99) blk.29.hc_attn_up.weight 0.9893 (hyper_connection)
- [0.95, 0.99) blk.29.hc_ffn_up.weight 0.9887 (hyper_connection)
- [0.95, 0.99) blk.3.hc_ffn_up.weight 0.9871 (hyper_connection)
- [0.95, 0.99) blk.30.hc_attn_up.weight 0.9879 (hyper_connection)
- [0.95, 0.99) blk.30.hc_ffn_up.weight 0.9881 (hyper_connection)
- [0.95, 0.99) blk.31.hc_ffn_up.weight 0.9859 (hyper_connection)
- [0.95, 0.99) blk.32.hc_attn_up.weight 0.9892 (hyper_connection)
- [0.95, 0.99) blk.32.hc_ffn_up.weight 0.9890 (hyper_connection)
- [0.95, 0.99) blk.33.hc_attn_up.weight 0.9892 (hyper_connection)
- [0.95, 0.99) blk.33.hc_ffn_up.weight 0.9872 (hyper_connection)
- [0.95, 0.99) blk.34.hc_attn_up.weight 0.9883 (hyper_connection)
- [0.95, 0.99) blk.34.hc_ffn_up.weight 0.9898 (hyper_connection)
- [0.95, 0.99) blk.35.hc_attn_up.weight 0.9891 (hyper_connection)
- [0.95, 0.99) blk.35.hc_ffn_up.weight 0.9864 (hyper_connection)
- [0.95, 0.99) blk.36.hc_attn_up.weight 0.9882 (hyper_connection)
- [0.95, 0.99) blk.36.hc_ffn_up.weight 0.9876 (hyper_connection)
- [0.95, 0.99) blk.37.hc_attn_up.weight 0.9898 (hyper_connection)
- [0.95, 0.99) blk.37.hc_ffn_up.weight 0.9878 (hyper_connection)
- [0.95, 0.99) blk.38.hc_attn_up.weight 0.9890 (hyper_connection)
- [0.95, 0.99) blk.38.hc_ffn_up.weight 0.9858 (hyper_connection)
- [0.95, 0.99) blk.39.hc_attn_up.weight 0.9881 (hyper_connection)
- [0.95, 0.99) blk.39.hc_ffn_up.weight 0.9871 (hyper_connection)
- [0.95, 0.99) blk.4.hc_ffn_up.weight 0.9873 (hyper_connection)
- [0.95, 0.99) blk.40.hc_attn_up.weight 0.9892 (hyper_connection)
- [0.95, 0.99) blk.40.hc_ffn_up.weight 0.9886 (hyper_connection)
- [0.95, 0.99) blk.41.hc_ffn_up.weight 0.9869 (hyper_connection)
- [0.95, 0.99) blk.42.hc_attn_up.weight 0.9899 (hyper_connection)
- [0.95, 0.99) blk.42.hc_ffn_up.weight 0.9849 (hyper_connection)
- [0.95, 0.99) blk.43.hc_ffn_up.weight 0.9876 (hyper_connection)
- [0.95, 0.99) blk.44.attn_qkv.weight 0.9896 (other)
- [0.95, 0.99) blk.44.hc_attn_up.weight 0.9867 (hyper_connection)
- [0.95, 0.99) blk.44.hc_ffn_up.weight 0.9889 (hyper_connection)
- [0.95, 0.99) blk.45.hc_ffn_up.weight 0.9890 (hyper_connection)
- [0.95, 0.99) blk.46.attn_gate.weight 0.9896 (other)
- [0.95, 0.99) blk.46.hc_attn_up.weight 0.9892 (hyper_connection)
- [0.95, 0.99) blk.46.hc_ffn_up.weight 0.9896 (hyper_connection)
- [0.95, 0.99) blk.47.hc_attn_up.weight 0.9885 (hyper_connection)
- [0.95, 0.99) blk.47.hc_ffn_up.weight 0.9883 (hyper_connection)
- [0.95, 0.99) blk.5.hc_attn_up.weight 0.9877 (hyper_connection)
- [0.95, 0.99) blk.5.hc_ffn_up.weight 0.9850 (hyper_connection)
- [0.95, 0.99) blk.6.hc_attn_up.weight 0.9872 (hyper_connection)
- [0.95, 0.99) blk.6.hc_ffn_up.weight 0.9890 (hyper_connection)
- [0.95, 0.99) blk.7.hc_attn_up.weight 0.9883 (hyper_connection)
- [0.95, 0.99) blk.7.hc_ffn_up.weight 0.9872 (hyper_connection)
- [0.95, 0.99) blk.8.hc_attn_up.weight 0.9882 (hyper_connection)
- [0.95, 0.99) blk.8.hc_ffn_up.weight 0.9857 (hyper_connection)
- [0.95, 0.99) blk.9.hc_attn_up.weight 0.9690 (hyper_connection)
- [0.95, 0.99) blk.9.hc_ffn_up.weight 0.9878 (hyper_connection)
- prefill_device=cpu (--no-op-offload: the CPU Q4_1xQ8 dot and the CUDA MMQ path differ by op order only)
- llama.cpp prompt processing 23.89 tok/s; memcg peak 47107 MiB; MemoryMax 47616 MiB (table 47616 MiB, MemFree at start 91747 MiB); memfree_min 41339 MiB; memfree_stop=false
- token ids sha256 e65198947ecb06833dcd69e042e406f52945dcc88073621d52c3e8e54a4d8e55, wikitext rows sha256 5c8877eeabb3b8afc09f8ec35194fdac8c55acaa25de738d27000876d7c73e4c

Verdict: PASS — engine and exact-Q4_1 llama.cpp agree within 2% (-0.05%) on 32 windows; sweep exact except the named exceptions; mapping floor 0.95, 88 tensors in [0.95, 0.99) listed; prefill_device=cpu; no MemFree soft stop
