# The exact Q4_1 reference

To check the engine's quality against a second implementation, the project builds a GGUF Q4_1 file from the same MLX 4-bit weights and scores it with llama.cpp. This page explains why that file can be exact, where it is not, what the check found, and how to run it again. The numbers are in [f9-reference.md](../db/records/measurements/f9-reference.md) and [f9-quality.md](../db/records/measurements/f9-quality.md).

## Why an exact conversion exists

An MLX affine 4-bit group is `w = q * scale + bias` with `q` in 0 to 15, in groups of 64 values (32 for the n-gram table). A ggml Q4_1 block is `y = q * d + m` with the same 4-bit `q`, in blocks of 32 values. So one MLX group of 64 becomes two Q4_1 blocks that share `d = scale` and `m = bias`. The codes are the same; only their order in the bytes changes (a permutation of the same 4-bit values). Nothing is requantized.

The one conversion that could lose information is the scale and bias type. MLX stores them as bf16, Q4_1 as fp16.

## bf16 to fp16 is exact on a range

bf16 keeps 7 stored mantissa bits and an 8-bit exponent; fp16 keeps 10 mantissa bits and a 5-bit exponent. Every bf16 value whose magnitude is in [2^-17, 65504], and zero, is representable in fp16, so the conversion changes nothing there. Outside that range the fp16 value is subnormal or overflows. `repack_q4_1.py` treats any such value as a hard error that names the tensor (`ExactnessError`), unless the tensor is one of the named exceptions below.

`tools/repack_q4_1.py --sweep` runs this check over the whole checkpoint without writing anything. On the recorded run it covered 3,215 tensors and found 7 that are not exact, listed in `slotstream_spark/config/q4_1_exceptions.json` ([f9-reference.md](../db/records/measurements/f9-reference.md)). The file is the contract: the job stops if the sweep finds a different set.

## The seven exceptions

Scale and bias tensors for which some values are below 2^-17. They are rounded to fp16 with an absolute error of at most 2^-24 (about 5.96e-8) per value. The largest error measured was 2.980e-08.

| Tensor | Values rounded | Largest error |
|---|---|---|
| `layers.1.linear_attn.in_proj_qkv.biases` | 1 | 2.980e-08 |
| `layers.1.linear_attn.in_proj_qkv.scales` | 556 | 2.980e-08 |
| `layers.1.linear_attn.in_proj_z.biases` | 39 | 2.887e-08 |
| `layers.1.linear_attn.in_proj_z.scales` | 103 | 2.980e-08 |
| `layers.1.ple.ple_embedding.ngram_embedding.shard_127.scales` | 450 | 1.909e-08 |
| `layers.2.linear_attn.in_proj_qkv.scales` | 120 | 1.909e-08 |
| `layers.2.linear_attn.in_proj_z.scales` | 120 | 1.909e-08 |

(Tensor names are shortened here; the full names start with `language_model.model.`.) This is why the claim reads "exact except the 7 named exceptions" and not "exact".

## Two tensors that are not Q4_1

- `blk.1.ple_conv1d.weight` is written as F32. It is the 1-D convolution of the n-gram layer; the template stores it as F32 (`type_overrides` in the manifest).
- 194 hyper-connection matrices (`hc_attn_down` and `_up`, `hc_ffn_down` and `_up`, `output_hc_down` and `_up`) are written as BF16, the MLX 4-bit values dequantized in fp32 and rounded to nearest even. Written as Q4_1 they made llama.cpp worse: llama.cpp quantizes the activations of a Q4_1 matrix product to 8 bits, and these gates are sensitive to it. The docstring of `tools/repack_q4_1.py` gives the cost of the diagnostic run (`bench/campaign/f9c_overlay_job.sh`) as 0.15 nats per token with Q4_1 hyper-connection tensors against 0.04 as BF16. That diagnostic run has no record in `db/records/`, so treat those two figures as a note from the code, not as a measurement of this page. The engine reads the same dequantized values, so the comparison is of equal weights.

## Mapping check

The reference file copies its tensor names, shapes and order from a template GGUF (a public Q4_1 build of the same model). Its values are never used. Every template tensor needs a rule that maps the MLX tensors onto it, with the same transformations the llama.cpp converter applies: reordering of the value heads of the gated delta net, `A_log` turned into `-exp(A_log)`, 1-D convolution kernels squeezed, the indexer projection split into q and k, the 128 n-gram shards concatenated.

A wrong mapping would still produce a file that loads. To catch it, `repack_q4_1.py` compares sampled rows of every tensor with the template's dequantized values: the Pearson correlation must be at least 0.95, and for tensors that are not 4-bit the mean offset must stay within 0.05 standard deviations (which is what a missing or doubled `+1` on a norm looks like). The recorded run checked 1,224 tensors, none under the floor of 0.95. 88 of them have a correlation between 0.95 and 0.99; the full list is in [f9-reference.md](../db/records/measurements/f9-reference.md), and 85 are hyper-connection matrices and 3 are other tensors. The lowest is `blk.25.hc_attn_up.weight` at 0.9663. The template's values are a different quantization of the original weights, so a correlation under 1 is expected; the floor exists to catch mapping errors, not to prove equality.

## What the check found

The criterion was fixed before the run: the perplexity of the engine and of llama.cpp on the first 32 wikitext windows (same token ids, same starts) may differ by at most 2 %. Prefill ran on the CPU in llama.cpp (`--no-op-offload`), so that its Q4_1 times Q8 dot product and the CUDA kernel differ only in the order of operations.

| Quantity | Value |
|---|---|
| PPL, engine | 4.7216 |
| PPL, llama.cpp on the exact repack | 4.7239 |
| PPL, transformers oracle ([f9-quality.md](../db/records/measurements/f9-quality.md)) | 4.7265 |
| Difference engine to llama.cpp | 0.05 % (limit 2 %) |

The combined statement is `engine 4.7216, llama.cpp 4.7239`, on 32 windows of wikitext-2 at 2,048 tokens each. Per window the two disagree more: the largest difference in negative log likelihood is +11.13 % at window 15, and the engine is above llama.cpp on 14 of the 32 windows. The close total is a sum of differences with both signs, not agreement window by window. One text, one tokenizer protocol and 32 windows is a small sample.

A separate figure, 5.7338 over 145 windows, is the engine's wikitext-2 perplexity against a reference of 5.3914 from the model card. It is 6.35 % higher. The card does not publish its text or window protocol, so the two are not comparable and the gate treats the difference as information only ([f9-quality.md](../db/records/measurements/f9-quality.md)).

## How to rerun it

On the Spark, with the checkpoint and the packed artifacts in place ([install-spark](install-spark.md)) and a finished engine capture (`f9-engine`) to take the token ids from:

1. Build the llama.cpp reference setup. `tools/spark/verify_artifacts.sh` expects llama.cpp at commit `23b0202a189c44a54625aadcb37a946dd1d6278d` under `$R/lb/llama.cpp` and the template GGUF under `$R/lb/gguf`. `tools/spark/dl_gguf.sh` downloads the GGUF files.
2. Start the job through the guard, with a footprint that leaves room for the CPU prefill:

```
tools/spark/jobs.sh start f9-ref --heavy f9-ref 57344 12600 --memcg slotstream-spark-f9-ref -- bash bench/campaign/f9_reference_job.sh
tools/spark/jobs.sh wait f9-ref --timeout 12600
```

3. The job runs in phases and stops on any failed one, writing the reason to `stop.txt` in its capture: preflight (150 GB free, finish by 23:45 unless pause mode says otherwise, wikitext and the id helper present), the exactness sweep, the mapping check, the build of `$R/model/q4_1-exact/model.gguf`, a smoke prompt through llama-server, and the 32-window perplexity.
4. `F9_REF_SMOKE=1` runs only the tooling check on the template GGUF with one window. That run is not a result.
5. Judge the capture with `tools/decide.py f9-diff`, which applies the 2 % rule to `result.json`.

To check only the conversion, without llama.cpp: `tools/spark/py.sh tools/repack_q4_1.py --model $R/model/qwen38-flash-next-mlx-4bit --sweep`. It reads the checkpoint, writes nothing and should report 3,215 tensors, the 7 named exceptions and nothing else. `tests/t0/test_repack_q4_1.py` covers the conversion on small synthetic inputs and, where `third_party/qwen38-flash-next-metadata/tensors.csv` is present, the tensor inventory.
