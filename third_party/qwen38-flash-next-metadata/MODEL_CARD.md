---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- mlx
- apple-silicon
- qwen4_exp
- mixture-of-experts
- 4-bit
pipeline_tag: text-generation
library_name: mlx
---

# Qwen3.8-Flash-Next-MLX-4bit

MLX (Apple Silicon) build of [**Qwen3.8-Flash-Next**](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) —
125B-A6B hybrid Gated-DeltaNet / sparse-attention MoE with a 51B-parameter hashed n-gram
embedding — quantized to **4-bit**.

**These files are modified**: the weights are converted to MLX and quantized; the architecture is
unchanged. The 4B multi-token-prediction head is not included. The vision tower is carried in
bfloat16 but the runtime below is text-only.

## Runtime

`qwen4_exp` is in **no released mlx-lm** (https://github.com/ml-explore/mlx-lm/pull/1788 is open and unmerged), so this repository ships its
own `qwen4_exp.py` and declares it via `model_file`:

```bash
pip install -U mlx-lm
mlx_lm.generate --model pipenetwork/Qwen3.8-Flash-Next-MLX-4bit --trust-remote-code \
  --prompt "Write a Python function that merges overlapping intervals." --max-tokens 300
```
```python
from mlx_lm import load, generate
model, tokenizer = load("pipenetwork/Qwen3.8-Flash-Next-MLX-4bit", trust_remote_code=True)
```

The bundled runtime is the open PR **with three numerical fixes** found while validating it against
`transformers` 5.16 (details and tests in [https://github.com/PipeNetwork/qwen38-flash-next-mlx](https://github.com/PipeNetwork/qwen38-flash-next-mlx)):

| what | reference | the PR as submitted | effect |
|---|---|---|---|
| RMSNorm variant | `x/rms · (1 + w)`, zero-initialised | `x/rms · w` | half the channels of every residual read sign-flipped |
| n-gram hash seed | `1234` (transformers default; not in config.json) | `0` | every bigram/trigram looks up an unrelated row |
| sparse-attention prefill | per-query blocks, own partial block visible, causal | global blocks; leaks future tokens, drops the query's own | wrong hidden states for prompts > 2048 tokens |

This checkpoint follows the mlx-lm convention for the norm fix: the `+1` is folded into the stored
norm weights at conversion, and the runtime multiplies by `w`. Tiny-config parity against
`transformers` is **1e-7** on the dense path, exact on cached decode and chunked prefill.

## Size and what is quantized

**103.8 GB** on disk (bfloat16 upstream: 360.0 GB).

| group | share of parameters | this build |
|---|---:|---|
| routed experts (`switch_mlp`) | 120.8B (96.6% of the 125B) | 4-bit, group 64 |
| n-gram embedding tables (128 shards × [2,500,012 × 160]) | 51.2B (separate) | 4-bit, group 32 |
| attention, DeltaNet, hyper-connections, shared experts, embeddings, `lm_head` | ~4.2B | 4-bit, group 64 |
| MoE router, `shared_expert_gate`, residual write gates, DeltaNet `in_proj_a/b`, indexer projection | 36M | bfloat16 |
| vision tower | 0.4B | bfloat16 (unused by this runtime) |

The n-gram tables need group size 32 because their row width (160) is not a multiple of 64; left
in bfloat16 they alone would be 102 GB.

## Quality

Perplexity on wikitext-2 (test), 296,815 tokens in 145 windows of 2048, every build scored
on **identical** windows through this runtime. Perplexity varies far more between windows than
between quants, so the comparison that means anything is paired: per-window NLL differences
against bfloat16, bootstrapped over one shared index set (20,000 resamples).

| build | size | perplexity | ΔNLL/token vs bf16 [95% CI] | windows worse |
|---|---:|---:|---|---:|
| bfloat16 (upstream) | 360.0 GB | 4.4708 | — | — |
| [8bit](https://huggingface.co/pipenetwork/Qwen3.8-Flash-Next-MLX-8bit) | 192.2 GB | 4.4749 | +0.0009 [−0.0003, +0.0021] | 73/145 |
| [6bit](https://huggingface.co/pipenetwork/Qwen3.8-Flash-Next-MLX-6bit) | 148.0 GB | 4.4767 | +0.0013 [−0.0003, +0.0029] | 81/145 |
| [mixed-4_8bit](https://huggingface.co/pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit) | 106.2 GB | 4.5286 | +0.0128 [+0.0109, +0.0148] | 128/145 |
| [4bit](https://huggingface.co/pipenetwork/Qwen3.8-Flash-Next-MLX-4bit) | 103.8 GB | 5.3914 | +0.1872 [+0.1778, +0.1968] | 145/145 |

Read the interval, not the point estimate: an interval that straddles zero is a build that is
statistically indistinguishable from bfloat16 on this corpus; "windows worse" counts how many of
the 145 windows the build lost outright.

**The mixed 4/8-bit build is the one to use at this size.** Routed experts are 96.6% of the parameters, yet quantizing everything to 4-bit costs +20.6% while keeping only the ~3% of non-expert weights (attention, DeltaNet, hyper-connections, shared experts, embeddings) at 8-bit brings that to +1.3% for 2.4 GB more — the uniform 4-bit build is dominated and is published for completeness. 6-bit and 8-bit are statistically indistinguishable from bfloat16 on this corpus.

### Where the 4-bit damage comes from

One group at a time moved back to 8-bit from the uniform 4-bit build, same windows, same runtime (the ablation builds are not published):

| everything 4-bit except… | perplexity | vs bfloat16 |
|---|---:|---:|
| — (uniform 4-bit) | 5.3914 | +20.6% |
| hyper-connection read gates (`input_mix_weight_down/up`, 0.6B) at 8-bit | 4.9744 | +11.3% |
| attention, DeltaNet, shared experts, PLE projections (~2.3B) at 8-bit | 4.8969 | +9.5% |
| `embed_tokens` and `lm_head` (1.3B) at 8-bit | 5.2843 | +18.2% |
| all three at 8-bit (= mixed-4_8bit) | 4.5286 | +1.3% |

No single group is responsible: the hyper-connection gates and the attention/DeltaNet projections each carry about half of the loss and the effects are roughly additive, so every non-expert weight is worth its 8 bits. Per parameter, these ~4B weights are roughly 20x more quantization-sensitive than the 121B of routed experts.

Greedy generation (a collapse detector, not a ranking) is coherent on every published build.

## License

[Qwen Community License 1.0](LICENSE), as the upstream model. Port code: [https://github.com/PipeNetwork/qwen38-flash-next-mlx](https://github.com/PipeNetwork/qwen38-flash-next-mlx).
