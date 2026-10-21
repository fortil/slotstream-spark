# Third-party notices

## slotstream
MIT, Copyright (c) 2026 Carlos Galarza. Submodule `third_party/slotstream` pinned at
8bd579385152d547f96c9f6285181fe7885dfe29. The Swift sources are used as a specification;
the Python tools are used as they are.
`tests/fixtures/mtp/mtp_parity.safetensors` and `mtp_parity_inputs.safetensors` (about 385 KB) are copied unchanged
from `third_party/slotstream/Tools/reference/fixtures/` (MIT, same copyright); `SHA256SUMS` there records their hashes.

## helgard-orlm/qwen-flash-next-8gb
Apache-2.0, commit 958d22e7ba0317750ccc38862b9667b1034a4dd1. The upstream project has no NOTICE file.
Ported pieces are listed in `third_party/vendor/helgard-958d22e7/PORTED.md`; the license is in
`third_party/vendor/helgard-958d22e7/LICENSE`.

## vLLM
Apache-2.0, commit 187a0eb98aa42341d703f83421d693fa7585581b. The Qwen4Exp QSA, indexer, prepare, hyper-connection
and QSA cache files are vendored in `third_party/vendor/vllm-187a0eb9/qwen4_exp_ops/`, modified by slotstream-spark
(imports redirected to `_shims.py`, PDL off, the CUDA top-k ops replaced by `kernels.topk_stable`). The license is
`third_party/vendor/vllm-187a0eb9/LICENSE`, the notice `NOTICE`, the exact differences `expected.diff` (checked by `diff.sh`).

The grouped affine MoE kernel `slotstream_spark/kernels/fused_moe_affine.py` is derived from vLLM's
`fused_moe_kernel_gptq_awq` (vLLM v0.28.0 image copy, not a vendored tree); origin, hash and modifications are in
`third_party/vendor/vllm-187a0eb9/fused_moe/ORIGIN.md`.

## flash-linear-attention
MIT, version 0.5.2 (the operators come from the `fla-core` 0.5.2 sdist). The gated-delta-rule closure is vendored in
`third_party/vendor/fla-0.5.2/fla_v052/`, modified by slotstream-spark (package renamed to `fla_v052`, three package
`__init__.py` files replaced by stubs). The license is `third_party/vendor/fla-0.5.2/LICENSE`, the notice `NOTICE`, the
exact differences `expected.diff` (checked by `diff.sh`).

## slotstream expert-lookahead corpus (F6 cohorts)
MIT. `bench/cohorts/b0.jsonl` and `b1.jsonl` hold the prompts of the test and validation splits of the corpus in
`third_party/slotstream/Tools/fixtures/expert-lookahead/` (code and text excerpts of the files that repository cites,
materialized by its `Tools/expert_lookahead_corpus.py`, every prompt checked against its `prompt_sha256`). The license is
`third_party/slotstream/LICENSE`.

## nanobind
BSD-3-Clause. Used by the native `slotio` module.

## doctest
MIT, version 2.4.11. Used by the native tests. The single header is kept unchanged in
`slotio/tests/doctest.h` (fetched from the v2.4.11 tag); `slotio/tests/doctest.h.sha256` records its hash.

## llama.cpp
MIT, commit 23b0202a189c44a54625aadcb37a946dd1d6278d. Not vendored; it is built on the Spark.

## Hugging Face transformers 5.19.0 (Qwen4Exp reference sources)
Apache-2.0, tag v5.19.0. `transformers-v5.19.0-configuration_qwen4_exp.py` and `transformers-v5.19.0-modeling_qwen4_exp.py`
are kept unchanged in `third_party/transformers-5.19.0-qwen4_exp/` as the numerical oracle. The license is
`third_party/transformers-5.19.0-qwen4_exp/LICENSE`, the notice `NOTICE`.

## Qwen3.8-Flash-Next weights and metadata
Qwen Community License 1.0. This repository has no weights. The model metadata (config, tokenizer files, chat template,
`qwen4_exp.py`) is in `third_party/qwen38-flash-next-metadata/`, with the license in `LICENSE` there. Section 2 of that
license (Model as a Service) binds whoever hosts the model and gives third parties access to inference, not this
repository. Derived artifacts (`experts.bin` and the other packed files) carry the same copyright notice and are never
committed.

## slotstream ports
Files below are ports of slotstream sources (MIT, Copyright (c) 2026 Carlos Galarza, commit
8bd579385152d547f96c9f6285181fe7885dfe29). Each carries `SPDX-License-Identifier: MIT` and a line naming its source.

- `slotstream_spark/engine/lookahead.py`
- `slotstream_spark/engine/mtp.py`
- `slotstream_spark/engine/persistent_prefix.py`
- `slotstream_spark/engine/prefill_schedule.py`
- `slotstream_spark/engine/prefix_cache.py`
- `slotstream_spark/engine/pressure.py`
- `slotstream_spark/engine/rollback.py`
- `slotstream_spark/engine/sampler.py`
- `slotstream_spark/engine/sweep.py`
- `slotstream_spark/kernels/forecast.py`
- `slotstream_spark/kernels/topk_stable.py`
- `slotstream_spark/memory/governor.py`
- `slotstream_spark/memory/ledger.py`
- `slotstream_spark/server/app.py`
- `slotstream_spark/server/bounded_output.py`
- `slotstream_spark/server/common.py`
- `slotstream_spark/server/dialects/anthropic.py`
- `slotstream_spark/server/dialects/ollama.py`
- `slotstream_spark/server/dialects/openai_chat.py`
- `slotstream_spark/server/dialects/responses.py`
- `slotstream_spark/server/http_limits.py`
- `slotstream_spark/server/output.py`
- `slotstream_spark/server/request_control.py`
- `slotstream_spark/server/think.py`
- `slotstream_spark/server/tool_calls.py`
- `tools/api_robustness.sh`
- `tools/claims_gate.py`
- `tools/mac/launch.sh`
- `tools/quality/quality_probe.sh`

Files that also carry helgard code (`SPDX-License-Identifier: MIT AND Apache-2.0`) are listed in
`third_party/vendor/helgard-958d22e7/PORTED.md`. `slotstream_spark/kernels/fused_moe_affine.py` is Apache-2.0 (vLLM).
