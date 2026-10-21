# Qwen3.8-Flash-Next metadata

Small files copied from the Hugging Face repository of the MLX affine 4-bit build of Qwen3.8-Flash-Next. The weights are not here. The engine needs these files to validate the checkpoint, render the chat template and check tensor shapes on a machine that has no weights.

Source and revision: see `REVISION` (a Hugging Face repository and commit; downloaded 2026-10-10).

| File | What it is |
|---|---|
| `config.json` | Model configuration; `slotstream_spark/model/config.py` validates it. |
| `generation_config.json` | Default sampling settings of the checkpoint. |
| `chat_template.jinja` | Chat template; its SHA-256 is pinned in `model.lock.json` and `server/template.py` renders it. |
| `tokenizer_config.json` | Tokenizer settings. |
| `preprocessor_config.json`, `video_preprocessor_config.json` | Image and video preprocessing settings, kept for completeness. |
| `model.safetensors.index.json` | Shard index of the checkpoint. |
| `tensors.csv` | One row per tensor (name, shard, dtype, shape, byte range, category, quantization), read from the safetensors headers. It was `tensores.csv` in earlier revisions of this repository. |
| `qwen4_exp.py` | The MLX port of the model code that ships with the checkpoint. |
| `MODEL_CARD.md` | The model card of the source repository (it was `README.md` there); `numeric_contract.toml` cites its line 83. |
| `README.md` | This file. |
| `LICENSE` | Qwen Community License 1.0. |

License: every file here, including `qwen4_exp.py`, is under the Qwen Community License 1.0 (see `LICENSE`). The Model as a Service clause in that license applies to anyone who serves the weights to third parties. The reference sources from transformers are in `../transformers-5.19.0-qwen4_exp/` under Apache-2.0.
