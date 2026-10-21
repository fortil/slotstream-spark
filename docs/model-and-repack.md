# The model and the repack

This page describes the checkpoint the engine serves and the files `tools/spark/repack_all.sh` builds from it. The commands to build them are in [install-spark](install-spark.md). The engine's use of these files at run time is in [architecture](architecture.md).

## The checkpoint

The engine serves one checkpoint: `pipenetwork/Qwen3.8-Flash-Next-MLX-4bit`, an MLX affine 4-bit build of Qwen3.8-Flash-Next, at revision `aa7c790e804bbf9d491ddb109c3d61bc4a555f7c`. It is 24 files and 103,793,508,077 B; 11 of them are `.safetensors` shards totaling 103,770,199,081 B ([checkpoint.md](../db/records/measurements/checkpoint.md), `model.lock.json`). `SHA256SUMS.model` holds the hash of every file. The model is released under the Qwen Community License 1.0; the copy that travels with the small metadata files is `third_party/qwen38-flash-next-metadata/LICENSE`, and that license has a clause for anyone who serves the weights to third parties, so read it before you do.

The weights are not in this repository. `third_party/qwen38-flash-next-metadata/` holds only the files a machine without weights needs: `config.json`, the tokenizer files, the chat template, the safetensors index, `tensors.csv` (one row per tensor, 3,215 rows) and the model card (`MODEL_CARD.md`). `third_party/transformers-5.19.0-qwen4_exp/` holds the reference model code from transformers 5.19.0, used by the oracle tests.

### Shape

Values below are read from `third_party/qwen38-flash-next-metadata/config.json`, `text_config` section.

| Property | Value | Line |
|---|---|---|
| Layers | 48 (`num_hidden_layers`) | 103 |
| Layer kinds | 36 `linear_attention` (gated delta net) and 12 `full_attention`, every fourth layer (`full_attention_interval` 4) | 14, 27-76 |
| Hidden size | 2,560 | 20 |
| Routed experts | 512 per layer, 10 chosen per token | 101-102 |
| Expert width | `moe_intermediate_size` 640; one shared expert of width 640 | 86, 127 |
| Attention | 24 heads of 256, 2 key-value heads | 17, 100, 104 |
| Gated delta net | 16 key heads and 48 value heads of 128, conv kernel 4 | 77-81 |
| Vocabulary | 248,320 | 131 |
| Context | 262,144 positions | 84 |
| MTP | one extra layer (`mtp_num_hidden_layers` 1) that proposes the next token | 87-96 |
| N-gram table | `ngram_size` 3, 128 shards (`split_ngram_parts`), added at layer index 1 | 98-99, 111-113, 128 |

The vision tower is part of the checkpoint (`vision_config`, depth 27) and the repack drops it: this engine serves text only. The runtime reads the same numbers through `slotstream_spark/model/config.py`.

The size comes from the experts. 24,576 expert records (48 layers x 512) are 67.9 GB. They are read from the NVMe drive on demand. The rest of the dense weights, 2.9 GB, stays in memory.

## Why a repack

The checkpoint stores each expert as nine separate tensors spread across shards, and the n-gram table as 128 shards of three tensors each. Reading one expert from it would take nine reads. The repack puts everything the engine reads per token in the shape it reads it: one record per expert, one row per n-gram id, one file for the dense weights. The MLX 4-bit values are copied byte for byte; nothing is requantized. `tools/verify_r0.py` proves that by comparing the packed bytes with the checkpoint ([r0.md](../db/records/measurements/r0.md): 0 mismatches).

## Expert records

`tools/repack_experts.py` writes `experts.bin`. The record for layer L and expert e starts at `(L * 512 + e) * 2,764,800` bytes. A record holds the nine pieces of one expert in this order:

| Piece | Size, B | Offset in the record |
|---|---|---|
| `gate_proj.weight` | 819,200 | 0 |
| `gate_proj.scales` | 51,200 | 819,200 |
| `gate_proj.biases` | 51,200 | 870,400 |
| `up_proj.weight` | 819,200 | 921,600 |
| `up_proj.scales` | 51,200 | 1,740,800 |
| `up_proj.biases` | 51,200 | 1,792,000 |
| `down_proj.weight` | 819,200 | 1,843,200 |
| `down_proj.scales` | 51,200 | 2,662,400 |
| `down_proj.biases` | 51,200 | 2,713,600 |

The sizes follow from the shape: a weight piece is 640 x 2,560 values at 4 bits, 819,200 B; scales and biases are bf16, one per group of 64 values, 640 x 40 x 2 = 51,200 B each (the down projection is 2,560 x 10 groups and has the same sizes). Three times (819,200 + 51,200 + 51,200) is 2,764,800 B. The offsets are computed from those sizes here; `experts.manifest.json` carries the authoritative `piece_offsets`, and the loader reads them from it.

The layout name in the manifest is `slotstream-pool-layer-expert-piece-v1`. The manifest also holds the checkpoint identity (the sha256 of the safetensors index), the record size, the alignment, a sha256 per layer and the source repo and revision. The MTP layer has its own file, `mtp_experts.bin`, with 512 records of the same size (1,415,577,600 B). The point of the format is one `pread` per expert instead of nine: on the recorded run, 1,000 random experts, a packed read took a median of 1,685.4 us against 3,872.5 us for the nine unpacked reads ([e1.md](../db/records/measurements/e1.md)).

The write is atomic and resumable. Data goes to `experts.bin.tmp`; a layer is added to a progress file only after it is written and synced; the temporary file is renamed only when every layer has a hash.

## N-gram rows

The n-gram table has 320,001,536 rows (128 shards x 2,500,012 rows) of 160 values at 4 bits with groups of 32 (`tensors.csv`, layer 1). `tools/repack_ngram.py` writes `ngram.bin` with one fixed-size row per id:

| Field | Bytes | Offset |
|---|---|---|
| weight (20 words of 32 bits) | 80 | 0 |
| scales (5 bf16) | 10 | 80 |
| biases (5 bf16) | 10 | 90 |
| padding up to `row_bytes` | 0 for 100 | 100 |

`row_bytes` is 100. A 128-byte row would be aligned but would make the file 40,960,196,608 B instead of 32,000,153,600 B. The choice was fixed before measuring: 128 would have been chosen if `0.1875 x (ratio - 1) > 0.05`, where ratio is the median latency of a 1,024 B read over a 512 B read at queue depth 1 with `O_DIRECT`, and 18.75 % is the share of 100-byte rows that straddle a 512-byte sector. The measured ratio was 1.0005, so 100 won ([ngram-format.md](../db/records/decisions/ngram-format.md), [io2.md](../db/records/measurements/io2.md)). Rows are read with `O_DIRECT`.

## Trunk

`tools/extract_trunk.py` copies every dense tensor into one safetensors file, `trunk.safetensors`: all `language_model.*` tensors except the routed experts and the n-gram shards, with `vision_tower.*` dropped. That is 2,066 tensors and 2,924,300,312 B. `mtp_trunk.safetensors` does the same for the `mtp.*` tensors of `mtp.safetensors`, except the MTP layer's experts: 59 tensors, 55,377,408 B.

The layout is: an 8-byte header length, a JSON header padded with spaces so that the data starts on a 4,096-byte boundary, then `embed_tokens.weight`, `embed_tokens.scales` and `embed_tokens.biases`, then every other tensor sorted by name, contiguous. Bytes are copied verbatim; no norm is changed. The safetensors metadata carries the checkpoint identity (`source_index_sha256`), which is how the script knows a finished trunk is current.

## The MTP file

The checkpoint does not include the MTP layer. `mtp.safetensors` (1,470,955,171 B, sha256 `c80b58faae46eeacb94dea49dd3453566ee05597fbd28c7c647eccb2862ab744`) is produced by converting the MTP tensors of `Qwen/Qwen3.8-Flash-Next` with the converter in `third_party/slotstream/Tools/mtp_convert.py`, or taken from a pinned mirror repository when conversion is not possible. `tools/spark/mtp_build.sh` accepts only a file with that hash. `model.lock.json` holds the hash, the source revisions and the mirror revision.

## Manifests and identity

Every packed file has a manifest or a header that names the checkpoint it came from. `repack_all.sh` skips a stage only when the existing output and its identity match the current checkpoint:

| Output | Identity check |
|---|---|
| `experts.bin` | `experts.manifest.json` `model_identity` equals the sha256 of `model.safetensors.index.json` |
| `mtp_experts.bin` | `mtp_experts.manifest.json` `model_identity` equals the hash of the `mtp.safetensors` header |
| `trunk.safetensors` | metadata `source_index_sha256` equals the index hash |
| `mtp_trunk.safetensors` | the same, against the MTP header hash |
| `ngram.bin` | `ngram.manifest.json` `row_bytes` equals the value in `platform.toml` and its source index hash matches |

If you change the checkpoint, every stage reruns. `tools/verify_r0.py` is the check after the build (it compares the bytes with the checkpoint), and `cli doctor` confirms that the packed files and manifests exist.

## What is not repacked

The vision tower, the tokenizer files and the chat template stay where they are. The tokenizer and the template are read from the checkpoint directory (`slotstream_spark/server/template.py`), and from `third_party/qwen38-flash-next-metadata/` on a machine without weights. The exact repack for the llama.cpp comparison is a different artifact; it is described in [exact-q4_1-reference](exact-q4_1-reference.md).
