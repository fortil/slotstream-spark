# Sources

Where the design, the reference code and the model data come from, and how to get them again. Nothing here is needed to run the engine; it is needed to repeat a comparison or to read a record that cites a reference file.

## slotstream

The engine is designed after [slotstream](https://github.com/carloslfu/slotstream) by Carlos Galarza (MIT), which runs Qwen3.8-Flash-Next on a Mac with the experts read from the SSD. This repository ports its server, dialects, tool-call handling, governor and launcher logic and cites it by file and line. The pinned commit is `8bd579385152d547f96c9f6285181fe7885dfe29`.

It is a read-only git submodule at `third_party/slotstream/`. The full repository carries about 1.2 GB of raw runs under `db/sources/` that this project does not use, so clone it with a partial clone and a sparse checkout:

```
git submodule update --init --depth 1 --filter=blob:limit=3m third_party/slotstream
git -C third_party/slotstream sparse-checkout set --no-cone '/*' '!/db/sources/'
```

To recreate it outside the repository at the pinned commit:

```
git clone --depth 1 --filter=blob:limit=3m --no-checkout \
  https://github.com/carloslfu/slotstream.git reference/slotstream
git -C reference/slotstream sparse-checkout set --no-cone '/*' '!/db/sources/'
git -C reference/slotstream fetch --depth 1 origin 8bd579385152d547f96c9f6285181fe7885dfe29
git -C reference/slotstream checkout --detach 8bd579385152d547f96c9f6285181fe7885dfe29
```

## qwen-flash-next-8gb

The engine of the `helgard-orlm` project that already reads this model's experts from NVMe on an 8 GB GPU (Apache-2.0). This project ported 8 pieces and the files carry a "Portions ported" header. Commit `958d22e7ba0317750ccc38862b9667b1034a4dd1`. It is a reference clone with its own `.git`, kept outside this repository; recreate it with:

```
git clone https://github.com/helgard-orlm/qwen-flash-next-8gb reference/qwen-flash-next-8gb
git -C reference/qwen-flash-next-8gb checkout --detach 958d22e7ba0317750ccc38862b9667b1034a4dd1
```

Tests that compare against it run only when `SLOTSTREAM_HELGARD_CHECKOUT` points at a clone; otherwise they skip.

## Model metadata

`third_party/qwen38-flash-next-metadata/` holds the files of the MLX affine 4-bit checkpoint that are not weights: `config.json`, the tokenizer, the chat template, the tensor list (`tensors.csv`), the Hugging Face model card (`MODEL_CARD.md`) and a `README.md` that lists them and the revision. The weights themselves are never in git. The files are under the Qwen Community License 1.0.

`third_party/transformers-5.19.0-qwen4_exp/` holds the reference model code of transformers v5.19.0 (Apache-2.0, with a `NOTICE`) that the oracle runs, plus the MLX port under the Qwen license. Licenses of everything vendored are listed in [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md).

## Captures

Raw run output lives in `db/sources/runs/YYYY/MM/<capture>/` and is mostly ignored by git. The captures that support current claims and gate inputs are tracked after scrubbing: host names, logins and addresses are replaced by `tools/evidence_scrub.py`, which rewrites text files only and never the files whose hashes the records check. Captures that are not published are listed in `db/sources/ARCHIVED.txt`, and `tools/db_check.py` accepts a record that points at one of them. How a record points at a capture is explained in [measurements](measurements.md).
