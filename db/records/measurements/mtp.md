# mtp: MTP head installed with the pinned SHA
Date: 2026-10-11
Capture: db/sources/runs/2026/10/20261011T183501-verify-artifacts/
Command: tools/spark/run.sh --heavy verify-artifacts 1024 900 --nolock -- tools/capture.sh verify-artifacts -- bash tools/spark/verify_artifacts.sh (installed by the mtp job, tools/spark/mtp_build.sh)

| item | value |
|---|---|
| installed mtp.safetensors sha256 | c80b58faae46eeacb94dea49dd3453566ee05597fbd28c7c647eccb2862ab744 |
| pinned sha256 (model.lock.json) | c80b58faae46eeacb94dea49dd3453566ee05597fbd28c7c647eccb2862ab744 |
| copy installed | pinned mirror copy |
| locally converted file sha256 | df1a0e94c0411ad561bc26dbd1337f1385c2a1a0d070fd366f674fa679acd43f (1,470,955,241 B; mirror copy 1,470,955,171 B) |
| converter | slotstream mtp_convert.py with MLX 0.32.3, source Qwen/Qwen3.8-Flash-Next revision de4b8e4d43b917e7706784d8bb445c9af86a3540 |
| mtp-compare.json | names_equal true, dtypes_shapes_equal true, 68/68 tensors identical, max abs dequant diff 0.0 |

The converted file differs from the mirror copy byte-wise (provenance metadata and file size) but holds the same 68 tensors with identical values.

Verdict: PASS — installed file has the pinned SHA c80b58fa
