# checkpoint: pinned MLX 4-bit checkpoint downloaded and verified
Date: 2026-10-11
Capture: db/sources/runs/2026/10/20261011T183501-verify-artifacts/
Command: tools/spark/run.sh --heavy verify-artifacts 1024 900 --nolock -- tools/capture.sh verify-artifacts -- bash tools/spark/verify_artifacts.sh (downloaded by the dl-ckpt job, tools/spark/dl_checkpoint.sh, revision aa7c790e804bbf9d491ddb109c3d61bc4a555f7c)

| item | value |
|---|---|
| files verified (sha256sum -c against SHA256SUMS.model) | 24/24 OK, 0 FAILED |
| total size of the 24 files | 103,793,508,077 B |
| hfcache-check files (nothing written to the shared Hugging Face cache) | 3 checked, 0 non-empty |
| first verification attempts | reported 0/24 because the OK count was locale dependent; fixed with LC_ALL=C and the job rerun |

Verdict: PASS — 24/24 files OK, 103,793,508,077 B
