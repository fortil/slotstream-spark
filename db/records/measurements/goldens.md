# goldens: reproducible oracle goldens
Date: 2026-10-12
Capture: db/sources/runs/2026/10/20261012T142519-oracle-chain/
Command: part of the oracle-chain job; tools/make_goldens.py ran twice and compared sha256 per file; manifest in tests/goldens/manifest.json

| item | value |
|---|---|
| prompts | 16 (8 dev, 8 reserved) |
| files per run | 2,400, same file set in both runs |
| differing files between the two runs | 0 |
| reproducible | true |
| oracle | bfloat16, GDN torch implementation, torch 2.13.0+cu130, transformers 5.19.0 |

The goldens data stays out of git (tests/goldens/data/); only the manifest with the sha256 list is tracked.

Verdict: PASS — 16 prompts, 2,400 files, two runs identical (difference 0), reproducible true
