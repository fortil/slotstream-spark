#!/usr/bin/env python3
"""Generate model.lock.json and SHA256SUMS.model from the pinned slotstream manifest.

Usage: gen_model_lock.py [--pinned PATH] [--check]
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PINNED = "third_party/slotstream/Sources/Slotstream/PinnedModel.swift"
ROW_RE = re.compile(r'File\(path: "([^"]+)", size: (\d+), sha256: "([0-9a-f]{64})"(, optional: true)?\)')


def build(pinned_text):
    files, mtp = [], None
    for path, size, sha, opt in ROW_RE.findall(pinned_text):
        row = {"path": path, "sha256": sha, "size": int(size)}
        if opt:
            mtp = row
        else:
            files.append(row)
    files.sort(key=lambda r: r["path"])
    lock = {
        "files": files,
        "generated_from": "PinnedModel.swift@8bd579385152d547f96c9f6285181fe7885dfe29",
        "mirror": {
            "repo": "carloslfu/Qwen3.8-Flash-Next-MLX-4bit",
            "revision": "e9d552f83de4665d243d5c9cf73201a1ca6c16d7",
        },
        "mtp": {
            "converter": "third_party/slotstream/Tools/mtp_convert.py",
            "path": mtp["path"],
            "sha256": mtp["sha256"],
            "size": mtp["size"],
            "source_bytes": 5214301696,
            "source_repo": "Qwen/Qwen3.8-Flash-Next",
            "source_revision": "de4b8e4d43b917e7706784d8bb445c9af86a3540",
            "source_tensors": 31,
        },
        "repo": "pipenetwork/Qwen3.8-Flash-Next-MLX-4bit",
        "revision": "aa7c790e804bbf9d491ddb109c3d61bc4a555f7c",
        "shards_bytes": sum(r["size"] for r in files if r["path"].endswith(".safetensors")),
        "total_bytes": sum(r["size"] for r in files),
    }
    sums = "".join(f"{r['sha256']}  {r['path']}\n" for r in files)
    return json.dumps(lock, indent=2, sort_keys=True) + "\n", sums


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pinned", default=DEFAULT_PINNED)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    pinned = Path(a.pinned)
    if not pinned.is_absolute():
        pinned = ROOT / pinned
    lock, sums = build(pinned.read_text())
    lp, sp = ROOT / "model.lock.json", ROOT / "SHA256SUMS.model"
    if a.check:
        bad = [p.name for p, t in ((lp, lock), (sp, sums)) if not p.exists() or p.read_text() != t]
        if bad:
            print("differs: " + ", ".join(bad))
            return 1
        return 0
    lp.write_text(lock)
    sp.write_text(sums)
    return 0


if __name__ == "__main__":
    sys.exit(main())
