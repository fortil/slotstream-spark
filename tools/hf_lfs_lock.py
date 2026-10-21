#!/usr/bin/env python3
"""Pin Hugging Face LFS files (path, size, sha256) into a lock JSON. Needs network.

  hf_lfs_lock.py --repo R --revision FULLSHA --include GLOB [--include ...] --key K --out FILE
"""
import argparse
import fnmatch
import json
import re
import urllib.request
from pathlib import Path


def tree(repo, rev):
    url = f"https://huggingface.co/api/models/{repo}/tree/{rev}?recursive=true"
    items = []
    while url:
        with urllib.request.urlopen(url, timeout=60) as r:
            items += json.load(r)
            link = r.headers.get("Link", "")
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else None
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--include", action="append", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", a.revision):
        raise SystemExit("--revision must be a full 40-hex commit")

    files = []
    for it in tree(a.repo, a.revision):
        if it.get("type") != "file" or "lfs" not in it:
            continue
        if any(fnmatch.fnmatch(it["path"], g) for g in a.include):
            files.append({"path": it["path"], "size": it["lfs"].get("size", it["size"]),
                          "sha256": it["lfs"]["oid"]})
    files.sort(key=lambda f: f["path"])
    out = Path(a.out)
    lock = json.loads(out.read_text()) if out.exists() else {}
    lock[a.key] = {"repo": a.repo, "revision": a.revision, "files": files,
                   "total_bytes": sum(f["size"] for f in files)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lock, indent=1, sort_keys=True) + "\n")
    print(f"{a.key}: {len(files)} files, {lock[a.key]['total_bytes']} bytes")


if __name__ == "__main__":
    main()
