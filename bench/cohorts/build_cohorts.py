#!/usr/bin/env python3
"""Freeze the F6 A/B cohorts from the expert-lookahead corpus (plan step 77). Runs on the Mac.

  build_cohorts.py --slotstream third_party/slotstream --out bench/cohorts

B0 = the corpus test split (33 requests), B1 = the validation split (32). Every prompt is materialized with the corpus script
(`materialize` from Tools/expert_lookahead_corpus.py, which checks `prompt_sha256` and raises on a mismatch) and its hash is
checked again here. Writes b0.jsonl, b1.jsonl (one request per line, chat messages included) and manifest.json, which pins the
16 pair prompts of each cohort (the first 16 by id whose token estimate is at most 7,680), the seed and the corpus git head.
`--slotstream` falls back to third_party/slotstream when the path has no corpus script.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = "Tools/expert_lookahead_corpus.py"
CORPUS = "Tools/fixtures/expert-lookahead/corpus.json"
COHORTS = {"B0": ("test", 33), "B1": ("validation", 32)}
PAIRS = 16
MAX_ESTIMATE = 7680
SEED = 1729


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locate(slotstream):
    """-> (root of the slotstream tree that holds the corpus script, label for the manifest)."""
    for cand, label in ((Path(slotstream), "slotstream"), (ROOT / "third_party" / "slotstream", "third_party/slotstream")):
        if (cand / SCRIPT).is_file() and (cand / CORPUS).is_file():
            return cand, label
    raise SystemExit(f"build_cohorts: no {SCRIPT} under {slotstream} or third_party/slotstream")


def load_module(path):
    spec = importlib.util.spec_from_file_location("expert_lookahead_corpus", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(root, out):
    mod = load_module(root / SCRIPT)
    corpus = json.loads((root / CORPUS).read_text())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": SEED, "corpus_version": corpus["version"], "corpus_git_head": corpus["git_head"],
                "corpus_sha256": sha256_file(root / CORPUS), "pair_count": PAIRS, "pair_max_token_estimate": MAX_ESTIMATE,
                "cohorts": {}}
    for name, (split, want) in COHORTS.items():
        reqs = sorted((r for r in corpus["requests"] if r["split"] == split), key=lambda r: r["id"])
        if len(reqs) != want:
            raise SystemExit(f"build_cohorts: {name} ({split}) has {len(reqs)} requests, expected {want}")
        rows, verified = [], 0
        for r in reqs:
            messages = mod.materialize(r)                          # raises ValueError on a prompt_sha256 mismatch
            digest = mod.sha256_text(json.dumps(messages, ensure_ascii=False, sort_keys=True))
            if digest != r["prompt_sha256"]:
                raise SystemExit(f"build_cohorts: {r['id']} prompt sha {digest} != frozen {r['prompt_sha256']}")
            verified += 1
            rows.append({"id": r["id"], "kind": r["kind"], "family": r["family"], "split": split, "cap": r["cap"],
                         "prompt_sha256": r["prompt_sha256"], "prompt_tokens_estimate": r.get("prompt_tokens_estimate"),
                         "messages": messages})
        path = out / f"{name.lower()}.jsonl"
        path.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in rows))
        pairs = [x["id"] for x in rows if x["prompt_tokens_estimate"] is not None
                 and x["prompt_tokens_estimate"] <= MAX_ESTIMATE][:PAIRS]
        if len(pairs) != PAIRS:
            raise SystemExit(f"build_cohorts: {name} has {len(pairs)} prompts under {MAX_ESTIMATE} tokens, need {PAIRS}")
        manifest["cohorts"][name] = {"split": split, "file": path.name, "requests": len(rows), "verified_sha": verified,
                                     "sha256": sha256_file(path), "pair_ids": pairs}
        print(f"{name}: {len(rows)} prompts ({split}), {verified} prompt_sha256 verified, {len(pairs)} pair prompts")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slotstream", default="third_party/slotstream")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent))
    a = ap.parse_args(argv)
    root, _ = locate(a.slotstream)
    build(root, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
