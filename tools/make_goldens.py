#!/usr/bin/env python3
"""Teacher-forced dumps of the oracle (tools/ref_layerwise.py) on the golden prompts, and their manifest.

Per prompt, resumable (a DONE marker is written last), DIR/<set>/<id>/ holds:
  tokens.json            {"id", "set", "ids"}
  h_in.f32               [T,10240]   hyper-width input of layer 0
  layer_{00..47}_out.f32 [T,10240]
  router_{LL}_ids.i32    [T,10]      router_{LL}_w.f32  [T,10]
  ngram_ids.i64          [T,16]      ngram_rows.f32     [T,16,160]
  logits.f32             [T,248320]
  meta.json              {dtype, experts_mode, gdn_impl, transformers, torch}
The oracle dtype comes from ref_layerwise's ORACLE_DTYPE env knob (unset = bf16 goldens, float32 = the fp32
calibration pass). This layout is the dump contract that the engine's Model.teacher_force(ids, dump_dir=...)
(step 47) writes too; load_golden(id) reads it.

  make_goldens.py --prompts P --out DIR [--state DIR] [--sets dev,reserved] [--experts-mode correct|swap|noexp]
  make_goldens.py --compare DIR_A DIR_B --manifest OUT [--prompts P]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TOOLS = Path(__file__).resolve().parent
LAYERS = 48
HIDDEN = 10240
TOPK = 10
NGRAM_HEADS = 16
NGRAM_DIM = 160
VOCAB = 248320
MAX_PROMPT_TOKENS = 64
DEFAULT_PROMPTS = ROOT / "tests" / "goldens" / "prompts.json"
DEFAULT_DATA = ROOT / "tests" / "goldens" / "data"

_SUFFIX = {".f32": np.float32, ".i32": np.int32, ".i64": np.int64}


def expected_files(layers=LAYERS):
    """File names (relative to a prompt dir) that a complete dump holds, DONE excluded."""
    names = ["tokens.json", "meta.json", "h_in.f32", "ngram_ids.i64", "ngram_rows.f32", "logits.f32"]
    for L in range(layers):
        names += [f"layer_{L:02d}_out.f32", f"router_{L:02d}_ids.i32", f"router_{L:02d}_w.f32"]
    return sorted(names)


def file_shape(name, T):
    """Shape of a raw dump file for a prompt of T tokens, or None for json files."""
    if name.endswith(".json"):
        return None
    if name.startswith("router_"):
        return [T, TOPK]
    if name == "ngram_ids.i64":
        return [T, NGRAM_HEADS]
    if name == "ngram_rows.f32":
        return [T, NGRAM_HEADS, NGRAM_DIM]
    if name == "logits.f32":
        return [T, VOCAB]
    return [T, HIDDEN]


def load_prompts(path):
    d = json.load(open(path))
    return d["prompts"] if isinstance(d, dict) else d


def find_prompt_dir(root, pid):
    for s in ("dev", "reserved"):
        p = Path(root) / s / pid
        if p.is_dir():
            return p
    raise FileNotFoundError(f"golden {pid} not found under {root}")


def load_golden(pid, root=None):
    """-> dict keyed by file stem ('h_in', 'layer_00_out', 'router_00_ids', 'ngram_ids', 'logits', ...),
    plus 'tokens' (list of ids) and 'meta'."""
    d = find_prompt_dir(root or DEFAULT_DATA, pid)
    tk = json.load(open(d / "tokens.json"))
    T = len(tk["ids"])
    out = {"tokens": tk["ids"], "meta": json.load(open(d / "meta.json"))}
    for f in sorted(d.iterdir()):
        if f.suffix in _SUFFIX:
            out[f.stem] = np.fromfile(f, dtype=_SUFFIX[f.suffix]).reshape(file_shape(f.name, T))
    return out


def sha256_file(p, buf=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _files(root):
    root = Path(root)
    return {str(p.relative_to(root)): p for p in sorted(root.rglob("*")) if p.is_file() and p.name != "DONE"
            and ".state" not in p.relative_to(root).parts}


def compare(dir_a, dir_b, prompts_path=None):
    """Manifest of DIR_A; reproducible only if DIR_B has the same file set and every sha256 is equal."""
    fa, fb = _files(dir_a), _files(dir_b)
    files, reproducible = {}, set(fa) == set(fb)
    for rel, p in fa.items():
        sha = sha256_file(p)
        if rel in fb and sha256_file(fb[rel]) != sha:
            reproducible = False
        name = p.name
        T = None
        if not name.endswith(".json"):
            tk = json.load(open(p.parent / "tokens.json"))
            T = len(tk["ids"])
        sfx = p.suffix
        files[rel] = {"sha256": sha, "bytes": p.stat().st_size,
                      "dtype": np.dtype(_SUFFIX[sfx]).name if sfx in _SUFFIX else "json",
                      "shape": file_shape(name, T) if T is not None else None}
    metas = sorted(Path(dir_a).rglob("meta.json"))
    m = json.load(open(metas[0])) if metas else {}
    pp = Path(prompts_path) if prompts_path else DEFAULT_PROMPTS
    return {"oracle": {"transformers": m.get("transformers"), "torch": m.get("torch"),
                       "gdn_impl": m.get("gdn_impl"), "dtype": m.get("dtype")},
            "prompts_sha256": sha256_file(pp) if pp.exists() else None,
            "reproducible": reproducible, "files": files}


def _tofile(path, arr, dtype):
    np.ascontiguousarray(arr, dtype=dtype).tofile(path)


def convert_dump(tmp, dst, T, nlayers=LAYERS):
    """ref_layerwise .npy dump (tmp) -> raw layout (dst)."""
    _tofile(dst / "h_in.f32", np.load(tmp / "h_in.npy").reshape(T, HIDDEN), np.float32)
    for L in range(nlayers):
        _tofile(dst / f"layer_{L:02d}_out.f32", np.load(tmp / f"layer_{L}.npy").reshape(T, HIDDEN), np.float32)
        ids = np.load(tmp / f"router_ids_{L}.npy").reshape(T, -1)
        w = np.load(tmp / f"router_weights_{L}.npy").reshape(T, -1)
        if ids.shape[1] != TOPK or w.shape[1] != TOPK:
            raise RuntimeError(f"layer {L}: router picks {ids.shape}/{w.shape}, expected [T,{TOPK}]")
        _tofile(dst / f"router_{L:02d}_ids.i32", ids, np.int32)
        _tofile(dst / f"router_{L:02d}_w.f32", w, np.float32)
    _tofile(dst / "ngram_ids.i64", np.load(tmp / "ngram_ids.npy").reshape(T, NGRAM_HEADS), np.int64)
    _tofile(dst / "ngram_rows.f32", np.load(tmp / "ngram_rows.npy").reshape(T, NGRAM_HEADS, NGRAM_DIM), np.float32)
    _tofile(dst / "logits.f32", np.load(tmp / "logits.npy").reshape(T, VOCAB), np.float32)


def tokenize(prompts, model_dir):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    out = {}
    for p in prompts:
        ids = [int(x) for x in tok(p["text"], add_special_tokens=False)["input_ids"]]
        if not 1 <= len(ids) <= MAX_PROMPT_TOKENS:
            raise SystemExit(f"prompt {p['id']} has {len(ids)} tokens (limit {MAX_PROMPT_TOKENS})")
        out[p["id"]] = ids
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    ap.add_argument("--out")
    ap.add_argument("--state")
    ap.add_argument("--sets", default="dev,reserved")
    ap.add_argument("--experts-mode", default="correct", choices=("correct", "swap", "noexp"))
    ap.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"))
    ap.add_argument("--manifest")
    a = ap.parse_args(argv)

    if a.compare:
        if not a.manifest:
            ap.error("--compare needs --manifest OUT")
        man = compare(a.compare[0], a.compare[1], a.prompts)
        Path(a.manifest).parent.mkdir(parents=True, exist_ok=True)
        with open(a.manifest, "w") as f:
            json.dump(man, f, indent=1, sort_keys=True)
            f.write("\n")
        print(json.dumps({"reproducible": man["reproducible"], "files": len(man["files"])}))
        return 0 if man["reproducible"] else 1
    if not a.out:
        ap.error("--out DIR is required")

    sys.path.insert(0, str(TOOLS))
    import ref_layerwise as ref  # heavy: torch, transformers, CUDA
    import torch
    import transformers
    from slotstream_spark.model.mlx_ckpt import Checkpoint

    sets = [s for s in a.sets.split(",") if s]
    prompts = [p for p in load_prompts(a.prompts) if p["set"] in sets]
    out = Path(a.out)
    state = Path(a.state) if a.state else out / ".state"
    mdir = ref.model_dir()
    ids_by = tokenize(prompts, mdir)
    cfg = ref.load_cfg(mdir)
    ck = Checkpoint(mdir)
    done = 0
    for p in prompts:
        dst = out / p["set"] / p["id"]
        if (dst / "DONE").exists():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        tmp = state / "dump"
        shutil.rmtree(tmp, ignore_errors=True)
        ids = ids_by[p["id"]]
        ref.teacher_force(ids, dump_dir=str(tmp), experts_mode=a.experts_mode, ck=ck, cfg=cfg)
        convert_dump(tmp, dst, len(ids))
        shutil.rmtree(tmp, ignore_errors=True)
        (dst / "tokens.json").write_text(json.dumps({"id": p["id"], "set": p["set"], "ids": ids}))
        (dst / "meta.json").write_text(json.dumps({
            "dtype": str(ref.DTYPE).replace("torch.", ""), "experts_mode": a.experts_mode, "gdn_impl": "torch",
            "transformers": transformers.__version__, "torch": torch.__version__}, sort_keys=True))
        (dst / "DONE").write_text("ok\n")
        done += 1
        print(f"golden {p['set']}/{p['id']}: {len(ids)} tokens", flush=True)
    ck.close()
    print(json.dumps({"prompts": len(prompts), "written": done, "experts_mode": a.experts_mode}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
