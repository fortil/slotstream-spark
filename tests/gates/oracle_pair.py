"""The oracle on arbitrary ids in bf16 and in fp32 (the "pair" of design decision 16), for the per-sequence limits.

Subprocess-only and importable without torch: the oracle runs in a child interpreter (tools/ref_layerwise.py
--teacher-force) and the dump is converted to the step-31 layout with make_goldens.convert_dump. ORACLE_DTYPE is read
by ref_layerwise at import, so it is set in the child's environment before the interpreter starts.

  fp32_dump(ids, dst, tag=)      oracle in fp32
  bf16_dump(ids, dst, tag=)      oracle in the checkpoint dtype
  pair_dumps(ids, root, tag)     -> (bf16_dir, fp32_dir) under root/<tag>/{bf16,fp32}, run one after the other

One model process at a time: the caller holds the guard and the two runs are sequential. A dump dir gets
tokens.json {"id": tag, "ids": ids}; an existing dst/tokens.json with equal ids is reused.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "tools" / "ref_layerwise.py"


def _done(dst, ids):
    p = Path(dst) / "tokens.json"
    if not p.exists():
        return False
    try:
        return [int(t) for t in json.loads(p.read_text())["ids"]] == [int(t) for t in ids]
    except (ValueError, KeyError, TypeError):
        return False


def _dump(ids, dst, tag, dtype):
    ids = [int(t) for t in ids]
    dst = Path(dst)
    if _done(dst, ids):
        return dst
    sys.path.insert(0, str(ROOT / "tools"))
    import make_goldens as mg
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    tmp = dst.parent / (dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    ids_json = tmp / "ids.json"
    ids_json.write_text(json.dumps(ids))
    env = dict(os.environ)
    if dtype == "float32":
        env["ORACLE_DTYPE"] = "float32"
    else:
        env.pop("ORACLE_DTYPE", None)
    subprocess.run([sys.executable, str(REF), "--teacher-force", str(ids_json), "--out", str(tmp / "dump")],
                   check=True, env=env)
    mg.convert_dump(tmp / "dump", dst, len(ids))
    shutil.rmtree(tmp, ignore_errors=True)
    (dst / "tokens.json").write_text(json.dumps({"id": tag, "ids": ids}))
    return dst


def fp32_dump(ids, dst, *, tag):
    return _dump(ids, dst, tag, "float32")


def bf16_dump(ids, dst, *, tag):
    return _dump(ids, dst, tag, "bfloat16")


def pair_dumps(ids, root, tag):
    root = Path(root) / tag
    return bf16_dump(ids, root / "bf16", tag=tag), fp32_dump(ids, root / "fp32", tag=tag)
