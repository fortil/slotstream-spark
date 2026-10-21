import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from synth_ckpt import write_synth_checkpoint, write_safetensors

ROOT = Path(__file__).resolve().parents[2]
TOOL = str(ROOT / "tools" / "repack_ngram.py")
P = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding.shard_"


def run(model, out, row_bytes, check=True):
    return subprocess.run([sys.executable, TOOL, "--model", str(model), "--out", str(out), "--row-bytes",
                           str(row_bytes), "--allow-buffered", "--max-mbps", "0"],
                          capture_output=True, text=True, check=check)


def expected_row(t, s, r, row_bytes):
    w, sc, b = (t[f"{P}{s}.{k}"][r] for k in ("weight", "scales", "biases"))
    row = w.astype("<u4").tobytes() + sc.astype("<u2").tobytes() + b.astype("<u2").tobytes()
    return row + b"\0" * (row_bytes - len(row))


@pytest.mark.parametrize("row_bytes", [100, 128])
def test_rows_equal_source(tmp_path, row_bytes):
    model = tmp_path / "m"
    # 1000-row shards x 100 B = 100,000 B: not a multiple of 4096, so the carry path runs
    exp = write_synth_checkpoint(model, ngram_shards=3, ngram_rows=1000)
    out = tmp_path / "o"
    r = run(model, out, row_bytes)
    assert json.loads(r.stdout)["rows"] == 3000
    data = (out / "ngram.bin").read_bytes()
    man = json.loads((out / "ngram.manifest.json").read_text())
    assert len(data) == 3000 * row_bytes == man["total_bytes"]
    assert man["layout"] == "slotstream-spark-ngram-rows-v1" and man["rows_per_shard"] == 1000
    assert man["shards"] == 3 and man["fields"] == {"weight": [0, 80], "scales": [80, 90], "biases": [90, 100]}
    assert man["source"]["index_sha256"] == exp["index_sha256"]
    for s in range(3):
        blk = data[s * 1000 * row_bytes:(s + 1) * 1000 * row_bytes]
        assert hashlib.sha256(blk).hexdigest() == man["shard_sha256"][s]
        for r_ in (0, 1, 499, 999):
            g = s * 1000 + r_
            assert data[g * row_bytes:(g + 1) * row_bytes] == expected_row(exp["tensors"], s, r_, row_bytes)
    assert not (out / "ngram.bin.tmp").exists()


def test_shard_gap_fails(tmp_path):
    model = tmp_path / "m"
    write_synth_checkpoint(model, ngram_shards=2)
    # shard_0 tensors still exist in the safetensors headers; rename shard_1 to shard_2 to make a gap
    import struct
    for fn in sorted(model.glob("*.safetensors")):
        raw = fn.read_bytes()
        (n,) = struct.unpack("<Q", raw[:8])
        h = raw[8:8 + n].decode()
        h2 = h.replace("shard_1.", "shard_2.")
        if h2 != h:
            assert len(h2) == len(h)
            fn.write_bytes(raw[:8] + h2.encode() + raw[8 + n:])
    r = run(model, tmp_path / "o", 100, check=False)
    assert r.returncode != 0 and "contiguous" in (r.stderr + r.stdout)
