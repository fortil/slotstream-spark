import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from synth_ckpt import write_synth_checkpoint

ROOT = Path(__file__).resolve().parents[2]
TOOL = str(ROOT / "tools" / "repack_experts.py")
PIECES = [f"{p}.{k}" for p in ("gate_proj", "up_proj", "down_proj") for k in ("weight", "scales", "biases")]


def run(model, out, *extra, check=True):
    return subprocess.run([sys.executable, TOOL, "--model", str(model), "--out", str(out), "--allow-buffered",
                           "--expect-record", "0", "--max-mbps", "0", *extra],
                          capture_output=True, text=True, check=check)


@pytest.fixture()
def synth(tmp_path):
    model = tmp_path / "model"
    exp = write_synth_checkpoint(model, layers=3, experts=2)
    return model, exp


def test_records_equal_source_slices_and_manifest(synth, tmp_path):
    model, exp = synth
    out = tmp_path / "out"
    r = run(model, out)
    assert json.loads(r.stdout)["layers"] == 3
    data = (out / "experts.bin").read_bytes()
    man = json.loads((out / "experts.manifest.json").read_text())
    rec = man["record_bytes"]
    assert rec == exp["expert_record_bytes"]
    assert len(data) == man["total_bytes"] == 3 * 2 * rec
    assert man["layout"] == "slotstream-pool-layer-expert-piece-v1"
    assert man["version"] == 1 and man["alignment"] == 4096
    assert man["piece_names"] == PIECES
    assert man["model_identity"] == exp["index_sha256"]
    assert man["pieces"] == [int(x) for x in np.diff(man["piece_offsets"] + [rec])]
    assert man["stamp"]["size"] == len(data)
    t = exp["tensors"]
    for L in range(3):
        base = f"language_model.model.layers.{L}.mlp.switch_mlp."
        for e in range(2):
            want = b"".join(np.ascontiguousarray(t[base + n][e]).tobytes() for n in PIECES)
            off = (L * 2 + e) * rec
            assert data[off:off + rec] == want
        assert man["layer_sha256"][L] == hashlib.sha256(data[L * 2 * rec:(L + 1) * 2 * rec]).hexdigest()
    assert not (out / "experts.bin.tmp").exists()


def test_resume_gives_identical_file(synth, tmp_path):
    model, _ = synth
    a, b = tmp_path / "a", tmp_path / "b"
    run(model, a)
    run(model, b)
    full = (b / "experts.bin").read_bytes()
    # simulate a crash after layer 1: tmp file back in place, last layer gone from progress
    (b / "experts.bin").rename(b / "experts.bin.tmp")
    (b / "experts.manifest.json").unlink()
    prog = json.loads((b / "experts.progress.json").read_text())
    prog.pop("2")
    (b / "experts.progress.json").write_text(json.dumps(prog))
    rec = json.loads((a / "experts.manifest.json").read_text())["record_bytes"]
    with open(b / "experts.bin.tmp", "r+b") as f:
        f.seek(2 * 2 * rec)
        f.write(b"\xff" * (2 * rec))  # damaged in-flight layer
    r = run(model, b)
    assert json.loads(r.stdout)["written_layers"] == [2]
    assert (b / "experts.bin").read_bytes() == full == (a / "experts.bin").read_bytes()


def test_alignment_violation_raises(tmp_path):
    model = tmp_path / "m"
    write_synth_checkpoint(model, layers=2, experts=2, hidden=520, ff=264)
    r = run(model, tmp_path / "o", check=False)
    assert r.returncode != 0
    assert "multiple of 4096" in (r.stderr + r.stdout)


def test_expect_record_mismatch(synth, tmp_path):
    model, _ = synth
    r = subprocess.run([sys.executable, TOOL, "--model", str(model), "--out", str(tmp_path / "o"),
                        "--allow-buffered", "--expect-record", "2764800"], capture_output=True, text=True)
    assert r.returncode != 0


def test_direct_probe_never_touches_the_data_file(tmp_path, monkeypatch):
    """Resuming reopens a half-written file; the O_DIRECT probe must not overwrite its first block.
    O_SYNC stands in for O_DIRECT so the branch runs on a Mac too."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("repack_experts_probe", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "O_DIRECT", os.O_SYNC)
    path = tmp_path / "experts.bin.tmp"
    size = 4 * mod.ALIGN
    path.write_bytes(bytes(range(256)) * (size // 256))
    before = path.read_bytes()
    slab = mod.SlabFile(str(path), size, allow_buffered=False)
    slab.close()
    assert path.read_bytes() == before
    assert not (tmp_path / "experts.bin.tmp.probe").exists()
