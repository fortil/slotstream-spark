import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from synth_ckpt import _quant, bf16, write_safetensors, write_synth_checkpoint

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "tools"


def sh(*args, check=True):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True, check=check)


@pytest.fixture(scope="module")
def packed_set(tmp_path_factory):
    d = tmp_path_factory.mktemp("r0")
    model, packed = d / "model", d / "packed"
    write_synth_checkpoint(model, layers=3, experts=2, ngram_shards=2, ngram_rows=1000)
    rng = np.random.default_rng(5)
    mt = {}
    for proj, (o, i) in (("gate_proj", (256, 512)), ("up_proj", (256, 512)), ("down_proj", (512, 256))):
        w, s, b = _quant(rng, o, i, 64, lead=(2,))
        for k, v in (("weight", w), ("scales", s), ("biases", b)):
            mt[f"mtp.layers.0.mlp.switch_mlp.{proj}.{k}"] = v
    mt["mtp.fc.weight"] = bf16(rng.standard_normal((4, 4)))
    write_safetensors(model / "mtp.safetensors", mt)
    common = ["--allow-buffered"]
    sh(T / "repack_experts.py", "--model", model, "--out", packed, "--expect-record", "0", "--max-mbps", "0", *common)
    sh(T / "repack_experts.py", "--model", model, "--out", packed, "--mtp", "--expect-record", "0", "--max-mbps", "0", *common)
    sh(T / "extract_trunk.py", "--model", model, "--out", packed / "trunk.safetensors", "--allow-unaligned", *common)
    sh(T / "extract_trunk.py", "--model", model, "--out", packed / "mtp_trunk.safetensors", "--mtp", "--allow-unaligned", *common)
    sh(T / "repack_ngram.py", "--model", model, "--out", packed, "--row-bytes", "100", "--max-mbps", "0", *common)
    return model, packed


def verify(model, packed, out):
    return sh(T / "verify_r0.py", "--model", model, "--packed", packed, "--e1", "20", "--max-mbps", "0",
              "--allow-buffered", "--json", out, check=False)


def test_clean_set_is_all_equal(packed_set, tmp_path):
    model, packed = packed_set
    r = verify(model, packed, tmp_path / "r0.json")
    assert r.returncode == 0, r.stdout + r.stderr
    j = json.loads((tmp_path / "r0.json").read_text())
    assert j["all_equal"] and j["experts"]["records"] == 6 and j["mtp_experts"]["records"] == 2
    assert j["ngram"]["rows"] == 2000 and j["e1"]["n"] == 20 and j["e1"]["mismatches"] == 0
    assert j["trunk"]["tensors"] > 5 and j["mtp_trunk"]["tensors"] == 1


@pytest.mark.parametrize("art,key,off", [
    ("experts.bin", "experts", 5), ("mtp_experts.bin", "mtp_experts", 5), ("ngram.bin", "ngram", 5),
    ("trunk.safetensors", "trunk", -1), ("mtp_trunk.safetensors", "mtp_trunk", -1)])
def test_one_flipped_byte_is_found(packed_set, tmp_path, art, key, off):
    model, packed = packed_set
    bad = tmp_path / "packed"
    shutil.copytree(packed, bad)
    p = bad / art
    data = bytearray(p.read_bytes())
    data[off] ^= 0xFF
    p.write_bytes(bytes(data))
    r = verify(model, bad, tmp_path / "r0.json")
    assert r.returncode == 1
    j = json.loads((tmp_path / "r0.json").read_text())
    assert j["all_equal"] is False and j[key]["mismatches"] == 1


def test_matching_layer_sha_does_not_hide_a_changed_byte(packed_set, tmp_path):
    """Falsifier: rewrite the manifest sha to the damaged layer; the byte comparison must still fail."""
    import hashlib
    model, packed = packed_set
    bad = tmp_path / "packed"
    shutil.copytree(packed, bad)
    p = bad / "experts.bin"
    data = bytearray(p.read_bytes())
    data[7] ^= 0x01
    p.write_bytes(bytes(data))
    man = json.loads((bad / "experts.manifest.json").read_text())
    lb = man["experts"] * man["record_bytes"]
    man["layer_sha256"][0] = hashlib.sha256(bytes(data[:lb])).hexdigest()
    (bad / "experts.manifest.json").write_text(json.dumps(man))
    assert verify(model, bad, tmp_path / "r0.json").returncode == 1
    assert json.loads((tmp_path / "r0.json").read_text())["experts"]["mismatches"] == 1


def test_repack_all_syntax():
    assert subprocess.run(["bash", "-n", str(T / "spark" / "repack_all.sh")]).returncode == 0
