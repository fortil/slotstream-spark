import json
import subprocess
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slotstream_spark.model.mlx_ckpt import Checkpoint  # noqa: E402
from synth_ckpt import write_synth_checkpoint, write_safetensors, bf16  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TOOL = str(ROOT / "tools" / "extract_trunk.py")


def run(model, out, *extra, check=True):
    return subprocess.run([sys.executable, TOOL, "--model", str(model), "--out", str(out), "--allow-buffered",
                           "--allow-unaligned", *extra], capture_output=True, text=True, check=check)


def test_main_trunk(tmp_path):
    model = tmp_path / "m"
    exp = write_synth_checkpoint(model, layers=2, experts=2, ngram_shards=2)
    out = tmp_path / "trunk.safetensors"
    s = json.loads(run(model, out).stdout)
    t = exp["tensors"]
    want = sorted(n for n in t if n.startswith("language_model.") and ".mlp.switch_mlp." not in n
                  and ".ngram_embedding.shard_" not in n)
    got = Checkpoint(out, allow_buffered=True)
    names = list(got.tensors())
    assert sorted(names) == want and s["tensors"] == len(want)
    assert not any(n.startswith("vision_tower.") or "switch_mlp" in n or "shard_" in n for n in names)
    # layout: embed first, then sorted, contiguous, header aligned
    (n,) = struct.unpack("<Q", out.read_bytes()[:8])
    assert (8 + n) % 4096 == 0
    hdr = json.loads(out.read_bytes()[8:8 + n])
    assert hdr["__metadata__"]["source_index_sha256"] == exp["index_sha256"]
    order = sorted((v["data_offsets"][0], k) for k, v in hdr.items() if k != "__metadata__")
    first3 = [k for _, k in order[:3]]
    assert first3 == ["language_model.model.embed_tokens." + x for x in ("weight", "scales", "biases")]
    assert [k for _, k in order[3:]] == sorted(k for _, k in order[3:])
    end = 0
    for off, k in order:
        assert off == end
        end = hdr[k]["data_offsets"][1]
    assert got.info("language_model.model.embed_tokens.weight").offset % 4096 == 0
    # bytes identical, centered norms untouched
    for k in want:
        assert np.array_equal(got.read_array(k), t[k]), k
    assert np.array_equal(got.read_array("language_model.model.layers.0.attn_hyper_connection.hc_norm.weight"),
                          t["language_model.model.layers.0.attn_hyper_connection.hc_norm.weight"])
    try:
        from safetensors.numpy import load_file
    except ImportError:
        return
    d = load_file(str(out))
    assert set(d) == set(want)


def test_mtp_trunk(tmp_path):
    model = tmp_path / "m"
    model.mkdir()
    rng = np.random.default_rng(1)
    t = {"mtp.layers.0.mlp.switch_mlp.gate_proj.weight": rng.integers(0, 2**32, (2, 8, 8), dtype=np.uint32),
         "mtp.layers.0.mlp.gate.weight": bf16(rng.standard_normal((2, 16))),
         "mtp.fc.weight": bf16(rng.standard_normal((4, 4))),
         "mtp.norm.weight": bf16(rng.standard_normal(4))}
    write_safetensors(model / "mtp.safetensors", t)
    out = tmp_path / "mtp_trunk.safetensors"
    run(model, out, "--mtp")
    got = Checkpoint(out, allow_buffered=True)
    assert sorted(got.tensors()) == ["mtp.fc.weight", "mtp.layers.0.mlp.gate.weight", "mtp.norm.weight"]
    for k in got.tensors():
        assert np.array_equal(got.read_array(k), t[k])


def test_misaligned_embed_rejected(tmp_path):
    model = tmp_path / "m"
    write_synth_checkpoint(model)
    r = subprocess.run([sys.executable, TOOL, "--model", str(model), "--out", str(tmp_path / "o"),
                        "--allow-buffered"], capture_output=True, text=True)
    assert r.returncode != 0 and "4096-aligned" in (r.stderr + r.stdout)
