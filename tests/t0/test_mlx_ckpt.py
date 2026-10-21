import os
import struct
import sys
import json
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synth_ckpt import bf16, write_synth_checkpoint  # noqa: E402

from slotstream_spark.model import mlx_ckpt as M  # noqa: E402

P = "language_model.model."


@pytest.fixture()
def synth(tmp_path):
    exp = write_synth_checkpoint(tmp_path / "ck")
    return tmp_path / "ck", exp


def test_record_size_is_54_pages(synth):
    _, exp = synth
    assert exp["expert_record_bytes"] == 221_184 == 54 * 4096


def test_headers_and_offsets(synth):
    root, exp = synth
    with M.Checkpoint(root, allow_buffered=True) as ck:
        ts = ck.tensors()
        assert set(ts) == set(exp["tensors"])
        assert ck.index_sha256 == exp["index_sha256"]
        for name, info in ts.items():
            a = exp["tensors"][name]
            assert info.shape == a.shape and info.nbytes == a.nbytes
            fn = root / info.file
            n = struct.unpack("<Q", fn.read_bytes()[:8])[0]
            hdr = json.loads(fn.read_bytes()[8:8 + n])
            assert info.offset == 8 + n + hdr[name]["data_offsets"][0]
            assert np.array_equal(ck.read_array(name), a)


def test_odd_offset_reads(synth):
    root, exp = synth
    name = P + "layers.0.mlp.switch_mlp.gate_proj.weight"
    raw = exp["tensors"][name].tobytes()
    with M.Checkpoint(root, allow_buffered=True) as ck:
        for start, length in ((0, 1), (1, 4095), (4093, 11), (12345, 5000), (len(raw) - 7, 7)):
            assert ck.read_bytes(name, start, length) == raw[start:start + length]
        rows = ck.read_rows(name, [1, 0, 1])
        assert np.array_equal(rows, exp["tensors"][name][[1, 0, 1]])
        with pytest.raises(ValueError):
            ck.read_bytes(name, len(raw) - 1, 2)


def test_single_file_root(synth):
    root, exp = synth
    one = root / "model-00001-of-00002.safetensors"
    with M.Checkpoint(one, allow_buffered=True) as ck:
        assert ck.tensors()
        name = next(iter(ck.tensors()))
        assert np.array_equal(ck.read_array(name), exp["tensors"][name])


def test_refuses_buffered_without_flag(synth):
    root, _ = synth
    if M.O_DIRECT:
        pytest.skip("platform has O_DIRECT")
    with pytest.raises(RuntimeError):
        M.Checkpoint(root)


def _loop_dequant(w, s, b, group):
    out = np.zeros(w.shape[:-1] + (w.shape[-1] * 8,), dtype=np.float32)
    ss, bb = M.bf16_to_f32(s), M.bf16_to_f32(b)
    for idx in np.ndindex(*w.shape[:-1]):
        for j in range(w.shape[-1] * 8):
            q = (int(w[idx + (j // 8,)]) >> (4 * (j % 8))) & 0xF
            out[idx + (j,)] = np.float32(q) * ss[idx + (j // group,)] + bb[idx + (j // group,)]
    return out


@pytest.mark.parametrize("group", [64, 32])
def test_dequant_matches_loop(group):
    rng = np.random.default_rng(1)
    inp = 128
    w = rng.integers(0, 2**32, size=(3, inp // 8), dtype=np.uint32)
    s = bf16(rng.uniform(0.01, 0.1, size=(3, inp // group)))
    b = bf16(rng.uniform(-0.5, 0.5, size=(3, inp // group)))
    assert np.array_equal(M.dequant_affine(w, s, b), _loop_dequant(w, s, b, group))


def test_nibble_order_and_bf16():
    w = np.array([[0x76543210]], dtype=np.uint32)
    one = bf16(np.array([[1.0]]))
    zero = bf16(np.array([[0.0]]))
    assert M.dequant_affine(w, one, zero).tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert M.bf16_to_f32(np.array([0x3F80], dtype=np.uint16)).tolist() == [1.0]


def test_hf_tensor_rules(synth):
    root, exp = synth
    with M.Checkpoint(root, allow_buffered=True) as ck:
        k, a = M.hf_tensor(P + "layers.0.attn_hyper_connection.hc_norm.weight", ck)
        assert k == "attn_hyper_connection.hc_norm.weight"
        raw = exp["tensors"][P + "layers.0.attn_hyper_connection.hc_norm.weight"]
        assert a.dtype == np.float32 and np.array_equal(a, M.bf16_to_f32(raw) - 1.0)
        k, a = M.hf_tensor(P + "layers.0.linear_attn.norm.weight", ck)
        assert k == "linear_attn.norm.weight"
        assert np.array_equal(a, exp["tensors"][P + "layers.0.linear_attn.norm.weight"])  # unchanged
        k, a = M.hf_tensor(P + "layers.0.linear_attn.conv1d.weight", ck)
        assert a.shape == (16, 1, 4)
        assert np.array_equal(a[:, 0, :], exp["tensors"][P + "layers.0.linear_attn.conv1d.weight"][:, :, 0])
        k, a = M.hf_tensor(P + "embed_tokens.weight", ck)
        assert k == "embed_tokens.weight" and a.shape == (64, 512) and a.dtype == np.float32
        k, a = M.hf_tensor(P + "layers.1.ple.ple_embedding.layer_multipliers", ck)
        assert a.dtype == np.int64 and a.tolist() == [3, 5, 7]
        k, a = M.hf_tensor(P + "layers.0.mlp.switch_mlp.gate_proj.weight", ck)
        assert k == "mlp.switch_mlp.gate_proj.weight" and a.shape == (2, 256, 512)
        k, a = M.hf_tensor(P + "layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight", ck)
        assert a.shape == (1000, 160)  # gs32 shard
        with pytest.raises(ValueError):
            M.hf_tensor("vision_tower.blocks.0.norm1.weight", ck)


def test_quantized_centered_norm_is_dequant_minus_one():
    name = P + "layers.0.q_norm.weight"
    base = name[:-len(".weight")]
    arrs = {
        name: np.array([[0x76543210]], dtype=np.uint32),
        base + ".scales": np.array([[0x3F80]], dtype=np.uint16),
        base + ".biases": np.array([[0]], dtype=np.uint16),
    }

    class Fake:
        def tensors(self):
            return arrs

        def read_array(self, n):
            return arrs[n]

    k, a = M.hf_tensor(name, Fake())
    assert k == "q_norm.weight"
    assert a.tolist() == [[float(i) - 1.0 for i in range(8)]]


def test_hf_key_prefixes():
    assert M.hf_key("language_model.model.layers.12.mlp.gate.weight") == "mlp.gate.weight"
    assert M.hf_key("language_model.model.hyper_connection_mixer.x.weight") == "hyper_connection_mixer.x.weight"
    assert M.hf_key("language_model.lm_head.weight") == "lm_head.weight"


def test_centered_norm_tuple():
    assert len(M.CENTERED_NORMS) == 8 and "hc_norm.weight" in M.CENTERED_NORMS
    assert "linear_attn.norm.weight" not in M.CENTERED_NORMS


@pytest.mark.parametrize("o_direct", [True, False])
def test_aligned_writer_roundtrip(tmp_path, o_direct):
    rng = np.random.default_rng(3)
    blocks = [rng.integers(0, 256, size=100_000, dtype=np.uint8).tobytes() for _ in range(90)]
    path = tmp_path / "out.bin"
    with M.AlignedWriter(path, o_direct=o_direct) as w:
        for b in blocks:
            w.write(b)
        assert w.tell == 9_000_000
    data = path.read_bytes()
    assert len(data) == 9_000_000 and data == b"".join(blocks)


def test_aligned_writer_seek(tmp_path):
    path = tmp_path / "s.bin"
    w = M.AlignedWriter(path, o_direct=False)
    w.write(b"\x01" * 8192)
    w.seek_aligned(16384)
    w.write(b"\x02" * 100)
    with pytest.raises(ValueError):
        w.seek_aligned(5)
    w.close()
    d = path.read_bytes()
    assert len(d) == 16384 + 100 and d[:8192] == b"\x01" * 8192 and d[16384:] == b"\x02" * 100
    w2 = M.AlignedWriter(tmp_path / "t.bin", o_direct=False)
    w2.write(b"x" * 10)
    with pytest.raises(ValueError):
        w2.seek_aligned(4096)
    w2.close()


def test_torch_dequant_matches_numpy():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    w = rng.integers(0, 2**32, size=(4, 16), dtype=np.uint32)
    s = bf16(rng.uniform(0.01, 0.1, size=(4, 2)))
    b = bf16(rng.uniform(-0.5, 0.5, size=(4, 2)))
    got = M.dequant_affine_torch(w, s, b, device="cpu").numpy()
    assert np.array_equal(got, M.dequant_affine(w, s, b))
