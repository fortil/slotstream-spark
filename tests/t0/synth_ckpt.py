"""Synthetic MLX-format checkpoint for t0 tests (numpy only, no weights needed)."""
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

_ST = {np.dtype(np.uint32): "U32", np.dtype(np.uint16): "BF16", np.dtype(np.int64): "I64",
       np.dtype(np.float32): "F32"}


def bf16(a):
    """float32 array -> raw bf16 bits (round to nearest even)."""
    u = np.asarray(a, dtype=np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + np.uint32(0x7FFF)
    return ((u + r) >> 16).astype(np.uint16)


def write_safetensors(path, tensors):
    """tensors: ordered dict name -> numpy array (uint32/uint16(bf16)/int64/float32)."""
    header, off = {}, 0
    for name, a in tensors.items():
        n = a.nbytes
        header[name] = {"dtype": _ST[a.dtype], "shape": list(a.shape), "data_offsets": [off, off + n]}
        off += n
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for a in tensors.values():
            f.write(np.ascontiguousarray(a).tobytes())


def _quant(rng, out, inp, group, lead=()):
    w = rng.integers(0, 2**32, size=lead + (out, inp // 8), dtype=np.uint32)
    s = bf16(rng.uniform(0.01, 0.1, size=lead + (out, inp // group)))
    b = bf16(rng.uniform(-0.5, 0.5, size=lead + (out, inp // group)))
    return w, s, b


def write_synth_checkpoint(root, layers=2, experts=2, hidden=512, ff=256, ngram_shards=2,
                           ngram_rows=1000, seed=0):
    """Write a 2-shard checkpoint + index under root; return what the tests compare against."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    t, expect = {}, {}
    p = "language_model.model."

    w, s, b = _quant(rng, 64, 512, 64)
    t[p + "embed_tokens.weight"], t[p + "embed_tokens.scales"], t[p + "embed_tokens.biases"] = w, s, b
    for L in range(layers):
        lp = f"{p}layers.{L}."
        for proj, (o, i) in (("gate_proj", (ff, hidden)), ("up_proj", (ff, hidden)), ("down_proj", (hidden, ff))):
            w, s, b = _quant(rng, o, i, 64, lead=(experts,))
            t[f"{lp}mlp.switch_mlp.{proj}.weight"] = w
            t[f"{lp}mlp.switch_mlp.{proj}.scales"] = s
            t[f"{lp}mlp.switch_mlp.{proj}.biases"] = b
        t[lp + "mlp.gate.weight"] = bf16(rng.standard_normal((experts, hidden)))
    t[p + "layers.0.attn_hyper_connection.hc_norm.weight"] = bf16(rng.uniform(0.5, 2.0, size=64))
    t[p + "layers.0.linear_attn.norm.weight"] = bf16(rng.uniform(0.5, 2.0, size=32))
    t[p + "layers.0.linear_attn.conv1d.weight"] = bf16(rng.standard_normal((16, 4, 1)))
    t[p + "layers.1.ple.ple_embedding.layer_multipliers"] = np.array([3, 5, 7], dtype=np.int64)
    t[p + "layers.1.ple.ple_embedding.ngram_heads_vocab_sizes"] = np.arange(16, dtype=np.int64) + 100
    t[p + "layers.1.ple.ple_embedding.ngram_heads_offsets"] = np.arange(16, dtype=np.int64) * 1000
    for sh in range(ngram_shards):
        base = f"{p}layers.1.ple.ple_embedding.ngram_embedding.shard_{sh}"
        w, s, b = _quant(rng, ngram_rows, 160, 32)
        t[base + ".weight"], t[base + ".scales"], t[base + ".biases"] = w, s, b
    t["vision_tower.blocks.0.norm1.weight"] = bf16(rng.standard_normal(8))

    names = list(t)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half],
              "model-00002-of-00002.safetensors": names[half:]}
    wm, total = {}, 0
    for fn, ns in shards.items():
        write_safetensors(root / fn, {n: t[n] for n in ns})
        for n in ns:
            wm[n] = fn
            total += t[n].nbytes
    index = {"metadata": {"total_size": total}, "weight_map": wm}
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    expect["tensors"] = t
    expect["weight_map"] = wm
    expect["index_sha256"] = hashlib.sha256((root / "model.safetensors.index.json").read_bytes()).hexdigest()
    expect["expert_record_bytes"] = sum(
        t[f"{p}layers.0.mlp.switch_mlp.{pr}.{k}"][0].nbytes
        for pr in ("gate_proj", "up_proj", "down_proj") for k in ("weight", "scales", "biases"))
    return expect
