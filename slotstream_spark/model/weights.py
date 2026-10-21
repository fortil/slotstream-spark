"""Trunk weights, embedding rows and n-gram rows for the engine.

`load_trunk` reads trunk.safetensors tensor by tensor into preallocated host buffers and moves them to the
device. Each tensor's pages are dropped from the page cache right after (POSIX_FADV_DONTNEED, rule 8 of the
plan), so loading the trunk does not leave 2.9 GB of cache behind. `embed_tokens` is skipped: EmbedRows serves
it from the same file. Quantized groups (`X.weight` uint32 + `X.scales` + `X.biases`) become one AffineWeight
under the key `X.weight`; dense tensors keep their name. Names lose the `language_model.model.` prefix
(`language_model.lm_head.*` becomes `lm_head.*`), which is what reference_torch.RefModel asks for.
"""
import json
import os
import struct
from pathlib import Path

import numpy as np
import torch

from slotstream_spark import config as _config
from slotstream_spark.kernels.gemv_affine import AffineWeight, DenseWeight

_DT = {"U32": torch.int32, "BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
       "I64": torch.int64, "I32": torch.int32, "U8": torch.uint8}
_ITEM = {"U32": 4, "BF16": 2, "F16": 2, "F32": 4, "I32": 4}
_PREFIXES = ("language_model.model.", "language_model.")
GS = 64


def strip_prefix(name):
    for p in _PREFIXES:
        if name.startswith(p):
            return name[len(p):]
    return name


def _layer_of(name):
    if name.startswith("layers."):
        return int(name.split(".", 2)[1])
    return None


def read_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return 8 + n, json.loads(f.read(n))


def _read_tensor(fd, offset, nbytes):
    buf = torch.empty(nbytes, dtype=torch.uint8)
    mv = memoryview(buf.numpy())
    got = 0
    while got < nbytes:
        r = os.preadv(fd, [mv[got:]], offset + got)
        if r <= 0:
            raise OSError(f"short read at offset {offset + got}")
        got += r
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, offset, nbytes, os.POSIX_FADV_DONTNEED)
    return buf


def load_trunk(path, layers=None, device="cuda"):
    """-> dict name -> AffineWeight | torch.Tensor. `layers` (iterable of layer indices) keeps only those
    layers plus every tensor outside `layers.N.` (final mixer, lm_head); None keeps all of them."""
    keep = None if layers is None else set(int(x) for x in layers)
    base, header = read_header(path)
    header = {strip_prefix(k): v for k, v in header.items() if k != "__metadata__"}
    names = sorted((k for k in header if not k.startswith("embed_tokens.")),
                   key=lambda k: header[k]["data_offsets"][0])
    names = [k for k in names if keep is None or _layer_of(k) is None or _layer_of(k) in keep]
    raw = {}
    fd = os.open(str(path), os.O_RDONLY)
    try:
        for k in names:
            meta = header[k]
            b, e = meta["data_offsets"]
            buf = _read_tensor(fd, base + b, e - b)
            t = buf.view(_DT[meta["dtype"]]).reshape(meta["shape"])
            raw[k] = t if t.dtype == torch.int64 else t.to(device)
    finally:
        os.close(fd)
    out = {}
    for k, t in raw.items():
        stem = k.rsplit(".", 1)[0]
        part = k.rsplit(".", 1)[1]
        if part in ("scales", "biases"):
            continue
        if part == "weight" and f"{stem}.scales" in raw:
            s, b = raw[f"{stem}.scales"], raw[f"{stem}.biases"]
            n, words = t.shape
            kdim = s.shape[1] * GS
            bits = words * 32 // kdim
            out[k] = AffineWeight(t, s, b, bits, GS, n, kdim)
        else:
            out[k] = t
    return out


def accessor(weights):
    """The `w(name)` callable of RefModel/Model over a loaded dict (a missing name is an error)."""
    def w(name):
        try:
            return weights[name]
        except KeyError:
            raise KeyError(f"trunk tensor {name!r} not loaded") from None
    return w


# ---------------------------------------------------------------- rows served by slotio
class Embedder:
    """embed_tokens rows through slotio.EmbedRows (pinned landing buffer) and the row dequant kernel -> bf16."""

    def __init__(self, reader, trunk_path, device="cuda", vocab=None):
        import slotio
        base, header = read_header(trunk_path)
        metas = {}
        for part in ("weight", "scales", "biases"):
            hits = [k for k in header if k.endswith(f"embed_tokens.{part}")]
            if len(hits) != 1:
                raise RuntimeError(f"expected exactly one embed_tokens.{part} in {trunk_path}, found {hits}")
            metas[part] = header[hits[0]]

        def row_bytes(m):
            return int(m["shape"][1]) * _ITEM[m["dtype"]]

        self.file = slotio.File.open_direct(str(trunk_path))     # EmbedRows keeps it alive too (keep_alive)
        off = {p: base + metas[p]["data_offsets"][0] for p in metas}
        self.rows = slotio.EmbedRows(reader, self.file, off["weight"], off["scales"], off["biases"],
                                     row_bytes(metas["weight"]), row_bytes(metas["scales"]),
                                     row_bytes(metas["biases"]), int(vocab or metas["weight"]["shape"][0]))
        self.row_bytes = self.rows.row_bytes
        self.device = device

    def __call__(self, ids):
        from slotstream_spark.kernels.rows import embed_rows_affine
        ids = np.ascontiguousarray(np.asarray([int(i) for i in ids], dtype=np.int64))
        buf = torch.empty((len(ids), self.row_bytes), dtype=torch.uint8).pin_memory()
        self.rows.read(ids, buf.numpy())
        return embed_rows_affine(buf.to(self.device, non_blocking=True))


def check_ngram_row_bytes(packed_dir, platform_row_bytes=None):
    """row_bytes of ngram.manifest.json must equal platform.toml ngram.row_bytes; fails loudly otherwise."""
    man = json.loads((Path(packed_dir) / "ngram.manifest.json").read_text())
    want = _config.get("ngram.row_bytes") if platform_row_bytes is None else platform_row_bytes
    got = int(man["row_bytes"])
    if got != int(want):
        raise RuntimeError(f"ngram.manifest.json row_bytes={got} differs from platform.toml ngram.row_bytes={want}")
    return got


class NgramSource:
    """The `ngram_rows(history, ids) -> bf16 [T, 2560]` accessor: ids from NgramHasher, 16 rows per token
    from ngram.bin through slotio.NgramRows, dequantized by the gs32 gather kernel."""

    def __init__(self, reader, packed_dir, weights, eos, device="cuda", cache_rows=400000):
        import slotio
        from slotstream_spark.model.ngram_hash import NgramHasher
        self.row_bytes = check_ngram_row_bytes(packed_dir)      # enforced at the construction site
        p = "layers.1.ple.ple_embedding."
        mult = weights[p + "layer_multipliers"].tolist()
        sizes = weights[p + "ngram_heads_vocab_sizes"].tolist()
        offs = weights[p + "ngram_heads_offsets"].tolist()
        self.hasher = NgramHasher.from_buffers(mult, sizes, offs, eos)
        self.eos = int(eos)
        self.file = slotio.File.open_direct(str(Path(packed_dir) / "ngram.bin"))
        self.rows = slotio.NgramRows(reader, self.file, self.row_bytes, cache_rows)
        if int(self.rows.row_bytes) != self.row_bytes:
            raise RuntimeError("NgramRows.row_bytes differs from the manifest")
        self.device = device
        self.last_ids = None

    def ids(self, history, ids):
        ctx = ([self.eos] * 2 + [int(t) for t in history])[-2:]
        return self.hasher.ids(ctx, [int(t) for t in ids])

    def __call__(self, history, ids):
        from slotstream_spark.kernels.rows import ple_gather_gs32
        gids = np.ascontiguousarray(self.ids(history, ids), dtype=np.int64)       # [T, 16]
        self.last_ids = gids
        buf = torch.empty((gids.size, self.row_bytes), dtype=torch.uint8).pin_memory()
        self.rows.read(gids.reshape(-1), buf.numpy())
        dev = buf.to(self.device, non_blocking=True).view(len(gids), 16, self.row_bytes)
        return torch.stack([ple_gather_gs32(dev[t].contiguous(), self.row_bytes) for t in range(len(gids))])
