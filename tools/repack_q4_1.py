#!/usr/bin/env python3
"""MLX affine 4-bit -> GGUF Q4_1, exact except the named exceptions (plan step 94). The weights are repacked, never requantized.

  repack_q4_1.py --model DIR --template DIR|FIRST_SHARD --out DIR [--max-mbps 2000] [--allow-buffered]
  repack_q4_1.py --model DIR --sweep [--json FILE]       fp16 exactness of every tensor of the checkpoint, nothing written
  repack_q4_1.py --template DIR --dump-template          the template's KV keys and tensors

Why it is exact. An MLX group of 64 (or 32 for the n-gram table) is `w = q * scale + bias` with q in 0..15. A ggml Q4_1
block of 32 is `y = q * d + m` with the same q. One MLX group of 64 becomes two Q4_1 blocks that share (d, m) = (scale,
bias), converted bf16 -> fp16. That conversion is exact for every |v| in [2^-17, 65504] (and for 0); anything else is a hard
error that names the tensor, except the scale/bias tensors named in slotstream_spark/config/q4_1_exceptions.json (7 tensors of
the real checkpoint, found by the read-only sweep): those are rounded to fp16 with an absolute error of at most 2^-24 per value,
listed by name in the sweep result and in the manifest (design decision 21). Negative scales are kept (MLX writes them for half of the groups; `q * d + m` does not need
d >= 0). The nibble order changes from MLX's (element i of a uint32 at bits 4i) to Q4_1's (`qs[j]` low nibble = element j,
high nibble = element j + 16); that is a permutation of the same 4-bit codes. The bartowski Q4_1 GGUF of step 12 is only the
template: its KV, tensor names, shapes and order are copied; its values are never used for the output.

Mapping. Every tensor of the template is rebuilt from the MLX tensor(s) named by the rule table below, with the same
transformations the llama.cpp converter (conversion/qwen4exp.py and qwen.py) applies: V heads of the gated delta net reordered
from grouped to tiled order (rows of in_proj_qkv V part, in_proj_z, in_proj_a/b, columns of out_proj, the conv channels,
A_log, dt_bias), `A_log -> -exp(A_log)`, the 1-D conv kernels squeezed, `index_qk_proj` split into the indexer q and k, the 128
n-gram shards concatenated, the norms copied as they are (the MLX checkpoint already holds the +1 that the converter adds).
The hyper-connection matrices (hc_*_down/up, output_hc_*) are the one exception to "stay Q4_1": they are written as BF16, the
MLX 4-bit values dequantized in fp32 (q * scale + bias) and rounded to nearest even. Written as Q4_1 they cost llama.cpp +0.15
nats per token against the original GGUF's own Q4_1 hc tensors (DIAG_F9c.md), because llama.cpp quantizes the activations of a
Q4_1 matmul to 8 bits and these gates are sensitive to it; as BF16 the same values cost 0.04. The engine reads the same values.
A template tensor with no rule is refused. A mapping that does not reproduce the template is caught by two checks on sampled
rows of every tensor: Pearson correlation with the template's dequantized values >= 0.95 (one floor for every family;
tensors in [0.95, 0.99) are listed as information), and (for tensors that are not
4-bit) the mean offset within 0.05 standard deviations, which is what a missing or doubled +1 looks like. The int64 hash
constants are read from the checkpoint and must equal the template's KV arrays, which are kept byte for byte.

Output: OUT/model.gguf (single file, data aligned to general.alignment), written through mlx_ckpt.AlignedWriter (O_DIRECT, the
page cache is dropped), atomically (tmp, fsync, rename), and OUT/manifest.json (sha256 of every tensor's bytes, the sources,
the checkpoint revision and the template's). `--max-mbps` limits both the reads and the writes.
"""
import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from slotstream_spark.model import mlx_ckpt  # noqa: E402

GGML_F32, GGML_F16, GGML_Q4_0, GGML_Q4_1, GGML_Q5_1, GGML_Q8_0, GGML_Q6_K, GGML_BF16 = 0, 1, 2, 3, 7, 8, 14, 30
# type -> (elements per block, bytes per block)
BLOCK = {GGML_F32: (1, 4), GGML_F16: (1, 2), GGML_BF16: (1, 2), GGML_Q4_1: (32, 20), GGML_Q5_1: (32, 24), GGML_Q8_0: (32, 34),
         GGML_Q6_K: (256, 210)}
Q4_1_BYTES = 20
MIN_CORRELATION = 0.95          # one floor for every family (BRIEF 2026-10-15 01:25); --min-corr changes it
BAND_UPPER = 0.99               # the former `other` floor, now only the upper edge of the informative band [floor, 0.99)
HC_RE = re.compile(r"^(blk\.\d+\.hc_(attn|ffn)_(down|up)|output_hc_(down|up))\.weight$")
BF16_AFFINE_RE = HC_RE          # 4-bit MLX tensors written as dequantized BF16 instead of Q4_1 (DIAG_F9c.md)
EXCEPTIONS_PATH = ROOT / "slotstream_spark/config/q4_1_exceptions.json"
TOLERATED = {}                  # name -> {"count", "max_abs_err"}: values the named exceptions rounded (reset per build/sweep)
MAX_MEAN_OFFSET_SD = 0.05
ROW_CORR_LOW = 0.90             # sampled rows correlating below this with the template are counted and listed (real data: none)
ROW_CORR_FAIL = 0.50            # one sampled row below this is a layout defect, not quantization noise (real data: lowest 0.945)
DEFAULT_ALIGNMENT = 32
SPLIT_KEYS = ("split.no", "split.count", "split.tensors.count")
DROP_PREFIXES = ("split.", "quantize.imatrix.")
REVISION = "aa7c790e"


class ExactnessError(RuntimeError):
    pass


class MappingError(RuntimeError):
    pass


# ================================================================================================ numeric core
def bf16_to_f32(u16):
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def load_exceptions(path=EXCEPTIONS_PATH):
    """-> {tensor name: absolute tolerance} from the tracked exceptions file."""
    d = json.loads(Path(path).read_text())
    if float(d["abs_tol"]) != 2.0 ** int(d["abs_tol_pow2"]):
        raise ValueError(f"{path}: abs_tol {d['abs_tol']} is not 2^{d['abs_tol_pow2']}")
    return {n: float(d["abs_tol"]) for n in d["tensors"]}


def bf16_to_f16_checked(u16, name="?", exceptions=None):
    """bf16 bits -> fp16 values (uint16 bits). Raises ExactnessError naming `name` unless every value converts without change.

    A `name` listed in `exceptions` ({name: abs_tol}) is rounded to fp16 instead and must stay finite with a maximum absolute
    error of at most abs_tol; what was rounded is recorded in TOLERATED."""
    f32 = bf16_to_f32(u16)
    if not np.isfinite(f32).all():
        raise ExactnessError(f"{name}: non-finite value")
    with np.errstate(over="ignore"):
        f16 = f32.astype(np.float16)
    bad = f16.astype(np.float32) != f32
    tol = (exceptions or {}).get(name)
    if bad.any() and tol is not None:
        back = f16.astype(np.float32)
        err = np.abs(back.astype(np.float64) - f32.astype(np.float64))
        if not np.isfinite(back).all():
            raise ExactnessError(f"{name}: {int(bad.sum())} values exceed the tolerated 2^-24 (overflow to a non-finite fp16)")
        worst = float(err[bad].max())
        if worst > tol:
            raise ExactnessError(f"{name}: {int((err > tol).sum())} values exceed the tolerated 2^-24 (max abs error {worst:.3e})")
        rec = TOLERATED.setdefault(name, {"count": 0, "max_abs_err": 0.0})
        rec["count"] += int(bad.sum())
        rec["max_abs_err"] = max(rec["max_abs_err"], worst)
        return f16.view(np.uint16)
    if bad.any():
        v = f32[bad]
        raise ExactnessError(f"{name}: {int(bad.sum())} of {f32.size} values are not exact in fp16 "
                             f"(|v| outside [2^-17, 65504]; smallest {float(np.abs(v).min()):.3e}, largest {float(np.abs(v).max()):.3e})")
    return f16.view(np.uint16)


def corr_floor(name):
    """-> (correlation floor, family) for a template tensor: MIN_CORRELATION for every family; the label is informative."""
    return MIN_CORRELATION, "hyper_connection" if HC_RE.match(name) else "other"


def unpack_mlx(w_u32):
    """uint32 [R, K/8] -> uint8 [R, K]; element i of word j is at bits 4i of that word."""
    w = np.asarray(w_u32, dtype=np.uint32)
    shifts = np.arange(8, dtype=np.uint32) * 4
    return ((w[..., None] >> shifts) & np.uint32(0xF)).astype(np.uint8).reshape(w.shape[:-1] + (w.shape[-1] * 8,))


def pack_q4_1(q, scales_bf16, biases_bf16, group_size, name="?", exceptions=None):
    """q uint8 [R, K] (0..15), scales/biases bf16 bits [R, K/group_size] -> uint8 [R, K/32, 20] Q4_1 blocks."""
    R, K = q.shape
    if K % 32 or group_size % 32 or K % group_size:
        raise ValueError(f"{name}: K={K} group={group_size} do not split into Q4_1 blocks")
    per = group_size // 32
    d16 = bf16_to_f16_checked(scales_bf16, name + ".scales", exceptions)
    m16 = bf16_to_f16_checked(biases_bf16, name + ".biases", exceptions)
    d16 = np.repeat(d16, per, axis=1)                                  # [R, K/32]
    m16 = np.repeat(m16, per, axis=1)
    nb = K // 32
    blk = q.reshape(R, nb, 2, 16)                                      # [..., half, j]: element 16*half + j
    qs = (blk[:, :, 0, :] | (blk[:, :, 1, :] << 4)).astype(np.uint8)    # qs[j] = element j (low) | element j + 16 (high)
    out = np.empty((R, nb, Q4_1_BYTES), dtype=np.uint8)
    out[:, :, 0:2] = d16.astype("<u2").view(np.uint8).reshape(R, nb, 2)
    out[:, :, 2:4] = m16.astype("<u2").view(np.uint8).reshape(R, nb, 2)
    out[:, :, 4:] = qs
    return out


def unpack_q4_1(blocks):
    """uint8 [R, nb, 20] -> (q uint8 [R, 32 nb], d float32 [R, nb], m float32 [R, nb])."""
    b = np.asarray(blocks, dtype=np.uint8)
    R, nb, _ = b.shape
    d = np.ascontiguousarray(b[:, :, 0:2]).view("<f2").reshape(R, nb).astype(np.float32)
    m = np.ascontiguousarray(b[:, :, 2:4]).view("<f2").reshape(R, nb).astype(np.float32)
    qs = b[:, :, 4:]
    q = np.concatenate([qs & 0xF, qs >> 4], axis=2).reshape(R, nb * 32)
    return q, d, m


def dequant_q4_1(blocks):
    """ggml's dequantize_row_q4_1: y = q * d + m in fp32."""
    q, d, m = unpack_q4_1(blocks)
    R, K = q.shape
    return q.astype(np.float32) * np.repeat(d, 32, axis=1) + np.repeat(m, 32, axis=1)


def dequant_q5_1(raw, R, K):
    """ggml's dequantize_row_q5_1 (24-byte blocks: d fp16, m fp16, qh uint32, qs[16]): y = x * d + m with a 5-bit x."""
    nb = K // 32
    b = np.frombuffer(raw, dtype=np.uint8).reshape(R, nb, 24)
    d = np.ascontiguousarray(b[:, :, 0:2]).view("<f2").reshape(R, nb).astype(np.float32)
    m = np.ascontiguousarray(b[:, :, 2:4]).view("<f2").reshape(R, nb).astype(np.float32)
    qh = np.ascontiguousarray(b[:, :, 4:8]).view("<u4").reshape(R, nb)
    qs = b[:, :, 8:]
    j = np.arange(16, dtype=np.uint32)
    x0 = (qs & 0xF).astype(np.uint32) | (((qh[:, :, None] >> j) & 1) << 4)
    x1 = (qs >> 4).astype(np.uint32) | (((qh[:, :, None] >> (j + 16)) & 1) << 4)
    x = np.concatenate([x0, x1], axis=2).astype(np.float32)
    return (x * d[:, :, None] + m[:, :, None]).reshape(R, K)


def dequant_q8_0(raw, R, K):
    b = np.frombuffer(raw, dtype=np.uint8).reshape(R, K // 32, 34)
    d = np.ascontiguousarray(b[:, :, 0:2]).view("<f2").reshape(R, K // 32).astype(np.float32)
    qs = np.ascontiguousarray(b[:, :, 2:]).view(np.int8).astype(np.float32)
    return (qs * d[:, :, None]).reshape(R, K)


def dequant_q6_k(raw, R, K):
    """ggml's dequantize_row_q6_K (210-byte blocks of 256: ql[128], qh[64], scales int8[16], d fp16)."""
    nb = K // 256
    b = np.frombuffer(raw, dtype=np.uint8).reshape(R, nb, 210)
    ql, qh = b[:, :, 0:128], b[:, :, 128:192]
    sc = np.ascontiguousarray(b[:, :, 192:208]).view(np.int8).astype(np.float32)
    d = np.ascontiguousarray(b[:, :, 208:210]).view("<f2").reshape(R, nb).astype(np.float32)
    y = np.empty((R, nb, 256), dtype=np.float32)
    for half in range(2):                                              # two runs of 128 values
        qlh, qhh, sch = ql[:, :, 64 * half:64 * half + 64], qh[:, :, 32 * half:32 * half + 32], sc[:, :, 8 * half:8 * half + 8]
        l = np.arange(32)
        isx = l // 16
        q1 = ((qlh[:, :, 0:32] & 0xF) | (((qhh >> 0) & 3) << 4)).astype(np.int8).astype(np.int32) - 32
        q2 = ((qlh[:, :, 32:64] & 0xF) | (((qhh >> 2) & 3) << 4)).astype(np.int8).astype(np.int32) - 32
        q3 = ((qlh[:, :, 0:32] >> 4) | (((qhh >> 4) & 3) << 4)).astype(np.int8).astype(np.int32) - 32
        q4 = ((qlh[:, :, 32:64] >> 4) | (((qhh >> 6) & 3) << 4)).astype(np.int8).astype(np.int32) - 32
        base = 128 * half
        y[:, :, base + 0:base + 32] = d[:, :, None] * sch[:, :, isx + 0] * q1
        y[:, :, base + 32:base + 64] = d[:, :, None] * sch[:, :, isx + 2] * q2
        y[:, :, base + 64:base + 96] = d[:, :, None] * sch[:, :, isx + 4] * q3
        y[:, :, base + 96:base + 128] = d[:, :, None] * sch[:, :, isx + 6] * q4
    return y.reshape(R, K)


def dequant_ggml(gtype, raw, R, K):
    """Float32 [R, K] of `R` rows of ggml type `gtype` and length `K` held in `raw`."""
    if gtype == GGML_F32:
        return np.frombuffer(raw, dtype="<f4").reshape(R, K).astype(np.float32)
    if gtype == GGML_F16:
        return np.frombuffer(raw, dtype="<f2").reshape(R, K).astype(np.float32)
    if gtype == GGML_BF16:
        return bf16_to_f32(np.frombuffer(raw, dtype="<u2")).reshape(R, K)
    if gtype == GGML_Q4_1:
        return dequant_q4_1(np.frombuffer(raw, dtype=np.uint8).reshape(R, K // 32, Q4_1_BYTES))
    if gtype == GGML_Q5_1:
        return dequant_q5_1(raw, R, K)
    if gtype == GGML_Q8_0:
        return dequant_q8_0(raw, R, K)
    if gtype == GGML_Q6_K:
        return dequant_q6_k(raw, R, K)
    raise MappingError(f"ggml type {gtype} is not supported by the sanity check")


def row_bytes(gtype, ne0):
    elems, nbytes = BLOCK[gtype]
    if ne0 % elems:
        raise MappingError(f"row length {ne0} is not a multiple of the block size {elems} of ggml type {gtype}")
    return ne0 // elems * nbytes


def v_reorder_index(n, num_k_heads, num_v_per_k, head_dim):
    """new[i] = old[idx[i]]: V heads from grouped-by-K-head order to tiled order (converter `_reorder_v_heads`)."""
    return np.arange(n).reshape(num_k_heads, num_v_per_k, head_dim).transpose(1, 0, 2).reshape(n)


def group_perm_from_cols(col_perm, group_size):
    """The permutation of whole groups that a column permutation made of group-aligned runs amounts to."""
    cp = np.asarray(col_perm).reshape(-1, group_size)
    starts = cp[:, 0]
    if not (cp == starts[:, None] + np.arange(group_size)).all() or (starts % group_size).any():
        raise MappingError("the column permutation does not move whole quantization groups")
    return starts // group_size


# ================================================================================================ GGUF
@dataclass
class GgufTensor:
    name: str
    ne: tuple
    gtype: int
    offset: int                      # absolute offset of the data in `file`
    file: str


@dataclass
class GgufFile:
    kv: list = field(default_factory=list)          # [(key, raw bytes of the whole KV record)]
    tensors: list = field(default_factory=list)
    alignment: int = DEFAULT_ALIGNMENT
    values: dict = field(default_factory=dict)      # key -> python value for the scalar / small array records


_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class _Reader:
    def __init__(self, f):
        self.f = f

    def take(self, fmt):
        n = struct.calcsize(fmt)
        return struct.unpack(fmt, self.f.read(n))[0]

    def string(self):
        n = self.take("<Q")
        return self.f.read(n).decode("utf-8", errors="replace")

    def value(self, t):
        if t == 8:
            return self.string()
        if t == 9:
            et, n = self.take("<I"), self.take("<Q")
            if et in _SCALAR:
                size = struct.calcsize(_SCALAR[et])
                raw = self.f.read(size * n)
                return list(struct.unpack("<" + _SCALAR[et][1] * n, raw)) if n <= 4096 else ("array", et, n)
            return [self.value(et) for _ in range(n)] if n <= 4096 else self._skip_strings(n)
        return self.take(_SCALAR[t])

    def _skip_strings(self, n):
        for _ in range(n):
            self.f.seek(self.take("<Q"), 1)
        return ("array", 8, n)


def read_gguf(paths):
    """Parse the header of one or more GGUF shards: every KV record as raw bytes, the tensors with absolute data offsets."""
    out = GgufFile()
    for i, p in enumerate(paths):
        with open(p, "rb") as f:
            if f.read(4) != b"GGUF":
                raise MappingError(f"{p}: not a GGUF file")
            r = _Reader(f)
            version, nt, nk = r.take("<I"), r.take("<Q"), r.take("<Q")
            if version != 3:
                raise MappingError(f"{p}: GGUF version {version}, expected 3")
            for _ in range(nk):
                start = f.tell()
                key, t = r.string(), r.take("<I")
                val = r.value(t)
                end = f.tell()
                if i == 0:
                    f.seek(start)
                    out.kv.append((key, f.read(end - start)))
                    out.values[key] = val
            infos = []
            for _ in range(nt):
                name, nd = r.string(), r.take("<I")
                ne = tuple(r.take("<Q") for _ in range(nd))
                gtype, off = r.take("<I"), r.take("<Q")
                infos.append((name, ne, gtype, off))
            align = int(out.values.get("general.alignment", DEFAULT_ALIGNMENT)) if i == 0 else out.alignment
            if i == 0:
                out.alignment = align
            base = (f.tell() + align - 1) // align * align
            for name, ne, gtype, off in infos:
                out.tensors.append(GgufTensor(name, ne, gtype, base + off, str(p)))
    return out


def template_paths(template):
    t = Path(template)
    if t.is_dir():
        found = sorted(t.glob("*.gguf"))
    else:
        m = re.match(r"(.*)-(\d+)-of-(\d+)\.gguf$", t.name)
        found = sorted(t.parent.glob(f"{m.group(1)}-*-of-{m.group(3)}.gguf")) if m else [t]
    if not found:
        raise MappingError(f"no GGUF under {template}")
    return found


def _gstr(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def gguf_header(kv_records, tensors, alignment):
    """Header bytes: magic, version 3, counts, KV records, tensor infos (offsets relative to the data start), padding."""
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv_records)))
    for _key, raw in kv_records:
        out += raw
    off = 0
    for name, ne, gtype, nbytes in tensors:
        out += _gstr(name) + struct.pack("<I", len(ne)) + b"".join(struct.pack("<Q", d) for d in ne)
        out += struct.pack("<IQ", gtype, off)
        off += (nbytes + alignment - 1) // alignment * alignment
    out += b"\0" * (-len(out) % alignment)
    return bytes(out)


def kv_record_u32(key, value):
    return key, _gstr(key) + struct.pack("<II", 4, value)


# ================================================================================================ sources
class Source:
    """Cached whole-tensor reads from the MLX checkpoint (every tensor is < 1 GiB)."""

    def __init__(self, ckpt, throttle=None):
        self.ckpt, self.throttle, self._cache = ckpt, throttle, {}
        self.exceptions = None          # {name: abs_tol} of the named fp16 exceptions; run_build sets it

    def get(self, name):
        if name not in self._cache:
            if len(self._cache) > 6:
                self._cache.clear()
            try:
                self.ckpt.info(name)
            except KeyError:
                raise MappingError(f"{name}: not in the checkpoint")
            a = self.ckpt.read_array(name)
            if self.throttle:
                self.throttle.spend(a.nbytes)
            self._cache[name] = a
        return self._cache[name]


class Spec:
    """One output tensor: `ne` and `gtype` as the template says, `nrows` rows of `rbytes` bytes, `rows(i0, i1)` -> uint8 [n, rbytes]."""
    sources = ()
    kind = ""
    sample_k = 24                    # rows compared with the template

    def __init__(self, name, ne, gtype):
        self.name, self.ne, self.gtype = name, tuple(ne), gtype
        self.rbytes = row_bytes(gtype, self.ne[0])
        self.nrows = int(np.prod(self.ne[1:], dtype=np.int64)) if len(self.ne) > 1 else 1

    @property
    def nbytes(self):
        return self.rbytes * self.nrows

    def rows(self, i0, i1):
        raise NotImplementedError


class AffineSpec(Spec):
    """MLX affine 4-bit `base.{weight,scales,biases}` -> Q4_1. Leading dimensions are flattened into rows."""
    kind = "q4_1"

    def __init__(self, name, ne, src, base, row_idx=None, col_perm=None):
        super().__init__(name, ne, GGML_Q4_1)
        self.src, self.base, self.row_idx, self.col_perm = src, base, row_idx, col_perm
        self.sources = (f"{base}.weight", f"{base}.scales", f"{base}.biases")

    def _arrays(self):
        w, s, b = (self.src.get(n) for n in self.sources)
        K = w.shape[-1] * 8
        w, s, b = w.reshape(-1, w.shape[-1]), s.reshape(-1, s.shape[-1]), b.reshape(-1, b.shape[-1])
        if K != self.ne[0]:
            raise MappingError(f"{self.name}: source rows hold {K} elements, the template says {self.ne[0]}")
        if w.shape[0] != self.nrows and self.row_idx is None:
            raise MappingError(f"{self.name}: source has {w.shape[0]} rows, the template says {self.nrows}")
        return w, s, b, K

    def rows(self, i0, i1):
        w, s, b, K = self._arrays()
        gs = K // s.shape[-1]
        idx = np.arange(i0, i1) if self.row_idx is None else self.row_idx[i0:i1]
        q = unpack_mlx(w[idx])
        sc, bi = s[idx], b[idx]
        if self.col_perm is not None:
            gp = group_perm_from_cols(self.col_perm, gs)
            q = q[:, self.col_perm]
            sc, bi = sc[:, gp], bi[:, gp]
        return pack_q4_1(q, sc, bi, gs, self.base, getattr(self.src, "exceptions", None)).reshape(i1 - i0, -1)


def f32_to_bf16_bits(a):
    """float32 -> bf16 bits (uint16), round to nearest even (finite values)."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    return ((u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))) >> 16).astype(np.uint16)


class AffineBf16Spec(AffineSpec):
    """MLX affine 4-bit `base.{weight,scales,biases}` written as BF16: y = q * scale + bias in fp32 (the scale and bias are bf16
    values, so nothing is rounded before the product), then rounded to bf16. No fp16 exactness limit applies."""
    kind = "affine_bf16"

    def __init__(self, name, ne, src, base, row_idx=None, col_perm=None):
        super().__init__(name, ne, src, base, row_idx=row_idx, col_perm=col_perm)
        self.gtype = GGML_BF16
        self.rbytes = row_bytes(self.gtype, self.ne[0])

    def rows(self, i0, i1):
        w, s, b, K = self._arrays()
        gs = K // s.shape[-1]
        idx = np.arange(i0, i1) if self.row_idx is None else self.row_idx[i0:i1]
        q = unpack_mlx(w[idx]).astype(np.float32)
        sc, bi = bf16_to_f32(s[idx]), bf16_to_f32(b[idx])
        if self.col_perm is not None:
            gp = group_perm_from_cols(self.col_perm, gs)
            q = q[:, self.col_perm]
            sc, bi = sc[:, gp], bi[:, gp]
        y = q * np.repeat(sc, gs, axis=1) + np.repeat(bi, gs, axis=1)
        if not np.isfinite(y).all():
            raise ExactnessError(f"{self.base}: non-finite dequantized value")
        return f32_to_bf16_bits(y).astype("<u2").view(np.uint8).reshape(i1 - i0, self.rbytes)


class NgramSpec(Spec):
    """The 128 n-gram shards (gs32, 160 values per row) concatenated in shard order."""
    kind = "q4_1"

    def __init__(self, name, ne, src, pattern, n_shards):
        super().__init__(name, ne, GGML_Q4_1)
        self.sample_k = 6            # every sampled row can come from another 225 MB shard
        self.src, self.pattern, self.n = src, pattern, n_shards
        self.sources = tuple(pattern.format(i=i) + sfx for i in range(n_shards) for sfx in (".weight", ".scales", ".biases"))
        self.bounds = None

    def _bounds(self):
        if self.bounds is None:
            rows = [self.src.ckpt.info(self.pattern.format(i=i) + ".weight").shape[0] for i in range(self.n)]
            self.bounds = np.concatenate([[0], np.cumsum(rows)])
            if int(self.bounds[-1]) != self.nrows:
                raise MappingError(f"{self.name}: the shards hold {int(self.bounds[-1])} rows, the template says {self.nrows}")
        return self.bounds

    def rows(self, i0, i1):
        bd = self._bounds()
        parts = []
        i = i0
        while i < i1:
            k = int(np.searchsorted(bd, i, side="right")) - 1
            j = min(i1, int(bd[k + 1]))
            base = self.pattern.format(i=k)
            w, s, b = (self.src.get(base + sfx) for sfx in (".weight", ".scales", ".biases"))
            a, z = i - int(bd[k]), j - int(bd[k])
            K = w.shape[-1] * 8
            parts.append(pack_q4_1(unpack_mlx(w[a:z]), s[a:z], b[a:z], K // s.shape[-1], base,
                                   getattr(self.src, "exceptions", None)).reshape(z - a, -1))
            i = j
        return np.concatenate(parts, 0)


class DenseSpec(Spec):
    """A tensor that is not 4-bit in MLX: bf16 (or int) data through `fn` (float32 -> float32), written as F32 / F16 / BF16.

    A template tensor of type F16 whose values are not all exact in fp16 is written as F32 instead (exact for any bf16 value);
    `override` names that change and the manifest records it."""
    kind = "dense"

    def __init__(self, name, ne, gtype, src, names, fn=None):
        super().__init__(name, ne, gtype)
        self.src, self.names, self.fn = src, tuple(names), fn
        self.sources = self.names
        self.override = None
        self._out = None
        if gtype == GGML_F16:
            self._array()

    def _values(self):
        raw = self.src.get(self.names[0])
        f32 = bf16_to_f32(raw) if raw.dtype == np.uint16 else raw.astype(np.float32)
        a = f32 if self.fn is None else self.fn(f32)
        shape = tuple(reversed(self.ne))
        if a.size != int(np.prod(shape, dtype=np.int64)):
            raise MappingError(f"{self.name}: {a.size} values, the template's shape {shape}")
        return raw, np.ascontiguousarray(a.reshape(shape)), shape

    def _array(self):
        if self._out is None:
            raw, a, shape = self._values()
            if self.gtype == GGML_F16:
                with np.errstate(over="ignore"):
                    h = a.astype(np.float16)
                if (h.astype(np.float32) != a).any():
                    self.override = "template F16, written F32: the bf16 values are not all exact in fp16"
                    self.gtype = GGML_F32
                    self.rbytes = row_bytes(self.gtype, self.ne[0])
                else:
                    out = h
            if self.gtype == GGML_F32:
                out = a.astype("<f4")
            elif self.gtype == GGML_BF16:
                if self.fn is not None or raw.dtype != np.uint16:
                    raise MappingError(f"{self.name}: BF16 output needs the bf16 source untouched")
                out = np.ascontiguousarray(raw.reshape(shape))
            elif self.gtype != GGML_F16:
                raise MappingError(f"{self.name}: dense output type {self.gtype}")
            self._out = out.reshape(self.nrows, -1).view(np.uint8).reshape(self.nrows, self.rbytes)
        return self._out

    def rows(self, i0, i1):
        return self._array()[i0:i1]


# ================================================================================================ rules
def _hp(cfg):
    return {"nk": int(cfg["linear_num_key_heads"]), "nv": int(cfg["linear_num_value_heads"]),
            "dk": int(cfg["linear_key_head_dim"]), "dv": int(cfg["linear_value_head_dim"]),
            "n_idx_q": int(cfg["indexer_n_heads"]) * int(cfg["indexer_head_dim"])}


def make_rules(src, cfg):
    """[(regex on the GGUF tensor name, builder(match, template tensor) -> Spec)]. Layer prefix: language_model.model.layers.N."""
    hp = _hp(cfg)
    nk, nv, dk, dv = hp["nk"], hp["nv"], hp["dk"], hp["dv"]
    npk = nv // nk
    lm = "language_model.model."
    qk_rows = 2 * nk * dk

    def aff(base, **kw):
        return lambda m, t: AffineSpec(t.name, t.ne, src, base.format(*m.groups()), **kw)

    def dense(srcname, fn=None):
        return lambda m, t: DenseSpec(t.name, t.ne, t.gtype, src, [srcname.format(*m.groups())], fn)

    def qkv(m, t):
        n = nv * dv
        idx = np.concatenate([np.arange(qk_rows), qk_rows + v_reorder_index(n, nk, npk, dv)])
        return AffineSpec(t.name, t.ne, src, f"{lm}layers.{m.group(1)}.linear_attn.in_proj_qkv", row_idx=idx)

    def zproj(m, t):
        return AffineSpec(t.name, t.ne, src, f"{lm}layers.{m.group(1)}.linear_attn.in_proj_z",
                          row_idx=v_reorder_index(nv * dv, nk, npk, dv))

    def outproj(m, t):
        return AffineSpec(t.name, t.ne, src, f"{lm}layers.{m.group(1)}.linear_attn.out_proj",
                          col_perm=v_reorder_index(nv * dv, nk, npk, dv))

    def reorder_rows():
        return lambda a: a.reshape(nv, -1)[v_reorder_index(nv, nk, npk, 1)]

    def reorder_vec(a):
        return a[v_reorder_index(nv, nk, npk, 1)]

    def conv(a):
        a = a.reshape(a.shape[0], -1)                                   # [C, 4, 1] -> [C, 4]
        out = a.copy()
        out[qk_rows:] = a[qk_rows:][v_reorder_index(nv * dv, nk, npk, dv)]
        return out

    def ple_conv(a):
        return a.reshape(a.shape[0], -1)

    def index_split(which):
        def build(m, t):
            base = f"{lm}layers.{m.group(1)}.self_attn.indexer.index_qk_proj.weight"
            raw = src.get(base)
            part = raw[:hp["n_idx_q"]] if which == "q" else raw[hp["n_idx_q"]:]
            return DenseSpec(t.name, t.ne, t.gtype, _Literal(base, part), [base])
        return build

    lay = lm + "layers.{}."
    rules = [
        (r"token_embd\.weight$", lambda m, t: AffineSpec(t.name, t.ne, src, lm + "embed_tokens")),
        (r"output\.weight$", lambda m, t: AffineSpec(t.name, t.ne, src, "language_model.lm_head")),
        (r"output_hc_down\.weight$", lambda m, t: AffineBf16Spec(t.name, t.ne, src, lm + "hyper_connection_mixer.input_mix_weight_down")),
        (r"output_hc_up\.weight$", lambda m, t: AffineBf16Spec(t.name, t.ne, src, lm + "hyper_connection_mixer.input_mix_weight_up")),
        (r"output_hc_norm\.weight$", dense(lm + "hyper_connection_mixer.hc_norm.weight")),
        (r"per_layer_token_embd\.weight$",
         lambda m, t: NgramSpec(t.name, t.ne, src, lm + "layers.1.ple.ple_embedding.ngram_embedding.shard_{i}", 128)),
        (r"blk\.(\d+)\.attn_qkv\.weight$", qkv),
        (r"blk\.(\d+)\.attn_gate\.weight$", zproj),
        (r"blk\.(\d+)\.ssm_out\.weight$", outproj),
        (r"blk\.(\d+)\.ssm_alpha\.weight$", dense(lay + "linear_attn.in_proj_a.weight", reorder_rows())),
        (r"blk\.(\d+)\.ssm_beta\.weight$", dense(lay + "linear_attn.in_proj_b.weight", reorder_rows())),
        (r"blk\.(\d+)\.ssm_a$", dense(lay + "linear_attn.A_log", lambda a: -np.exp(reorder_vec(a)))),
        (r"blk\.(\d+)\.ssm_dt\.bias$", dense(lay + "linear_attn.dt_bias", reorder_vec)),
        (r"blk\.(\d+)\.ssm_conv1d\.weight$", dense(lay + "linear_attn.conv1d.weight", conv)),
        (r"blk\.(\d+)\.ssm_norm\.weight$", dense(lay + "linear_attn.norm.weight")),
        (r"blk\.(\d+)\.attn_q\.weight$", aff(lay + "self_attn.q_proj")),
        (r"blk\.(\d+)\.attn_k\.weight$", aff(lay + "self_attn.k_proj")),
        (r"blk\.(\d+)\.attn_v\.weight$", aff(lay + "self_attn.v_proj")),
        (r"blk\.(\d+)\.attn_output\.weight$", aff(lay + "self_attn.o_proj")),
        (r"blk\.(\d+)\.attn_q_norm\.weight$", dense(lay + "self_attn.q_norm.weight")),
        (r"blk\.(\d+)\.attn_k_norm\.weight$", dense(lay + "self_attn.k_norm.weight")),
        (r"blk\.(\d+)\.indexer\.q_proj\.weight$", index_split("q")),
        (r"blk\.(\d+)\.indexer\.k_proj\.weight$", index_split("k")),
        (r"blk\.(\d+)\.indexer\.q_norm\.weight$", dense(lay + "self_attn.indexer.q_layernorm.weight")),
        (r"blk\.(\d+)\.indexer\.k_norm\.weight$", dense(lay + "self_attn.indexer.k_layernorm.weight")),
        (r"blk\.(\d+)\.ffn_gate_inp\.weight$", dense(lay + "mlp.gate.weight")),
        (r"blk\.(\d+)\.ffn_gate_inp_shexp\.weight$", dense(lay + "mlp.shared_expert_gate.weight")),
        (r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", lambda m, t: AffineSpec(t.name, t.ne, src, f"{lm}layers.{m.group(1)}.mlp.switch_mlp.{m.group(2)}_proj")),
        (r"blk\.(\d+)\.ffn_(gate|up|down)_shexp\.weight$", lambda m, t: AffineSpec(t.name, t.ne, src, f"{lm}layers.{m.group(1)}.mlp.shared_expert.{m.group(2)}_proj")),
        (r"blk\.(\d+)\.hc_(attn|ffn)_(down|up)\.weight$", lambda m, t: AffineBf16Spec(
            t.name, t.ne, src, f"{lm}layers.{m.group(1)}.{'attn' if m.group(2) == 'attn' else 'mlp'}_hyper_connection.input_mix_weight_{m.group(3)}")),
        (r"blk\.(\d+)\.hc_(attn|ffn)_inject\.weight$", lambda m, t: DenseSpec(
            t.name, t.ne, t.gtype, src, [f"{lm}layers.{m.group(1)}.{'attn' if m.group(2) == 'attn' else 'mlp'}_hyper_connection.block_inject_weight.weight"])),
        (r"blk\.(\d+)\.hc_(attn|ffn)_norm\.weight$", lambda m, t: DenseSpec(
            t.name, t.ne, t.gtype, src, [f"{lm}layers.{m.group(1)}.{'attn' if m.group(2) == 'attn' else 'mlp'}_hyper_connection.hc_norm.weight"])),
        (r"blk\.(\d+)\.ple_conv1d\.weight$", dense(lay + "ple.conv1d.weight", ple_conv)),
        (r"blk\.(\d+)\.ple_key\.weight$", aff(lay + "ple.key_proj")),
        (r"blk\.(\d+)\.ple_value\.weight$", aff(lay + "ple.value_proj")),
        (r"blk\.(\d+)\.ple_norm_(conv|key|query)\.weight$", lambda m, t: DenseSpec(
            t.name, t.ne, t.gtype, src, [f"{lm}layers.{m.group(1)}.ple.norm_{m.group(2)}.weight"])),
    ]
    return rules


class _Literal:
    """A Source stand-in for a slice of a tensor already read (the indexer split)."""

    def __init__(self, name, arr):
        self.name, self.arr = name, arr

    def get(self, name):
        return self.arr


def plan_tensors(template, rules, src):
    """One Spec per template tensor in the template's order. A tensor with no rule is refused; so is a shape the rule cannot fill."""
    specs, missing = [], []
    for t in template.tensors:
        for rx, build in rules:
            m = re.match(rx, t.name)
            if m:
                specs.append(build(m, t))
                break
        else:
            missing.append(t.name)
    if missing:
        raise MappingError(f"{len(missing)} template tensors have no mapping rule, refused: {missing[:8]}")
    return specs


# ================================================================================================ checks
def sample_rows(n, k=24):
    if n <= k:
        return list(range(n))
    return sorted({0, n - 1} | {int(i * (n - 1) / (k - 1)) for i in range(k)})


def read_template_rows(t, rows):
    """Float32 [len(rows), ne0] of rows of a template tensor."""
    rb = row_bytes(t.gtype, t.ne[0])
    out = []
    with open(t.file, "rb") as f:
        for r in rows:
            f.seek(t.offset + r * rb)
            out.append(dequant_ggml(t.gtype, f.read(rb), 1, t.ne[0])[0])
    return np.stack(out)


def row_correlations(ours, theirs):
    """Per-row Pearson correlation of [n, K] arrays -> {"row_corr_min", "row_corr_p05", "rows_below_090", "rows_below_050", "rows_scored"}.
    A pooled correlation is dominated by the rows with the largest variance; one mis-mapped head or expert among many rows
    barely moves it, but its rows correlate near 0 one by one. Constant rows (either side) are not scored."""
    a = np.asarray(ours, dtype=np.float64).reshape(len(ours), -1)
    b = np.asarray(theirs, dtype=np.float64).reshape(len(theirs), -1)
    if a.shape[1] < 8:
        return {"row_corr_min": None, "row_corr_p05": None, "rows_below_090": 0, "rows_below_050": 0, "rows_scored": 0}
    a, b = a - a.mean(1, keepdims=True), b - b.mean(1, keepdims=True)
    den = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    ok = den > 0
    if not ok.any():
        return {"row_corr_min": None, "row_corr_p05": None, "rows_below_090": 0, "rows_below_050": 0, "rows_scored": 0}
    c = (a[ok] * b[ok]).sum(1) / den[ok]
    return {"row_corr_min": float(c.min()), "row_corr_p05": float(np.percentile(c, 5)),
            "rows_below_090": int((c < ROW_CORR_LOW).sum()), "rows_below_050": int((c < ROW_CORR_FAIL).sum()),
            "rows_scored": int(ok.sum())}


def check_against_template(spec, t, rows=None, strict=True):
    """Pearson correlation >= the floor (corr_floor) on sampled rows; for non-4-bit tensors also the mean offset.
    -> dict of the measurements, with `corr_floor`, `family`, `ok` and `reason`; strict=True raises MappingError(reason) instead."""
    nrows = spec.nrows
    idx = sample_rows(nrows, spec.sample_k) if rows is None else rows
    ours_bytes = np.concatenate([spec.rows(i, i + 1) for i in idx], 0)
    ours = dequant_ggml(spec.gtype, ours_bytes.tobytes(), len(idx), spec.ne[0])
    theirs = read_template_rows(t, idx)
    a, b = ours.reshape(-1).astype(np.float64), theirs.reshape(-1).astype(np.float64)
    sd = float(b.std())
    out = {"rows": len(idx), "mean_offset": float(a.mean() - b.mean()), "template_sd": sd}
    out.update(row_correlations(ours, theirs))
    if sd == 0 or float(a.std()) == 0:
        corr = 1.0 if np.allclose(a, b, atol=1e-6) else 0.0
    else:
        corr = float(np.corrcoef(a, b)[0, 1])
    floor, family = corr_floor(spec.name)
    out["corr"], out["corr_floor"], out["family"] = corr, floor, family
    out["ok"], out["reason"] = True, None
    if not corr >= floor:
        out["ok"], out["reason"] = False, (f"{spec.name}: correlation {corr:.4f} with the template is below {floor} (family {family}) "
                                           f"(sources {list(spec.sources)[:3]})")
    elif out["rows_below_050"] > 0:
        out["ok"], out["reason"] = False, (f"{spec.name}: {out['rows_below_050']} of {out['rows_scored']} sampled rows correlate below "
                                           f"{ROW_CORR_FAIL} with the template (lowest {out['row_corr_min']:.4f}) although the pooled correlation "
                                           f"is {corr:.4f}: part of the tensor is laid out differently (sources {list(spec.sources)[:3]})")
    elif spec.kind == "dense" and abs(out["mean_offset"]) > MAX_MEAN_OFFSET_SD * max(sd, 1e-3):
        out["ok"], out["reason"] = False, (f"{spec.name}: mean offset {out['mean_offset']:.4f} from the template exceeds {MAX_MEAN_OFFSET_SD} x sd "
                                           f"{sd:.4f} (a missing or doubled +1 looks like this)")
    if strict and not out["ok"]:
        raise MappingError(out["reason"])
    return out


def check_hash_constants(src, template, ckpt):
    """The int64 PLE constants of the checkpoint equal the template's KV arrays (exact, never through float)."""
    tail = "language_model.model.layers.1.ple.ple_embedding."
    pairs = {"qwen4exp.ple.layer_multipliers": tail + "layer_multipliers",
             "qwen4exp.ple.head_offsets": tail + "ngram_heads_offsets",
             "qwen4exp.ple.head_vocab_sizes": tail + "ngram_heads_vocab_sizes"}
    for key, tname in pairs.items():
        if key not in template.values:
            continue
        got = [int(x) for x in ckpt.read_array(tname).reshape(-1)]
        want = template.values[key]
        if isinstance(want, tuple):                                    # long array kept unparsed: compare through the raw record
            continue
        if got != [int(x) for x in want]:
            raise MappingError(f"{key}: the checkpoint's int64 constants differ from the template's")
    return True


# ================================================================================================ sweep
def sweep(ckpt, throttle=None, log=print, exceptions=None):
    """fp16 exactness of every tensor of the checkpoint (nothing is written). -> dict; `bad` lists the tensors that are not exact,
    `tolerated` the named exceptions that were not exact and stayed within their tolerance (`exceptions` None = the tracked file)."""
    if exceptions is None:
        exceptions = load_exceptions()
    TOLERATED.clear()
    names = sorted(ckpt.tensors())
    bad, tolerated, n_sb, n_other = [], [], 0, 0
    for i, name in enumerate(names):
        info = ckpt.info(name)
        a = ckpt.read_array(name)
        if throttle:
            throttle.spend(a.nbytes)
        if name.endswith((".scales", ".biases")) and info.dtype == "BF16":
            n_sb += 1
            try:
                bf16_to_f16_checked(a, name, exceptions)
            except ExactnessError as e:
                bad.append(str(e))
            if name in TOLERATED:
                tolerated.append({"name": name, **TOLERATED[name]})
        else:
            n_other += 1
            if info.dtype == "BF16" and not np.isfinite(bf16_to_f32(a)).all():
                bad.append(f"{name}: non-finite value")
        if (i + 1) % 400 == 0:
            log(f"sweep: {i + 1}/{len(names)} tensors")
    return {"tensors": len(names), "scales_biases": n_sb, "other": n_other, "bad": bad, "exact": not bad,
            "tolerated": tolerated, "exceptions_file": str(EXCEPTIONS_PATH.relative_to(ROOT)) if EXCEPTIONS_PATH.is_relative_to(ROOT) else str(EXCEPTIONS_PATH),
            "abs_tol": max(exceptions.values()) if exceptions else None}


# ================================================================================================ build
def _throttle(mbps):
    from repack_experts import Throttle
    return Throttle(mbps)


def run_build(model, template_path, out_dir, *, max_mbps=2000, allow_buffered=False, rules_fn=None, o_direct=True, log=print,
              cfg=None, chunk_bytes=256 << 20, write=True, exceptions=None):
    if exceptions is None:
        exceptions = load_exceptions()
    TOLERATED.clear()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = mlx_ckpt.Checkpoint(model, allow_buffered=allow_buffered)
    try:
        throttle = _throttle(max_mbps)
        src = Source(ckpt, throttle)
        src.exceptions = exceptions
        tpaths = template_paths(template_path) if not isinstance(template_path, (list, tuple)) else list(template_path)
        template = read_gguf(tpaths)
        if cfg is None:
            from slotstream_spark.model.config import TextConfig
            cfg = TextConfig.from_checkpoint(model)
        rules = (rules_fn or make_rules)(src, cfg)
        specs = plan_tensors(template, rules, src)
        by_name = {t.name: t for t in template.tensors}
        for s in specs:
            t = by_name[s.name]
            if tuple(s.ne) != tuple(t.ne):
                raise MappingError(f"{s.name}: shape {s.ne} differs from the template's {t.ne}")
        check_hash_constants(src, template, ckpt)
        kv = [(k, raw) for k, raw in template.kv if not k.startswith(DROP_PREFIXES)]
        align = template.alignment
        header = gguf_header(kv, [(s.name, s.ne, s.gtype, s.nbytes) for s in specs], align)
        tmp, final = out_dir / "model.gguf.tmp", out_dir / "model.gguf"
        log(f"{'building' if write else 'checking'} {len(specs)} tensors, {sum(s.nbytes for s in specs) / 1e9:.1f} GB -> {final}")
        manifest = {"version": 1, "tool": "tools/repack_q4_1.py", "checkpoint_revision": REVISION,
                    "template": {"files": [Path(p).name for p in tpaths]}, "alignment": align, "written": bool(write),
                    "tensors": []}
        failures = []
        written = 0
        t0 = time.time()
        w = mlx_ckpt.AlignedWriter(tmp, o_direct=o_direct) if write else None
        try:
            if w:
                w.write(header)
            for n, s in enumerate(specs):
                kept = {k: dict(v) for k, v in TOLERATED.items()}
                chk = check_against_template(s, by_name[s.name], strict=write)
                if not chk["ok"]:
                    failures.append({"name": s.name, "corr": round(chk["corr"], 6), "corr_floor": chk["corr_floor"],
                                     "family": chk["family"], "reason": chk["reason"], "sources": list(s.sources)[:3]})
                if w:                                   # the sampled rows are read again by the write pass: count that one only
                    TOLERATED.clear()
                    TOLERATED.update(kept)
                h = hashlib.sha256()
                if w:
                    per = max(1, chunk_bytes // max(s.rbytes, 1))
                    for i0 in range(0, s.nrows, per):
                        i1 = min(s.nrows, i0 + per)
                        data = np.ascontiguousarray(s.rows(i0, i1)).tobytes()
                        h.update(data)
                        w.write(data)
                        throttle.spend(len(data))
                    pad = -s.nbytes % align
                    if pad:
                        w.write(b"\0" * pad)
                written += s.nbytes
                manifest["tensors"].append({"name": s.name, "ne": list(s.ne), "ggml_type": s.gtype, "nbytes": s.nbytes,
                                            "sha256": h.hexdigest() if w else None,
                                            "sources": list(s.sources)[:3] + (["..."] if len(s.sources) > 3 else []),
                                            "n_sources": len(s.sources), "corr": round(chk["corr"], 6), "corr_floor": chk["corr_floor"], "family": chk["family"],
                                            "row_corr_min": chk["row_corr_min"], "row_corr_p05": chk["row_corr_p05"],
                                            "rows_below_090": chk["rows_below_090"], "rows_below_050": chk["rows_below_050"], "rows_scored": chk["rows_scored"],
                                            "type_override": getattr(s, "override", None),
                                            "mean_offset": round(chk["mean_offset"], 6)})
                if (n + 1) % 100 == 0 or n + 1 == len(specs):
                    log(f"{n + 1}/{len(specs)} tensors, {written / 1e9:.1f} GB, {time.time() - t0:.0f} s")
        finally:
            if w:
                w.close()
        manifest["exceptions"] = [{"name": n, **v} for n, v in sorted(TOLERATED.items())]
        manifest["type_overrides"] = [t["name"] for t in manifest["tensors"] if t["type_override"]]
        manifest["dequantized_bf16"] = [s.name for s in specs if getattr(s, "kind", "") == "affine_bf16"]
        manifest["corr_floor"] = MIN_CORRELATION
        manifest["failures"] = failures
        manifest["below_099"] = sorted(({"name": x["name"], "corr": x["corr"], "family": x["family"]}
                                        for x in manifest["tensors"] if MIN_CORRELATION <= x["corr"] < BAND_UPPER),
                                       key=lambda x: x["name"])
        if write:
            os.replace(tmp, final)
            manifest["file_bytes"] = final.stat().st_size
            (out_dir / "manifest.json.tmp").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
            os.replace(out_dir / "manifest.json.tmp", out_dir / "manifest.json")
        else:
            manifest["file_bytes"] = None
            (out_dir / "check.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
            if failures:
                raise MappingError(f"{len(failures)} tensors under the {MIN_CORRELATION} floor: "
                                   + "; ".join(f"{f['name']} corr {f['corr']:.4f} ({f['family']})" for f in failures))
        return manifest
    finally:
        ckpt.close()


def dump_template(template_path):
    t = read_gguf(template_paths(template_path))
    for k, raw in t.kv:
        v = t.values.get(k)
        print("KV", k, (v if not isinstance(v, (list, str)) or len(str(v)) < 120 else str(v)[:120]))
    for x in t.tensors:
        print("T", x.name, list(x.ne), x.gtype)
    print(f"{len(t.kv)} KV, {len(t.tensors)} tensors, alignment {t.alignment}")


def main(argv=None):
    global MIN_CORRELATION
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model")
    ap.add_argument("--template")
    ap.add_argument("--out")
    ap.add_argument("--max-mbps", type=float, default=2000)
    ap.add_argument("--allow-buffered", action="store_true")
    ap.add_argument("--min-corr", type=float, default=MIN_CORRELATION,
                    help=f"correlation floor for every family (plan: 0.95); fp16 exact except the tensors named "
                         f"in slotstream_spark/config/q4_1_exceptions.json")
    ap.add_argument("--sample-rows", type=int, default=Spec.sample_k,
                    help="rows per tensor compared with the template (default 24; the n-gram table keeps 6)")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--check-only", action="store_true", help="plan the mapping and compare sampled rows with the template; write nothing but OUT/check.json")
    ap.add_argument("--dump-template", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    MIN_CORRELATION = a.min_corr
    Spec.sample_k = a.sample_rows
    if a.dump_template:
        dump_template(a.template)
        return 0
    if not a.model:
        ap.error("--model is required")
    if a.sweep:
        with mlx_ckpt.Checkpoint(a.model, allow_buffered=a.allow_buffered) as ck:
            res = sweep(ck, _throttle(a.max_mbps))
        print(json.dumps({k: v for k, v in res.items() if k not in ("bad", "tolerated")}
                         | {"bad_count": len(res["bad"]), "tolerated_count": len(res["tolerated"])}))
        for t in res["tolerated"]:
            print("TOLERATED:", t["name"], f"count {t['count']}", f"max abs err {t['max_abs_err']:.3e}")
        for b in res["bad"][:50]:
            print("NOT EXACT:", b, file=sys.stderr)
        if a.json:
            Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
        elif os.environ.get("CAP"):
            Path(os.environ["CAP"], "repack_q4_1_sweep.json").write_text(json.dumps(res, indent=1) + "\n")
        return 0 if res["exact"] else 1
    if not (a.template and a.out):
        ap.error("--template and --out are required to build")
    try:
        man = run_build(a.model, a.template, a.out, max_mbps=a.max_mbps, allow_buffered=a.allow_buffered, write=not a.check_only)
    except MappingError as e:
        if not a.check_only:
            raise
        cj = Path(a.out) / "check.json"
        fails = json.loads(cj.read_text()).get("failures") if cj.exists() else None
        for f in fails or []:
            print(f"BELOW FLOOR: {f['name']} corr {f['corr']:.4f} floor {f['corr_floor']} family {f['family']} sources {f['sources']}")
        print(f"MappingError: {e}", file=sys.stderr)
        return 1
    print(json.dumps({"tensors": len(man["tensors"]), "file_bytes": man["file_bytes"]} |
                     ({"below_099": len(man["below_099"])} if a.check_only else {})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
