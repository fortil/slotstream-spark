"""MLX affine 4-bit checkpoint access: index, O_DIRECT reader, dequantizer, aligned writer.

The checkpoint is a set of safetensors files plus ``model.safetensors.index.json``.
Reads go through ``Checkpoint.read_bytes`` (aligned-superset ``preadv`` into an anonymous
page-aligned buffer, O_DIRECT where the platform has it). The affine dequantizer follows
MLX: element j of each uint32 sits at bits [bits*j, bits*(j+1)), value = q * scale + bias in fp32.

numpy only; torch is imported lazily by the ``*_torch`` helpers.
"""

import errno
import hashlib
import json
import mmap
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

O_DIRECT = getattr(os, "O_DIRECT", 0)
ALIGN = 4096

# RMSNorms that transformers computes as x/rms * (1 + w). The MLX checkpoint folded the +1
# into w, so the transformers-side weight is w - 1. `linear_attn.norm` is not in this list.
# Same tuple as qwen4_exp.py:823-827.
CENTERED_NORMS = (
    "hc_norm.weight", "q_norm.weight", "k_norm.weight",
    "indexer.q_layernorm.weight", "indexer.k_layernorm.weight",
    "ple.norm_key.weight", "ple.norm_query.weight", "ple.norm_conv.weight",
)

_DTYPES = {
    "U32": np.dtype(np.uint32),
    "BF16": np.dtype(np.uint16),  # raw bits
    "F16": np.dtype(np.float16),
    "I64": np.dtype(np.int64),
    "I32": np.dtype(np.int32),
    "U8": np.dtype(np.uint8),
    "F32": np.dtype(np.float32),
}


@dataclass(frozen=True)
class TensorInfo:
    name: str
    file: str
    dtype: str
    shape: tuple
    offset: int  # absolute offset in the file: 8 + header length + data_offsets[0]
    nbytes: int


def _align_down(x, a=ALIGN):
    return x - (x % a)


def _align_up(x, a=ALIGN):
    return _align_down(x + a - 1, a)


def _read_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return n, header


class Checkpoint:
    def __init__(self, root, allow_buffered=False):
        self.root = Path(root)
        self.allow_buffered = allow_buffered
        if not O_DIRECT and not allow_buffered:
            raise RuntimeError("O_DIRECT is not available on this platform; pass allow_buffered=True")
        self._fds = {}
        self._tensors = {}
        if self.root.is_file():
            files = {self.root.name: self.root}
            self._dir = self.root.parent
            self._order = [self.root.name]
            n, _ = _read_header(self.root)
            with open(self.root, "rb") as f:
                self.index_sha256 = hashlib.sha256(f.read(8 + n)).hexdigest()
        else:
            self._dir = self.root
            idx = self.root / "model.safetensors.index.json"
            if idx.exists():
                raw = idx.read_bytes()
                self.index_sha256 = hashlib.sha256(raw).hexdigest()
                wm = json.loads(raw)["weight_map"]
                self._order = sorted(set(wm.values()))
            else:
                self._order = sorted(p.name for p in self.root.glob("*.safetensors"))
                if not self._order:
                    raise FileNotFoundError(f"no safetensors files under {self.root}")
                self.index_sha256 = hashlib.sha256("".join(self._order).encode()).hexdigest()
        for fn in self._order:
            path = self._dir / fn
            n, header = _read_header(path)
            for name, meta in header.items():
                if name == "__metadata__":
                    continue
                b, e = meta["data_offsets"]
                self._tensors[name] = TensorInfo(name, fn, meta["dtype"], tuple(meta["shape"]),
                                                 8 + n + b, e - b)

    # ---- index
    def tensors(self):
        return dict(self._tensors)

    def info(self, name):
        return self._tensors[name]

    # ---- file access
    def _open(self, path):
        if O_DIRECT:
            try:
                fd = os.open(path, os.O_RDONLY | O_DIRECT)
                probe = mmap.mmap(-1, ALIGN)
                try:
                    # Some filesystems (tmpfs) accept the flag at open and fail at read time.
                    os.preadv(fd, [probe], 0)
                    return fd
                except OSError:
                    os.close(fd)
                    raise
                finally:
                    probe.close()
            except OSError as e:
                if e.errno != errno.EINVAL:
                    raise
            if not self.allow_buffered:
                raise RuntimeError(f"O_DIRECT is not supported for {path}; pass allow_buffered=True")
        return os.open(path, os.O_RDONLY)

    def _fd(self, fn):
        fd = self._fds.get(fn)
        if fd is None:
            fd = self._fds[fn] = self._open(str(self._dir / fn))
        return fd

    def _pread_at(self, fn, offset, length):
        """Return bytes [offset, offset+length) of file fn via an aligned superset read."""
        if length == 0:
            return b""
        fd = self._fd(fn)
        start = _align_down(offset)
        end = _align_up(offset + length)
        size = end - start
        buf = mmap.mmap(-1, size)
        try:
            mv = memoryview(buf)
            got = 0
            while got < size:
                n = _preadv(fd, mv[got:], start + got)
                if n == 0:
                    break
                got += n
            need = offset + length - start
            if got < need:
                raise EOFError(f"short read in {fn}: wanted {need} bytes from {start}, got {got}")
            out = bytes(mv[offset - start: offset - start + length])
            mv.release()
        finally:
            buf.close()
        return out

    def read_bytes(self, name, start=0, length=None):
        t = self._tensors[name]
        if length is None:
            length = t.nbytes - start
        if start < 0 or start + length > t.nbytes:
            raise ValueError(f"{name}: range {start}+{length} outside {t.nbytes} bytes")
        return self._pread_at(t.file, t.offset + start, length)

    def read_array(self, name):
        t = self._tensors[name]
        return np.frombuffer(self.read_bytes(name), dtype=_DTYPES[t.dtype]).reshape(t.shape)

    def read_rows(self, name, rows):
        """Rows along dim 0, in the order given; adjacent rows share one read."""
        t = self._tensors[name]
        if not t.shape:
            raise ValueError(f"{name} is a scalar")
        dt = _DTYPES[t.dtype]
        rowbytes = int(np.prod(t.shape[1:], dtype=np.int64)) * dt.itemsize
        rows = [int(r) for r in rows]
        for r in rows:
            if not 0 <= r < t.shape[0]:
                raise IndexError(f"{name}: row {r} outside {t.shape[0]}")
        cache = {}
        uniq = sorted(set(rows))
        i = 0
        while i < len(uniq):
            j = i
            while j + 1 < len(uniq) and uniq[j + 1] == uniq[j] + 1:
                j += 1
            blob = self.read_bytes(name, uniq[i] * rowbytes, (uniq[j] - uniq[i] + 1) * rowbytes)
            for k in range(i, j + 1):
                cache[uniq[k]] = blob[(uniq[k] - uniq[i]) * rowbytes:(uniq[k] - uniq[i] + 1) * rowbytes]
            i = j + 1
        data = b"".join(cache[r] for r in rows)
        return np.frombuffer(data, dtype=dt).reshape((len(rows),) + t.shape[1:])

    def close(self):
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _preadv(fd, mv, offset):
    if hasattr(os, "preadv"):
        return os.preadv(fd, [mv], offset)
    data = os.pread(fd, len(mv), offset)
    mv[:len(data)] = data
    return len(data)


# ---- numerics

def bf16_to_f32(u16):
    u = np.asarray(u16, dtype=np.uint16)
    return (u.astype(np.uint32) << 16).view(np.float32)


def dequant_affine(w_u32, scales_bf16, biases_bf16, bits=4):
    """w_u32 [..., in/(32/bits)], scales/biases [..., in/group] (BF16 raw) -> float32 [..., in]."""
    if 32 % bits:
        raise ValueError("bits must divide 32")
    per = 32 // bits
    w = np.asarray(w_u32, dtype=np.uint32)
    shifts = (np.arange(per, dtype=np.uint32) * bits)
    q = (w[..., None] >> shifts) & np.uint32((1 << bits) - 1)
    q = q.reshape(w.shape[:-1] + (w.shape[-1] * per,)).astype(np.float32)
    n_in = q.shape[-1]
    ngroups = scales_bf16.shape[-1]
    if n_in % ngroups:
        raise ValueError(f"{n_in} inputs do not split into {ngroups} groups")
    group = n_in // ngroups
    s = np.repeat(bf16_to_f32(scales_bf16), group, axis=-1)
    b = np.repeat(bf16_to_f32(biases_bf16), group, axis=-1)
    return q * s + b


def dequant_affine_torch(w_u32, scales_bf16, biases_bf16, bits=4, device="cuda"):
    """Same math on a torch device; inputs are the numpy arrays read from the checkpoint."""
    import torch
    per = 32 // bits
    w = torch.from_numpy(np.asarray(w_u32, dtype=np.uint32).astype(np.int64)).to(device)
    shifts = torch.arange(per, device=device, dtype=torch.int64) * bits
    q = (w[..., None] >> shifts) & ((1 << bits) - 1)
    q = q.reshape(w.shape[:-1] + (w.shape[-1] * per,)).to(torch.float32)
    group = q.shape[-1] // scales_bf16.shape[-1]
    s = torch.from_numpy(bf16_to_f32(scales_bf16)).to(device).repeat_interleave(group, dim=-1)
    b = torch.from_numpy(bf16_to_f32(biases_bf16)).to(device).repeat_interleave(group, dim=-1)
    return q * s + b


# ---- MLX -> transformers names and values

def hf_key(name):
    """Key relative to the module the oracle builds (layer, text model, or the LM head)."""
    if name.startswith("vision_tower."):
        raise ValueError(f"vision tensors are not part of the text model: {name}")
    if name.startswith("language_model.model.layers."):
        rest = name[len("language_model.model.layers."):]
        _, _, tail = rest.partition(".")
        return tail
    if name.startswith("language_model.model."):
        return name[len("language_model.model."):]
    if name.startswith("language_model."):
        return name[len("language_model."):]
    return name


def hf_tensor(name, ckpt):
    """Return (hf_key, array) for one checkpoint tensor.

    - quantized ``.weight`` (sibling ``.scales``/``.biases``): dequantized float32
    - keys ending in CENTERED_NORMS: float32 ``w - 1`` (transformers adds 1 back)
    - conv1d ``(C, K, 1)`` -> ``(C, 1, K)``
    - everything else (int64 buffers, plain BF16 as raw uint16, ...) unchanged
    """
    key = hf_key(name)
    tensors = ckpt.tensors() if hasattr(ckpt, "tensors") else ckpt
    t = tensors[name]
    arr = None
    if name.endswith(".weight"):
        base = name[:-len(".weight")]
        if base + ".scales" in tensors and base + ".biases" in tensors:
            w = ckpt.read_array(name)
            arr = dequant_affine(w, ckpt.read_array(base + ".scales"), ckpt.read_array(base + ".biases"))
            if key.endswith(CENTERED_NORMS):
                arr = arr - np.float32(1.0)
            return key, arr
    arr = ckpt.read_array(name)
    if key.endswith(CENTERED_NORMS):
        return key, bf16_to_f32(arr) - np.float32(1.0)
    if "conv1d.weight" in key and arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.ascontiguousarray(arr.transpose(0, 2, 1))
    return key, arr


# ---- writer

class AlignedWriter:
    """Sequential writer that issues 4 MiB aligned writes (O_DIRECT) and flushes the tail buffered."""

    CHUNK = 4 << 20

    def __init__(self, path, o_direct=True):
        self.path = str(path)
        self._buf = mmap.mmap(-1, self.CHUNK)
        self._fill = 0
        self._pos = 0  # file offset of the start of the buffer
        self._direct = bool(o_direct and O_DIRECT)
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        self._fd = None
        if self._direct:
            try:
                self._fd = os.open(self.path, flags | O_DIRECT, 0o644)
                self._probe()
            except OSError as e:
                if e.errno != errno.EINVAL:
                    raise
                if self._fd is not None:
                    os.close(self._fd)
                self._fd = None
                self._direct = False
        if self._fd is None:
            self._fd = os.open(self.path, flags, 0o644)
        self._closed = False

    def _probe(self):
        # tmpfs accepts O_DIRECT at open but rejects the write.
        probe = mmap.mmap(-1, ALIGN)
        try:
            os.pwrite(self._fd, probe, 0)
        finally:
            probe.close()
        os.ftruncate(self._fd, 0)

    def write(self, data):
        mv = memoryview(data).cast("B")
        i = 0
        while i < len(mv):
            n = min(self.CHUNK - self._fill, len(mv) - i)
            self._buf[self._fill:self._fill + n] = mv[i:i + n]
            self._fill += n
            i += n
            if self._fill == self.CHUNK:
                self._flush_aligned(self._fill)

    def _flush_aligned(self, n):
        """Write the first n bytes (a multiple of ALIGN) of the buffer; keep the remainder."""
        assert n % ALIGN == 0
        if n:
            view = memoryview(self._buf)[:n]
            done = 0
            while done < n:
                done += os.pwrite(self._fd, view[done:], self._pos + done)
            view.release()
            rest = self._fill - n
            if rest:
                self._buf[:rest] = self._buf[n:self._fill]
            self._pos += n
            self._fill = rest

    @property
    def tell(self):
        return self._pos + self._fill

    def seek_aligned(self, off):
        if off % ALIGN:
            raise ValueError(f"seek_aligned: {off} is not a multiple of {ALIGN}")
        if self._fill % ALIGN:
            raise ValueError("seek_aligned: pending data is not aligned")
        self._flush_aligned(self._fill)
        self._pos = off

    def close(self):
        if self._closed:
            return
        self._closed = True
        whole = _align_down(self._fill)
        self._flush_aligned(whole)
        tail = bytes(self._buf[:self._fill])
        pos = self._pos
        os.close(self._fd)
        self._buf.close()
        fd = os.open(self.path, os.O_WRONLY)
        try:
            if tail:
                os.pwrite(fd, tail, pos)
            os.fsync(fd)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
