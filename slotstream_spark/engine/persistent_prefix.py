# SPDX-License-Identifier: MIT
# Port of PersistentPrefix*.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""Persistent prefix tier (format 3): committed conversation states kept on disk (port of PersistentPrefix*.swift).

Pure code: numpy in, numpy out, no GPU and no torch. The engine hands a `StateView` (host arrays of the committed
state) to `save` and gets one back from `restore`; step 86 does the device copies around it.

A persisted state is one head (`<40 hex>.slotprefix`: token ids, the fixed recurrent arrays, the draft head's last row
and, for every sequence array, the ordered extents of its live rows) plus segments (`<32 hex>.slotseg`: absolute row
ranges of sequence arrays, written once, never modified). A turn that continues a persisted state writes a new head and
one segment with only the rows that are new; its other extents point at earlier segments.

Container, little-endian: `"SLOTPFX1"`, payloads back to back, a sorted-key JSON header (offset, byteCount, crc32 per
payload) and a 32-byte footer: header offset u64, header length u64, header crc32 u32, reserved u32, `"SLOTEND1"`.

Rules that matter (each has a test):
- A head name is sha256(identity || token ids as int64 LE)[:40]. Rows of a parent are referenced only through a lineage
  bound to that head name and to this tier instance, never because two token lists happen to be equal.
- Only a missing file (ENOENT) or a format error deletes anything. An I/O or permission error keeps the file.
- The directory is mode 0700 and locked with flock(LOCK_EX | LOCK_NB) on `.lock`.
"""
import errno
import fcntl
import glob
import hashlib
import json
import os
import struct
import threading
import time
import uuid
import zlib
from dataclasses import dataclass, field, replace
from enum import IntEnum
from pathlib import Path
from typing import Callable, Optional

import numpy as np

FORMAT_VERSION = 3
HEAD_EXT = "slotprefix"
SEGMENT_EXT = "slotseg"
MAGIC = b"SLOTPFX1"
FOOTER_MAGIC = b"SLOTEND1"
FOOTER_BYTES = 32
MAX_HEADER_BYTES = 16 << 20
MAX_ARRAY_BYTES = 16 << 30
INT32_MAX = (1 << 31) - 1
INT64_MAX = (1 << 63) - 1
INT64_MIN = -(1 << 63)
MODEL_LIMIT = 1 << 20

DEFAULT_MAX_BYTES = 20_000_000_000
DEFAULT_MINIMUM_TOKENS = 1024
DEFAULT_MAX_AGE_DAYS = 30
DEFAULT_MAX_AGE = DEFAULT_MAX_AGE_DAYS * 86_400.0
MINIMUM_FREE_BYTES = 2_000_000_000
MAXIMUM_SEGMENTS = 32
SAMPLE_BYTES = 4 << 20

ITEM_BYTES = {"bool": 1, "uint8": 1, "uint32": 4, "int32": 4, "int64": 8, "float16": 2, "bfloat16": 2, "float32": 4}
NUMPY_DTYPE = {"bool": np.bool_, "uint8": np.uint8, "uint32": np.uint32, "int32": np.int32, "int64": np.int64,
               "float16": np.float16, "bfloat16": np.uint16, "float32": np.float32}   # bfloat16 travels as raw uint16


class FileError(Exception):
    """A file that cannot describe a valid state for this identity: removed on sight."""

    def __init__(self, description, segments=()):
        super().__init__(description)
        self.description, self.segments = description, frozenset(segments)


class ReadFailure(Exception):
    """The system would not let us read or write (permissions, I/O): says nothing about the state's validity."""


class InaccessibleFile(Exception):
    def __init__(self, name, reason):
        super().__init__(reason if name in reason else f"prefix cache file {name}: {reason}")
        self.name, self.reason = name, reason


class DirectoryInUse(Exception):
    pass


class IdentityError(Exception):
    pass


# ----------------------------------------------------------------------------------------------- configuration
@dataclass
class Configuration:
    directory: Path
    max_bytes: int = DEFAULT_MAX_BYTES
    minimum_tokens: int = DEFAULT_MINIMUM_TOKENS
    max_age: Optional[float] = DEFAULT_MAX_AGE

    def __post_init__(self):
        self.directory = Path(self.directory).expanduser()


# --------------------------------------------------------------------------------------------------- identity
def _sha(data):
    return hashlib.sha256(data).hexdigest()


class Identity:
    """sha256 over sorted `k=v` lines; the only identity written to files."""

    def __init__(self, components):
        self.components = dict(components)
        canonical = "\n".join(f"{k}={self.components[k]}" for k in sorted(self.components))
        self.digest = _sha(canonical.encode())


def code_digest(package_dir, extra_files=()):
    """sha256 over every `*.py` under `package_dir` (sorted, relative path and bytes) and the extra binaries (the slotio
    .so). A change in any of them makes every state written before it foreign."""
    h = hashlib.sha256()
    root = Path(package_dir)
    for p in sorted(root.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        h.update(p.relative_to(root).as_posix().encode() + b"\0")
        h.update(p.read_bytes())
    for p in sorted(Path(x) for x in extra_files):
        h.update(p.name.encode() + b"\0")
        h.update(p.read_bytes())
    return h.hexdigest()


def sampled_digest(path):
    """`name:size:sha256(first 4 MiB + last 4 MiB)`: content of the ends, not modification times, so a copied model
    keeps its cache and another checkpoint of the same size does not inherit it."""
    p = Path(path)
    size = p.stat().st_size
    window = min(SAMPLE_BYTES, size)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for start in sorted({0, size - window}):
            f.seek(start)
            h.update(f.read(window))
    return f"{p.name}:{size}:{h.hexdigest()}"


def packed_files(packed_dir):
    """Every regular file of the packed directory, sorted: experts, trunk, n-gram table and the MTP files."""
    return sorted(Path(p) for p in glob.glob(str(Path(packed_dir) / "*")) if Path(p).is_file())


def backend_components(probe=None, require_gpu=True):
    """torch, triton, driver, NVML GPU name and compute capability. `probe()` returns that dict (tests inject it); the
    default asks torch / triton / pynvml and refuses to continue without a GPU when `require_gpu`."""
    if probe is not None:
        got = dict(probe())
    else:
        got = {"backend": "torch"}
        try:
            import torch
            got["torch"] = torch.__version__
            got["cuda"] = str(torch.version.cuda)
            if torch.cuda.is_available():
                major, minor = torch.cuda.get_device_capability()
                got["compute_capability"] = f"{major}.{minor}"
                got["gpu_name"] = torch.cuda.get_device_name()
        except ImportError:
            got["torch"] = "absent"
        try:
            import triton
            got["triton"] = triton.__version__
        except ImportError:
            got["triton"] = "absent"
        try:
            import pynvml
            pynvml.nvmlInit()
            got["driver"] = _text(pynvml.nvmlSystemGetDriverVersion())
            got["gpu_name"] = _text(pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0)))
        except Exception:  # noqa: BLE001 - NVML is one of two sources for the name
            got.setdefault("driver", _driver_from_proc())
    if require_gpu:
        for k in ("gpu_name", "compute_capability", "driver", "torch", "triton"):
            if not got.get(k) or got[k] == "absent":
                raise IdentityError(f"cannot identify the backend: {k} is not available")
    return {f"backend.{k}": str(v) for k, v in sorted(got.items())}


def _text(v):
    return v.decode() if isinstance(v, bytes) else str(v)


def _driver_from_proc():
    try:
        return Path("/proc/driver/nvidia/version").read_text().splitlines()[0].strip()
    except (OSError, IndexError):
        return "absent"


def identity_for(*, code, config_sha, weights, geometry, optimizations, envelope_id, arithmetic_version, backend):
    """The identity of the engine that wrote a state. `weights` is the list of `sampled_digest` strings,
    `optimizations` a dict (serialized with sorted keys), `backend` the dict from `backend_components`."""
    comps = {
        "format": str(FORMAT_VERSION),
        "code": code,
        "config": config_sha,
        "weights": ",".join(weights),
        "geometry": ",".join(str(g) for g in geometry),
        "optimizations": json.dumps(optimizations, sort_keys=True, separators=(",", ":")),
        "envelope": envelope_id,
        "context_arithmetic": str(arithmetic_version),
    }
    comps.update(backend)
    return Identity(comps)


# ------------------------------------------------------------------------------------------------- records
@dataclass
class ArrayRecord:
    name: str
    dtype: str
    shape: list
    axis: int
    length: int
    offset: int
    byteCount: int
    crc32: int


@dataclass
class IndexerRecord:
    offset: int
    rawBase: int
    compactRaw: bool
    pooledCount: int
    pooledRatio: int


@dataclass
class LinearRecord:
    layer: int
    ngramContext: list


@dataclass
class AttentionRecord:
    layer: int
    kvOffset: int
    indexer: IndexerRecord


@dataclass
class DraftRecord:
    kvOffset: int
    indexer: IndexerRecord


@dataclass(frozen=True)
class Extent:
    segment: str
    start: int
    end: int


@dataclass
class SequenceRecord:
    name: str
    dtype: str
    shape: list
    axis: int
    base: int
    live: int
    extents: list = field(default_factory=list)

    @property
    def end(self):
        return self.base + self.live

    @property
    def leading(self):
        return list(self.shape[:self.axis]) if 0 <= self.axis < len(self.shape) else []

    @property
    def trailing(self):
        return list(self.shape[self.axis + 1:]) if 0 <= self.axis < len(self.shape) else []


@dataclass
class RowsRecord:
    name: str
    dtype: str
    leading: list
    trailing: list
    start: int
    end: int
    offset: int
    byteCount: int
    crc32: int


@dataclass
class Head:
    identity: str
    tokenCount: int
    continued: bool = False
    shared: Optional[bool] = None
    compactStateWindows: bool = False
    ngramContext: list = field(default_factory=list)
    linear: list = field(default_factory=list)
    attention: list = field(default_factory=list)
    draft: Optional[DraftRecord] = None
    arrays: list = field(default_factory=list)
    sequences: list = field(default_factory=list)
    sequenceBytes: int = 0
    residentBytes: int = 0
    splicingTokens: Optional[list] = None
    prefillChunk: Optional[int] = None
    format: int = FORMAT_VERSION
    kind: str = "head"


@dataclass
class Segment:
    identity: str
    rows: list
    format: int = FORMAT_VERSION
    kind: str = "segment"


def _prune(obj):
    """Dataclasses to plain JSON values; None fields are absent from the file."""
    if isinstance(obj, (Extent, SequenceRecord, RowsRecord, ArrayRecord, IndexerRecord, LinearRecord, AttentionRecord,
                        DraftRecord, Head, Segment)):
        return {k: _prune(v) for k, v in obj.__dict__.items() if v is not None}
    if isinstance(obj, (list, tuple)):
        return [_prune(v) for v in obj]
    return obj


def encode_header(header):
    data = json.dumps(_prune(header), sort_keys=True, separators=(",", ":")).encode()
    if len(data) > MAX_HEADER_BYTES:
        raise ValueError("prefix state header is too large")
    return data


def _need(d, key, kinds, what):
    v = d.get(key) if isinstance(d, dict) else None
    if isinstance(v, bool) and bool not in kinds:
        raise FileError(f"unreadable {what}")
    if not isinstance(v, kinds):
        raise FileError(f"unreadable {what}")
    return v


def _i64(d, key, what):
    v = _need(d, key, (int,), what)
    if not INT64_MIN <= v <= INT64_MAX:
        raise FileError(f"unreadable {what}")
    return v


def _int_list(d, key, what):
    v = _need(d, key, (list,), what)
    for x in v:
        if isinstance(x, bool) or not isinstance(x, int) or not INT64_MIN <= x <= INT64_MAX:
            raise FileError(f"unreadable {what}")
    return list(v)


def _indexer(d, what):
    if not isinstance(d, dict):
        raise FileError(f"unreadable {what}")
    return IndexerRecord(_i64(d, "offset", what), _i64(d, "rawBase", what), _need(d, "compactRaw", (bool,), what),
                         _i64(d, "pooledCount", what), _i64(d, "pooledRatio", what))


def _sequence(d):
    w = "head"
    exts = []
    for e in _need(d, "extents", (list,), w):
        exts.append(Extent(_need(e, "segment", (str,), w), _i64(e, "start", w), _i64(e, "end", w)))
    return SequenceRecord(_need(d, "name", (str,), w), _need(d, "dtype", (str,), w), _int_list(d, "shape", w),
                          _i64(d, "axis", w), _i64(d, "base", w), _i64(d, "live", w), exts)


def head_from_json(d):
    w = "head"
    if not isinstance(d, dict):
        raise FileError("unreadable head")
    arrays = [ArrayRecord(_need(a, "name", (str,), w), _need(a, "dtype", (str,), w), _int_list(a, "shape", w),
                          _i64(a, "axis", w), _i64(a, "length", w), _i64(a, "offset", w), _i64(a, "byteCount", w),
                          _i64(a, "crc32", w)) for a in _need(d, "arrays", (list,), w)]
    linear = [LinearRecord(_i64(r, "layer", w), _int_list(r, "ngramContext", w)) for r in _need(d, "linear", (list,), w)]
    attention = [AttentionRecord(_i64(r, "layer", w), _i64(r, "kvOffset", w), _indexer(r.get("indexer"), w))
                 for r in _need(d, "attention", (list,), w)]
    draft = None
    if d.get("draft") is not None:
        draft = DraftRecord(_i64(d["draft"], "kvOffset", w), _indexer(d["draft"].get("indexer"), w))
    shared = d.get("shared")
    if shared is not None and not isinstance(shared, bool):
        raise FileError("unreadable head")
    splicing = _int_list(d, "splicingTokens", w) if d.get("splicingTokens") is not None else None
    chunk = _i64(d, "prefillChunk", w) if d.get("prefillChunk") is not None else None
    return Head(identity=_need(d, "identity", (str,), w), tokenCount=_i64(d, "tokenCount", w),
                continued=_need(d, "continued", (bool,), w), shared=shared,
                compactStateWindows=_need(d, "compactStateWindows", (bool,), w),
                ngramContext=_int_list(d, "ngramContext", w), linear=linear, attention=attention, draft=draft,
                arrays=arrays, sequences=[_sequence(s) for s in _need(d, "sequences", (list,), w)],
                sequenceBytes=_i64(d, "sequenceBytes", w), residentBytes=_i64(d, "residentBytes", w),
                splicingTokens=splicing, prefillChunk=chunk, format=_i64(d, "format", w), kind=_need(d, "kind", (str,), w))


def segment_from_json(d):
    w = "segment"
    if not isinstance(d, dict):
        raise FileError("unreadable segment")
    rows = [RowsRecord(_need(r, "name", (str,), w), _need(r, "dtype", (str,), w), _int_list(r, "leading", w),
                       _int_list(r, "trailing", w), _i64(r, "start", w), _i64(r, "end", w), _i64(r, "offset", w),
                       _i64(r, "byteCount", w), _i64(r, "crc32", w)) for r in _need(d, "rows", (list,), w)]
    return Segment(_need(d, "identity", (str,), w), rows, _i64(d, "format", w), _need(d, "kind", (str,), w))


# --------------------------------------------------------------------------------------------- byte geometry
def _product(values):
    r = 1
    for v in values:
        r *= v
        if r > INT64_MAX:
            return None
    return r


def fixed_capacity(shape, item_bytes):
    """Bytes of a fixed payload (its whole shape). Zero-size arrays and scalars are valid geometry."""
    if not all(isinstance(s, int) and 0 <= s <= INT32_MAX for s in shape) or item_bytes <= 0:
        return None
    b = _product(list(shape) + [item_bytes])
    return b if b is not None and b <= MAX_ARRAY_BYTES else None


class Layout:
    __slots__ = ("leading", "row_bytes", "logical_bytes", "capacity_bytes")

    def __init__(self, leading, row_bytes, logical_bytes, capacity_bytes):
        self.leading, self.row_bytes, self.logical_bytes, self.capacity_bytes = leading, row_bytes, logical_bytes, capacity_bytes

    def __eq__(self, o):
        return isinstance(o, Layout) and all(getattr(self, k) == getattr(o, k) for k in self.__slots__)


def layout(shape, axis, length, item_bytes):
    """Byte geometry of one array, or None for an invalid, overflowing or oversized shape."""
    if not shape or not all(isinstance(s, int) and 0 < s <= INT32_MAX for s in shape):
        return None
    if not (0 <= axis < len(shape)) or not (0 <= length <= shape[axis]) or item_bytes <= 0:
        return None
    leading, trailing = _product(shape[:axis]), _product(shape[axis + 1:])
    if leading is None or trailing is None:
        return None
    row_bytes = _product([trailing, item_bytes])
    chunk = None if row_bytes is None else _product([leading, row_bytes])
    logical = None if chunk is None else _product([chunk, length])
    capacity = None if chunk is None else _product([chunk, shape[axis]])
    if logical is None or capacity is None or capacity > MAX_ARRAY_BYTES:
        return None
    return Layout(leading, row_bytes, logical, capacity)


def rows_geometry(leading, trailing, item_bytes, count):
    """(chunks, row_bytes, bytes) for `count` rows, or None."""
    if not all(x > 0 for x in leading) or not all(x > 0 for x in trailing) or item_bytes <= 0 or count < 0:
        return None
    chunks = _product(leading)
    row_bytes = _product(list(trailing) + [item_bytes])
    total = None if chunks is None or row_bytes is None else _product([chunks, row_bytes, count])
    return None if total is None or total > MAX_ARRAY_BYTES else (chunks, row_bytes, total)


def footer(header_offset, header_length, header_crc):
    return struct.pack("<QQII", header_offset, header_length, header_crc, 0) + FOOTER_MAGIC


def parse_footer(data, file_size):
    """(offset, length, crc) unless the footer is intact and the header ends exactly where the footer begins."""
    if len(data) != FOOTER_BYTES or data[24:32] != FOOTER_MAGIC:
        return None
    offset, length, crc, _reserved = struct.unpack("<QQII", data[:24])
    if not (offset >= len(MAGIC) and 0 < length <= MAX_HEADER_BYTES and file_size >= len(MAGIC) + FOOTER_BYTES
            and offset <= file_size and offset + length == file_size - FOOTER_BYTES):
        return None
    return offset, length, crc


def file_name(identity, tokens):
    """`<40 hex>.slotprefix`, deterministic per identity and token ids."""
    h = hashlib.sha256()
    h.update(identity.encode())
    h.update(np.asarray(list(tokens), dtype="<i8").tobytes())
    return h.hexdigest()[:40] + "." + HEAD_EXT


def new_segment_name():
    return uuid.uuid4().hex + "." + SEGMENT_EXT


def _is_name(name, digits, ext):
    if not name.endswith("." + ext):
        return False
    stem = name[:-(len(ext) + 1)]
    return len(stem) == digits and all(c in "0123456789abcdef" for c in stem)


def is_head_name(name):
    return _is_name(name, 40, HEAD_EXT)


def is_segment_name(name):
    return _is_name(name, 32, SEGMENT_EXT)


def token_bytes(tokens):
    return np.asarray(list(tokens), dtype="<i4").tobytes()


def crc32(data, seed=0):
    return zlib.crc32(data, seed) & 0xFFFFFFFF


def _buf(x):
    """Bytes-like view of a payload (bytes, memoryview or a C-contiguous array)."""
    if callable(x):
        x = x()
    if isinstance(x, np.ndarray):
        x = np.ascontiguousarray(x)
        return memoryview(x).cast("B") if x.size else memoryview(b"")
    return memoryview(x)


def _read_exact(fd, count, offset):
    out = bytearray()
    while len(out) < count:
        try:
            chunk = os.pread(fd, min(count - len(out), 1 << 30), offset + len(out))
        except InterruptedError:
            continue
        except OSError as e:
            raise ReadFailure(f"prefix cache read failed: {e.strerror}") from e
        if not chunk:
            raise FileError("file ends before its recorded data")
        out += chunk
    return bytes(out)


def _write_all(fd, data):
    view = memoryview(data)
    done = 0
    while done < len(view):
        try:
            n = os.write(fd, view[done:done + (1 << 30)])
        except InterruptedError:
            continue
        except OSError as e:
            raise ReadFailure(f"prefix cache write failed: {e.strerror}") from e
        if n <= 0:
            raise ReadFailure("prefix cache write failed")
        done += n


def write_container(path, payloads, expected, header_fn):
    """Write magic, payloads, the header built from where each payload landed (`header_fn(placed) -> bytes`, `placed` =
    [(offset, byteCount, crc32)]) and the footer to a NEW file (O_EXCL). Returns the file size. A payload whose size
    differs from `expected` fails the write."""
    if len(payloads) != len(expected):
        raise ValueError("prefix state payloads are inconsistent")
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
    except OSError as e:
        raise ReadFailure(f"cannot create {path}: {e.strerror}") from e
    try:
        _write_all(fd, MAGIC)
        offset, placed = len(MAGIC), []
        for i, p in enumerate(payloads):
            raw = _buf(p)
            if raw.nbytes != expected[i]:
                raise ValueError(f"prefix state payload {i} has {raw.nbytes} bytes, expected {expected[i]}")
            crc = crc32(raw)
            if raw.nbytes:
                _write_all(fd, raw)
            placed.append((offset, raw.nbytes, crc))
            offset += raw.nbytes
        js = header_fn(placed)
        if len(js) > MAX_HEADER_BYTES:
            raise ValueError("prefix state header is too large")
        _write_all(fd, js)
        tail = footer(offset, len(js), crc32(js))
        _write_all(fd, tail)
        return offset + len(js) + len(tail)
    finally:
        os.close(fd)


def read_header_data(fd, size):
    """(header JSON bytes, payload end) of an intact container."""
    if size < len(MAGIC) + FOOTER_BYTES:
        raise FileError("file is too short")
    if _read_exact(fd, len(MAGIC), 0) != MAGIC:
        raise FileError("not a prefix state file")
    parsed = parse_footer(_read_exact(fd, FOOTER_BYTES, size - FOOTER_BYTES), size)
    if parsed is None:
        raise FileError("damaged footer")
    offset, length, crc = parsed
    data = _read_exact(fd, length, offset)
    if crc32(data) != crc:
        raise FileError("header checksum mismatch")
    return data, offset


# ------------------------------------------------------------------------------------------------- the state
@dataclass
class FixedArray:
    dtype: str
    data: np.ndarray


@dataclass
class SeqArray:
    dtype: str
    data: np.ndarray          # allocated shape; rows along `axis`
    axis: int
    base: int
    live: int


@dataclass
class Lineage:
    """The persisted head a live state's committed rows are known to equal, because the state was written as that head or
    restored from it and has only grown since."""
    tier: str
    head: str
    token_count: int
    sequences: dict


@dataclass
class StateView:
    """Host view of a committed state. Arrays of dtype bfloat16 travel as raw uint16."""
    token_count: int
    ngram_context: list = field(default_factory=list)
    compact_state_windows: bool = False
    linear: list = field(default_factory=list)       # LinearRecord
    attention: list = field(default_factory=list)    # AttentionRecord
    draft: Optional[DraftRecord] = None
    fixed: dict = field(default_factory=dict)        # name -> FixedArray, in file order
    sequences: dict = field(default_factory=dict)    # name -> SeqArray, in file order
    committed: bool = True
    recording: bool = False
    lineage: Optional[Lineage] = None

    def drop_lineage_below(self, tokens, draft_rows=None):
        """A rewind below the persisted boundary may rewrite those rows with other bits."""
        lin = self.lineage
        if lin is None:
            return
        draft = lin.sequences.get("draft.kv.keys")
        if lin.token_count > tokens or (draft_rows is not None and draft is not None and draft_rows < draft.live):
            self.lineage = None


@dataclass(frozen=True)
class Layout3:
    """The cache layers a loaded model allocates; a restored head must match them exactly."""
    linear_layers: frozenset
    attention_layers: frozenset
    compact_indexer_raw: bool


# ------------------------------------------------------------------------------------------------- the index
@dataclass
class Entry:
    file: str
    identity: str
    tokens: list
    bytes: int
    last_used: float
    sequence_bytes: int = 0
    resident_bytes: int = 0
    has_draft: bool = False
    continued: bool = False
    shared: bool = False
    prefill_chunk: Optional[int] = None
    sequences: dict = field(default_factory=dict)
    splicing_tokens: Optional[list] = None

    @property
    def segments(self):
        return {e.segment for s in self.sequences.values() for e in s.extents}


@dataclass
class SegmentEntry:
    file: str
    identity: str
    bytes: int
    rows: dict


@dataclass
class Reuse:
    extents: dict = field(default_factory=dict)
    first_new_row: dict = field(default_factory=dict)
    segments: set = field(default_factory=set)
    compacted: bool = False


class Value(IntEnum):
    """Eviction classes, in removal order."""
    FOREIGN = 0
    EXPIRED = 1
    ONE_OFF = 2
    PARENT = 3
    CONVERSATION = 4
    SHARED = 5

    @property
    def label(self):
        return {0: "other build", 1: "expired", 2: "one-off", 3: "parent", 4: "conversation", 5: "shared prefix"}[self.value]


# ------------------------------------------------------------------------------------------------- the policy
def is_expired(entry, now, max_age):
    return max_age is not None and entry.last_used < now - max_age


def best_match(entries, identity, prompt, longer_than, require_draft, now, max_age, boundaries=None, prefill_chunk=None):
    """The longest own unexpired state the prompt STRICTLY extends and that is longer than memory's. A request that will
    speculate needs the draft cache. `boundaries` are the incoming prompt's own prefill pass boundaries: a state of any
    other length would continue the read from a position it never reads through; its `prefill_chunk` must equal the
    request's."""
    best = None
    prompt = list(prompt)
    for e in entries:
        n = len(e.tokens)
        if (e.identity == identity and n and n > longer_than and len(prompt) > n and (not require_draft or e.has_draft)
                and not is_expired(e, now, max_age) and (boundaries is None or n in boundaries)
                and (boundaries is None or (prefill_chunk is not None and e.prefill_chunk == prefill_chunk))
                and prompt[:n] == e.tokens):
            if best is None or n > len(best.tokens):
                best = e
    return best


def extensions(entries, identity, prefix, now, max_age):
    """Metadata candidates: ids of own unexpired states strictly extending `prefix` (spliced ids when recorded)."""
    out = []
    prefix = list(prefix)
    for e in entries:
        if e.identity != identity or is_expired(e, now, max_age):
            continue
        ids = e.splicing_tokens if e.splicing_tokens is not None else e.tokens
        if len(ids) > len(prefix) and ids[:len(prefix)] == prefix:
            out.append(ids)
    return out


def longest_extension(entries, identity, prefix, now, max_age):
    best = None
    for ids in extensions(entries, identity, prefix, now, max_age):
        if best is None or len(ids) > len(best):
            best = ids
    return best


def redundant_ancestors(entries, identity, tokens):
    """Own states a save of `tokens` makes redundant: every strict prefix except the longest. A shared prefix never is."""
    tokens = list(tokens)
    prefixes = [e for e in entries if e.identity == identity and e.tokens and not e.shared
                and len(e.tokens) < len(tokens) and tokens[:len(e.tokens)] == e.tokens]
    if not prefixes:
        return []
    parent = max(prefixes, key=lambda e: (len(e.tokens), _Rev(e.file)))
    return [e for e in prefixes if e.file != parent.file]


class _Rev:
    """Orders names descending, so `max` ties on token count pick the smallest name (Swift: `($0.count, $1.file) < ...`)."""

    def __init__(self, s):
        self.s = s

    def __lt__(self, o):
        return self.s > o.s

    def __eq__(self, o):
        return self.s == o.s


def common_prefix_length(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def longest_common_prefix(entries, identity, prompt, now, max_age, diverging=False):
    best = 0
    for e in entries:
        if e.identity != identity or not e.tokens or is_expired(e, now, max_age):
            continue
        shared = common_prefix_length(e.tokens, prompt)
        if diverging and shared == len(e.tokens):
            continue
        best = max(best, shared)
    return best


def system_prefix_boundary(ids, header, turn_end):
    """Index just past the first turn end after the system header, when the prompt starts with one."""
    ids = list(ids)
    if not header or not turn_end or ids[:len(header)] != list(header):
        return None
    i = len(header)
    while i + len(turn_end) <= len(ids):
        if ids[i:i + len(turn_end)] == list(turn_end):
            return i + len(turn_end)
        i += 1
    return None


def _extends(child, parent):
    return (child.identity == parent.identity and child.file != parent.file and bool(parent.tokens)
            and len(child.tokens) > len(parent.tokens)
            and child.tokens[len(parent.tokens) - 1] == parent.tokens[len(parent.tokens) - 1]
            and child.tokens[:len(parent.tokens)] == parent.tokens)


def value_of(entry, entries, identity, now, max_age):
    if entry.identity != identity or not entry.tokens:
        return Value.FOREIGN
    if is_expired(entry, now, max_age):
        return Value.EXPIRED
    children = [e for e in entries if _extends(e, entry)]
    lineages = [c for c in children if not any(_extends(o, c) for o in children)]
    if len(lineages) >= 2:
        return Value.SHARED
    if children:
        return Value.PARENT
    return Value.CONVERSATION if entry.continued and not entry.shared else Value.ONE_OFF


def eviction_order(entries, identity, now, max_age):
    """Class, then least recently used, then name."""
    scored = [(value_of(e, entries, identity, now, max_age), e) for e in entries]
    scored.sort(key=lambda t: (t[0], t[1].last_used, t[1].file))
    return [e for _, e in scored]


def held_bytes(entries, segments, pinned):
    """Bytes held with these heads kept: their files plus every segment they or the incoming save reference, once."""
    referenced = set(pinned)
    for e in entries:
        referenced |= e.segments
    return sum(e.bytes for e in entries) + sum(segments.get(s, 0) for s in referenced)


def write_fits_quota(charged_bytes, replacing_head_bytes, incoming, quota):
    """Final write admission. Only an atomic same-name head replacement is credited."""
    if not (charged_bytes >= 0 and 0 <= replacing_head_bytes <= charged_bytes and incoming >= 0 and quota >= 0
            and incoming <= quota):
        return False
    return charged_bytes - replacing_head_bytes <= quota - incoming


def eviction_victims(entries, segments, identity, quota, incoming, pinned, freed, now, max_age):
    """Heads to remove so `incoming` new bytes fit the quota once `freed` heads are gone; None when not even removing
    every head makes room."""
    if incoming < 0 or held_bytes([], segments, pinned) + incoming > quota:
        return None
    kept = [e for e in entries if e.file not in freed]
    victims = []
    for victim in eviction_order(kept, identity, now, max_age):
        if held_bytes(kept, segments, pinned) + incoming <= quota:
            break
        kept = [e for e in kept if e.file != victim.file]
        victims.append(victim)
    return victims if held_bytes(kept, segments, pinned) + incoming <= quota else None


def unreferenced_segments(entries, segments):
    referenced = set()
    for e in entries:
        referenced |= e.segments
    return sorted(s for s in segments if s not in referenced)


def tiles(record):
    """Do the extents tile rows base..<base+live exactly, in order, over valid segment names?"""
    if record.base < 0 or record.live < 0 or record.base > INT64_MAX - record.live:
        return False
    nxt = record.base
    for e in record.extents:
        if not is_segment_name(e.segment) or e.start != nxt or e.end <= e.start:
            return False
        nxt = e.end
    return nxt == record.end


def clip(extents, start, end):
    """Extents clipped to rows start..<end, or None unless they tile it exactly."""
    if start > end:
        return None
    result, nxt = [], start
    for e in extents:
        if e.end > start and e.start < end:
            lo, hi = max(e.start, start), min(e.end, end)
            if lo != nxt:
                return None
            result.append(Extent(e.segment, lo, hi))
            nxt = hi
    return result if nxt == end else None


def dangling_reason(entry, segments):
    """Why an own head cannot be restored from these segments, or None."""
    for record in sorted(entry.sequences.values(), key=lambda r: r.name):
        if not tiles(record):
            return f"{record.name} extents do not tile its live rows"
        for e in record.extents:
            seg = segments.get(e.segment)
            if seg is None or seg.identity != entry.identity:
                return f"segment {e.segment} is missing"
            rows = seg.rows.get(record.name)
            if (rows is None or rows.dtype != record.dtype or rows.leading != record.leading
                    or rows.trailing != record.trailing or not (rows.start <= e.start and e.end <= rows.end)):
                return f"segment {e.segment} does not hold {record.name} rows {e.start}..<{e.end}"
    return None


def reuse_plan(child, parent, segment_exists, maximum_segments):
    """For each array of a new state, the parent extents it references and the first row it writes. A parent array is
    reused only with the same dtype and row geometry, a base no later than the child's, live rows the child still
    covers and every segment present. Past `maximum_segments` referenced segments the save writes every row."""
    plan, writes_rows = Reuse(), False
    for a in child:
        plan.extents[a.name] = []
        plan.first_new_row[a.name] = a.base
        old = (parent or {}).get(a.name)
        if old is not None and (old.dtype == a.dtype and old.axis == a.axis and old.leading == a.leading
                                and old.trailing == a.trailing and old.base <= a.base and old.end <= a.end
                                and tiles(old)):
            kept = clip(old.extents, a.base, old.end)
            if kept is not None and all(segment_exists(e.segment) for e in kept):
                plan.extents[a.name] = kept
                plan.first_new_row[a.name] = max(a.base, old.end)
                plan.segments |= {e.segment for e in kept}
                writes_rows = writes_rows or a.end > max(a.base, old.end)
                continue
        writes_rows = writes_rows or a.live > 0
    if len(plan.segments) + (1 if writes_rows else 0) > max(1, maximum_segments):
        compacted = Reuse(compacted=True)
        for a in child:
            compacted.extents[a.name] = []
            compacted.first_new_row[a.name] = a.base
        return compacted
    return plan


# ----------------------------------------------------------------------------------------- read / validate
def validate_head(header, payload_end):
    """Payloads sit back to back from the magic to the header, names are unique, every sequence array's extents tile its
    live rows and the recorded memory accounting equals the array capacities."""
    if not (0 < header.tokenCount <= MODEL_LIMIT):
        raise FileError("token count is invalid")
    expected, names, resident = len(MAGIC), set(), 0
    for r in header.arrays:
        if not (r.offset == expected and r.byteCount >= 0 and expected <= payload_end
                and r.byteCount <= payload_end - expected and r.name not in names):
            raise FileError(f"array {r.name} is out of place")
        names.add(r.name)
        item = ITEM_BYTES.get(r.dtype)
        cap = None if item is None else fixed_capacity(r.shape, item)
        if item is None or r.axis != 0 or r.length != (r.shape[0] if r.shape else 1) or cap is None or cap != r.byteCount:
            raise FileError(f"array {r.name} has an invalid fixed layout")
        if r.name != "tokens":
            resident += cap
        expected += r.byteCount
    if expected != payload_end:
        raise FileError("arrays do not end at the header")
    sequence_bytes = 0
    for r in header.sequences:
        item = ITEM_BYTES.get(r.dtype)
        lay = None if item is None else layout(r.shape, r.axis, r.live, item)
        if r.name in names or lay is None or not tiles(r):
            raise FileError(f"sequence {r.name} is invalid")
        names.add(r.name)
        sequence_bytes += lay.capacity_bytes
        resident += lay.capacity_bytes
    if header.sequenceBytes != sequence_bytes or header.residentBytes != resident:
        raise FileError("recorded memory accounting differs from array capacity")


def read_tokens(fd, header):
    n = header.tokenCount
    rec = next((a for a in header.arrays if a.name == "tokens"), None)
    if rec is None or rec.dtype != "int32" or rec.shape != [n] or rec.axis != 0 or rec.length != n or rec.byteCount != n * 4:
        raise FileError("token ids are missing")
    raw = _read_exact(fd, n * 4, rec.offset)
    if crc32(raw) != rec.crc32:
        raise FileError("token checksum mismatch")
    return np.frombuffer(raw, dtype="<i4").astype(np.int64).tolist()


def _mtime(st):
    return st.st_mtime


def read_file(directory, name):
    """Read and validate one file's header (and a head's token ids). -> ("head", Entry) | ("segment", SegmentEntry) |
    ("other", bytes). Raises FileError for a file that is invalid, ReadFailure/OSError for one we could not read."""
    path = Path(directory) / name
    try:
        fd = os.open(str(path), os.O_RDONLY | os.O_CLOEXEC)
    except FileNotFoundError:
        raise FileError(f"file {name} is missing") from None
    except OSError as e:
        raise ReadFailure(f"cannot open prefix cache file {name}: {e.strerror}") from e
    try:
        st = os.fstat(fd)
        size = st.st_size
        data, payload_end = read_header_data(fd, size)
        try:
            raw = json.loads(data)
        except (ValueError, RecursionError):
            raise FileError("unreadable header") from None
        if not isinstance(raw, dict) or isinstance(raw.get("format"), bool) or not isinstance(raw.get("format"), int):
            raise FileError("unreadable header")
        if raw["format"] != FORMAT_VERSION:
            return "other", size
        if name.endswith("." + HEAD_EXT):
            if raw.get("kind") != "head" or not is_head_name(name):
                raise FileError("unreadable head")
            header = head_from_json(raw)
            validate_head(header, payload_end)
            tokens = read_tokens(fd, header)
            if header.splicingTokens is not None:
                ids = header.splicingTokens
                if not (len(ids) <= MODEL_LIMIT and ids[:len(tokens)] == tokens
                        and all(0 <= t <= INT32_MAX for t in ids)):
                    raise FileError("invalid conversation token metadata")
            if name != file_name(header.identity, tokens):
                raise FileError("file name does not match its contents")
            entry = Entry(name, header.identity, tokens, size, _mtime(st), header.sequenceBytes, header.residentBytes,
                          header.draft is not None, header.continued, bool(header.shared), header.prefillChunk,
                          {s.name: s for s in header.sequences}, header.splicingTokens)
            return "head", entry
        if raw.get("kind") != "segment" or not is_segment_name(name):
            raise FileError("unreadable segment")
        seg = segment_from_json(raw)
        expected, rows = len(MAGIC), {}
        for r in seg.rows:
            item = ITEM_BYTES.get(r.dtype)
            geo = None if item is None else rows_geometry(r.leading, r.trailing, item, r.end - r.start)
            if not (r.offset == expected and r.name not in rows and r.start >= 0 and r.end > r.start
                    and geo is not None and geo[2] == r.byteCount):
                raise FileError(f"segment rows {r.name} are invalid")
            rows[r.name] = r
            expected += r.byteCount
        if expected != payload_end:
            raise FileError("segment rows do not end at the header")
        return "segment", SegmentEntry(name, seg.identity, size, rows)
    finally:
        os.close(fd)


def _now():
    return time.time()


def _file_size(path):
    try:
        return os.stat(path).st_size
    except OSError:
        return 0


def _megabytes(n):
    return f"{n / 1e6:.1f} MB"


def is_state_file(name):
    return name.endswith("." + HEAD_EXT) or name.endswith("." + SEGMENT_EXT) or (name.startswith(".") and name.endswith(".tmp"))


@dataclass
class Maintenance:
    other_builds: int = 0
    expired: int = 0
    unreadable: int = 0
    incomplete: int = 0
    orphan_segments: int = 0
    over_quota: int = 0
    bytes: int = 0

    @property
    def files(self):
        return (self.other_builds + self.expired + self.unreadable + self.incomplete + self.orphan_segments
                + self.over_quota)


@dataclass
class SaveResult:
    outcome: str                 # saved | present | skipped: <why> | failed: <why>
    tokens: int
    bytes: int = 0
    reused_bytes: int = 0
    compacted: bool = False
    seconds: float = 0.0
    removed_files: int = 0


@dataclass
class RestoreResult:
    state: StateView
    tokens: int
    bytes: int
    seconds: float


def prepare_directory(directory):
    """Create the directory owner-only and prove it is writable, without indexing it."""
    d = Path(directory)
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    if not d.is_dir() or not os.access(d, os.R_OK | os.W_OK | os.X_OK):
        raise ReadFailure(f"prefix cache directory {d} is not a writable directory")


def lock_directory(directory):
    fd = os.open(str(Path(directory) / ".lock"), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        raise DirectoryInUse(f"prefix cache directory {directory} is in use by another process or cache") from None
    return fd


# --------------------------------------------------------------------------------------------- the tier
class PersistentPrefix:
    """Disk tier of the conversation prefix cache. One instance per directory and process."""

    def __init__(self, configuration, identity, *, on_event=None, free_bytes_fn=None):
        c = configuration
        if c.max_bytes <= 0:
            raise ValueError("the prefix cache disk quota must be positive")
        if not 1 <= c.minimum_tokens <= MODEL_LIMIT:
            raise ValueError(f"the persisted prefix minimum must be between 1 and {MODEL_LIMIT} tokens")
        if c.max_age is not None and not (c.max_age > 0 and c.max_age != float("inf")):
            raise ValueError("the persisted prefix maximum age must be positive")
        self.configuration = c
        self.identity = identity
        self.on_event = on_event
        self.instance = uuid.uuid4().hex
        self.maximum_segments = MAXIMUM_SEGMENTS
        self._free_bytes = free_bytes_fn or self._statvfs_free
        self.lock = threading.Lock()
        self.operations = threading.RLock()
        self.heads = []
        self.segments = {}
        self.counters = dict(restores=0, restored_tokens=0, restored_bytes=0, restore_failures=0, saves=0,
                             delta_saves=0, compactions=0, written_bytes=0, reused_bytes=0, shared_saves=0,
                             skipped_saves=0, failed_saves=0, evictions=0, expired=0, replaced=0, rejected=0,
                             deleted=0, removed_segments=0)
        prepare_directory(c.directory)
        self._lock_fd = lock_directory(c.directory)
        try:
            self.maintenance = self._open_directory()
        except BaseException:
            self.close()
            raise

    # ---- lifecycle
    def close(self):
        fd, self._lock_fd = getattr(self, "_lock_fd", None), None
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def _statvfs_free(self):
        st = os.statvfs(self.configuration.directory)
        return st.f_bavail * st.f_frsize

    def _path(self, file):
        return str(self.configuration.directory / file)

    def _report(self, line):
        if self.on_event:
            self.on_event(line)

    # ---- opening
    def _open_directory(self):
        directory, result = self.configuration.directory, Maintenance()
        found, found_segments = [], {}

        def discard(name, size, reason):
            try:
                os.unlink(directory / name)
            except FileNotFoundError:
                return
            except OSError as e:
                raise InaccessibleFile(name, f"cannot remove it: {e.strerror}") from e
            setattr(result, reason, getattr(result, reason) + 1)
            result.bytes += size

        try:
            names = sorted(os.listdir(directory))
        except OSError:
            names = []
        for name in names:
            path = directory / name
            if name.startswith(".") and name.endswith(".tmp"):
                discard(name, _file_size(path), "incomplete")      # an interrupted write: the lock proves no writer is live
                continue
            if not (name.endswith("." + HEAD_EXT) or name.endswith("." + SEGMENT_EXT)):
                continue
            try:
                kind, item = read_file(directory, name)
            except FileError:
                discard(name, _file_size(path), "unreadable")
                continue
            except (ReadFailure, OSError) as e:
                raise InaccessibleFile(name, str(e)) from e
            if kind == "head" and item.identity == self.identity.digest:
                found.append(item)
            elif kind == "segment" and item.identity == self.identity.digest:
                found_segments[name] = item
            elif kind == "other":
                discard(name, item, "other_builds")
            else:
                discard(name, item.bytes, "other_builds")
        now, max_age = _now(), self.configuration.max_age
        keep = []
        for head in found:
            if is_expired(head, now, max_age):
                discard(head.file, head.bytes, "expired")
            elif dangling_reason(head, found_segments) is not None:
                discard(head.file, head.bytes, "incomplete")
            else:
                keep.append(head)
        found = keep
        victims = eviction_victims(found, {k: v.bytes for k, v in found_segments.items()}, self.identity.digest,
                                   self.configuration.max_bytes, 0, set(), set(), now, max_age)
        if victims:
            files = {v.file for v in victims}
            for v in victims:
                discard(v.file, v.bytes, "over_quota")
            found = [h for h in found if h.file not in files]
        for name in unreferenced_segments(found, list(found_segments)):
            discard(name, found_segments[name].bytes, "orphan_segments")
            del found_segments[name]
        with self.lock:
            self.heads, self.segments = found, found_segments
        return result

    # ---- index
    @property
    def stored_states(self):
        with self.lock:
            return sum(1 for h in self.heads if h.identity == self.identity.digest)

    @property
    def stored_segments(self):
        with self.lock:
            return len(self.segments)

    @property
    def stored_bytes(self):
        with self.lock:
            return sum(h.bytes for h in self.heads) + sum(s.bytes for s in self.segments.values())

    @property
    def indexed_entries(self):
        with self.lock:
            return list(self.heads)

    @property
    def indexed_segments(self):
        with self.lock:
            return dict(self.segments)

    def can_write(self, incoming, replacing_head):
        with self.lock:
            charged = sum(h.bytes for h in self.heads) + sum(s.bytes for s in self.segments.values())
            replacement = next((h.bytes for h in self.heads if h.file == replacing_head), 0)
            return write_fits_quota(charged, replacement, incoming, self.configuration.max_bytes)

    def best_match(self, prompt, longer_than=0, require_draft=False, boundaries=None, prefill_chunk=None):
        now = _now()
        with self.lock:
            return best_match(self.heads, self.identity.digest, prompt, longer_than, require_draft, now,
                              self.configuration.max_age, boundaries, prefill_chunk)

    def longest_extension(self, prefix):
        now = _now()
        with self.lock:
            return longest_extension(self.heads, self.identity.digest, prefix, now, self.configuration.max_age)

    def extensions(self, prefix):
        now = _now()
        with self.lock:
            return extensions(self.heads, self.identity.digest, prefix, now, self.configuration.max_age)

    def longest_common_prefix(self, prompt, diverging=False):
        now = _now()
        with self.lock:
            return longest_common_prefix(self.heads, self.identity.digest, prompt, now, self.configuration.max_age,
                                         diverging)

    @property
    def stored_shared_states(self):
        with self.lock:
            return sum(1 for h in self.heads if h.identity == self.identity.digest and h.shared)

    def value(self, file):
        now = _now()
        with self.lock:
            e = next((h for h in self.heads if h.file == file), None)
            return None if e is None else value_of(e, self.heads, self.identity.digest, now, self.configuration.max_age)

    # ---- removal
    def _remove_file(self, file):
        """-> (gone, removed). Only a successful unlink or an already absent file releases its charge."""
        try:
            os.unlink(self._path(file))
            return True, True
        except FileNotFoundError:
            return True, False
        except OSError as e:
            self._report(f"could not remove prefix cache file {file}: {e.strerror}")
            return False, False

    def _remove(self, files, reason, keeping=frozenset()):
        removed = 0
        for file in files:
            gone, did = self._remove_file(file)
            if not gone:
                continue
            removed += did
            with self.lock:
                self.heads = [h for h in self.heads if h.file != file]
                self.counters[reason] += 1
        return removed + self._collect_garbage(keeping)

    def _collect_garbage(self, keeping=frozenset()):
        with self.lock:
            orphans = [s for s in unreferenced_segments(self.heads, list(self.segments)) if s not in keeping]
        removed = 0
        for name in orphans:
            gone, did = self._remove_file(name)
            if not gone:
                continue
            removed += did
            with self.lock:
                self.segments.pop(name, None)
                if did:
                    self.counters["removed_segments"] += 1
        return removed

    def _remove_expired(self, keeping=frozenset()):
        now = _now()
        with self.lock:
            expired = [h.file for h in self.heads if is_expired(h, now, self.configuration.max_age)]
        if not expired:
            return 0
        removed = self._remove(expired, "expired", keeping)
        with self.lock:
            remaining = {h.file for h in self.heads}
        forgotten = sum(1 for f in expired if f not in remaining)
        self._report(f"forgot {forgotten} state{'' if forgotten == 1 else 's'} unused past the maximum age")
        return removed

    def _touch(self, file):
        try:
            os.utime(self._path(file))
        except OSError:
            pass
        now = _now()
        with self.lock:
            for h in self.heads:
                if h.file == file:
                    h.last_used = now

    def clear(self):
        """Remove every state and segment in the directory, including files this build could not use."""
        with self.operations:
            try:
                names = os.listdir(self.configuration.directory)
            except OSError:
                names = []
            with self.lock:
                head_files = {n for n in names if n.endswith("." + HEAD_EXT)} | {h.file for h in self.heads}
            removed = self._remove(sorted(head_files), "deleted")
            with self.lock:
                referenced = set()
                for h in self.heads:
                    referenced |= h.segments
            try:
                remaining = os.listdir(self.configuration.directory)
            except OSError:
                remaining = []
            for name in remaining:
                if is_state_file(name) and name not in head_files and name not in referenced:
                    gone, did = self._remove_file(name)
                    if not gone:
                        continue
                    removed += did
                    with self.lock:
                        if self.segments.pop(name, None) is not None and did:
                            self.counters["removed_segments"] += 1
            self._report(f"cleared {removed} file{'' if removed == 1 else 's'}")
            return removed

    def remove_states_overlapping(self, tokens):
        """Remove own states whose ids are a prefix of `tokens` or begin with them (a conversation being deleted)."""
        tokens = list(tokens)
        if not tokens:
            return 0
        with self.operations:
            with self.lock:
                files = [h.file for h in self.heads if h.identity == self.identity.digest and h.tokens
                         and (h.tokens[:len(tokens)] == tokens or tokens[:len(h.tokens)] == h.tokens)]
            if not files:
                return 0
            self._remove(files, "deleted")
            with self.lock:
                remaining = {h.file for h in self.heads}
            removed = sum(1 for f in files if f not in remaining)
            self._report(f"removed {removed} state{'' if removed == 1 else 's'} of a deleted conversation")
            return removed

    # ---- save
    def save(self, view, tokens, *, shared=False, splicing=None, prefill_chunk=None):
        """Write `view`, which consumed exactly `tokens`, unless a head already holds the same ids with at least the same
        draft state. Rows below the view's lineage boundary are referenced, not written. Never raises: a state that
        cannot be written leaves the request and memory unchanged."""
        with self.operations:
            return self._save(view, list(tokens), shared, splicing, prefill_chunk)

    def _plan(self, view, tokens):
        """-> (payloads, array records, sequence records with buffers, head)."""
        if (not view.committed or view.recording or view.token_count != len(tokens)
                or not all(0 <= t <= INT32_MAX for t in tokens)):
            raise ValueError("state is not at a committed boundary")
        head = Head(identity=self.identity.digest, tokenCount=len(tokens), compactStateWindows=view.compact_state_windows,
                    ngramContext=list(view.ngram_context), linear=list(view.linear), attention=list(view.attention),
                    draft=view.draft)
        payloads = [token_bytes(tokens)]
        arrays = [ArrayRecord("tokens", "int32", [len(tokens)], 0, len(tokens), 0, len(tokens) * 4, 0)]
        for name, fa in view.fixed.items():
            item = ITEM_BYTES.get(fa.dtype)
            shape = list(fa.data.shape)
            lay = None if item is None else fixed_capacity(shape, item)
            if lay is None or fa.data.dtype.itemsize != item:
                raise ValueError(f"cannot persist {name}: {fa.dtype} {shape}")
            payloads.append(fa.data)
            arrays.append(ArrayRecord(name, fa.dtype, shape, 0, shape[0] if shape else 1, 0, lay, 0))
            head.residentBytes += lay
        sequences = []
        for name, sa in view.sequences.items():
            item = ITEM_BYTES.get(sa.dtype)
            shape = list(sa.data.shape)
            lay = None if item is None or sa.base < 0 else layout(shape, sa.axis, sa.live, item)
            if lay is None or sa.data.dtype.itemsize != item:
                raise ValueError(f"cannot persist {name}: {sa.dtype} {shape} with {sa.live} live rows")
            sequences.append((SequenceRecord(name, sa.dtype, shape, sa.axis, sa.base, sa.live, []), sa.data))
            head.sequenceBytes += lay.capacity_bytes
            head.residentBytes += lay.capacity_bytes
        return payloads, arrays, sequences, head

    @staticmethod
    def _row_payload(data, record, first):
        """Rows first..<end of one sequence array as one contiguous buffer, chunk by chunk."""
        lay = layout(record.shape, record.axis, record.live, ITEM_BYTES[record.dtype])
        lead = lay.leading
        view = data.reshape(lead, record.shape[record.axis], -1)
        return lambda: np.ascontiguousarray(view[:, first - record.base:record.live, :])

    def _save(self, view, tokens, shared, splicing, prefill_chunk):
        start = time.monotonic()
        removed, written, reused, compacted = 0, 0, 0, False

        def finish(outcome):
            saved = outcome == "saved"
            if shared and (outcome.startswith("skipped") or outcome.startswith("failed")):
                self._report(f"kept no shared {len(tokens)}-token prefix: {outcome}")
            with self.lock:
                c = self.counters
                if saved:
                    c["saves"] += 1
                    c["written_bytes"] += written
                    c["reused_bytes"] += reused
                    c["delta_saves"] += reused > 0
                    c["compactions"] += compacted
                    c["shared_saves"] += bool(shared)
                elif outcome.startswith("skipped"):
                    c["skipped_saves"] += 1
                elif outcome.startswith("failed"):
                    c["failed_saves"] += 1
            return SaveResult(outcome, len(tokens), written if saved else 0, reused if saved else 0,
                              saved and compacted, time.monotonic() - start, removed)

        cfg = self.configuration
        if len(tokens) < cfg.minimum_tokens:
            return finish(f"skipped: shorter than {cfg.minimum_tokens} tokens")
        removed += self._remove_expired()
        include_draft = view.draft is not None
        try:
            payloads, arrays, sequences, head = self._plan(view, tokens)
        except ValueError as e:
            return finish(f"skipped: {e}")
        identity = self.identity.digest
        name = file_name(identity, tokens)
        with self.lock:
            existing = next((h for h in self.heads if h.file == name and h.tokens == tokens), None)
        if (existing is not None and existing.prefill_chunk == prefill_chunk and (existing.has_draft or not include_draft)
                and (existing.shared or not shared)):
            self._touch(name)
            return finish("present")
        keeps_shared = shared or (existing is not None and existing.shared)

        parent = None
        lin = view.lineage
        if (lin is not None and lin.tier == self.instance and lin.token_count <= len(tokens)
                and lin.head == file_name(identity, tokens[:lin.token_count])):
            parent = lin.sequences
        known = self.indexed_segments
        reuse = reuse_plan([r for r, _ in sequences], parent, lambda s: s in known and known[s].identity == identity,
                           self.maximum_segments)
        compacted = reuse.compacted

        segment_name = new_segment_name()
        segment_payloads, row_records, seq_out = [], [], []
        for record, data in sequences:
            item = ITEM_BYTES[record.dtype]
            extents = list(reuse.extents.get(record.name, []))
            for e in extents:
                g = rows_geometry(record.leading, record.trailing, item, e.end - e.start)
                reused += g[2] if g else 0
            first = reuse.first_new_row.get(record.name, record.base)
            if record.end > first:
                g = rows_geometry(record.leading, record.trailing, item, record.end - first)
                if g is None:
                    return finish(f"skipped: cannot persist rows of {record.name}")
                segment_payloads.append(self._row_payload(data, record, first))
                row_records.append(RowsRecord(record.name, record.dtype, record.leading, record.trailing, first,
                                              record.end, 0, g[2], 0))
                extents.append(Extent(segment_name, first, record.end))
            seq_out.append(replace(record, extents=extents))

        overhead = len(MAGIC) + FOOTER_BYTES + 65_536
        extent_count = sum(len(s.extents) for s in seq_out)
        segment_estimate = 0 if not row_records else sum(r.byteCount + 512 for r in row_records) + overhead
        head_estimate = overhead + sum(a.byteCount + 512 for a in arrays) + 512 * (len(seq_out) + extent_count)
        estimate = head_estimate + segment_estimate
        with self.lock:
            redundant = redundant_ancestors(self.heads, identity, tokens)
        continued = parent is not None or self._strict_prefixes(tokens) > 0
        now = _now()
        with self.lock:
            orphan_bytes = sum(self.segments[s].bytes for s in unreferenced_segments(self.heads, list(self.segments))
                               if s not in reuse.segments)
            victims = eviction_victims(self.heads, {k: v.bytes for k, v in self.segments.items()}, identity,
                                       cfg.max_bytes - orphan_bytes, estimate, reuse.segments,
                                       {r.file for r in redundant} | {name}, now, cfg.max_age)
        if victims is None:
            return finish(f"skipped: a {_megabytes(estimate)} write exceeds the {_megabytes(cfg.max_bytes)} quota")
        reclaimed = sum(v.bytes for v in victims)
        if self._free_bytes() + reclaimed < estimate + MINIMUM_FREE_BYTES:
            return finish(f"skipped: less than {_megabytes(estimate + MINIMUM_FREE_BYTES)} free on the volume")
        if victims:
            removed += self._remove([v.file for v in victims], "evictions", reuse.segments)
            with self.lock:
                remaining = {h.file for h in self.heads}
            for v in victims:
                if v.file not in remaining:
                    self._report(f"evicted {len(v.tokens)}-token state {v.file} ({_megabytes(v.bytes)})")
            if any(v.file in remaining for v in victims):
                return finish("failed: could not remove prefix cache states to make room")

        def temp(file):
            return self.configuration.directory / f".{file}.{os.getpid()}.{uuid.uuid4().hex}.tmp"

        older = [r.file for r in redundant if r.file != name]
        if not self.can_write(estimate, name):
            removed += self._remove(older, "replaced", reuse.segments)
        if not self.can_write(estimate, name):
            return finish("failed: could not reclaim prefix cache disk quota")

        new_segment = None
        try:
            if segment_payloads:
                tmp = temp(segment_name)
                try:
                    placed_rows = [replace(r) for r in row_records]

                    def seg_header(placed):
                        for i, r in enumerate(placed_rows):
                            r.offset, r.crc32 = placed[i][0], placed[i][2]
                        return encode_header(Segment(identity, placed_rows))

                    size = write_container(tmp, segment_payloads, [r.byteCount for r in row_records], seg_header)
                    os.rename(tmp, self._path(segment_name))
                finally:
                    _unlink_quiet(tmp)
                new_segment = SegmentEntry(segment_name, identity, size, {r.name: r for r in placed_rows})
                with self.lock:
                    self.segments[segment_name] = new_segment
                written += size
            head.continued = continued
            head.shared = True if keeps_shared else None
            head.splicingTokens = list(splicing) if splicing is not None else (existing.splicing_tokens if existing else None)
            head.prefillChunk = prefill_chunk
            head.sequences = seq_out
            placed_arrays = [replace(a) for a in arrays]

            def head_header(placed):
                for i, a in enumerate(placed_arrays):
                    a.offset, a.crc32 = placed[i][0], placed[i][2]
                head.arrays = placed_arrays
                return encode_header(head)

            tmp = temp(name)
            try:
                size = write_container(tmp, payloads, [a.byteCount for a in arrays], head_header)
                os.rename(tmp, self._path(name))
            finally:
                _unlink_quiet(tmp)
            written += size
            entry = Entry(name, identity, tokens, size, _now(), head.sequenceBytes, head.residentBytes,
                          head.draft is not None, continued, keeps_shared, prefill_chunk, {s.name: s for s in seq_out},
                          head.splicingTokens)
        except (ReadFailure, OSError, ValueError) as e:
            if new_segment is not None and self._remove_file(new_segment.file)[0]:
                with self.lock:
                    self.segments.pop(new_segment.file, None)
            return finish(f"failed: {e}")
        with self.lock:
            self.heads = [h for h in self.heads if h.file != name] + [entry]
        view.lineage = Lineage(self.instance, name, len(tokens), dict(entry.sequences))
        with self.lock:
            remaining_older = sorted({h.file for h in self.heads} & set(older))
        removed += self._remove(remaining_older, "replaced")
        result = finish("saved")
        self._report((f"saved shared {len(tokens)}-token prefix (" if shared else f"saved {len(tokens)} tokens (")
                      + f"{_megabytes(written)} written" + (f", {_megabytes(reused)} of rows reused" if reused else "")
                      + (", rows rewritten" if compacted else "") + f") in {result.seconds:.2f} s"
                      + (f", removed {removed} older file{'' if removed == 1 else 's'}" if removed else ""))
        return result

    def _strict_prefixes(self, tokens):
        with self.lock:
            return sum(1 for h in self.heads if h.identity == self.identity.digest and len(h.tokens) < len(tokens)
                       and tokens[:len(h.tokens)] == h.tokens)

    # ---- restore
    def restore(self, entry, allocate=None, *, layout=None, include_draft=False):
        """Read `entry` into a new StateView. `allocate(shape, dtype_name) -> np.ndarray` provides each buffer (zeroed by
        us); the default is host memory. A head or segment that fails a check is removed, with every head that uses it,
        before the error is rethrown; an I/O failure keeps the files for a later request."""
        with self.operations:
            start = time.monotonic()
            try:
                state, nbytes = self._read(entry, allocate or _default_allocate, layout, include_draft)
            except FileError as e:
                with self.lock:
                    files = [h.file for h in self.heads if h.file == entry.file or not h.segments.isdisjoint(e.segments)]
                self._remove(files, "rejected")
                with self.lock:
                    self.counters["restore_failures"] += 1
                    remaining = {h.file for h in self.heads}
                gone = sum(1 for f in files if f not in remaining)
                self._report(f"removed {gone} unusable state{'' if gone == 1 else 's'}: {e}")
                raise
            except Exception:
                with self.lock:
                    self.counters["restore_failures"] += 1
                raise
            self._touch(entry.file)
            seconds = time.monotonic() - start
            with self.lock:
                self.counters["restores"] += 1
                self.counters["restored_tokens"] += len(entry.tokens)
                self.counters["restored_bytes"] += nbytes
            self._report(f"restored {len(entry.tokens)} tokens ({_megabytes(nbytes)}) in {seconds:.2f} s")
            return RestoreResult(state, len(entry.tokens), nbytes, seconds)

    def _read(self, entry, allocate, expected, include_draft):
        try:
            fd = os.open(self._path(entry.file), os.O_RDONLY | os.O_CLOEXEC)
        except FileNotFoundError:
            raise FileError("file is missing") from None
        except OSError as e:
            raise ReadFailure(f"cannot open {entry.file}: {e.strerror}") from e
        readers = {}
        try:
            if os.fstat(fd).st_size != entry.bytes:
                raise FileError("file changed on disk")
            data, payload_end = read_header_data(fd, entry.bytes)
            try:
                raw = json.loads(data)
                header = head_from_json(raw)
            except (ValueError, RecursionError):
                raise FileError("file identity changed") from None
            if header.format != FORMAT_VERSION or header.kind != "head" or header.identity != self.identity.digest:
                raise FileError("file identity changed")
            validate_head(header, payload_end)
            if header.prefillChunk != entry.prefill_chunk:
                raise FileError("producing prefill arithmetic changed")
            if read_tokens(fd, header) != entry.tokens:
                raise FileError("token ids changed")
            if expected is not None:
                lin, att = [r.layer for r in header.linear], [r.layer for r in header.attention]
                if (sorted(lin) != sorted(expected.linear_layers) or sorted(att) != sorted(expected.attention_layers)
                        or len(set(lin)) != len(lin) or len(set(att)) != len(att)):
                    raise FileError("layer structure differs from the loaded model")
            fixed = {a.name: a for a in header.arrays}
            seqs = {s.name: s for s in header.sequences}
            known = self.indexed_segments
            nbytes = header.tokenCount * 4
            restored = {}
            view = StateView(header.tokenCount, list(header.ngramContext), header.compactStateWindows,
                             list(header.linear), list(header.attention), None)

            def array(name):
                nonlocal nbytes
                rec = fixed.get(name)
                if rec is None:
                    return None
                if rec.axis != 0 or (rec.shape and rec.shape[0] != rec.length):
                    raise FileError(f"{name} does not match its recorded state")
                nbytes += rec.byteCount
                view.fixed[name] = FixedArray(rec.dtype, self._read_array(fd, rec, allocate))
                return view.fixed[name]

            def rows(name, axis, base, live, rank):
                nonlocal nbytes
                rec = seqs.get(name)
                if rec is None:
                    return None
                if not (rec.axis == axis and rec.base == base and rec.live == live and len(rec.shape) == rank):
                    raise FileError(f"{name} does not match its recorded state")
                buf, read = self._read_sequence(rec, known, readers, allocate)
                nbytes += read
                restored[name] = rec
                view.sequences[name] = SeqArray(rec.dtype, buf, rec.axis, rec.base, rec.live)
                return buf

            def indexer(rec, prefix, committed):
                if not (rec.offset == committed and 0 <= rec.rawBase <= rec.offset and (rec.compactRaw or rec.rawBase == 0)
                        and rec.pooledRatio >= 1 and 0 <= rec.pooledCount <= rec.offset // rec.pooledRatio):
                    raise FileError(f"{prefix} ranges are invalid")
                raw = rows(f"{prefix}.raw", 1, rec.rawBase, rec.offset - rec.rawBase, 3)
                pooled = rows(f"{prefix}.pooled", 1, 0, rec.pooledCount, 3)
                if (raw is None and rec.offset != rec.rawBase) or (pooled is None and rec.pooledCount != 0):
                    raise FileError(f"{prefix} rows are missing")

            for rec in header.linear:
                for part in ("conv", "ssm", "ple"):
                    array(f"linear.{rec.layer}.{part}")
            for rec in header.attention:
                if rec.kvOffset != header.tokenCount or (expected is not None
                                                          and rec.indexer.compactRaw != expected.compact_indexer_raw):
                    raise FileError(f"attention layer {rec.layer} is not at the committed boundary")
                k = rows(f"kv.{rec.layer}.keys", 2, 0, rec.kvOffset, 4)
                v = rows(f"kv.{rec.layer}.values", 2, 0, rec.kvOffset, 4)
                if (k is None) != (v is None) or (k is None and rec.kvOffset != 0):
                    raise FileError(f"attention layer {rec.layer} rows are missing")
                indexer(rec.indexer, f"indexer.{rec.layer}", header.tokenCount)
            if include_draft and header.draft is not None:
                d = header.draft
                if not (header.tokenCount > 0 and d.kvOffset == header.tokenCount - 1 and not d.indexer.compactRaw):
                    raise FileError("draft cache is not aligned")
                rows("draft.kv.keys", 2, 0, d.kvOffset, 4)
                rows("draft.kv.values", 2, 0, d.kvOffset, 4)
                indexer(d.indexer, "draft.indexer", d.kvOffset)
                if array("draft.last_multi") is None:
                    raise FileError("draft cache is not aligned")
                view.draft = d
            view.lineage = Lineage(self.instance, entry.file, header.tokenCount, restored)
            return view, nbytes
        finally:
            os.close(fd)
            for r in readers.values():
                os.close(r)

    @staticmethod
    def _read_array(fd, rec, allocate):
        item = ITEM_BYTES.get(rec.dtype)
        if item is None or fixed_capacity(rec.shape, item) != rec.byteCount:
            raise FileError(f"{rec.name} has an invalid layout")
        raw = _read_exact(fd, rec.byteCount, rec.offset)
        if crc32(raw) != rec.crc32:
            raise FileError(f"{rec.name} checksum mismatch")
        out = allocate(list(rec.shape), rec.dtype)
        out[...] = np.frombuffer(raw, dtype=NUMPY_DTYPE[rec.dtype]).reshape(rec.shape)
        return out

    def _segment_fd(self, name, readers):
        if name in readers:
            return readers[name]
        if not is_segment_name(name):
            raise FileError("invalid segment name")
        try:
            readers[name] = os.open(self._path(name), os.O_RDONLY | os.O_CLOEXEC)
        except FileNotFoundError:
            raise FileError(f"segment {name} is missing") from None
        except OSError as e:
            raise ReadFailure(f"cannot open segment {name}: {e.strerror}") from e
        return readers[name]

    def _read_sequence(self, record, known, readers, allocate):
        """One sequence array assembled from its extents into a buffer of the recorded allocated shape, dead capacity
        zero. -> (buffer, bytes read)."""
        item = ITEM_BYTES.get(record.dtype)
        lay = None if item is None else layout(record.shape, record.axis, record.live, item)
        if lay is None or not tiles(record):
            raise FileError(f"{record.name} has an invalid layout")
        buf = allocate(list(record.shape), record.dtype)
        buf[...] = 0
        grid = buf.reshape(lay.leading, record.shape[record.axis], -1)
        read = 0
        for e in record.extents:
            seg = known.get(e.segment)
            rows = None if seg is None or seg.identity != self.identity.digest else seg.rows.get(record.name)
            if (rows is None or rows.dtype != record.dtype or rows.leading != record.leading
                    or rows.trailing != record.trailing or not (rows.start <= e.start and e.end <= rows.end)
                    or lay.leading * (rows.end - rows.start) * lay.row_bytes != rows.byteCount):
                raise FileError(f"segment {e.segment} does not hold {record.name} rows {e.start}..<{e.end}",
                                segments={e.segment})
            try:
                fd = self._segment_fd(e.segment, readers)
                payload = _read_exact(fd, rows.byteCount, rows.offset)
                if crc32(payload) != rows.crc32:
                    raise FileError(f"{record.name} checksum mismatch in segment {e.segment}")
            except FileError as err:
                if err.segments:
                    raise
                raise FileError(err.description, segments={e.segment}) from None
            count = rows.end - rows.start
            src = np.frombuffer(payload, dtype=NUMPY_DTYPE[record.dtype]).reshape(lay.leading, count, -1)
            grid[:, e.start - record.base:e.end - record.base, :] = src[:, e.start - rows.start:e.end - rows.start, :]
            read += lay.leading * (e.end - e.start) * lay.row_bytes
        return buf, read

    # ---- conversation ids
    def remember_conversation(self, tokens):
        """Attach exact generated ids to the longest aligned checkpoint they extend. Only extension lookup reads these ids;
        restore still uses the checkpoint's own tokens."""
        tokens = list(tokens)
        if not tokens or len(tokens) > MODEL_LIMIT or not all(0 <= t <= INT32_MAX for t in tokens):
            return False
        with self.operations:
            now = _now()
            with self.lock:
                cands = [h for h in self.heads if h.identity == self.identity.digest and tokens[:len(h.tokens)] == h.tokens
                         and not is_expired(h, now, self.configuration.max_age)]
            if not cands:
                return False
            entry = max(cands, key=lambda h: len(h.tokens))
            if entry.splicing_tokens == tokens:
                return True
            source = self._path(entry.file)
            tmp = self._path(f".{entry.file}.{uuid.uuid4().hex}.tmp")
            try:
                fd = os.open(source, os.O_RDONLY | os.O_CLOEXEC)
                try:
                    data, payload_end = read_header_data(fd, entry.bytes)
                    header = head_from_json(json.loads(data))
                    header.splicingTokens = tokens
                    js = encode_header(header)
                    size = payload_end + len(js) + FOOTER_BYTES
                    if (self.stored_bytes - entry.bytes + size > self.configuration.max_bytes
                            or self._free_bytes() < size + MINIMUM_FREE_BYTES):
                        self._report("conversation token metadata skipped: disk quota or free-space limit")
                        return False
                    out = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
                    try:
                        offset = 0
                        while offset < payload_end:
                            n = min(64 << 10, payload_end - offset)
                            _write_all(out, _read_exact(fd, n, offset))
                            offset += n
                        _write_all(out, js)
                        _write_all(out, footer(payload_end, len(js), crc32(js)))
                    finally:
                        os.close(out)
                finally:
                    os.close(fd)
                os.rename(tmp, source)
            except (FileError, ReadFailure, OSError, ValueError) as e:
                self._report(f"conversation token metadata not saved: {e}")
                return False
            finally:
                _unlink_quiet(tmp)
            with self.lock:
                for h in self.heads:
                    if h.file == entry.file:
                        h.splicing_tokens, h.bytes, h.last_used = tokens, size, now
                self.counters["written_bytes"] += size
            return True

    # ---- reports
    def json(self):
        now = _now()
        with self.lock:
            own = [h for h in self.heads if h.identity == self.identity.digest]
            classes = {}
            for e in own:
                v = value_of(e, own, self.identity.digest, now, self.configuration.max_age)
                classes[v] = classes.get(v, 0) + 1
            head_bytes = sum(h.bytes for h in self.heads)
            segment_bytes = sum(s.bytes for s in self.segments.values())
            m, c = self.maintenance, self.counters
            return {
                "directory": str(self.configuration.directory), "states": len(own),
                "conversations": classes.get(Value.CONVERSATION, 0), "parents": classes.get(Value.PARENT, 0),
                "one_off": classes.get(Value.ONE_OFF, 0), "shared": classes.get(Value.SHARED, 0),
                "shared_prefixes": sum(1 for h in own if h.shared), "held_tokens": sum(len(h.tokens) for h in own),
                "segments": len(self.segments), "bytes": head_bytes + segment_bytes, "head_bytes": head_bytes,
                "segment_bytes": segment_bytes, "max_bytes": self.configuration.max_bytes,
                "minimum_tokens": self.configuration.minimum_tokens,
                "max_age_seconds": self.configuration.max_age or 0,
                "restores": c["restores"], "restored_tokens": c["restored_tokens"], "restored_bytes": c["restored_bytes"],
                "restore_failures": c["restore_failures"], "saves": c["saves"], "delta_saves": c["delta_saves"],
                "shared_saves": c["shared_saves"], "compactions": c["compactions"], "written_bytes": c["written_bytes"],
                "reused_bytes": c["reused_bytes"], "skipped_saves": c["skipped_saves"], "failed_saves": c["failed_saves"],
                "evictions": c["evictions"], "expired_states": c["expired"], "replaced_states": c["replaced"],
                "rejected_files": c["rejected"], "deleted_states": c["deleted"], "removed_segments": c["removed_segments"],
                "opened": {"removed_other_builds": m.other_builds, "removed_expired": m.expired,
                           "removed_unreadable": m.unreadable, "removed_incomplete": m.incomplete,
                           "removed_orphan_segments": m.orphan_segments, "removed_over_quota": m.over_quota,
                           "removed_bytes": m.bytes},
            }


def _default_allocate(shape, dtype):
    return np.zeros(shape, dtype=NUMPY_DTYPE[dtype])


def _unlink_quiet(path):
    try:
        os.unlink(path)
    except OSError:
        pass


# ------------------------------------------------------------------- directory tools (no identity, no model)
def inspect(directory):
    """A directory's contents as any build sees them. -> dict(states, segments, segment_bytes, other_format_files,
    other_format_bytes, unreadable_files, unreadable_bytes, in_use, total_bytes)."""
    d = Path(directory)
    if not d.is_dir():
        raise ReadFailure(f"{d} is not a directory")
    rep = dict(states=[], segments=0, segment_bytes=0, other_format_files=0, other_format_bytes=0, unreadable_files=0,
               unreadable_bytes=0, in_use=False)
    try:
        fd = os.open(str(d / ".lock"), os.O_RDONLY | os.O_CLOEXEC)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            rep["in_use"] = True
        finally:
            os.close(fd)
    except OSError:
        pass
    for name in sorted(os.listdir(d)):
        if not (name.endswith("." + HEAD_EXT) or name.endswith("." + SEGMENT_EXT)):
            continue
        try:
            kind, item = read_file(d, name)
        except Exception:  # noqa: BLE001 - anything we cannot read is reported as unreadable
            rep["unreadable_files"] += 1
            rep["unreadable_bytes"] += _file_size(d / name)
            continue
        if kind == "head":
            rep["states"].append(dict(identity=item.identity, tokens=len(item.tokens), head_bytes=item.bytes,
                                      last_used=item.last_used, continued=item.continued, has_draft=item.has_draft,
                                      shared=item.shared, segments=len(item.segments)))
        elif kind == "segment":
            rep["segments"] += 1
            rep["segment_bytes"] += item.bytes
        else:
            rep["other_format_files"] += 1
            rep["other_format_bytes"] += item
    rep["total_bytes"] = (sum(s["head_bytes"] for s in rep["states"]) + rep["segment_bytes"] + rep["other_format_bytes"]
                          + rep["unreadable_bytes"])
    return rep


def clear_directory(directory):
    """Remove every state file from a directory no process is using. -> (files, bytes). Removes every file it can, then
    names each one it could not."""
    d = Path(directory)
    if not d.is_dir():
        raise ReadFailure(f"{d} is not a directory")
    fd = lock_directory(d)
    try:
        files, total, failures = 0, 0, []
        for name in sorted(os.listdir(d)):
            if not is_state_file(name):
                continue
            path, size = d / name, _file_size(d / name)
            try:
                os.unlink(path)
                files += 1
                total += size
            except FileNotFoundError:
                pass
            except OSError as e:
                failures.append(f"{path}: {e.strerror}")
        if failures:
            raise ReadFailure(f"removed {files} prefix cache file{'' if files == 1 else 's'} but could not remove "
                              f"{len(failures)}: " + "; ".join(failures))
        return files, total
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
