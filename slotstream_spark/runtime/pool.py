"""Python face of slotio: the expert slot pool, row readers and the sequence-word wait."""

import json
import struct
from pathlib import Path

from slotstream_spark import config as _config

# platform.toml memory.pool_memory -> slotio mem_type
_MEM_TYPES = {"host_alloc": "host_alloc", "vmm_host_numa": "vmm_host_numa"}


def platform_mem_type():
    """The pool memory type decided in platform.toml (step 24)."""
    value = _config.load()["memory"]["pool_memory"]
    if value not in _MEM_TYPES:
        raise RuntimeError(f"platform.toml memory.pool_memory={value!r} is not a host-pool type")
    return _MEM_TYPES[value]


def _slotio():
    import slotio  # built by tools/build_slotio.sh, on PYTHONPATH through tools/spark/py.sh

    return slotio


class ExpertPool:
    """Slots of packed expert records streamed from experts.bin with an exact, transactional ensure."""

    def __init__(self, packed_dir, slots, mem_type=None, *, experts_name="experts.bin",
                 manifest_name="experts.manifest.json", chunk_bytes=1 << 30, max_slots=25088, reader=None):
        sl = _slotio()
        packed = Path(packed_dir)
        man = json.loads((packed / manifest_name).read_text())
        self.record_bytes = int(man["record_bytes"])
        layers, experts = int(man["layers"]), int(man["experts"])
        # the MTP layer (index 48) shares the table: 49 layers in the map, 48 in the manifest
        map_layers = layers + 1 if layers == 48 else layers
        self.mem_type = mem_type or platform_mem_type()
        self.reader = reader or sl.Reader()
        self.file = sl.File.open_direct(str(packed / experts_name), int(man.get("alignment", 4096)))
        self.pool = sl.Pool(self.mem_type, record_bytes=self.record_bytes,
                            slots_per_chunk=chunk_bytes // self.record_bytes, chunk_bytes=chunk_bytes,
                            layers=map_layers, experts=experts, max_slots=max_slots)
        self._ensurer = sl.Ensurer(self.pool, self.reader, self.file)
        self.resize_slots(slots)
        self._gen = 0
        self._prefetch = None

    def ensure(self, layer, ids, gen=None):
        """Slot of every id (int32 array, input order). Raises slotio.Exhausted / slotio.IoError, publishing nothing."""
        import numpy as np

        if gen is None:
            self._gen += 1
            gen = self._gen
        slots, timing = self._ensurer.ensure(layer, np.ascontiguousarray(ids, dtype=np.int32), gen)
        self.last_timing = timing
        return slots

    def resize_slots(self, n):
        self._ensurer.resize(n)

    def enable_prefetch(self, slot_cap=64, min_evictable=256):
        """Attach the speculative prefetcher (slotio.Prefetch) to the ensurer; returns self."""
        if self._prefetch is None:
            self._prefetch = _slotio().Prefetch(self._ensurer, self.reader, self.file, slot_cap, min_evictable)
        return self

    def disable_prefetch(self):
        if self._prefetch is not None:
            self._prefetch.expire_all()
            self._prefetch = None            # the destructor detaches it from the ensurer

    def issue(self, layer, ids):
        import numpy as np
        return self._prefetch.issue(int(layer), np.ascontiguousarray(ids, dtype=np.int32))

    def expire_layer(self, layer):
        return self._prefetch.expire_layer(int(layer))

    def expire_all(self):
        return self._prefetch.expire_all() if self._prefetch is not None else 0

    def prefetch_stats(self):
        return dict(self._prefetch.stats(), outstanding=self._prefetch.outstanding()) if self._prefetch else None

    @property
    def map_tensor(self):
        """int32[layers*experts] on the GPU, aliasing the pinned table (no copy)."""
        import torch

        return torch.as_tensor(self.pool.map_table(), device="cuda")

    @property
    def slot_ptr_tensor(self):
        """uint64 slot pointers on the GPU, aliasing the pinned table (no copy)."""
        import torch

        return torch.as_tensor(self.pool.slot_ptr_table(), device="cuda")

    def set_progress_ptr(self, addr):
        self._ensurer.set_progress_ptr(int(addr))

    def owned_bytes(self):
        return self.pool.owned_bytes()

    def stats(self):
        s = dict(self.pool.stats())
        s["reader"] = self.reader.stats()
        s["pinned"] = self._ensurer.pinned()
        return s


def wait_seq(addr, target, timeout_ns):
    """Spin until the uint64 at `addr` (an int address) is >= target; raises slotio.Timeout."""
    _slotio().wait_seq(int(addr), int(target), int(timeout_ns))


def safetensors_tensor_offsets(path, names):
    """Absolute file offsets of each named tensor's data in a .safetensors file."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return {k: 8 + n + header[k]["data_offsets"][0] for k in names}, header


def embed_rows(reader, trunk_path, prefix="embed_tokens", vocab=None):
    """EmbedRows over the embed_tokens weight/scales/biases of trunk.safetensors (offsets from its header)."""
    sl = _slotio()
    with open(trunk_path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    keys = {}
    for part in ("weight", "scales", "biases"):
        hits = [k for k in header if k.endswith(f"{prefix}.{part}")]
        if len(hits) != 1:
            raise RuntimeError(f"expected exactly one {prefix}.{part} in {trunk_path}, found {hits}")
        keys[part] = hits[0]

    def row_bytes(meta):
        shape, dt = meta["shape"], meta["dtype"]
        size = {"U32": 4, "BF16": 2, "F16": 2, "F32": 4, "I32": 4}[dt]
        return int(shape[1]) * size

    metas = {p: header[k] for p, k in keys.items()}
    base = 8 + n
    f = sl.File.open_direct(str(trunk_path))
    rows = sl.EmbedRows(reader, f, base + metas["weight"]["data_offsets"][0], base + metas["scales"]["data_offsets"][0],
                        base + metas["biases"]["data_offsets"][0], row_bytes(metas["weight"]), row_bytes(metas["scales"]),
                        row_bytes(metas["biases"]), int(vocab or metas["weight"]["shape"][0]))
    rows._file = f  # keep the file object alive with the rows
    return rows
