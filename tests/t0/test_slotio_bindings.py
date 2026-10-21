"""slotio from Python: pool coherence with the GPU, wait_seq, timeout margin. Run on the Spark under the guard."""

import json
import os
import tempfile
import time
import zlib
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
import triton.language as tl  # noqa: E402

slotio = pytest.importorskip("slotio")
from slotstream_spark.runtime import pool as rpool  # noqa: E402

pytestmark = pytest.mark.cuda

REC = 65536  # synthetic record size; the pool takes it from the manifest
NREC = 64
LAYERS = 48


def _tmp_dir(tmp_path_factory):
    """A scratch dir under $SLOTSTREAM_R/tmp on the Spark (pytest's tmp otherwise)."""
    r = os.environ.get("SLOTSTREAM_R")
    if not r:
        return tmp_path_factory.mktemp("slotio")
    base = Path(r) / "tmp"
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="slotio-", dir=base))


@pytest.fixture(scope="module")
def packed(tmp_path_factory):
    d = _tmp_dir(tmp_path_factory)
    rng = np.random.default_rng(1234)
    data = rng.integers(0, 256, size=NREC * REC, dtype=np.uint8)
    (d / "experts.bin").write_bytes(data.tobytes())
    (d / "experts.manifest.json").write_text(json.dumps(
        {"record_bytes": REC, "layers": LAYERS, "experts": 512, "alignment": 4096}))
    yield d, data
    import shutil

    shutil.rmtree(d, ignore_errors=True)


def _rec(data, i):
    return data[i * REC:(i + 1) * REC]


def _crc_words(b):
    """Additive 64-bit checksum of the record as little-endian uint32 words (the GPU kernel computes the same)."""
    return int(b.view(np.uint32).astype(np.int64).sum())


@triton.jit
def _checksum_kernel(map_ptr, slot_ptr, idx_ptr, out_ptr, nwords, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    idx = tl.load(idx_ptr + pid)
    slot = tl.load(map_ptr + idx)
    base = tl.load(slot_ptr + slot).to(tl.pointer_type(tl.uint32))
    acc = tl.zeros([BLOCK], dtype=tl.int64)
    for off in range(0, nwords, BLOCK):
        o = off + tl.arange(0, BLOCK)
        w = tl.load(base + o, mask=o < nwords, other=0)
        acc += w.to(tl.int64)
    tl.store(out_ptr + pid, tl.sum(acc, axis=0))


def _gpu_checksums(ep, idxs):
    idx_t = torch.tensor(idxs, dtype=torch.int32, device="cuda")
    out = torch.zeros(len(idxs), dtype=torch.int64, device="cuda")
    _checksum_kernel[(len(idxs),)](ep.map_tensor, ep.slot_ptr_tensor, idx_t, out, REC // 4, BLOCK=1024)
    torch.cuda.synchronize()
    return out.cpu().tolist()


@pytest.mark.parametrize("mem", ["host_alloc", None], ids=["host_alloc", "platform"])
def test_pool_bytes_and_gpu_checksum(packed, mem):
    d, data = packed
    ep = rpool.ExpertPool(d, 16, mem_type=mem, chunk_bytes=1 << 26)
    try:
        ids = np.array([3, 9, 40, 9], dtype=np.int32)
        slots = ep.ensure(0, ids)
        assert slots[1] == slots[3]
        for e, s in zip(ids, slots):
            assert bytes(ep.pool.slot_view(int(s))) == bytes(_rec(data, int(e))), (e, s)
        assert [_crc_words(np.array(ep.pool.slot_view(int(s)))) for s in slots[:3]] == \
            _gpu_checksums(ep, [3, 9, 40])
        # crc of the whole record through zlib as the CPU-side identity of the file bytes
        assert zlib.crc32(bytes(ep.pool.slot_view(int(slots[0])))) == zlib.crc32(bytes(_rec(data, 3)))
        # evict and reload different records: once progress passes the pins every slot is replaced
        prog = torch.zeros(1, dtype=torch.int64).pin_memory()
        prog[0] = 1000
        ep.set_progress_ptr(prog.data_ptr())
        more = np.arange(10, 26, dtype=np.int32)
        slots2 = ep.ensure(0, more, gen=2000)
        want = [_crc_words(_rec(data, int(e)).copy()) for e in more]
        assert _gpu_checksums(ep, [int(e) for e in more]) == want
        assert ep.pool.map_numpy()[3] == -1  # the first records were evicted, not left stale
        assert ep.owned_bytes() == (1 << 26)
        assert len(slots2) == 16
    finally:
        del ep


def test_overwrite_slot_is_visible_to_the_gpu(packed):
    d, data = packed
    ep = rpool.ExpertPool(d, 4, mem_type="host_alloc", chunk_bytes=1 << 26)
    first = ep.ensure(0, np.array([1, 2, 3, 4], dtype=np.int32), gen=1)
    assert _gpu_checksums(ep, [1, 2, 3, 4]) == [_crc_words(_rec(data, i).copy()) for i in (1, 2, 3, 4)]
    # progress passes gen 1: pins retire, so new ids reuse the slots
    prog = torch.zeros(1, dtype=torch.int64).pin_memory()
    prog[0] = 2
    ep.set_progress_ptr(prog.data_ptr())
    second = ep.ensure(0, np.array([11, 12, 13, 14], dtype=np.int32), gen=2)
    assert set(second.tolist()) == set(first.tolist())
    assert _gpu_checksums(ep, [11, 12, 13, 14]) == [_crc_words(_rec(data, i).copy()) for i in (11, 12, 13, 14)]
    # the evicted ids are unmapped, not stale
    assert ep.pool.map_numpy()[1] == -1


def test_exhausted_changes_nothing(packed):
    d, _ = packed
    ep = rpool.ExpertPool(d, 4, mem_type="host_alloc", chunk_bytes=1 << 26)
    ep.ensure(0, np.array([1, 2, 3, 4], dtype=np.int32), gen=1)
    before = ep.pool.map_numpy().copy()
    with pytest.raises(slotio.Exhausted):
        ep.ensure(0, np.array([7], dtype=np.int32), gen=2)
    assert (ep.pool.map_numpy() == before).all()


@triton.jit
def _release_store(ptr, value):
    tl.atomic_xchg(ptr, value, sem="release", scope="sys")


@triton.jit
def _slow_then_store(ptr, scratch, n, value):
    acc = tl.zeros([1024], dtype=tl.float32)
    for i in range(0, n):
        acc += tl.load(scratch + tl.arange(0, 1024)) * 1.0001
    tl.store(scratch + tl.arange(0, 1024), acc)
    tl.atomic_xchg(ptr, value, sem="release", scope="sys")


def test_wait_seq_returns_when_a_kernel_writes_seq():
    seq = torch.zeros(1, dtype=torch.int64).pin_memory()
    scratch = torch.ones(1024, dtype=torch.float32, device="cuda")
    _slow_then_store[(1,)](seq, scratch, 20000, 5)
    t0 = time.perf_counter_ns()
    rpool.wait_seq(seq.data_ptr(), 5, 5_000_000_000)
    assert int(seq[0]) == 5
    assert time.perf_counter_ns() - t0 < 5_000_000_000
    torch.cuda.synchronize()


def test_wait_seq_timeout_margin():
    seq = torch.zeros(1, dtype=torch.int64).pin_memory()
    timeout = 200_000_000  # >= 100 ms: the spin yields, the call overhead stays inside the margin
    t0 = time.perf_counter_ns()
    with pytest.raises(slotio.Timeout):
        rpool.wait_seq(seq.data_ptr(), 1, timeout)
    dt = time.perf_counter_ns() - t0
    assert dt >= timeout
    assert dt <= timeout + max(0.1 * timeout, 5_000_000), dt - timeout


def test_resize_keeps_surviving_slots(packed):
    d, data = packed
    ep = rpool.ExpertPool(d, 8, mem_type="host_alloc", chunk_bytes=1 << 26)
    slots = ep.ensure(0, np.array([5, 6], dtype=np.int32), gen=1)
    ep.resize_slots(20000 // 8)  # grow to 2500 slots (1 chunk = 1024 slots at 64 KiB)
    assert bytes(ep.pool.slot_view(int(slots[0]))) == bytes(_rec(data, 5))
    ep.resize_slots(8)
    assert ep.pool.slots == 8


def test_platform_memory_real_geometry():
    """The platform.toml pool type with the production geometry: 65 GiB of VA, 388 records of 2,764,800 B per 1 GiB chunk."""
    mem = rpool.platform_mem_type()
    p = slotio.Pool(mem)
    p.resize_slots(389)  # two chunks: slot 388 opens chunk 1
    assert p.chunks == 2 and p.owned_bytes() == 2 << 30
    view = p.slot_view(388)
    view[:16] = np.arange(16, dtype=np.uint8)
    assert bytes(p.slot_view(388)[:16]) == bytes(range(16))
    assert p.slot_dev(388) - p.slot_dev(387) != 2764800  # chunk boundary, not the next record
    p.resize_slots(100)
    assert p.chunks == 1 and len(p.stats()["free_ns"]) == 1
    p.resize_slots(0)
