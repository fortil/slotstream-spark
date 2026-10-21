"""NgramRows prefetch/wait from Python on a synthetic n-gram file, and the NgramSource split used by the decode loop."""
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

slotio = pytest.importorskip("slotio")
if not hasattr(slotio, "NgramRows"):                           # the source directory on the Mac, not the built module
    pytest.skip("slotio native module not built", allow_module_level=True)

pytestmark = pytest.mark.cuda                                  # runs on the Spark with the other native tests

ROW = 100
NROWS = 5000


@pytest.fixture(scope="module")
def table():
    base = Path(os.environ["SLOTSTREAM_R"]) / "tmp" if os.environ.get("SLOTSTREAM_R") else None
    d = tempfile.mkdtemp(prefix="rowscache-", dir=base)
    data = np.random.default_rng(5).integers(0, 256, NROWS * ROW, dtype=np.uint8)
    path = Path(d) / "ngram.bin"
    data.tofile(path)
    f = slotio.File.open_direct(str(path))
    yield data, f, slotio.Reader()
    path.unlink()
    os.rmdir(d)


def _direct(data, gids):
    return np.concatenate([data[g * ROW:(g + 1) * ROW] for g in gids])


def test_prefetched_bytes_equal_direct_pread(table):
    data, f, reader = table
    rows = slotio.NgramRows(reader, f, ROW, 1000)
    gids = np.array([(i * 313 + 7) % NROWS for i in range(16)], dtype=np.int64)
    t = rows.prefetch(gids)
    assert t.size == 16 and t.misses == 16 and t.pending
    buf = np.empty(16 * ROW, dtype=np.uint8)
    rows.wait(t, buf)
    assert not t.pending
    assert np.array_equal(buf, _direct(data, gids))
    with pytest.raises(Exception):
        rows.wait(t, buf)                                      # a ticket is single-use


def test_hit_counter_rises_on_repeated_text(table):
    data, f, reader = table
    rows = slotio.NgramRows(reader, f, ROW, 1000)
    gids = np.arange(16, dtype=np.int64) * 7
    buf = np.empty(16 * ROW, dtype=np.uint8)
    rows.wait(rows.prefetch(gids), buf)
    h0 = rows.cache_stats()[0]
    t2 = rows.prefetch(gids)
    assert t2.misses == 0
    rows.wait(t2, buf)
    assert rows.cache_stats()[0] == h0 + 16
    assert np.array_equal(buf, _direct(data, gids))


def test_a_cache_entry_is_never_reused_for_another_gid(table):
    data, f, reader = table
    rows = slotio.NgramRows(reader, f, ROW, 4)               # tiny FIFO: eviction pressure
    buf = np.empty(ROW, dtype=np.uint8)
    for rnd in range(3):
        for g in (10, 11, 12, 13, 14, 15, 10, 11):
            rows.wait(rows.prefetch(np.array([g], dtype=np.int64)), buf)
            assert np.array_equal(buf, data[g * ROW:(g + 1) * ROW]), (rnd, g)


def test_out_of_range_and_short_destination(table):
    _, f, reader = table
    rows = slotio.NgramRows(reader, f, ROW, 10)
    with pytest.raises(Exception):
        rows.prefetch(np.array([NROWS], dtype=np.int64))
    t = rows.prefetch(np.array([1, 2], dtype=np.int64))
    with pytest.raises(Exception):
        rows.wait(t, np.empty(ROW, dtype=np.uint8))          # destination too small for 2 rows
    rows.wait(t, np.empty(2 * ROW, dtype=np.uint8))          # the ticket is still usable


def test_ticket_dropped_without_wait_is_safe(table):
    data, f, reader = table
    rows = slotio.NgramRows(reader, f, ROW, 100)
    t = rows.prefetch(np.arange(32, dtype=np.int64))
    del t
    buf = np.empty(ROW, dtype=np.uint8)
    rows.wait(rows.prefetch(np.array([3], dtype=np.int64)), buf)
    assert np.array_equal(buf, data[3 * ROW:4 * ROW])
