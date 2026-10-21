"""Soak tool dry-run: 2,000 reads --no-gpu on a synthetic experts file (fake backend; real pool when slotio is built)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import soak_safe_path as sp  # noqa: E402

REC, LAYERS, EXPERTS = 8192, 3, 64


@pytest.fixture()
def packed(tmp_path):
    data = np.random.default_rng(3).integers(0, 256, LAYERS * EXPERTS * REC, dtype=np.uint8)
    (tmp_path / "experts.bin").write_bytes(data.tobytes())
    (tmp_path / "experts.manifest.json").write_text(json.dumps(
        {"record_bytes": REC, "layers": LAYERS, "experts": EXPERTS, "alignment": 4096}))
    return tmp_path, data


class FakeBackend:
    def __init__(self, path, data, corrupt=False):
        self.path, self.data, self.record_bytes = str(path), data, REC
        self.layers, self.experts, self.slots, self.corrupt = LAYERS, EXPERTS, {}, corrupt

    def ensure(self, layer, ids):
        out = []
        for i in ids:
            self.n = getattr(self, "n", 0) + 1
            self.slots[self.n % 16] = (layer * EXPERTS + int(i)) * REC
            out.append(self.n % 16)
        return out, len(ids), 1_000_000

    def slot_bytes(self, slot):
        b = bytearray(self.data[self.slots[slot]:self.slots[slot] + REC].tobytes())
        if self.corrupt:
            b[0] ^= 1
        return bytes(b)


def args(**kw):
    a = sp.parse(["--reads", str(kw.pop("reads", 2000)), "--no-gpu", "--rate-gbps", "1000", "--temp-limit-mc", "50000"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def fast(**kw):
    return dict(sleep=lambda s: None, temp=lambda p: 30000, **kw)


def test_two_thousand_reads_no_mismatch(packed):
    d, data = packed
    res = sp.run(FakeBackend(d / "experts.bin", data), args(), **fast())
    assert res["reads"] >= 2000 and res["crc_checks"] >= 1
    assert res["errors"] == res["timeouts"] == res["checksum_mismatches"] == 0
    assert res["max_read_ms"] > 0 and res["p99_read_ms"] >= res["p50_read_ms"]


def test_corrupt_slot_is_a_mismatch(packed):
    d, data = packed
    res = sp.run(FakeBackend(d / "experts.bin", data, corrupt=True), args(), **fast())
    assert res["checksum_mismatches"] >= 1


def test_pause_when_hot_and_rate_cap(packed):
    d, data = packed
    sleeps, temps = [], iter([60000, 60000, 30000] + [30000] * 100000)
    res = sp.run(FakeBackend(d / "experts.bin", data), args(reads=100),
                 sleep=sleeps.append, temp=lambda p: next(temps))
    assert res["pauses"] == 2 and sleeps[:2] == [30.0, 30.0]
    t = {"v": 0.0}
    slept = []

    def sl(s):
        slept.append(s)
        t["v"] += s
    res = sp.run(FakeBackend(d / "experts.bin", data), args(reads=200, rate_gbps=0.1), now=lambda: t["v"], sleep=sl,
                 temp=lambda p: 30000)
    assert res["gbps"] <= 0.1 + 1e-6 and slept


def test_real_pool_when_slotio_is_built(packed):
    pytest.importorskip("slotio")
    pytest.importorskip("numpy")
    d, _ = packed
    try:
        b = sp.PoolBackend(d / "experts.bin", chunks=1, slots=32)
    except Exception as e:                                       # pool memory needs the CUDA runtime
        pytest.skip(f"pool unavailable: {e}")
    res = sp.run(b, args(reads=300), **fast())
    assert res["errors"] == 0 and res["checksum_mismatches"] == 0 and res["reads"] >= 300, res.get("last_error")


def test_temp_limit_comes_from_the_device(monkeypatch):
    from bench import eligibility
    monkeypatch.setattr(eligibility, "nvme_limit_mc", lambda *a, **k: (77850, "temp1_max-5000"))
    assert sp.temp_limit_mc() == 77850


def test_unreadable_device_limit_exits_2(monkeypatch):
    from bench import eligibility
    monkeypatch.setattr(eligibility, "nvme_limit_mc", lambda *a, **k: (None, "unreadable"))
    with pytest.raises(SystemExit) as e:
        sp.parse(["--reads", "10", "--no-gpu"])
    assert e.value.code == 2
