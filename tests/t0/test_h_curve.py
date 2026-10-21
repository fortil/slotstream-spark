import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import h_curve  # noqa: E402


def test_clock_evicts_unreferenced_first():
    c = h_curve.Clock(2)
    assert [c.access(k) for k in (1, 2, 1, 3, 1)] == [False, False, True, False, True]
    assert c.access(2) is False


def test_curve_monotone_and_full_cache_only_cold_misses(tmp_path):
    rng = np.random.default_rng(0)
    layers, topk, steps = 4, 2, 200
    hot = rng.integers(0, 6, size=(steps, layers, topk))
    for i in range(3):
        np.savez(tmp_path / f"trace-{i:02d}.npz", ids=hot.astype(np.int16))
    keys, nl, n = h_curve.load_keys(tmp_path)
    assert (nl, n) == (layers, 3)
    rows = h_curve.curve(keys, nl, points=(1, 3, 6, 512))
    lru = [r["lru"] for r in rows]
    assert lru == sorted(lru)
    total = sum(len(k) for k in keys)
    assert rows[-1]["lru"] == pytest.approx(1 - len(set(k for ks in keys for k in ks)) / total)
    assert rows[-1]["clock"] == pytest.approx(rows[-1]["lru"])
