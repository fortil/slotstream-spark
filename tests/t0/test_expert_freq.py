import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import expert_freq as ef  # noqa: E402


def _traces(tmp_path):
    rng = np.random.default_rng(3)
    for i in range(3):
        ids = rng.integers(0, 6, size=(20, 4, 2)).astype(np.int16)
        np.savez(tmp_path / f"trace-{i:02d}.npz", ids=ids)


def test_build_counts_every_occurrence(tmp_path):
    _traces(tmp_path)
    counts, n, steps = ef.build(tmp_path)
    assert counts.shape == (4, 6) and n == 3 and steps == 60
    assert int(counts.sum()) == 3 * 20 * 4 * 2


def test_select_orders_by_count_then_id_and_skips_zero():
    counts = np.array([[5, 5, 9, 0, 1, 0], [0, 0, 0, 0, 7, 0]])
    got = ef.select(counts, 2)
    assert got[0].tolist() == [2, 0]            # 9 first, tie 5/5 broken by lower id
    assert got[1].tolist() == [4]               # only one expert ever appears
    assert got[0].dtype == np.int32


def test_check_table_rejects_wrong_layers():
    ok = {"layers": 48, "experts": 512, "counts": [[0] * 512 for _ in range(48)]}
    ef.check_table(ok)
    with pytest.raises(ValueError):
        ef.check_table(dict(ok, layers=47))
    with pytest.raises(ValueError):
        ef.check_table(dict(ok, counts=ok["counts"][:47]))


def test_coverage_388_arithmetic():
    c = np.zeros((48, 512), dtype=np.int64)
    c[:, :388] = 2
    c[:, 388:] = 1
    want = (48 * 388 * 2) / (48 * 388 * 2 + 48 * 124)
    assert ef.coverage(c) == pytest.approx(want)


def test_main_writes_table(tmp_path):
    cap = tmp_path / "20260101T000000-x"
    (cap / "traces").mkdir(parents=True)
    ids = np.zeros((5, 48, 10), dtype=np.int16)
    np.savez(cap / "traces" / "trace-00.npz", ids=ids)
    out = tmp_path / "t.json"
    assert ef.main([str(cap / "traces"), "--out", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["source"] == cap.name and d["accesses"] == 5 * 480 and d["traces"] == 1
