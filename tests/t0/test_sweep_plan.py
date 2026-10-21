import numpy as np
import pytest

from slotstream_spark.engine import sweep


def _routing(seed):
    rng = np.random.default_rng(seed)
    r = int(rng.integers(1, 3000))
    n_exp = int(rng.integers(1, 512))
    pool = rng.choice(512, size=n_exp, replace=False)
    ids = pool[rng.integers(0, n_exp, size=r)].astype(np.int32)
    resident = rng.random(512) < rng.random()
    return ids, resident


def test_1000_random_routings():
    for seed in range(1000):
        ids, resident = _routing(seed)
        plan = sweep.plan_sweep(ids, resident)
        r = ids.size
        # a permutation of the rows
        assert np.array_equal(np.sort(plan.order), np.arange(r))
        # residents first: no resident row after a non-resident one
        is_res = resident[ids[plan.order]]
        assert not np.any(~is_res[:-1] & is_res[1:]), seed
        seen_nonres = False
        cover = 0
        for g in plan.groups:
            assert 1 <= g.experts.size <= 32
            assert not (g.resident and seen_nonres)
            seen_nonres |= not g.resident
            rows = ids[plan.order[g.row_lo:g.row_hi]]
            # homogeneous: every expert of the group has the group's residency, and the rows are exactly its experts' rows
            assert all(bool(resident[e]) == g.resident for e in g.experts)
            assert set(rows.tolist()) == set(g.experts.tolist())
            assert g.row_lo == cover
            cover = g.row_hi
        assert cover == r
        # stable: inside one expert the rows keep their original order
        for e in np.unique(ids):
            sel = plan.order[ids[plan.order] == e]
            assert np.all(np.diff(sel) > 0)


def test_groups_never_exceed_group_size_and_split_at_32():
    ids = np.repeat(np.arange(70, dtype=np.int32), 2)
    plan = sweep.plan_sweep(ids, np.zeros(512, dtype=bool))
    assert [g.experts.size for g in plan.groups] == [32, 32, 6]
    assert not any(g.resident for g in plan.groups)


def test_resident_and_nonresident_do_not_mix():
    ids = np.array([5, 6, 7, 8], dtype=np.int32)
    res = np.zeros(512, dtype=bool)
    res[[6, 8]] = True
    plan = sweep.plan_sweep(ids, res)
    assert [(g.resident, g.experts.tolist()) for g in plan.groups] == [(True, [6, 8]), (False, [5, 7])]


def test_admission_quota():
    assert [sweep.admission_quota(s) for s in (776, 3880, 19012)] == [16, 80, 396]
    assert sweep.admission_quota(1) == 1
    assert sweep.admission_quota(10 ** 9) == 512


def test_hottest_ties_by_id():
    counts = np.zeros(512, dtype=np.int64)
    counts[[9, 3, 40, 7]] = [5, 5, 9, 1]
    assert sweep.hottest(counts, 3).tolist() == [40, 3, 9]
    assert sweep.hottest(counts, 10).tolist() == [40, 3, 9, 7]   # experts with zero rows are never admitted
    assert sweep.hottest(counts, 0).tolist() == []


class _Ev:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def synchronize(self):
        self.log.append(("sync", self.name))


def test_iter_groups_alternates_buffers_and_waits_for_the_event():
    ids = np.repeat(np.arange(32 * 4, dtype=np.int32), 1)
    plan = sweep.plan_sweep(ids, np.zeros(512, dtype=bool))
    log = []
    it = sweep.GroupIterator(plan, buffers=2)
    bufs = []
    for i, (buf, g) in enumerate(it):
        bufs.append(buf)
        log.append(("use", i, buf))
        it.events[buf] = _Ev(log, i)
    assert bufs == [0, 1, 0, 1]
    # group 2 reuses buffer 0 only after the event of group 0, group 3 after the event of group 1
    assert log.index(("sync", 0)) < log.index(("use", 2, 0))
    assert log.index(("sync", 1)) < log.index(("use", 3, 1))
    assert it.max_alive <= 2


def test_iter_groups_four_buffers_keep_four_alive():
    ids = np.repeat(np.arange(32 * 4, dtype=np.int32), 1)
    plan = sweep.plan_sweep(ids, np.zeros(512, dtype=bool))
    it = sweep.GroupIterator(plan, buffers=4)
    bufs = []
    for buf, g in it:
        bufs.append(buf)
        it.events[buf] = object()
    assert bufs == [0, 1, 2, 3]
    assert it.max_alive <= 4


def test_iter_groups_resident_groups_need_no_buffer():
    ids = np.array([1, 2, 300], dtype=np.int32)
    res = np.zeros(512, dtype=bool)
    res[[1, 2]] = True
    got = list(sweep.GroupIterator(sweep.plan_sweep(ids, res)))
    assert [b for b, _ in got] == [None, 0]


def test_bad_inputs():
    with pytest.raises(ValueError):
        sweep.plan_sweep(np.array([600], dtype=np.int32), np.zeros(512, dtype=bool))
    with pytest.raises(ValueError):
        sweep.plan_sweep(np.array([1], dtype=np.int64), np.zeros(512, dtype=bool))
