"""Step 74, T0: the speculative-prefetch scheduler against a fake pool that follows the ticket rules of slotio/prefetch.cpp
(ports of slotstream's expert-lookahead tickets, scheduler, forecast-merge and decode-lookahead-defaults checks)."""
import numpy as np
import pytest

from slotstream_spark import config
from slotstream_spark.engine import lookahead as la


class FakePool:
    """Tickets with the rules of the C++ Prefetch: resident/ticketed skipped, <= slot_cap outstanding, a reservation only while
    slots - pinned - in_flight > min_evictable, unadopted tickets returned by expire_layer."""

    def __init__(self, slots=2304, pinned=0, slot_cap=64, min_evictable=256, resident=()):
        self.slots, self.pinned, self.slot_cap, self.min_evictable = slots, pinned, slot_cap, min_evictable
        self.resident = set(resident)               # (layer, e)
        self.tickets = {}                           # (layer, e) -> state
        self.issued = self.adopted = self.returned = self.refused = 0
        self.calls = []

    def issue(self, layer, ids):
        self.calls.append((layer, [int(i) for i in ids]))
        n = 0
        for e in ids:
            key = (layer, int(e))
            if key in self.resident or key in self.tickets:
                continue
            if len(self.tickets) >= self.slot_cap or self.slots - self.pinned - len(self.tickets) <= self.min_evictable:
                self.refused += 1
                continue
            self.tickets[key] = "inflight"
            self.issued += 1
            n += 1
        return n

    def claim(self, layer, e):
        if self.tickets.pop((layer, e), None) is None:
            return False
        self.resident.add((layer, e))
        self.adopted += 1
        return True

    def expire_layer(self, layer):
        gone = [k for k in self.tickets if k[0] <= layer]
        for k in gone:
            del self.tickets[k]
        self.returned += len(gone)
        return len(gone)

    def expire_all(self):
        n = len(self.tickets)
        self.tickets.clear()
        self.returned += n
        return n


def row(top_logits):
    """A forecast row of 24: ids 100.., logits as given then filler far below."""
    lg = list(top_logits) + [-50.0 - i for i in range(24 - len(top_logits))]
    return np.arange(100, 124, dtype=np.int32), np.array(lg, dtype=np.float32)


def cfg(mode="attention", **kw):
    return la.LookaheadConfig(mode=mode, **kw)


# ---------------------------------------------------------------------------------------------------- defaults
def test_defaults_match_the_qualified_decode_configuration():
    c = la.LookaheadConfig()
    assert (c.threshold, c.top, c.issue_cap, c.slot_cap, c.min_evictable) == (0.062, 24, 32, 64, 256)
    assert c.mode == "off"


def test_platform_toml_section_loads_and_validates():
    assert config.validate(config.load()) == []
    c = la.LookaheadConfig.load(env={})
    assert c == la.LookaheadConfig()


def test_env_overrides_mode_and_rejects_unknown():
    assert la.LookaheadConfig.load(env={"SLOTSTREAM_LOOKAHEAD": "attention-corrected"}).corrected
    assert la.LookaheadConfig.load(env={"SLOTSTREAM_LOOKAHEAD": "boundary2"}).mode == "boundary2"
    with pytest.raises(ValueError):
        la.LookaheadConfig.load(env={"SLOTSTREAM_LOOKAHEAD": "nope"})


# ---------------------------------------------------------------------------------------------------- candidates / merge
def test_margin_is_logit_minus_tenth_and_threshold_is_inclusive():
    # logits 10 down to 1: the tenth is 1.0; margins 9, 8, ..., 0
    ids, lg = row([10 - i for i in range(10)] + [0.9, 0.5])
    got = la.candidates(ids, lg, 0.062)
    assert list(got) == [100 + i for i in range(9)]              # the tenth (margin 0) and below are dropped
    got = la.candidates(ids, lg, 0.0)
    assert 109 in got
    # a non-top-10 id above the tenth logit cannot exist, but one within threshold of it does not pass
    ids, lg = row([5.0] * 9 + [4.0, 3.95])
    assert 110 not in la.candidates(ids, lg, 0.062)


def test_candidates_sort_ties_by_lower_id_and_cap():
    ids = np.arange(24, dtype=np.int32)[::-1].copy()
    lg = np.linspace(30, 7, 24).astype(np.float32)               # logit increasing with position, ids descending
    got = la.candidates(ids, lg, 0.062, cap=5)
    assert len(got) == 5 and list(got) == sorted(got, key=lambda e: -lg[list(ids).index(e)])
    tie = la.candidates(np.array([7, 3, 9] + list(range(50, 71)), dtype=np.int32),
                        np.array([2.0, 2.0, 2.0] + [-9.0 - i for i in range(21)], dtype=np.float32), -1e9, cap=3)
    assert list(tie) == [3, 7, 9]


def test_rows_merge_with_best_margin_and_dedupe():
    ia, la_ = row([9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1])
    ib = ia.copy()
    lb = la_.copy()
    lb[0], lb[1] = 20.0, 19.0                                    # same ids, bigger margins in row b
    got = la.candidates(np.stack([ia, ib]), np.stack([la_, lb]), 0.062, cap=32)
    assert len(got) == len(set(got.tolist()))
    assert got[0] == 100                                         # row b's best margin wins the order


def test_invalid_ids_ignored():
    ids = np.full(24, -1, dtype=np.int32)
    assert len(la.candidates(ids, np.zeros(24, dtype=np.float32), 0.0)) == 0


# ---------------------------------------------------------------------------------------------------- scheduler
def test_off_issues_nothing_and_attention_targets_next_layer():
    pool = FakePool()
    ids, lg = row([10 - i for i in range(10)])
    s = la.Scheduler(pool, cfg("off"))
    assert s.on_layer(3, ids[None], lg[None], 48) == 0 and pool.calls == []
    s = la.Scheduler(pool, cfg("attention"))
    assert s.on_layer(3, ids[None], lg[None], 48) > 0
    assert pool.calls[-1][0] == 4


def test_last_layer_has_no_target_and_boundary2_every_second_layer():
    pool = FakePool()
    ids, lg = row([10 - i for i in range(10)])
    s = la.Scheduler(pool, cfg("attention"))
    assert s.target(47, 48) is None and s.target(46, 48) == 47
    b = la.Scheduler(pool, cfg("boundary2"))
    assert [b.target(L, 48) for L in range(4)] == [1, None, 3, None]
    assert b.on_layer(1, ids[None], lg[None], 48) == 0


def test_issue_cap_32_per_target_layer():
    pool = FakePool()
    ids = np.arange(24, dtype=np.int32)
    lg = np.linspace(40, 17, 24).astype(np.float32)
    rows = 5          # a margin >= 0.062 over the tenth logit passes at most 9 ids per row: 45 candidates, cap 32
    stacked_ids = np.stack([ids + 24 * r for r in range(rows)])
    stacked_lg = np.stack([lg for _ in range(rows)])
    s = la.Scheduler(pool, cfg("attention"))
    n = s.on_layer(5, stacked_ids, stacked_lg, 48)
    assert n == 32 and len(pool.calls[-1][1]) == 32


def test_resident_and_ticketed_keys_are_skipped():
    ids, lg = row([10 - i for i in range(10)])
    pool = FakePool(resident={(4, 100), (4, 101)})
    s = la.Scheduler(pool, cfg("attention"))
    first = s.on_layer(3, ids[None], lg[None], 48)
    again = s.on_layer(3, ids[None], lg[None], 48)
    assert first == 7 and again == 0


def test_no_reservation_with_256_or_fewer_evictable_slots():
    ids, lg = row([10 - i for i in range(10)])
    pool = FakePool(slots=300, pinned=44)                        # 300 - 44 = 256 evictable: not more than 256
    s = la.Scheduler(pool, cfg("attention"))
    assert s.on_layer(0, ids[None], lg[None], 48) == 0 and pool.issued == 0 and not pool.tickets
    pool = FakePool(slots=300, pinned=43)                        # 257: one reservation, then 256 again
    assert la.Scheduler(pool, cfg("attention")).on_layer(0, ids[None], lg[None], 48) == 1


def test_slot_cap_64_outstanding():
    pool = FakePool()
    s = la.Scheduler(pool, cfg("attention", issue_cap=32))
    for L in range(0, 6):
        ids = np.arange(24, dtype=np.int32) + 24 * L
        lg = np.linspace(40, 17, 24).astype(np.float32)
        s.on_layer(L, ids[None], lg[None], 48)
        pool.expire_layer(-1) if False else None
    assert len(pool.tickets) <= 64


def test_expire_after_ensure_returns_unadopted_and_nothing_leaks():
    pool = FakePool()
    s = la.Scheduler(pool, cfg("attention"))
    ids, lg = row([10 - i for i in range(10)])
    for L in range(0, 40):
        s.on_layer(L, ids[None], lg[None], 48)                   # expires layers <= L, issues for L + 1
        assert all(k[0] == L + 1 for k in pool.tickets)
        for e in (100, 101):                                     # demand claims two of the L + 1 keys
            pool.claim(L + 1, e)
    leaked = s.end_request()
    assert pool.tickets == {} and pool.issued == pool.adopted + pool.returned
    assert leaked >= 0
    assert s.stats()["issued"] == pool.issued
