"""Policy tests of the extend-only prefix cache (step 71). Fake states, no torch."""
from types import SimpleNamespace

import pytest

from slotstream_spark.engine import prefill_schedule as ps
from slotstream_spark.engine import prefix_cache as pc
from slotstream_spark.engine.prefix_cache import CheckpointKey, Hit, Miss, PrefixCache
from slotstream_spark.memory import governor as G

KEY = CheckpointKey("m1", "{}", 4096, True)
OTHER = CheckpointKey("m1", "{}", 2048, True)
MAXT = 36_000
LOGIT_BYTES = 248_320 * 4


def st():
    return SimpleNamespace()


def prompt(n, seed=0):
    return [seed * 1_000_000 + i for i in range(n)]


def snapshot(c, toks, **kw):
    return c.store_checkpoint(st(), toks, len(toks) * pc.BYTES_PER_TOKEN, fresh_equivalent=True, produced_key=KEY, **kw)


def complete(c, toks):
    return c.store_checkpoint(st(), toks, len(toks) * pc.BYTES_PER_TOKEN, logits=object(), logits_bytes=LOGIT_BYTES,
                              prompt_key=KEY, fresh_equivalent=True, produced_key=KEY)


def test_constants_and_charge_arithmetic():
    assert pc.FIXED_BYTES_PER_ENTRY == 113_246_208 and pc.BYTES_PER_TOKEN == 27_648 and pc.MAX_ENTRIES == 4
    e = pc.Entry(st(), list(range(1000)), 1000 * 27_648, 1)
    assert pc.charge(e) == 1000                                   # no logits
    e.seq_bytes = 5000 * 27_648 + 1                               # allocated capacity above the token count wins
    assert pc.charge(e) == 5001
    e.last_logits, e.logits_bytes = object(), LOGIT_BYTES
    assert pc.charge(e) == 5001 + -(-(LOGIT_BYTES + 16_384) // 27_648)
    c = PrefixCache(max_tokens=MAXT)
    assert complete(c, prompt(100))
    assert c.owned_bytes() == 100 * 27_648 + 113_246_208 + LOGIT_BYTES + 16_384


def test_open_webui_pattern_hits_with_four_entries_and_not_with_one():
    chat, title = prompt(6000, 1), prompt(300, 2)
    turn2 = chat[:5000] + prompt(2000, 3)
    results = {}
    for n in (4, 1):
        c = PrefixCache(max_entries=n, max_tokens=MAXT)
        b1 = set(ps.resume_boundaries(len(chat), 4096))
        assert isinstance(c.take(chat, KEY, b1), Miss)
        snapshot(c, chat[:4096])
        complete(c, chat)
        assert isinstance(c.take(title, KEY, set()), Miss)        # the title request in between
        complete(c, title)
        b2 = set(ps.resume_boundaries(len(turn2), 4096))
        h = c.take(turn2, KEY, b2)
        results[n] = h
    assert isinstance(results[4], Hit) and results[4].reused == 4096 and results[4].logits is None
    assert isinstance(results[1], Miss)


def test_extend_only_and_equal_length():
    c = PrefixCache(max_tokens=MAXT)
    p = prompt(5000)
    complete(c, p)
    assert isinstance(c.take(p[:4000], KEY, {4000}), Miss)                    # shorter than the entry
    assert isinstance(c.take(p[:-1] + [-1], KEY, set()), Miss)                # diverges at the last token
    h = c.take(p, KEY, set())                                                 # equal length: the retained logits
    assert isinstance(h, Hit) and h.reused == 5000 and h.logits is not None
    assert isinstance(c.take(p, OTHER, set()), Miss)                          # another key never resumes it
    c2 = PrefixCache(max_tokens=MAXT)
    snapshot(c2, p[:4096])
    assert isinstance(c2.take(p[:4096], KEY, {4096}), Miss)                   # equal length needs retained logits


def test_non_boundary_length_never_offered():
    c = PrefixCache(max_tokens=MAXT)
    p = prompt(6000)
    snapshot(c, p[:4000])
    assert isinstance(c.take(p, KEY, {4096}), Miss)
    assert isinstance(c.take(p, KEY, {4000}), Hit)


def test_decode_end_state_is_never_offered_and_goes_first():
    c = PrefixCache(max_tokens=MAXT)
    p = prompt(5000)
    c.store(st(), p, len(p) * 27_648, fresh_equivalent=False, key=KEY, input_tokens=4990)
    assert isinstance(c.take(p + [1], KEY, {len(p)}), Miss)
    snapshot(c, p[:4096])
    complete(c, prompt(300, 4))
    complete(c, prompt(200, 5))                                               # the fourth slot is the producer's
    kinds = [(e.reusable, e.fresh_equivalent) for e in c.entries()]
    assert (False, False) not in kinds and len(c.entries()) == 3


def test_eviction_order_unused_then_superseded():
    c = PrefixCache(max_tokens=MAXT)
    a = prompt(9000, 1)
    snapshot(c, a[:4096])
    h = c.take(a, KEY, {4096})
    assert isinstance(h, Hit)                                                 # the snapshot has standing now
    deeper = snapshot(c, a[:8192])
    assert deeper
    # a snapshot nothing has used is evicted before one a hit has used
    b = prompt(5000, 2)
    snapshot(c, b[:4096])
    complete(c, prompt(100, 3))
    c.take(prompt(10, 9), KEY, set())                                         # a miss makes room before the caller allocates
    assert len(c.entries()) < pc.MAX_ENTRIES
    lens = sorted(len(e.tokens) for e in c.entries())
    assert 4096 in lens or 8192 in lens


def test_miss_evicts_before_the_caller_allocates_and_budget_holds():
    c = PrefixCache(max_tokens=30_000)
    for s in range(3):
        assert complete(c, prompt(5000, s))
    assert len(c.entries()) == 3
    c.take(prompt(16_000, 7), KEY, set())
    assert len(c.entries()) == 2 and c.charged_tokens() <= 30_000 - 16_000
    assert not complete(PrefixCache(max_tokens=100), prompt(5000))            # larger than the whole budget


def test_checkpoint_needs_a_free_slot_for_its_producer():
    c = PrefixCache(max_entries=1, max_tokens=MAXT)
    assert not snapshot(c, prompt(4096))


def test_drop_and_governor_hook():
    c = PrefixCache(max_tokens=MAXT)
    complete(c, prompt(5000))
    order = []

    class Spy(PrefixCache):
        def drop(self):
            order.append("drop")
            super().drop()
    s = Spy(max_tokens=MAXT)
    complete(s, prompt(5000))
    dec = G.ShrinkTo(4, "cap", drop_prefix=True)
    assert G.apply_shrink(dec, s, lambda n: order.append(("resize", n))) is True
    assert order == ["drop", ("resize", 4)] and s.entries() == []
    assert G.apply_shrink(G.ShrinkTo(3, "x"), c, lambda n: order.append(("resize", n))) is False
    assert len(c.entries()) == 1


def test_snapshot_end_is_a_pass_boundary_inside_the_new_tokens():
    p = list(range(6000))
    b = ps.resume_boundaries(len(p), 4096)
    assert b == {4096}
    assert pc.snapshot_end(p, 5990, 0, 4096, b) == 4096
    assert pc.snapshot_end(p, 3000, 0, 4096, b) is None                       # the first pass reaches past the target
    assert pc.snapshot_end(p, 5990, 4096, 4096, b) is None                    # nothing new before the target
    assert pc.snapshot_target([1, 2, 7, 3, 7, 4], 7) == 4 and pc.snapshot_target([1, 2], 7) is None


# ---- the window-scaled cache: charge_active=False ---------------------------------------------------------------
WINDOW = 131_072
LONG = 78_000


def scaled(max_tokens=WINDOW, **kw):
    return PrefixCache(max_tokens=max_tokens, charge_active=False, **kw)


def long_turns(seed=1, extra=300):
    """A LONG-token conversation: turn 1, its pass-end snapshot length from the schedule, turns 2 and 3."""
    p1 = prompt(LONG, seed)
    end = max(b for b in ps.resume_boundaries(LONG, 4096) if b < LONG)
    p2 = p1 + prompt(extra, seed + 10)
    p3 = p2 + prompt(extra, seed + 20)
    return p1, end, p2, p3


def boundaries(p):
    return ps.resume_boundaries(len(p), 4096)


def test_scaled_cache_reports_its_rule_and_the_design_stays_default():
    assert PrefixCache(max_tokens=MAXT).charge_active is True and PrefixCache(max_tokens=MAXT).stats()["charge_active"]
    assert scaled().charge_active is False and scaled().stats()["charge_active"] is False


def test_long_conversation_keeps_its_boundary_snapshot_across_turns():
    p1, end, p2, p3 = long_turns()
    assert LONG - 2048 < end < LONG                                     # the last tail-aware pass end before the trailing tokens
    c = scaled()
    assert isinstance(c.take(p1, KEY, boundaries(p1), reserve_tokens=LONG + 32_768), Miss)
    assert snapshot(c, p1[:end], reserve_tokens=LONG + 32_768)          # S1, whatever the request reserves
    assert not complete(c, p1)                                          # F1: no room beside S1 (76,800 + 78,037 > 131,072)
    assert [len(e.tokens) for e in c.entries()] == [end]
    assert not c.would_store(p1 + prompt(50, 99))                       # the decode-end state would be evicted at once
    h = c.take(p2, KEY, boundaries(p2), reserve_tokens=len(p2) + 32_768)
    assert isinstance(h, Hit) and h.reused == end
    assert [len(e.tokens) for e in c.entries()] == [end]                # a hit never evicts the entry it hands out
    assert not complete(c, p2)                                          # F2 must not supersede the used S1
    assert [len(e.tokens) for e in c.entries()] == [end]
    h3 = c.take(p3, KEY, boundaries(p3))
    assert isinstance(h3, Hit) and h3.reused == end
    assert c.charged_tokens() <= WINDOW and len(c.entries()) <= pc.MAX_ENTRIES


def test_the_design_charges_the_active_request_and_refuses_the_long_snapshot():
    p1, end, _, _ = long_turns()
    c = PrefixCache(max_tokens=WINDOW)                                  # charge_active=True at the same budget
    assert isinstance(c.take(p1, KEY, boundaries(p1), reserve_tokens=LONG + 32_768), Miss)
    assert not snapshot(c, p1[:end], reserve_tokens=LONG + 32_768)      # 110,768 reserved leaves 20,304
    assert snapshot(scaled(), p1[:end], reserve_tokens=LONG + 32_768)   # the reserve is ignored without the charge


def test_a_short_prompt_sits_beside_the_long_snapshot_and_a_second_long_one_makes_room():
    p1, end, p2, _ = long_turns(1)
    q1, qend, _, _ = long_turns(2)
    c = scaled()
    c.take(p1, KEY, boundaries(p1))
    assert snapshot(c, p1[:end]) and isinstance(c.take(p2, KEY, boundaries(p2)), Hit)
    short = prompt(20_000, 5)
    assert isinstance(c.take(short, KEY, set()), Miss) and complete(c, short)
    assert sorted(len(e.tokens) for e in c.entries()) == [20_000, end]
    assert isinstance(c.take(q1, KEY, boundaries(q1)), Miss)
    assert c.charged_tokens() <= WINDOW - LONG                          # the miss made room for B's own snapshot
    assert snapshot(c, q1[:qend]) and qend in [len(e.tokens) for e in c.entries()]
    assert c.charged_tokens() <= WINDOW


def test_a_hit_never_evicts_the_entry_it_hands_out_without_the_charge():
    p = prompt(9_000)
    b = boundaries(p)
    end = max(x for x in b if x < len(p))
    c = scaled(max_tokens=17_000)
    assert snapshot(c, p[:end])
    h = c.take(p, KEY, b)                                               # 9,000 reserved leaves 8,000 < 8,192: the design evicts
    assert isinstance(h, Hit) and h.reused == end and len(c.entries()) == 1
    c2 = PrefixCache(max_tokens=17_000)
    assert snapshot(c2, p[:end])
    assert isinstance(c2.take(p, KEY, b), Hit) and c2.entries() == []   # the design: handed out, then evicted


def test_a_deeper_boundary_snapshot_still_supersedes_a_used_shallower_one():
    c = scaled(max_tokens=15_000)
    p = prompt(16_000)
    assert snapshot(c, p[:4096])
    assert isinstance(c.take(p, KEY, {4096, 8192, 12288}), Hit)
    assert snapshot(c, p[:12288])                                       # 4,096 + 12,288 > 15,000: the used one goes
    assert [len(e.tokens) for e in c.entries()] == [12288]


def test_would_store_checkpoint_predicts_store_checkpoint():
    for charge_active in (True, False):
        c = PrefixCache(max_tokens=30_000, charge_active=charge_active)
        p = prompt(20_000)
        c.take(p, KEY, {4096}, reserve_tokens=21_000)
        kw = dict(fresh_equivalent=True, produced_key=KEY, reserve_tokens=21_000)
        want = c.would_store_checkpoint(p[:4096], **kw)
        assert want == c.store_checkpoint(st(), p[:4096], 4096 * pc.BYTES_PER_TOKEN, **kw)
        kw2 = dict(logits_bytes=LOGIT_BYTES, prompt_key=KEY, fresh_equivalent=True, produced_key=KEY, reserve_tokens=21_000)
        want = c.would_store_checkpoint(p, logits=True, **kw2)
        assert want == c.store_checkpoint(st(), p, len(p) * pc.BYTES_PER_TOKEN, logits=object(), **kw2)
        assert want is (not charge_active)                              # 4,096 + 20,037 fits 30,000 only without the charge
        assert c.charged_tokens() <= 30_000 and len(c.entries()) <= pc.MAX_ENTRIES
    assert not scaled().would_store_checkpoint([])


def test_would_store_predicts_the_decode_end_store():
    c = PrefixCache(max_tokens=MAXT)
    assert complete(c, prompt(15_000, 1))
    big = prompt(25_000, 2)
    assert not c.would_store(big)
    assert c.store(st(), big, len(big) * 27_648, key=KEY, input_tokens=24_000)
    assert [len(e.tokens) for e in c.entries()] == [15_000] and c.entries()[0].reusable   # the new state went first
    small = prompt(5_000, 3)
    assert c.would_store(small)
    assert c.store(st(), small, 5_000 * 27_648, key=KEY, input_tokens=4_000)
    assert sorted(len(e.tokens) for e in c.entries()) == [5_000, 15_000]
    assert c.would_store(small)                                         # a duplicate of an unresumable entry is replaced
    assert not c.would_store(prompt(MAXT + 1, 4)) and not c.would_store([])


def test_budget_and_slot_invariants_hold_after_every_call():
    c = scaled(max_tokens=50_000)
    for seed, n in enumerate((30_000, 12_000, 45_000, 8_000, 30_000, 20_000, 50_000)):
        p = prompt(n, seed)
        b = boundaries(p)
        c.take(p, KEY, b, reserve_tokens=len(p) + 32_768)
        assert len(c.entries()) < pc.MAX_ENTRIES and c.charged_tokens() <= 50_000
        ends = [x for x in b if x < len(p)]
        if ends:
            snapshot(c, p[:max(ends)])
        complete(c, p)
        assert c.charged_tokens() <= 50_000 and len(c.entries()) <= pc.MAX_ENTRIES
        if c.would_store(p + [7]):
            c.store(st(), p + [7], (len(p) + 1) * pc.BYTES_PER_TOKEN, key=KEY, input_tokens=len(p))
        assert c.charged_tokens() <= 50_000 and len(c.entries()) <= pc.MAX_ENTRIES
