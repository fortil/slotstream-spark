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
