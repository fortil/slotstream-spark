import pytest

from slotstream_spark.config import modes
from slotstream_spark.memory import planner
from slotstream_spark.memory.ledger import Ledger
from slotstream_spark.memory.planner import plan

AVAIL = 81_920


def cap_at_ceiling(name):
    return modes.get(name).ceiling_mib << 20


def cap_at_avail(name):
    return modes.dynamic_cap_mib(modes.get(name), AVAIL) << 20


@pytest.mark.parametrize("name,chunks", [("D1", 10), ("D2", 13), ("D5", 48)])
def test_chunks_at_the_ceilings(name, chunks):
    p = plan(name, cap_at_ceiling(name))
    assert p.chunks == chunks and p.slots == chunks * modes.SLOTS_PER_CHUNK
    assert p.expected_peak <= cap_at_ceiling(name)


@pytest.mark.parametrize("name,chunks", [("D1", 9), ("D2", 12), ("D5", 48)])
def test_chunks_at_81920_mib(name, chunks):
    p = plan(name, cap_at_avail(name))
    assert p.chunks == chunks
    assert p.expected_peak <= cap_at_avail(name)


def test_d5_has_48_chunks_at_the_ceiling_since_the_4x32_staging():
    p = plan("D5", cap_at_ceiling("D5"))
    assert p.chunks == 48 and cap_at_ceiling("D5") - p.expected_peak == 898_583_708
    old = plan("D5", cap_at_ceiling("D5"), staging_groups=2)
    assert old.chunks == 49 and cap_at_ceiling("D5") - old.expected_peak == 1_789_084


def test_design_constants_would_give_more_chunks():
    # modes.chunks_for keeps the design answer (fixed bytes with 1.75e9 of runtime): 11/14/49 at the ceilings
    got = [modes.chunks_for(modes.get(n), modes.get(n).ceiling_mib) for n in ("D1", "D2", "D5")]
    assert got == [11, 14, 49]
    got = [plan(n, cap_at_ceiling(n), measured={"runtime_bytes": 1_750_000_000}).chunks for n in ("D1", "D2", "D5")]
    assert got == [11, 14, 49]


@pytest.mark.parametrize("name,tokens", [("D1", 1024), ("D2", 2048), ("D5", 4096)])
def test_pass_sizes(name, tokens):
    assert plan(name, cap_at_ceiling(name)).pass_tokens == tokens


def test_pass_shrinks_with_a_small_pool():
    p = plan("D5", Ledger("D5").fixed_bytes() + 2 * (1 << 30))
    assert p.chunks == 2 and p.pass_tokens == 256 and 1_300_000 * 256 <= 2 * (1 << 30) // 4
    p = plan("D5", Ledger("D5").fixed_bytes() + 10 * (1 << 30))
    assert p.chunks == 10 and p.pass_tokens == 2048        # 1.3e6 * 2048 = 2.66e9 <= 10 GiB / 4 = 2.68e9 < 1.3e6 * 4096


def test_chunks_never_negative_and_clamped_to_the_ceiling():
    assert plan("D5", 1).chunks == 0
    assert plan("D5", 1 << 50).chunks == modes.get("D5").ceiling_chunks


def test_plan_defaults_and_mtp():
    p = plan("D5", cap_at_ceiling("D5"))
    assert (p.mtp_enabled, p.lookahead, p.window) == (False, False, 32768)
    assert p.mtp_reserved == 1_470_946_816 and plan("D1", cap_at_ceiling("D1")).mtp_reserved == 0
    assert p.expected_peak == Ledger("D5").expected_peak_bytes(p.chunks)


def test_integer_only_types():
    p = plan("D5", cap_at_ceiling("D5"))
    for v in (p.chunks, p.slots, p.pass_tokens, p.window, p.mtp_reserved, p.expected_peak):
        assert type(v) is int
    assert p.est_tok_s is None


CURVE = [(2000, 0.60), (6000, 0.80), (19012, 0.97)]


def test_estimate_needs_a_curve_and_timings():
    cap = cap_at_ceiling("D5")
    assert plan("D5", cap, t_comp=0.03, tau=1e-4, t_memcpy=1e-4).est_tok_s is None
    assert plan("D5", cap, h_curve=CURVE).est_tok_s is None
    assert plan("D5", cap, h_curve=CURVE, t_comp=0.03).est_tok_s is None


def test_estimate_formula_and_range():
    cap = cap_at_ceiling("D5")
    p = plan("D5", cap, h_curve=CURVE, t_comp=0.03, tau=1e-4, t_memcpy=1e-4, staging_groups=2)
    h = 0.97                                           # 49 * 388 = 19012 slots: the end of the curve (2x32 staging ledger)
    assert p.slots == 19012
    t_io = (1 - h) * 1.327e9 / 11.1e9 + 48 * (1 - h ** 10) * 1e-4
    assert p.est_tok_s == pytest.approx(1.0 / (0.03 + 48 * 1e-4 + t_io))
    # below the first curve point (2 chunks = 776 slots) and above the last one: no estimate
    low = plan("D5", Ledger("D5").fixed_bytes() + 2 * (1 << 30), h_curve=CURVE, t_comp=0.03, tau=1e-4, t_memcpy=1e-4)
    assert low.slots == 776 and low.est_tok_s is None
    assert planner.estimate_tok_s(CURVE, 19013, 0.03, 1e-4, 1e-4) is None
    assert planner.estimate_tok_s(CURVE, 4000, 0.03, 1e-4, 1e-4) is not None


def test_estimate_grows_with_the_hit_rate():
    a = planner.estimate_tok_s(CURVE, 2000, 0.03, 1e-4, 1e-4)
    b = planner.estimate_tok_s(CURVE, 6000, 0.03, 1e-4, 1e-4)
    c = planner.estimate_tok_s(CURVE, 19012, 0.03, 1e-4, 1e-4)
    assert a < b < c


def test_plan_at_a_131072_window_has_40_chunks_at_the_ceiling():
    p = plan("D5", cap_at_ceiling("D5"), window=131072)
    led = Ledger("D5", window=131072)
    assert p.chunks == 40 and p.window == 131072 and p.slots == 40 * modes.SLOTS_PER_CHUNK
    assert p.expected_peak == led.expected_peak_bytes(40) <= cap_at_ceiling("D5")
    assert plan("D5", cap_at_avail("D5"), window=131072).chunks == 39
    assert plan("D5", cap_at_ceiling("D5")).chunks == 48                          # the default window is unchanged
