"""Table tests of the governor policy (rules 1-8 of section 3.7, D5 default). No clock, no network."""
from dataclasses import replace

import pytest

from slotstream_spark.config import modes
from slotstream_spark.memory import governor as G
from slotstream_spark.memory.governor import (Exit, GovInputs, GrowTo, Hold, Resident, ShrinkTo, decide,
                                             may_start)

# the A fraction is taken of MemTotal; pin it so the table does not depend on the machine running the test
CFG = replace(G.load_config(), memfree_total_mib=G.MEMTOTAL_FALLBACK_MIB)
NOON = 12 * 60
FLOOR = CFG.floor_chunks
A = (Resident("large-test", "loading"),)
A_READY = (Resident("large-test", "ready"),)


def inputs(mode="D5", **kw):
    base = dict(mode=mode, now_min=NOON, running=(), chunks=20, footprint_mib=30000,
                mem_available_mib=60000, mem_free_mib=60000, calm_s=0)
    base.update(kw)
    return GovInputs(**base)


def avail_for_target(mode, chunks, own, cfg=CFG):
    """MemAvailable that puts the empty-running cap exactly at `chunks` whole chunks."""
    fixed = G.Ledger(mode).fixed_bytes()
    cap = -(-(fixed + chunks * modes.CHUNK_BYTES) // G.MIB)
    return cap + modes.get(mode).need_mib + cfg.margin_mib - own


def test_config_loads_from_the_records():
    assert CFG.a_transient_mib == 2527 and CFG.a1_load_s == 180 and CFG.memfree_cap is False
    assert CFG.d5_exit_after_s == 5 and CFG.grow_after_s == 60 and CFG.emergency_mib == 10240
    assert CFG.stable_band_mib == 512 and CFG.tick_s == 1 and CFG.running_poll_ms == 250


def test_resident_classes():
    c = lambda n: G.classify(n, CFG)
    assert c("large-test") == "A" and c("vision-main") == "B"
    assert c("candidate-27b") == "27B" and c("small-reasoning") == "C"
    assert c("something-else") == "C"


def test_residents_local_toml_replaces_classes_by_name(tmp_path, monkeypatch):
    import shutil
    for n in ("governor.toml", "residents.toml"):
        shutil.copy(G.CONFIG_DIR / n, tmp_path / n)
    (tmp_path / "residents.local.toml").write_text('[classes]\nA = ["prod-main-*"]\n')
    monkeypatch.setattr(G, "CONFIG_DIR", tmp_path)
    cfg = G.load_config()
    assert G.classify("prod-main-1", cfg) == "A" and G.classify("large-test", cfg) == "C"
    assert G.classify("vision-main", cfg) == "B"                   # classes the local file does not name stay


def test_memfree_total_zero_reads_memtotal(tmp_path):
    mi = tmp_path / "meminfo"
    mi.write_text("MemTotal: %d kB\n" % (100000 * 1024))
    assert G._memfree_total_mib(0, str(mi)) == 100000 and G._memfree_total_mib(5, str(mi)) == 5
    assert G._memfree_total_mib(0, str(tmp_path / "none")) == G.MEMTOTAL_FALLBACK_MIB


def test_d5_empty_holds_then_any_entry_shrinks_to_the_floor_first_tick():
    full = inputs(chunks=48, footprint_mib=65000, mem_available_mib=avail_for_target("D5", 48, 65000))
    assert decide(full, CFG) == Hold()
    r = decide(replace(full, running=(Resident("vision-main"),)), CFG)
    assert r == ShrinkTo(FLOOR, "running not empty", cancel=True, drop_prefix=True)


def test_d5_exit_only_after_5_s_at_the_floor():
    run = (Resident("large-test"),)
    assert decide(inputs(chunks=FLOOR, running=run, busy_at_floor_s=4), CFG) == Hold()
    assert isinstance(decide(inputs(chunks=FLOOR, running=run, busy_at_floor_s=5), CFG), Exit)


def test_d5_cleared_holds_then_grows_one_chunk_per_60_s():
    base = inputs(chunks=FLOOR, footprint_mib=5000, mem_available_mib=70000)
    assert decide(replace(base, calm_s=59), CFG) == Hold()
    assert decide(replace(base, calm_s=60), CFG) == GrowTo(FLOOR + 1)


def test_never_grows_within_60_s_of_a_shrink():
    for s in range(0, 60):
        assert not isinstance(decide(inputs(chunks=FLOOR, footprint_mib=5000, mem_available_mib=70000, calm_s=s), CFG), GrowTo)


def test_failed_get_counts_as_non_empty():
    r = decide(inputs(running=None), CFG)
    assert r == ShrinkTo(FLOOR, "running not empty", cancel=True, drop_prefix=True)
    d1 = decide(inputs("D1", running=None, chunks=8, mem_available_mib=30000), CFG)
    assert isinstance(d1, ShrinkTo) and d1.chunks == FLOOR


def test_d1_a_starting_goes_to_the_floor_and_holds_there():
    r = decide(inputs("D1", running=A, chunks=8, mem_available_mib=30000), CFG)
    assert r == ShrinkTo(FLOOR, "A starting", cancel=True, drop_prefix=True)
    assert decide(inputs("D1", running=A, chunks=FLOOR, mem_available_mib=30000), CFG) == Hold()


def test_d1_a_ready_regrows_slowly_to_the_previous_size():
    base = inputs("D1", running=A_READY, chunks=FLOOR, footprint_mib=3000, mem_available_mib=30000,
                  a_stable_s=60, prev_chunks=8, calm_s=60)
    assert decide(base, CFG) == GrowTo(FLOOR + 1)
    assert decide(replace(base, calm_s=59), CFG) == Hold()
    assert decide(replace(base, chunks=8), CFG) == Hold()
    # state ready but MemAvailable not yet stable: still treated as starting
    assert decide(replace(base, a_stable_s=59), CFG) == Hold()


def test_d1_a_without_state_field_waits_load_time_plus_60_s():
    run = (Resident("large-test", None),)
    base = inputs("D1", running=run, chunks=FLOOR, footprint_mib=3000, mem_available_mib=30000,
                  a_stable_s=60, prev_chunks=8, calm_s=60)
    assert decide(replace(base, a_since_s=CFG.a1_load_s + 59), CFG) == Hold()
    assert decide(replace(base, a_since_s=CFG.a1_load_s + 60), CFG) == GrowTo(FLOOR + 1)


def test_d1_a_transient_that_does_not_leave_8_gib_exits_even_at_the_floor():
    low = CFG.min_avail_mib + CFG.a_transient_mib - 1
    assert isinstance(decide(inputs("D1", running=A, chunks=FLOOR, mem_available_mib=low), CFG), Exit)
    ok = low + 1
    assert decide(inputs("D1", running=A, chunks=FLOOR, mem_available_mib=ok), CFG) == Hold()


@pytest.mark.parametrize("name", ["vision-main", "candidate-27b", "small-reasoning", "mystery"])
def test_d1_other_residents_exit(name):
    assert isinstance(decide(inputs("D1", running=(Resident(name),), chunks=8, mem_available_mib=30000), CFG), Exit)


def test_d2_a_starting_goes_to_the_floor():
    r = decide(inputs("D2", running=A, chunks=10, mem_available_mib=30000), CFG)
    assert isinstance(r, ShrinkTo) and r.chunks == FLOOR and r.cancel


def test_residents_grow_4_gib_shrinks_4_chunks_and_drops_the_prefix():
    own = 16000
    avail = avail_for_target("D1", 7, own)
    g = inputs("D1", chunks=7, footprint_mib=own, mem_available_mib=avail, mem_free_mib=avail)
    assert decide(g, CFG) == Hold()
    r = decide(replace(g, mem_available_mib=avail - 4096), CFG)
    assert r == ShrinkTo(3, "cap", drop_prefix=True)


def test_emergency_below_10_gib_shrinks_one_chunk_at_a_time():
    g = inputs("D5", chunks=20, footprint_mib=30000, mem_available_mib=CFG.emergency_mib - 1)
    r = decide(g, CFG)
    assert isinstance(r, ShrinkTo) and r.chunks == 19 and r.cancel


def test_cap_below_the_floor_exits():
    g = inputs("D1", chunks=FLOOR, footprint_mib=3000, mem_available_mib=20000)
    assert decide(g, CFG) == Exit("cap below the floor")


def test_23_59_and_the_night_exit():
    assert isinstance(decide(inputs(now_min=23 * 60 + 59), CFG), Exit)
    assert isinstance(decide(inputs(now_min=3 * 60 + 59), CFG), Exit)


def test_d1prime_memfree_below_a_need_shrinks():
    cfg = replace(CFG, memfree_cap=True)
    own = 16000
    avail = avail_for_target("D1", 7, own, cfg)
    g = inputs("D1", chunks=7, footprint_mib=own, mem_available_mib=avail + 5000, mem_free_mib=47720)
    # MemAvailable alone would allow growth; MemFree + own - 0.40 * total - margin does not
    assert decide(replace(g, calm_s=0), CFG) == Hold()
    r = decide(g, cfg)
    assert isinstance(r, ShrinkTo) and r.chunks == 3      # 4 before the 4x32 staging (+176,947,200 B of fixed bytes)
    # without the D1' flag the same inputs never shrink
    assert not isinstance(decide(g, CFG), ShrinkTo)


def test_grow_needs_a_chunk_and_512_mib_of_headroom():
    own = 20000
    g = inputs("D5", chunks=10, footprint_mib=own, mem_available_mib=avail_for_target("D5", 30, own), calm_s=60)
    need = modes.CHUNK_BYTES // G.MIB + CFG.margin_mib
    assert G._grow(g, CFG, own + need, 30) == GrowTo(11)
    assert G._grow(g, CFG, own + need - 1, 30) == Hold()


def test_same_avail_plus_own_gives_the_same_target():
    a = inputs("D1", chunks=12, footprint_mib=15000, mem_available_mib=14000, mem_free_mib=14000)
    b = inputs("D1", chunks=12, footprint_mib=10000, mem_available_mib=19000, mem_free_mib=19000)
    ca, cb = G.cap_mib(a, CFG), G.cap_mib(b, CFG)
    assert ca == cb and G.target_chunks(a, CFG, ca) == G.target_chunks(b, CFG, cb)


def test_fixed_bytes_do_not_depend_on_the_pool_for_the_floor():
    assert G.floor_mib(inputs("D5"), CFG) == -(-(G.Ledger("D5").fixed_bytes() + FLOOR * modes.CHUNK_BYTES) // G.MIB)


def test_may_start_windows_and_preconditions():
    floor = G.floor_mib(inputs("D5"), CFG)
    cap = floor + 1
    assert may_start(3 * 60 + 59, (), cap, floor, CFG) == (False, "before 04:00")
    assert may_start(4 * 60, (), cap, floor, CFG) == (True, "ok")
    assert may_start(23 * 60 + 44, (), cap, floor, CFG)[0] is True
    assert may_start(23 * 60 + 45, (), cap, floor, CFG) == (False, "after 23:45")
    assert may_start(NOON, A, cap, floor, CFG) == (False, "running not empty")
    assert may_start(NOON, None, cap, floor, CFG)[0] is False
    assert may_start(NOON, (), floor - 1, floor, CFG) == (False, "cap below the floor")
