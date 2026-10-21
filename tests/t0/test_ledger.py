import inspect

import pytest

from slotstream_spark import config
from slotstream_spark.config import modes
from slotstream_spark.memory import ledger
from slotstream_spark.memory.ledger import Ledger

DESIGN_RUNTIME = 1_750_000_000          # the design's runtime figure; only used to reproduce modes.py


def runtime_from_toml():
    return int(config.get("measured.runtime_footprint_mib")) << 20


@pytest.mark.parametrize("name,expected", [("D1", 8_821_658_084), ("D2", 12_123_804_900), ("D5", 14_786_204_900)])
def test_design_is_reproduced_with_the_design_runtime(name, expected):
    assert Ledger(name, measured={"runtime_bytes": DESIGN_RUNTIME}, staging_groups=2).fixed_bytes() == expected
    assert expected == modes.get(name).fixed_bytes


@pytest.mark.parametrize("name", ["D1", "D2", "D5"])
def test_default_runtime_is_the_platform_measurement(name):
    got = Ledger(name, staging_groups=2).fixed_bytes()
    assert got == modes.get(name).fixed_bytes + runtime_from_toml() - DESIGN_RUNTIME
    assert Ledger(name).runtime == runtime_from_toml()
    assert got - modes.get(name).fixed_bytes == 888_217_216          # 2516 MiB - 1.75e9


def test_no_design_runtime_default_in_the_ledger_source():
    src = inspect.getsource(ledger)
    assert "1750000000" not in src and "1_750_000_000" not in src and "1.75e9" not in src
    assert "fixed_bytes" in src and "modes.get" not in src          # never reads modes.fixed_bytes


def test_unmeasured_runtime_is_refused(monkeypatch):
    monkeypatch.setattr(ledger.config, "decided", lambda key, cfg=None: False)
    with pytest.raises(ValueError, match="not measured"):
        Ledger("D5")
    assert Ledger("D5", measured={"runtime_bytes": 1}).runtime == 1


def test_fixed_bytes_does_not_change_with_chunks():
    led = Ledger("D5")
    fixed = led.fixed_bytes()
    gib = 1 << 30
    for c in (0, 1, 2, 13, 48, 49):
        assert led.expected_peak_bytes(c) == fixed + c * gib
        assert led.fixed_bytes() == fixed
    assert led.expected_peak_bytes(0) == fixed
    assert led.expected_peak_bytes(49) - led.expected_peak_bytes(48) == gib
    assert "chunks" not in inspect.signature(Ledger.fixed_bytes).parameters
    assert "pool" not in led.terms()                                # no pool byte sits inside the fixed terms


def test_terms_and_integer_only_types():
    led = Ledger("D5")
    t = led.terms()
    assert t["trunk"] == 2_566_424_320 and t["gdn"] == 115_642_368 and t["read_scope"] == 234_881_024
    assert t["staging_scratch_rows"] == 556_947_200 and t["sidecar"] == 37_540_708 and t["margin"] == 1_000_000_000
    assert t["kv"] == 27_648 * 32768 and t["pass"] == 1_300_000 * 4096 and t["mtp"] == 1_470_946_816
    assert t["prefix"] == 1_000_000_000 and t["runtime"] == 2516 << 20
    assert all(type(v) is int for v in t.values())
    assert type(led.fixed_bytes()) is int and type(led.expected_peak_bytes(49)) is int
    assert Ledger("D1").terms()["prefix"] == 500_000_000 and Ledger("D1").terms()["mtp"] == 0
    assert Ledger("D2").terms()["mtp"] == 1_470_946_816
    assert Ledger("D5", window=65536).terms()["kv"] == 27_648 * 65536


@pytest.mark.parametrize("name", ["D1", "D2", "D5"])
def test_staging_term_is_groups_times_32_records(name):
    assert Ledger(name).fixed_bytes() - Ledger(name, staging_groups=2).fixed_bytes() == 176_947_200
    assert Ledger(name, staging_groups=2).terms()["staging_scratch_rows"] == 380_000_000
    assert Ledger("D5", staging_groups=4).terms()["staging_scratch_rows"] == 203_052_800 + 4 * 88_473_600


def test_unknown_mode():
    with pytest.raises(ValueError):
        Ledger("D9")
