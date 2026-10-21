import tomllib
from pathlib import Path

import pytest

from slotstream_spark import config
from slotstream_spark.config import modes

AVAIL = 81_920       # MiB, the MemAvailable the plan's table is written for


def test_thresholds_come_from_config_modes():
    for name, spec in config.MODES.items():
        m = modes.get(name)
        assert (m.ceiling_mib, m.need_mib) == (spec["ceiling_mib"], spec["need_mib"])


def test_fixed_bytes_and_ceiling_chunks():
    got = {n: (modes.get(n).fixed_bytes, modes.get(n).ceiling_chunks) for n in ("D1", "D2", "D5")}
    assert got == {"D1": (8_821_658_084, 11), "D2": (12_123_804_900, 14), "D5": (14_786_204_900, 49)}
    assert modes.CHUNK_BYTES == 1 << 30 and modes.SLOTS_PER_CHUNK == 388 and modes.FLOOR_CHUNKS == 2


@pytest.mark.parametrize("name,cap,chunks", [("D1", 19_276, 10), ("D2", 25_507, 13), ("D5", 65_024, 49)])
def test_caps_and_chunks_at_81920(name, cap, chunks):
    m = modes.get(name)
    assert modes.dynamic_cap_mib(m, AVAIL) == cap
    assert modes.chunks_for(m, cap) == chunks


def test_cap_grows_with_own_memory_and_is_clamped_by_the_ceiling():
    m = modes.get("D5")
    assert modes.dynamic_cap_mib(m, 20_000, own=30_000) == 20_000 + 30_000 - m.need_mib
    assert modes.dynamic_cap_mib(m, 200_000) == m.ceiling_mib
    assert modes.chunks_for(m, 10_000) < modes.FLOOR_CHUNKS


def test_footprint_is_the_guard_auto_formula(tmp_path):
    mi = tmp_path / "meminfo"
    mi.write_text(f"MemTotal: 1 kB\nMemAvailable: {AVAIL * 1024} kB\nMemFree: 5 kB\n")
    assert modes.footprint_mib(modes.get("D5"), str(mi)) == 65_024 - 256
    assert modes.footprint_mib(modes.get("D1"), str(mi)) == 19_276 - 256


def test_main_prints_cap_and_chunks(capsys):
    assert modes.main(["cap", "D5", str(AVAIL)]) == 0
    assert modes.main(["chunks", "D2", str(AVAIL)]) == 0
    assert capsys.readouterr().out.split() == ["65024", "13"]
    assert modes.main(["nope"]) == 2


def test_modes_agree_with_guard_script():
    guard = (Path(__file__).resolve().parents[2] / "deploy" / "guard.sh").read_text()
    for name, spec in config.MODES.items():
        assert f"MODE_CEILING_MIB_{name}={spec['ceiling_mib']}" in guard
        assert f"MODE_NEED_MIB_{name}={spec['need_mib']}" in guard


def test_cli_refuses_an_uncalibrated_contract(tmp_path, monkeypatch, capsys):
    from slotstream_spark import cli
    toml = tmp_path / "c.toml"
    toml.write_text('[fixture]\nkl_max_nats = -1.0\n')
    monkeypatch.setattr(cli, "CONTRACT", toml)
    assert cli.fixture_kl_max() is None
    assert cli.main(["run", "--golden-prompt", "dev0", "--check-first-token"]) == 2
    assert "contract not calibrated" in capsys.readouterr().err
    real = tomllib.loads((Path(cli.__file__).parent / "config" / "numeric_contract.toml").read_text())
    monkeypatch.undo()
    assert cli.fixture_kl_max() == real["fixture"]["kl_max_nats"] > 0


def test_cli_stubs_and_kl():
    import numpy as np
    from slotstream_spark import cli
    assert cli.main(["bench"]) == 2 and callable(cli.cmd_serve)   # serve is implemented since step 63; it needs the Spark
    a = np.array([1.0, 2.0, 3.0])
    assert cli.kl_nats(a, a) == pytest.approx(0.0, abs=1e-12)
    assert cli.kl_nats(a, a[::-1]) > 0.5
    src = (Path(cli.__file__)).read_text()
    assert "0.2" not in src, "no calibrated limit as a literal in cli.py"
