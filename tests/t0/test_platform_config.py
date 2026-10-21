import copy
import subprocess
import sys
from pathlib import Path

from slotstream_spark import config

ROOT = Path(__file__).resolve().parents[2]


def test_platform_parses_and_validates():
    cfg = config.load()
    assert config.validate(cfg) == []


def test_fixed_values():
    cfg = config.load()
    assert config.get("memory.mode_default", cfg) == "D5"
    assert config.get("memory.write_path", cfg) == "safe"
    assert config.get("ngram.read", cfg) == "odirect"


def test_bad_values_invalid():
    cfg = copy.deepcopy(config.load())
    cfg["ngram"]["row_bytes"] = 99
    cfg["kernels"]["plan"] = "Z"
    errs = config.validate(cfg)
    assert any("ngram.row_bytes=99" in e for e in errs)
    assert any("kernels.plan='Z'" in e for e in errs)


def test_modes_exact():
    assert config.MODES == {
        "D1": {"ceiling_mib": 19866, "need_mib": 62644},
        "D2": {"ceiling_mib": 26112, "need_mib": 56413},
        "D5": {"ceiling_mib": 65126, "need_mib": 16896},
    }


def test_decided():
    assert config.decided("memory.mode_default")
    assert config.decided("kernels.plan")
    cfg = config.load()
    cfg["runtime"]["wait_mechanism"] = "unmeasured"
    assert not config.decided("runtime.wait_mechanism", cfg)


def _run(*args):
    return subprocess.run([sys.executable, "-m", "slotstream_spark.config", *args],
                          cwd=ROOT, capture_output=True, text=True)


def test_cli_require_undecided(tmp_path):
    txt = (ROOT / "slotstream_spark/config/platform.toml").read_text()
    out = []
    for line in txt.splitlines():
        if line.startswith("wait_mechanism"):
            line = 'wait_mechanism = "unmeasured"'
        out.append(line)
    f = tmp_path / "platform.toml"
    f.write_text("\n".join(out) + "\n")
    r = _run("--path", str(f), "--require", "runtime.wait_mechanism")
    assert r.returncode == 1
    assert r.stdout.strip() == "undecided: runtime.wait_mechanism"


def test_cli_validate_and_require_ok():
    assert _run("--validate").returncode == 0
    assert _run("--require", "memory.mode_default").returncode == 0


def test_nvme_keys_are_records_not_limits():
    assert config.get("measured.nvme_temp_no_throttle_mc") == 56850
    assert "measured.nvme_temp_limit_mc" not in config.ALLOWED


def test_pool_prefill_key():
    assert config.get("pool.prefill") in {"off", "table"}
    assert config.get("pool.prefill") == "table"
