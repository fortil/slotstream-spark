"""The host-knob example documents every variable env.sh exports, and AGENTS.md is the only
project instruction file."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "tools" / "spark" / "env.local.sh.example"
ENV_SH = ROOT / "tools" / "spark" / "env.sh"
KNOBS = ("SPARK_HOST", "SLOTSTREAM_RUNNING_URL", "EXPECTED_CONTAINERS", "SLOTSTREAM_NVME_DEVICE",
         "SLOTSTREAM_HWMON_INPUT", "SLOTSTREAM_TOOLS_VENV")


def test_example_documents_every_knob():
    text = EXAMPLE.read_text()
    for knob in KNOBS:
        assert re.search(rf"^#? ?export {knob}=", text, re.M), knob


def test_example_has_generic_defaults():
    text = EXAMPLE.read_text()
    assert re.search(r"^export SLOTSTREAM_RUNNING_URL=$", text, re.M)
    assert re.search(r"^export EXPECTED_CONTAINERS=$", text, re.M)
    assert re.search(r"^export SLOTSTREAM_NVME_DEVICE=nvme0n1$", text, re.M)


def test_env_sh_defaults_without_local_file():
    out = subprocess.run(
        ["bash", "-c", f'unset SPARK_HOST; SLOTSTREAM_SKIP_LOCAL=1; export SLOTSTREAM_SKIP_LOCAL; '
                       f'source "{ENV_SH}"; echo "$SPARK_HOST"'],
        capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "spark"


def test_no_claude_md():
    assert not (ROOT / "CLAUDE.md").exists()
    ls = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                        cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    assert [p for p in ls if Path(p).name == "CLAUDE.md"] == []
