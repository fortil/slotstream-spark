import importlib
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SUBS = ["model", "kernels", "runtime", "memory", "engine", "server", "guard", "config"]


@pytest.mark.parametrize("sub", SUBS)
def test_subpackage_imports(sub):
    importlib.import_module(f"slotstream_spark.{sub}")


def test_license_exists():
    assert (ROOT / "LICENSE").is_file()


def test_notices_have_both_shas():
    text = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    assert "8bd579385152d547f96c9f6285181fe7885dfe29" in text
    assert "958d22e7ba0317750ccc38862b9667b1034a4dd1" in text


def test_vendor_license_matches_upstream():
    checkout = os.environ.get("SLOTSTREAM_HELGARD_CHECKOUT")
    if not checkout or not (Path(checkout) / "LICENSE").exists():
        pytest.skip("set SLOTSTREAM_HELGARD_CHECKOUT to a helgard checkout at 958d22e7 to compare")
    up = Path(checkout) / "LICENSE"
    mine = ROOT / "third_party/vendor/helgard-958d22e7/LICENSE"
    assert mine.read_bytes() == up.read_bytes()
