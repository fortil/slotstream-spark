import re
from pathlib import Path

from slotstream_spark.config import MODES

GUARD = Path(__file__).resolve().parents[2] / "deploy" / "guard.sh"


def test_guard_mode_constants_match_config():
    found = {}
    for kind, mode, val in re.findall(r"^MODE_(CEILING|NEED)_MIB_(D[125])=(\d+)$",
                                      GUARD.read_text(), re.M):
        found[(mode, kind.lower())] = int(val)
    assert len(found) == 6
    for mode, d in MODES.items():
        assert found[(mode, "ceiling")] == d["ceiling_mib"]
        assert found[(mode, "need")] == d["need_mib"]
