import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import yardstick  # noqa: E402

FIX = """# x
```toml
yardstick = "vara_b"

[vara]
arm = "q41-a-d5"
decode_tps_median = 10.3
prefill_tps_8k = 37.5
op_offload = false
prefill_device = "cpu"

[vara_b]
arm = "q41-b-d5"
decode_tps_median = 32.9
%s
op_offload = false
prefill_device = "cpu"
```
"""


def test_fixture_yardstick(tmp_path):
    p = tmp_path / "v.md"
    p.write_text(FIX % "prefill_tps_8k = 32.1")
    assert yardstick.yardstick(str(p))["arm"] == "q41-b-d5"


def test_missing_prefill_is_named(tmp_path):
    p = tmp_path / "v.md"
    p.write_text(FIX % "")
    with pytest.raises(ValueError, match="prefill_tps_8k"):
        yardstick.yardstick(str(p))


def test_missing_yardstick_key(tmp_path):
    p = tmp_path / "v.md"
    p.write_text(FIX.replace('yardstick = "vara_b"\n', "") % "prefill_tps_8k = 1")
    with pytest.raises(ValueError, match="yardstick"):
        yardstick.yardstick(str(p))


def test_tracked_record():
    d = yardstick.load_record(str(ROOT / "db/records/decisions/vara-d5.md"))
    assert d["yardstick"] == "vara_b"
    y = yardstick.yardstick(str(ROOT / "db/records/decisions/vara-d5.md"))
    assert y["arm"] == "q41-b-d5" and y["decode_tps_median"] == 32.9 and y["prefill_tps_8k"] == 32.1
    assert y["op_offload"] is False and y["prefill_device"] == "cpu"
    assert d["vara"]["arm"] == "q41-a-d5" and d["vara"]["decode_tps_median"] == 10.3
