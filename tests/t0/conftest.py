import tomllib
from pathlib import Path

import pytest

PLATFORM = Path(__file__).resolve().parents[2] / "slotstream_spark" / "config" / "platform.toml"


@pytest.fixture(autouse=True)
def _kernel_plan_gate(request):
    """CUDA tests assume Triton plan A; any other measured plan stops the run for a decision."""
    if request.node.get_closest_marker("cuda") is None:
        return
    plan = tomllib.loads(PLATFORM.read_text())["kernels"]["plan"]
    if plan != "A":
        pytest.fail(f"DECISION: platform.toml kernels.plan={plan}; Part B assumes Triton plan A "
                    "(see docs/architecture.md); ask the user", pytrace=False)
