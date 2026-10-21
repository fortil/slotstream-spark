"""Fixtures of the gate tests: model and packed artifacts on the Spark, goldens, the capture directory.

Missing files make a test FAIL, never skip: a gate that cannot run is not a gate that passed.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tests" / "gates"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _dir_from_env(*names):
    for n in names:
        v = os.environ.get(n)
        if v:
            return Path(v)
    pytest.fail(f"none of {', '.join(names)} is set (run through tools/spark/py.sh)", pytrace=False)


@pytest.fixture(scope="session")
def model_dir():
    p = _dir_from_env("SLOTSTREAM_MODEL_DIR", "SLOTSTREAM_MODEL")
    if not p.is_dir():
        pytest.fail(f"model directory missing: {p}", pytrace=False)
    return p


@pytest.fixture(scope="session")
def packed_dir():
    p = _dir_from_env("SLOTSTREAM_PACKED")
    for name in ("trunk.safetensors", "experts.bin", "experts.manifest.json", "ngram.bin", "ngram.manifest.json"):
        if not (p / name).exists():
            pytest.fail(f"packed artifact missing: {p / name}", pytrace=False)
    return p


@pytest.fixture(scope="session")
def golden():
    import make_goldens as mg

    def load(pid):
        try:
            return mg.load_golden(pid)
        except (FileNotFoundError, OSError) as e:
            pytest.fail(f"golden {pid} unreadable: {e}", pytrace=False)
    return load


@pytest.fixture(scope="session")
def cap_dir():
    v = os.environ.get("CAP")
    return Path(v) if v else None
