#!/usr/bin/env python3
"""Shim: the expert-frequency tool lives in slotstream_spark.memory.expert_freq."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slotstream_spark.memory import expert_freq as _m  # noqa: E402

if __name__ == "__main__":
    sys.exit(_m.main())
sys.modules[__name__] = _m
