#!/usr/bin/env python3
"""Reader of the D5 yardstick record: the only place that knows where the figures live (stdlib only).

  yardstick.py [--path P] [--key K]     prints the table named K (default: the yardstick) as JSON
"""
import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

# Table and record names (vara-d5.md, [vara], [vara_b]) are part of the recorded decision.
DEFAULT = "db/records/decisions/vara-d5.md"
REQUIRED = ("arm", "decode_tps_median", "prefill_tps_8k", "op_offload", "prefill_device")


def _path(path):
    p = Path(path)
    return p if p.is_absolute() or p.exists() else Path(__file__).resolve().parents[1] / path


def load_record(path=DEFAULT):
    """The first fenced toml block of the record, parsed."""
    m = re.search(r"```toml\n(.*?)\n```", _path(path).read_text(), re.S)
    if not m:
        raise ValueError("yardstick: no toml block")
    return tomllib.loads(m.group(1))


def yardstick(path=DEFAULT):
    d = load_record(path)
    if "yardstick" not in d:
        raise ValueError("yardstick: missing yardstick")
    t = d.get(d["yardstick"])
    if not isinstance(t, dict):
        raise ValueError(f"yardstick: missing {d['yardstick']}")
    for k in REQUIRED:
        if k not in t:
            raise ValueError(f"yardstick: missing {k}")
    return t


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT)
    ap.add_argument("--key")
    a = ap.parse_args(argv)
    d = load_record(a.path)
    print(json.dumps(d[a.key] if a.key else yardstick(a.path), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
