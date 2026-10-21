import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.gate
def test_oracle_reproduces_parity31():
    p = subprocess.run([sys.executable, "tools/ref_layerwise.py", "--parity31", "third_party/slotstream/bench/parity31"],
                       cwd=ROOT, capture_output=True, text=True, env=os.environ.copy())
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    assert lines, f"oracle printed no result (rc {p.returncode}): {p.stderr[-2000:]}"
    r = json.loads(lines[-1])
    per = "; ".join(f"L{x['layer']} {x['rel_l2']:.4g}/{x['limit']:.4g} sep={x['separable']}" for x in r["layers"])
    ok = (r["pass"] and r["ngram_ids_equal"] and r["dequant_row_exact"]
          and all(x["separable"] and x["rel_l2"] <= x["limit"] for x in r["layers"]))
    assert ok, ("DESIGN GATE FAILED: oracle does not reproduce parity31 — stop and ask the user "
                f"(layers {per}; ids_equal {r['ngram_ids_equal']}; row_exact {r['dequant_row_exact']})")
    assert p.returncode == 0
