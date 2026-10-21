import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOL = str(ROOT / "tools" / "parity31_limits.py")
LIMITS = ROOT / "tests" / "gates" / "parity31_limits.json"


def run(tmp_path, layers):
    res = tmp_path / "result.json"
    res.write_text(json.dumps({"layers": [
        {"layer": i, "spread_bf16_fp32": s, "control_noexp": c, "control_swap": c + 0.1,
         "bf16_vs_golden": 0.01, "fp32_vs_golden": 0.01} for i, (s, c) in enumerate(layers)]}))
    out = tmp_path / "limits.json"
    p = subprocess.run([sys.executable, TOOL, str(res), "--write", str(out), "--record", str(tmp_path / "rec.md"),
                        "--capture", str(tmp_path)], capture_output=True, text=True)
    return p, out, tmp_path / "rec.md"


def test_limits_formula_and_roundtrip(tmp_path):
    p, out, rec = run(tmp_path, [(0.01, 0.4), (0.02, 0.4), (0.03, 0.4), (0.07, 0.4)])
    assert p.returncode == 0, p.stderr
    d = json.loads(out.read_text())
    for x, s in zip(d["layers"], (0.01, 0.02, 0.03, 0.07)):
        assert abs(x["limit"] - (2 * s + 0.01)) < 1e-12 and x["separable"]
    assert json.loads(json.dumps(d)) == d
    assert "Verdict: INFO — limits written, 4/4 layers separable" in rec.read_text()


def test_not_separable_noise(tmp_path):
    p, _, rec = run(tmp_path, [(0.01, 0.4), (0.1, 0.39), (0.01, 0.4), (0.01, 0.4)])
    assert p.returncode == 1
    assert "DESIGN GATE FAILED: per-layer rel_l2 cannot separate noise from a wrong expert" in p.stderr
    assert "Verdict: FAIL" in rec.read_text()


def test_not_separable_vs_limit(tmp_path):
    # c = 4.5*s passes the first test but c < 2*L
    s = 0.01
    p, _, _ = run(tmp_path, [(0.01, 0.4), (0.01, 0.4), (0.01, 0.4), (s, 4.5 * s)])
    assert p.returncode == 1 and "DESIGN GATE FAILED" in p.stderr


def test_fewer_than_four_layers(tmp_path):
    p, _, _ = run(tmp_path, [(0.01, 0.4)] * 3)
    assert p.returncode == 1


def test_tracked_limits_file():
    if not LIMITS.exists():
        return
    d = json.loads(LIMITS.read_text())
    assert len(d["layers"]) == 4
    for x in d["layers"]:
        assert x["separable"] and x["limit"] >= 0.01
        assert abs(x["limit"] - (2 * x["spread"] + 0.01)) < 1e-12
