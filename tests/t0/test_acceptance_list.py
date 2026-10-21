"""tools/acceptance.sh: the list of checks, its shell syntax, and the record checks it embeds (run here on the Mac
against the tracked records, and on mutated copies)."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/acceptance.sh"
REQUIRED_ROWS = ["oracle-parity31-calibration", "parity31-limits-rebind", "lb-yardstick-consistency", "contract-rebind"]


def sh(*args):
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, cwd=ROOT)


@pytest.fixture(scope="module")
def checks_py(tmp_path_factory):
    p = tmp_path_factory.mktemp("acc") / "accept_checks.py"
    p.write_text(sh("--emit-py").stdout)
    return p


def run_check(checks_py, cmd, root, *extra):
    r = subprocess.run([sys.executable, str(checks_py), cmd, "--root", str(root), *extra], capture_output=True, text=True)
    return r.returncode, r.stdout


def test_list_names_the_required_rows():
    r = sh("--list")
    assert r.returncode == 0
    names = r.stdout.split()
    for row in REQUIRED_ROWS:
        assert row in names
    assert len(names) == len(set(names))
    for n in ("claims-gate", "runbook-check", "docs-check", "required-records", "t0-suite", "sweep-check", "live-acceptance",
              "ppl-16-windows", "self-impact"):
        assert n in names


def test_every_listed_check_has_a_function():
    text = SCRIPT.read_text()
    for name in sh("--list").stdout.split():
        assert re.search(r"^c_%s\(\)" % name.replace("-", "_"), text, re.M), name


def test_shell_syntax_and_usage():
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    assert subprocess.run(["bash", "-n", str(ROOT / "tools/serve_gates.sh")]).returncode == 0
    assert sh().returncode == 2
    assert sh("bogus").returncode == 2


def test_embedded_python_compiles(checks_py):
    compile(checks_py.read_text(), "accept_checks.py", "exec")


@pytest.mark.parametrize("cmd", ["info", "parity31", "lb", "contract", "records"])
def test_record_checks_pass_on_the_tracked_tree(checks_py, cmd):
    rc, out = run_check(checks_py, cmd, ROOT)
    assert rc == 0, out
    assert "RESULT: PASS" in out.splitlines()[-1]


def mutated(tmp_path):
    """A root whose records and config are copies and everything else points at the real tree."""
    (tmp_path / "db").mkdir()
    shutil.copytree(ROOT / "db/records", tmp_path / "db/records")
    (tmp_path / "db/sources").symlink_to(ROOT / "db/sources")
    (tmp_path / "tools").symlink_to(ROOT / "tools")
    (tmp_path / "slotstream_spark").mkdir()
    for child in (ROOT / "slotstream_spark").iterdir():
        if child.name not in ("config", "__pycache__"):
            (tmp_path / "slotstream_spark" / child.name).symlink_to(child)
    shutil.copytree(ROOT / "slotstream_spark/config", tmp_path / "slotstream_spark/config")
    (tmp_path / "tests/gates").mkdir(parents=True)
    shutil.copy(ROOT / "tests/gates/parity31_limits.json", tmp_path / "tests/gates/parity31_limits.json")
    return tmp_path


def edit(path, old, new):
    t = path.read_text()
    assert old in t
    path.write_text(t.replace(old, new, 1))


def test_records_fail_without_a_rule_line_and_with_a_failed_verdict(checks_py, tmp_path):
    r = mutated(tmp_path)
    edit(r / "db/records/measurements/f5-sweep-check.md", "Logits rule:", "Logits ruleX:")
    rc, out = run_check(checks_py, "records", r)
    assert rc == 1 and "f5-sweep-check: missing line" in out
    r2 = mutated(tmp_path / "b") if (tmp_path / "b").mkdir() is None else None
    edit(r2 / "db/records/measurements/h3.md", "\nVerdict: PASS", "\nVerdict: FAIL")
    rc, out = run_check(checks_py, "records", r2)
    assert rc == 1 and "h3: Verdict is not PASS" in out


def test_records_fail_when_the_h5_tree_changed(checks_py, tmp_path):
    r = mutated(tmp_path)
    edit(r / "db/records/measurements/h5-agents.md", "server_tree_sha256: ", "server_tree_sha256: 0")
    rc, out = run_check(checks_py, "records", r)
    assert rc == 1 and "h5-agents" in out


def test_parity31_fails_when_the_limits_move(checks_py, tmp_path):
    r = mutated(tmp_path)
    p = r / "tests/gates/parity31_limits.json"
    d = json.loads(p.read_text())
    d["layers"][0]["limit"] += 1e-6
    p.write_text(json.dumps(d, indent=2) + "\n")
    rc, out = run_check(checks_py, "parity31", r)
    assert rc == 1 and "limit is not 2*spread" in out and "limits_sha256" in out


def test_contract_fails_when_an_array_or_the_rule_changes(checks_py, tmp_path):
    r = mutated(tmp_path)
    toml = r / "slotstream_spark/config/numeric_contract.toml"
    edit(toml, "seq_factor = 2.0", "seq_factor = 3.0")
    rc, out = run_check(checks_py, "contract", r)
    assert rc == 1 and "seq_factor" in out
    r2 = mutated(tmp_path / "c") if (tmp_path / "c").mkdir() is None else None
    edit(r2 / "slotstream_spark/config/numeric_contract.toml", "top1_min = 0.889763779527559", "top1_min = 0.88")
    rc, out = run_check(checks_py, "contract", r2)
    assert rc == 1 and "top1_min" in out


def test_lb_fails_without_a_justification(checks_py, tmp_path):
    r = mutated(tmp_path)
    edit(r / "db/records/measurements/f9-campaign-d5.md", "Justification: ttft", "Justification: none")
    rc, out = run_check(checks_py, "lb", r)
    assert rc == 1 and "Justification" in out


def test_ppl16_requires_bit_identical_windows(checks_py, tmp_path):
    cap = tmp_path / "cap"
    (cap / "quality").mkdir(parents=True)
    rows = [{"nll": 100.0 + i * 0.1234567890123, "start": 2048 * i, "tokens": 2047, "window": i} for i in range(145)]
    (cap / "quality/ppl.json").write_text(json.dumps({"per_window": rows, "ids_sha256": "x"}))
    (tmp_path / "db/records/measurements").mkdir(parents=True)
    (tmp_path / "db/records/measurements/f9-quality.md").write_text("Capture: cap/\n")
    new = tmp_path / "new.json"
    new.write_text(json.dumps({"per_window": rows[:16], "ids_sha256": "x"}))
    rc, out = run_check(checks_py, "ppl16", tmp_path, str(new))
    assert rc == 0, out
    bad = [dict(r) for r in rows[:16]]
    bad[7]["nll"] += 1e-12
    new.write_text(json.dumps({"per_window": bad, "ids_sha256": "x"}))
    rc, out = run_check(checks_py, "ppl16", tmp_path, str(new))
    assert rc == 1 and "windows [7]" in out
