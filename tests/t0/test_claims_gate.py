"""tools/claims_gate.py and tools/runbook_check.py: unit cases on a throwaway tree, plus the real repository."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import claims_gate  # noqa: E402
import runbook_check  # noqa: E402


def claim(root, cid, needle="42.5 tok/s", status="current", surfaces="README.md", supported="rec.md", body="body"):
    d = root / "db/records/claims"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"claim-{cid}.md").write_text(
        f"---\ntype: claim\nid: {cid}\nsummary: s\nbasis: measured\nneedle: {needle}\n"
        f"supported_by: {supported}\nsurfaces: {surfaces}\nstatus: {status}\n---\n{body}\n")


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "README.md").write_text("# x\n\n## Speed\n\nDecode 42.5\ntok/s today.\n")
    (tmp_path / "rec.md").write_text("decode 42.5 tok/s measured\n")
    return tmp_path


def test_passes_and_ignores_wrapping(tree, capsys):
    claim(tree, "a")
    assert claims_gate.run(str(tree)) == 0
    assert "0 failures" in capsys.readouterr().out


def test_needle_missing_from_surface_fails(tree, capsys):
    claim(tree, "a", needle="43.5 tok/s", supported="")
    (tree / "rec.md").write_text("43.5 tok/s\n")
    claim(tree, "a", needle="43.5 tok/s")
    assert claims_gate.run(str(tree)) == 1
    assert "missing from README.md" in capsys.readouterr().err


def test_needle_missing_from_supporting_record_fails(tree, capsys):
    claim(tree, "a")
    (tree / "rec.md").write_text("decode 42.4 tok/s measured\n")
    assert claims_gate.run(str(tree)) == 1
    assert "missing from supporting record" in capsys.readouterr().err


def test_withdrawn_needle_still_present_fails(tree, capsys):
    claim(tree, "a", status="withdrawn")
    assert claims_gate.run(str(tree)) == 1
    assert "still on README.md" in capsys.readouterr().err


def test_current_claim_without_supported_by_fails(tree, capsys):
    claim(tree, "a", supported="")
    assert claims_gate.run(str(tree)) == 1
    assert "needs supported_by" in capsys.readouterr().err


def test_readme_speed_number_without_claim_fails(tree, capsys):
    claim(tree, "a")
    (tree / "README.md").write_text("## Speed\n\n42.5 tok/s and 99.9 tok/s\n")
    assert claims_gate.run(str(tree)) == 1
    assert "number 99.9 has no claim" in capsys.readouterr().err


def test_numbers_outside_judged_sections_are_not_judged(tree):
    claim(tree, "a")
    (tree / "README.md").write_text("103.8 GB\n\n## Speed\n\n42.5 tok/s\n\n## Status and limits\n\n77 things\n")
    assert claims_gate.run(str(tree)) == 0


def test_quality_number_without_claim_fails_and_other_sections_are_not_judged(tree, capsys):
    claim(tree, "a")
    (tree / "README.md").write_text(
        "## Speed\n\n42.5 tok/s\n\n## Quality\n\nppl 7.7\n\n## Status and limits\n\n88 things\n")
    assert claims_gate.run(str(tree)) == 1
    err = capsys.readouterr().err
    assert "README ## Quality: number 7.7 has no claim" in err
    assert "88" not in err


def test_real_workload_number_without_claim_fails(tree, capsys):
    claim(tree, "a")
    (tree / "README.md").write_text("## Speed\n\n42.5 tok/s\n\n## Real workload\n\n61.3 s per call\n")
    assert claims_gate.run(str(tree)) == 1
    assert "README ## Real workload: number 61.3 has no claim" in capsys.readouterr().err


def test_judged_numbers_reads_all_four_headings():
    text = ("## Speed\n\n1.5\n\n## Quality\n\n2.5\n\n## Real workload\n\n5.5\n\n"
            "## Memory and coexistence\n\n3.5\n\n## Other\n\n4.5\n")
    assert claims_gate.judged_numbers(text) == {
        "## Speed": ["1.5"], "## Quality": ["2.5"], "## Real workload": ["5.5"],
        "## Memory and coexistence": ["3.5"]}
    assert claims_gate.judged_numbers("## Other\n\n4.5\n") == {}


def test_missing_claims_folder_fails(tmp_path):
    assert claims_gate.run(str(tmp_path)) == 1


def test_repository_claims_gate_passes(capsys):
    if not (ROOT / "docs/measurements.md").exists():
        pytest.skip("docs/measurements.md is a claim surface; it is written in the docs step")
    if "\n## Speed" not in (ROOT / "README.md").read_text(encoding="utf-8"):
        pytest.skip("README.md and RUNBOOK.md carry the claim needles after the README rewrite step")
    assert claims_gate.run(str(ROOT)) == 0
    out = capsys.readouterr().out
    assert "0 failures" in out
    n = len(list((ROOT / "db/records/claims").glob("claim-*.md")))
    assert 15 <= n <= 30
    assert (ROOT / "db/records/claims/claim-vs-llamacpp-arm-b.md").exists()


def _runbook(root, extra="", skip=()):
    heads = [h for h in runbook_check.REQUIRED if h not in skip]
    (root / "deploy").mkdir(exist_ok=True)
    (root / "deploy/a.service").write_text("")
    (root / "RUNBOOK.md").write_text("".join(f"## {h}\n\ntext\n\n" for h in heads) + extra)
    return root / "RUNBOOK.md"


def test_runbook_ok(tmp_path):
    rb = _runbook(tmp_path, "See `deploy/a.service` and `a.service` and `$R/run/STOP`.\n")
    assert runbook_check.check(str(rb), str(tmp_path)) == []


def test_runbook_missing_heading(tmp_path):
    rb = _runbook(tmp_path, skip=("Rollback",))
    assert runbook_check.check(str(rb), str(tmp_path)) == ["missing heading: Rollback"]


def test_runbook_missing_path_and_unit(tmp_path):
    rb = _runbook(tmp_path, "Run `tools/nothing.sh` and start `b.service`.\n")
    errs = runbook_check.check(str(rb), str(tmp_path))
    assert "path does not exist: tools/nothing.sh" in errs
    assert "unit not in deploy/: b.service" in errs


def test_repository_runbook_passes(capsys):
    if "## Coexistence with other model servers" not in (ROOT / "RUNBOOK.md").read_text(encoding="utf-8"):
        pytest.skip("RUNBOOK.md gets the renamed heading in the RUNBOOK rewrite step")
    assert runbook_check.main([str(ROOT / "RUNBOOK.md")]) == 0
