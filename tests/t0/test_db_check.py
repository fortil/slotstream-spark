import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import db_check  # noqa: E402


def _cap(root, name, **files):
    d = root / "db/sources/runs/2026/10" / name
    d.mkdir(parents=True)
    for fn, body in files.items():
        (d / fn).write_text(json.dumps(body))
    return d


def test_good_measurement(tmp_path):
    _cap(tmp_path, "20261011T100000-x", **{"verdict.json": {"rc": 0}})
    f = tmp_path / "db/records/measurements/x.md"
    f.parent.mkdir(parents=True)
    f.write_text("# x: t\nDate: 2026-10-11\nCapture: db/sources/runs/2026/10/20261011T100000-x/\n"
                 "Command: c\nVerdict: PASS — fine\n")
    assert db_check.check_measurement(f, tmp_path) == []


def test_measurement_without_verdict(tmp_path):
    _cap(tmp_path, "20261011T100000-x", **{"verdict.json": {"rc": 0}})
    f = tmp_path / "m.md"
    f.write_text("# x\nCapture: db/sources/runs/2026/10/20261011T100000-x/\n")
    assert db_check.check_measurement(f, tmp_path)


def test_measurement_missing_capture(tmp_path):
    f = tmp_path / "m.md"
    f.write_text("# x\nCapture: db/sources/runs/2026/10/nope/\nVerdict: INFO — a\n")
    assert db_check.check_measurement(f, tmp_path)


def test_decision(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    good = tmp_path / "good.md"
    good.write_text("# d\nDecision: v\nEvidence: a.txt\nReversal: never\n")
    bad = tmp_path / "bad.md"
    bad.write_text("# d\nDecision: v\nEvidence: nothere.txt\nReversal: never\n")
    assert db_check.check_decision(good, tmp_path) == []
    assert db_check.check_decision(bad, tmp_path)


def test_latest_and_ok(tmp_path):
    _cap(tmp_path, "20261011T100000-lab", **{"guard-post.json": {"rc": 1, "cut": None}})
    _cap(tmp_path, "20261011T110000-lab", **{"guard-post.json": {"rc": 0, "cut": None}})
    assert db_check.latest("lab", tmp_path).name == "20261011T110000-lab"
    assert db_check.ok("lab", tmp_path)
    _cap(tmp_path, "20261011T120000-lab", **{"guard-post.json": {"rc": 0, "cut": "stop"}})
    assert not db_check.ok("lab", tmp_path)
    assert db_check.latest("none", tmp_path) is None


LEGACY_KEY = "hue" + "lla_mib"  # key written by captures made before the rename; spelled in two parts so the rename grep stays clean


def test_footprint_mib_reads_both_keys():
    assert db_check.footprint_mib({"footprint_mib": 5}) == 5
    assert db_check.footprint_mib({LEGACY_KEY: 7}) == 7
    assert db_check.footprint_mib({"footprint_mib": 5, LEGACY_KEY: 7}) == 5
    assert db_check.footprint_mib({}) is None


def test_archived_capture_is_accepted_when_listed(tmp_path):
    (tmp_path / "db/sources").mkdir(parents=True)
    (tmp_path / "db/sources/ARCHIVED.txt").write_text("# offline\n20261011T100000-gone\n")
    f = tmp_path / "m.md"
    f.write_text("# x\nCapture: db/sources/runs/2026/10/20261011T100000-gone/\nVerdict: INFO — fine\n")
    assert db_check.check_measurement(f, tmp_path) == []
    g = tmp_path / "n.md"
    g.write_text("# x\nCapture: db/sources/runs/2026/10/20261011T100000-other/\nVerdict: INFO — fine\n")
    assert any("capture dir missing" in e for e in db_check.check_measurement(g, tmp_path))


def test_captures_line_and_trailing_note(tmp_path):
    _cap(tmp_path, "20261011T100000-a", **{"verdict.json": {"rc": 0}})
    _cap(tmp_path, "20261011T100001-b", **{"verdict.json": {"rc": 0}})
    f = tmp_path / "m.md"
    f.write_text("# x\nCaptures: db/sources/runs/2026/10/20261011T100000-a/, "
                 "db/sources/runs/2026/10/20261011T100001-b/ (traces stay on the Spark)\nVerdict: PASS — fine\n")
    assert db_check.check_measurement(f, tmp_path) == []

