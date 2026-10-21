import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import evidence_scrub as es  # noqa: E402
import hostinfo  # noqa: E402

HOME = "/ho" + "me/someone/"          # built in pieces so the test file holds no path of its own
RULES = [("db-box-1", "postgres-1"), ("cache-box-1", "other-1"), ("asr-box", "asr-1"), ("lab-host", "host")]
CMD = (
    "bash job.sh\n"
    f"SLOTSTREAM_R={HOME}slotstream-spark\n"
    f"CAP={HOME}slotstream-spark/runs/2026/10/20261014T000000-x\n"
    "host" + "name=lab-host\n"
)


def _cap(tmp_path):
    d = tmp_path / "20261014T000000-x"
    d.mkdir()
    (d / "command.txt").write_text(CMD)
    (d / "host-before.json").write_text(json.dumps({
        "uname": "Linux lab-host 6.17.0 #29 aarch64",
        "docker": {"db-box-1": "Up 1 hour", "asr-box": "Up", "cache-box-1": "Up"},
    }))
    (d / "prod-psi.csv").write_text("1,db-box-1,0.0,0.0\n1,asr-box,0.0,0.0\n")
    return d


def test_text_files_are_scrubbed(tmp_path):
    d = _cap(tmp_path)
    es.scrub_dir(d, rules=RULES)
    cmd = (d / "command.txt").read_text()
    assert "/ho" + "me/" not in cmd and "$R/runs/2026/10" in cmd and "hostname=redacted" in cmd
    host = json.loads((d / "host-before.json").read_text())
    assert host["uname"].startswith("Linux host ")
    assert set(host["docker"]) == {"postgres-1", "asr-1", "other-1"}
    psi = (d / "prod-psi.csv").read_text()
    assert "postgres-1" in psi and "asr-1" in psi and "box" not in psi


def test_builtin_rules_work_without_a_rules_file(tmp_path):
    d = _cap(tmp_path)
    es.scrub_dir(d)
    assert "$R/runs" in (d / "command.txt").read_text()
    assert json.loads((d / "host-before.json").read_text())["uname"].startswith("Linux host ")


def test_scrub_is_idempotent(tmp_path):
    d = _cap(tmp_path)
    es.scrub_dir(d, rules=RULES)
    assert es.scrub_dir(d, rules=RULES) == []


def test_check_mode_writes_nothing(tmp_path):
    d = _cap(tmp_path)
    rules = tmp_path / "rules.txt"
    rules.write_text("# comment\ndb-box-1 => postgres-1\n")
    before = (d / "command.txt").read_text()
    assert es.main(["--check", "--rules", str(rules), str(d)]) == 1
    assert (d / "command.txt").read_text() == before
    es.main(["--rules", str(rules), str(d)])
    assert es.main(["--check", "--rules", str(rules), str(d)]) == 0


def test_rules_file_parsing(tmp_path):
    rules = tmp_path / "rules.txt"
    rules.write_text("ab => x\nabcd => y\n\n# note\n")
    assert es.load_rules(rules) == [("abcd", "y"), ("ab", "x")]
    rules.write_text("no arrow here\n")
    try:
        es.load_rules(rules)
    except ValueError:
        pass
    else:
        raise AssertionError("a rule without => must be rejected")


def test_hashed_and_binary_files_are_not_touched(tmp_path):
    d = _cap(tmp_path)
    raw = HOME.encode()
    limits = d / "limits_p31.json"
    limits.write_text('{"p": "%sx"}' % HOME)
    npz = d / "trace.npz"
    npz.write_bytes(b"\x00" + raw + b"\xff")
    odd = d / "blob.log"
    odd.write_bytes(b"\xff\xfe" + raw)
    es.scrub_dir(d, rules=RULES)
    assert limits.read_text() == '{"p": "%sx"}' % HOME
    assert npz.read_bytes() == b"\x00" + raw + b"\xff"
    assert odd.read_bytes() == b"\xff\xfe" + raw


def test_result_json_only_loses_home_paths(tmp_path):
    d = _cap(tmp_path)
    body = {"model": f"{HOME}slotstream-spark/model/x", "host": "lab-host", "n": 3, "sha": "ab"}
    (d / "result.json").write_text(json.dumps(body))
    es.scrub_dir(d, rules=RULES)
    got = json.loads((d / "result.json").read_text())
    assert got == {**body, "model": "$R/model/x"}


def test_hostinfo_reports_families_not_names():
    assert [hostinfo.container_family(n) for n in ("job-asr", "x-embeddings-1", "a-qdrant-1", "db-postgres-1", "zzz")] == [
        "asr", "embeddings", "qdrant", "postgres", "other"]
