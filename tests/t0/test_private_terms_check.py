import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import private_terms_check as ptc  # noqa: E402

# Leaky values are assembled from pieces so this file passes its own scan.
HOME = "/ho" + "me/someone/"
MAC = "/Us" + "ers/someone/"
TERM = "zq" + "-private-name"


TERMS_REL = "TERMS"


def terms_of(repo):
    return repo.parent / (repo.name + "-terms.txt")


def run(args, cwd, env=None):
    e = dict(os.environ, **(env or {}))
    e.pop("SLOTSTREAM_PRIVATE_TERMS", None)
    e.update(env or {})
    args = [str(terms_of(Path(cwd))) if a == TERMS_REL else a for a in args]
    return subprocess.run([sys.executable, str(ROOT / "tools/private_terms_check.py"), *args],
                          cwd=cwd, capture_output=True, text=True, env=e)


@pytest.fixture
def repo(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.org"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    # the terms file sits next to the repository, not inside it
    (tmp_path.parent / (tmp_path.name + "-terms.txt")).write_text(f"# project terms\n\n{TERM}\n")
    return tmp_path


def stage(repo, name, text):
    p = repo / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    subprocess.run(["git", "add", name], cwd=repo, check=True)


def test_leak_in_content(repo):
    stage(repo, "a.md", f"fine\nsee {HOME}project\n")
    r = run(["--staged", "--terms-file", TERMS_REL], repo)
    assert r.returncode == 1 and "a.md:2: home-path" in r.stdout


def test_leak_in_file_name(repo):
    stage(repo, f"notes-{TERM}.md", "clean text\n")
    r = run(["--staged", "--terms-file", TERMS_REL], repo)
    assert r.returncode == 1 and ":0: terms-file:3" in r.stdout


def test_terms_file_match_is_reported_by_id_not_by_term(repo):
    stage(repo, "b.md", f"x {TERM} y\n")
    r = run(["--staged", "--terms-file", TERMS_REL], repo)
    assert r.returncode == 1
    assert "b.md:1: terms-file:3" in r.stdout
    assert TERM not in r.stdout and TERM not in r.stderr


def test_placeholders_and_owner_values_pass(repo):
    stage(repo, "c.md", "\n".join([
        "paths look like /ho" + "me/<user>/slotstream-spark and /Us" + "ers/<name>/x",
        "ss" + "h <login>@<host> and ss" + "h ${USER}@host",
        "hostname=redacted and hostname=$(hostname)",
        "gateway 172.17.0.1, example 203.0.113.9, timestamp 10.37.423.047, pin pkg==10.4.0.35",
        "contact billalpeza@" + "gmail.com",
    ]) + "\n")
    r = run(["--staged", "--terms-file", TERMS_REL], repo)
    assert r.returncode == 0, r.stdout


@pytest.mark.parametrize("line,pid", [
    ("ss" + "h dev@box", "ssh-login"),
    ("ss" + "h user@host", "ssh-login"),
    ("host" + 'name="lab1"', "hostname-value"),
    ("host" + "name='lab1'", "hostname-value"),
    ("host" + "name={lab1}", "hostname-value"),
    ("ts " + "box.tail1234.ts" + ".net", "tailnet-name"),
    ("ip " + "100.82" + ".1.2", "cgnat-ip"),
    ("ip " + "10.1" + ".2.3", "rfc1918-ip"),
    ("ip " + "192.168" + ".0.9", "rfc1918-ip"),
    ("ip " + "172.20" + ".0.1", "rfc1918-ip"),
    ("host" + "name=lab1", "hostname-value"),
    ("at " + MAC + "x", "users-path"),
])
def test_generic_patterns(repo, line, pid):
    stage(repo, "d.md", line + "\n")
    r = run(["--staged", "--terms-file", TERMS_REL], repo)
    assert r.returncode == 1 and pid in r.stdout


def test_tracked_mode_skips_third_party_and_binaries(repo):
    stage(repo, "third_party/x/LICENSE", f"{HOME}x\n")
    (repo / "blob.bin").write_bytes(b"\0" + HOME.encode())
    subprocess.run(["git", "add", "blob.bin"], cwd=repo, check=True)
    stage(repo, "ok.md", "nothing here\n")
    assert run(["--tracked", "--terms-file", TERMS_REL], repo).returncode == 0
    stage(repo, "bad.md", f"{HOME}y\n")
    assert run(["--tracked", "--terms-file", TERMS_REL], repo).returncode == 1


def test_paths_mode_and_missing_terms_file(repo):
    (repo / "e.md").write_text(f"{HOME}z\n")
    assert run(["--paths", "e.md", "--terms-file", TERMS_REL], repo).returncode == 1
    r = run(["--paths", "e.md", "--terms-file", "nope.txt"], repo)
    assert r.returncode == 2 and "terms file not found" in r.stderr


def test_invalid_term_regex_is_rejected(repo):
    (repo / "bad-terms.txt").write_text("(unclosed\n")
    r = run(["--paths", ".", "--terms-file", "bad-terms.txt"], repo)
    assert r.returncode == 2 and "line 1" in r.stderr


def test_history_mode_reports_added_lines(repo):
    stage(repo, "f.md", "first\n")
    subprocess.run(["git", "commit", "-q", "-m", "one"], cwd=repo, check=True)
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True).stdout.strip()
    stage(repo, "f.md", f"first\nsecond {HOME}p\n")
    subprocess.run(["git", "commit", "-q", "-m", "two"], cwd=repo, check=True)
    r = run(["--history", f"{base}..HEAD", "--terms-file", TERMS_REL], repo)
    assert r.returncode == 1 and "f.md:2: home-path" in r.stdout


def test_hook_refuses_a_staged_leak(repo):
    hooks = repo / ".githooks"
    hooks.mkdir()
    (hooks / "pre-commit").write_text((ROOT / ".githooks/pre-commit").read_text())
    (hooks / "pre-commit").chmod(0o755)
    (repo / "tools").mkdir()
    (repo / "tools/private_terms_check.py").write_text((ROOT / "tools/private_terms_check.py").read_text())
    subprocess.run(["git", "config", "core.hooksPath", ".githooks"], cwd=repo, check=True)
    stage(repo, "g.md", f"{HOME}q\n")
    env = dict(os.environ, SLOTSTREAM_PRIVATE_TERMS=str(terms_of(repo)))
    r = subprocess.run(["git", "commit", "-q", "-m", "x"], cwd=repo, capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "home-path" in r.stdout + r.stderr
    stage(repo, "g.md", "clean\n")
    r = subprocess.run(["git", "commit", "-q", "-m", "x"], cwd=repo, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr


def test_scan_text_unit():
    assert list(ptc.scan_text("nothing\n")) == []
    assert list(ptc.scan_text(f"a {HOME}b")) == [(1, "home-path")]
