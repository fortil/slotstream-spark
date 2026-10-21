#!/usr/bin/env python3
"""Scrub host-specific text from curated capture directories (stdlib only).

Usage:
  evidence_scrub.py [--check] [--rules FILE] CAPDIR...

Rewrites text files in place. Built-in rules: the project tree under a home
directory becomes `$R`, other home paths become `/home/<user>/`, the host name key
becomes `hostname=redacted`, the host name in a `uname` line becomes `host` and
the user in `pytest-of-<user>` becomes `user`.

Names that only this machine knows (container names, job names, the host name
outside `uname`) come from a private rules file, never from this source:
`--rules FILE`, else `$SLOTSTREAM_SCRUB_RULES`, else `private/scrub-rules.txt`
under the repository root. One rule per line, `old => new`, `#` starts a comment,
for example `my-db-container => postgres-1`. Longer `old` strings apply first.
Container labels follow the family names that tools/impact_report.py matches by
substring (asr, embedding, postgres, qdrant; anything else is `other-N`).

Never touched: binary files, `*.npz`, `*.npy` and `limits*.json` (hashed by the
rebind checks). `result.json` and `verdict.json` get only the home-path rule,
which changes string values and never keys, numbers or hashes.

With --check nothing is written; the exit code is 1 when a file would change.
"""
import fnmatch
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = {".txt", ".json", ".jsonl", ".log", ".csv", ".md", ".toml", ".yaml", ".yml"}
NEVER = ("*.npz", "*.npy", "limits*.json")
PATH_ONLY = {"result.json", "verdict.json"}

HOME_PROJECT = re.compile(r"/home/[A-Za-z0-9_.-]+/slotstream-spark")
HOME_OTHER = re.compile(r"/home/(?!<user>)[A-Za-z0-9_.-]+/")
HOSTNAME_KV = re.compile(r"host" r"name=\S+")
UNAME_HOST = re.compile(r"(Linux )(?!version\b)(?!host )\S+(?= \d+\.\d+)")
PYTEST_USER = re.compile(r"pytest-of-(?!user\b)[A-Za-z0-9_.-]+")


def scrub_paths(text):
    text = HOME_PROJECT.sub("$R", text)
    return HOME_OTHER.sub("/home/<user>/", text)


def load_rules(path=None):
    """Private `old => new` rules as a list sorted longest `old` first; [] when no file exists."""
    path = path or os.environ.get("SLOTSTREAM_SCRUB_RULES") or ROOT / "private" / "scrub-rules.txt"
    p = Path(path)
    if not p.is_file():
        return []
    rules = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        old, sep, new = line.partition("=>")
        if not sep or not old.strip():
            raise ValueError(f"{p}: bad rule (expected 'old => new'): line {len(rules) + 1}")
        rules.append((old.strip(), new.strip()))
    return sorted(rules, key=lambda r: -len(r[0]))


def scrub_text(text, rules=()):
    text = scrub_paths(text)
    text = HOSTNAME_KV.sub("hostname=redacted", text)
    text = UNAME_HOST.sub(r"\1host", text)
    text = PYTEST_USER.sub("pytest-of-user", text)
    for old, new in rules:
        text = text.replace(old, new)
    return text


def files_of(capdir):
    for p in sorted(Path(capdir).rglob("*")):
        if not p.is_file() or p.suffix not in TEXT_SUFFIXES:
            continue
        if any(fnmatch.fnmatch(p.name, pat) for pat in NEVER):
            continue
        yield p


def scrub_dir(capdir, write=True, rules=()):
    """Return the list of files that changed (or would change when write is False)."""
    changed = []
    for p in files_of(capdir):
        try:
            old = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        new = scrub_paths(old) if p.name in PATH_ONLY else scrub_text(old, rules)
        if new != old:
            changed.append(p)
            if write:
                p.write_text(new, encoding="utf-8")
    return changed


def main(argv):
    argv = list(argv)
    check = "--check" in argv
    argv = [a for a in argv if a != "--check"]
    rules_path = None
    if "--rules" in argv:
        i = argv.index("--rules")
        rules_path = argv[i + 1] if i + 1 < len(argv) else None
        del argv[i:i + 2]
        if rules_path is None:
            print(__doc__)
            return 2
    dirs = argv
    if not dirs:
        print(__doc__)
        return 2
    rules = load_rules(rules_path)
    if not rules:
        print("evidence_scrub: no private rules file; only the built-in rules apply", file=sys.stderr)
    rc = 0
    for d in dirs:
        if not Path(d).is_dir():
            print(f"{d}: not a directory", file=sys.stderr)
            return 2
        changed = scrub_dir(d, write=not check, rules=rules)
        for p in changed:
            print(("would change " if check else "scrubbed ") + str(p))
        if check and changed:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
