#!/usr/bin/env python3
"""Block host names, logins, addresses and private project terms from tracked files.

Usage:
  private_terms_check.py [--terms-file FILE] --staged
  private_terms_check.py [--terms-file FILE] --tracked
  private_terms_check.py [--terms-file FILE] --paths PATH...
  private_terms_check.py [--terms-file FILE] --history RANGE

Two sources of patterns:
  * generic patterns committed here (home directories of a person, private and
    carrier-grade NAT addresses, tailnet names, an ssh login, a host name key);
  * a terms file that is never committed: `--terms-file`, else
    $SLOTSTREAM_PRIVATE_TERMS, else ~/.config/git/private-terms.txt. One extended
    regular expression per line, `#` starts a comment, matching is case-insensitive.
    In CI the file is written from a repository secret.

File names are checked as well as content. Output is `path:line: <pattern-id>`
(line 0 for a file name); a match from the terms file is reported as
`terms-file:<line number of the term>`, and the term itself is never printed.

Modes: --staged checks the staged blobs, --tracked checks the index plus new files
that are not ignored (files deleted in the work tree are skipped), --paths checks
files and directories as given, --history checks the lines added by RANGE
(for example origin/main..HEAD). Git modes skip third_party/, binary files and
files over 5 MB. Exit 0 when clean, 1 on a match, 2 on a usage or terms-file error.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024
SKIP_PREFIXES = ("third_party/", "tools/probes/")   # vendored trees and probe sources pinned by SHA256SUMS
DEFAULT_TERMS = "~/.config/git/private-terms.txt"

# Values that are generic on purpose and may appear in public files.
ALLOWED_LITERALS = (
    "172.17.0.1",            # default gateway of the Docker bridge network, the same on every host
    "billalpeza@gmail.com",  # the maintainer's public contact address (LICENSE, SECURITY.md)
)

SSH_PLACEHOLDER = re.compile(r"^(<[^>]*>|\$\{?\w+\}?)@")

_OCT = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"   # one valid octet; keeps log timestamps and version pins out

GENERIC = (
    ("home-path", re.compile(r"/home/[A-Za-z0-9_.-]+/")),
    ("users-path", re.compile(r"/Users/[A-Za-z0-9_.-]+/")),
    ("cgnat-ip", re.compile(r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b")),
    ("rfc1918-ip", re.compile(rf"(?<![\d.=])(?:10\.{_OCT}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.{_OCT}\.{_OCT}(?![\d.]?\d)")),
    ("tailnet-name", re.compile(r"\.ts\.net\b")),
    ("ssh-login", re.compile(r"\bssh\s+(?:-\S+\s+)*(\S+@)")),
    ("hostname-value", re.compile(r"host" r"name=(?!redacted\b)(?![$<`])\S")),
)


class TermsError(Exception):
    pass


def load_terms(path):
    """Compiled terms as [(line number, regex)]; a missing explicit file is an error."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise TermsError(f"terms file not found: {path}")
    out = []
    for n, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        try:
            out.append((n, re.compile(t, re.IGNORECASE)))
        except re.error as e:
            raise TermsError(f"terms file line {n}: not a valid regular expression ({e.msg})")
    return out


def resolve_terms_path(arg):
    return arg or os.environ.get("SLOTSTREAM_PRIVATE_TERMS") or DEFAULT_TERMS


def scan_text(text, terms=()):
    """Yield (line number, pattern id) for each line of text; one id per pattern per line."""
    for ln, line in enumerate(text.splitlines(), 1):
        for lit in ALLOWED_LITERALS:
            line = line.replace(lit, "")
        for pid, rx in GENERIC:
            for m in rx.finditer(line):
                if pid == "ssh-login" and SSH_PLACEHOLDER.match(m.group(1)):
                    continue
                yield ln, pid
                break
        for tn, rx in terms:
            if rx.search(line):
                yield ln, f"terms-file:{tn}"


def scan_name(path, terms=()):
    for _, pid in scan_text(path.replace("/", "\n"), terms):
        yield 0, pid


def is_binary(data):
    return b"\0" in data[:8192]


def scan_bytes(path, data, terms):
    found = list(scan_name(path, terms))
    if len(data) <= MAX_BYTES and not is_binary(data):
        found += list(scan_text(data.decode("utf-8", errors="replace"), terms))
    return found


def git(*args, check=True):
    r = subprocess.run(["git", *args], capture_output=True, check=False)
    if check and r.returncode != 0:
        raise RuntimeError(r.stderr.decode(errors="replace").strip() or "git failed")
    return r.stdout


def skip(path):
    return path.startswith(SKIP_PREFIXES)


def report(path, found, out):
    for ln, pid in found:
        print(f"{path}:{ln}: {pid}", file=out)
    return len(found)


def check_tracked(terms, out):
    names = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")
    n = 0
    for name in sorted(set(x for x in names if x)):
        if skip(name) or not os.path.isfile(name):
            continue
        n += report(name, scan_bytes(name, Path(name).read_bytes(), terms), out)
    return n


def check_staged(terms, out):
    names = git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z").decode().split("\0")
    n = 0
    for name in sorted(x for x in names if x):
        if skip(name):
            continue
        n += report(name, scan_bytes(name, git("show", f":{name}"), terms), out)
    return n


def check_paths(paths, terms, out):
    n = 0
    for root in paths:
        p = Path(root)
        files = sorted(x for x in p.rglob("*") if x.is_file()) if p.is_dir() else [p]
        for f in files:
            try:
                data = f.read_bytes()
            except OSError as e:
                print(f"{f}: unreadable ({e.strerror})", file=sys.stderr)
                return -1
            n += report(str(f), scan_bytes(str(f), data, terms), out)
    return n


def check_history(rng, terms, out):
    """Lines added by the commits in RANGE, reported as `path:line:` of the new file."""
    log = git("log", "-p", "-U0", "--no-color", "--format=", "--diff-filter=AMR", rng).decode("utf-8", errors="replace")
    n = 0
    path, new_line = None, 0
    for line in log.splitlines():
        if line.startswith("+++ "):
            path = line[6:] if line.startswith("+++ b/") else None
            if path and not skip(path):
                n += report(path, list(scan_name(path, terms)), out)
        elif line.startswith("@@"):
            m = re.search(r"\+(\d+)", line)
            new_line = int(m.group(1)) if m else 0
        elif line.startswith("+") and not line.startswith("+++") and path and not skip(path):
            for _, pid in scan_text(line[1:], terms):
                n += report(path, [(new_line, pid)], out)
            new_line += 1
    return n


def main(argv, out=sys.stdout):
    argv = list(argv)
    terms_arg = None
    if "--terms-file" in argv:
        i = argv.index("--terms-file")
        if i + 1 >= len(argv):
            print(__doc__, file=sys.stderr)
            return 2
        terms_arg = argv[i + 1]
        del argv[i:i + 2]
    modes = [a for a in argv if a in ("--staged", "--tracked", "--paths", "--history")]
    if len(modes) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    mode = modes[0]
    rest = [a for a in argv if a != mode]
    explicit = terms_arg is not None or "SLOTSTREAM_PRIVATE_TERMS" in os.environ
    try:
        terms = load_terms(resolve_terms_path(terms_arg))
    except TermsError as e:
        if explicit:
            print(f"private_terms_check: {e}", file=sys.stderr)
            return 2
        print("private_terms_check: no terms file; only the generic patterns apply", file=sys.stderr)
        terms = []
    try:
        if mode == "--tracked":
            n = check_tracked(terms, out)
        elif mode == "--staged":
            n = check_staged(terms, out)
        elif mode == "--paths":
            if not rest:
                print(__doc__, file=sys.stderr)
                return 2
            n = check_paths(rest, terms, out)
        else:
            if len(rest) != 1:
                print(__doc__, file=sys.stderr)
                return 2
            n = check_history(rest[0], terms, out)
    except RuntimeError as e:
        print(f"private_terms_check: {e}", file=sys.stderr)
        return 2
    if n < 0:
        return 2
    if n:
        print(f"private_terms_check: {n} match(es)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
