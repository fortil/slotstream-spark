#!/usr/bin/env python3
"""Check the operator runbook against the repository.

  runbook_check.py [RUNBOOK.md] [--root DIR]

1. Every required heading is present (REQUIRED below).
2. Every path in a code span that looks like a repository path (it contains a slash and no wildcard, placeholder or
   scheme, or it ends in a known file extension) exists under the root. A span with `$`, `<`, `*`, `~` or `://` is
   a template or a remote path and is skipped.
3. Every systemd unit named in a code span (`*.service`, `*.socket`, `*.timer`) exists in deploy/.
Standard library only.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUIRED = [
    "Overview", "Modes", "Schedule", "Mac access", "Health", "Coexistence with other model servers", "Emergency stop", "Updating",
    "Measurements and A/B", "Incidents", "Rollback",
]
EXTS = (".sh", ".py", ".md", ".toml", ".json", ".service", ".socket", ".timer")
SPAN = re.compile(r"`([^`\n]+)`")
UNIT = re.compile(r"^[\w@.-]+\.(?:service|socket|timer)$")


def headings(text):
    return [m.group(2).strip() for m in re.finditer(r"^(#{1,6})\s+(.*)$", text, re.M)]


def check(runbook, root):
    errs = []
    with open(runbook, encoding="utf-8") as fh:
        text = fh.read()
    have = headings(text)
    for req in REQUIRED:
        if not any(h.lower() == req.lower() for h in have):
            errs.append("missing heading: %s" % req)
    seen = set()
    for span in SPAN.findall(text):
        for tok in span.split():
            tok = tok.strip(",;:()")
            if not tok or tok in seen:
                continue
            seen.add(tok)
            if any(c in tok for c in "$<>*~=") or "://" in tok or tok.startswith("-") or tok.startswith("/"):
                continue
            if UNIT.match(tok):
                if not os.path.isfile(os.path.join(root, "deploy", tok)):
                    errs.append("unit not in deploy/: %s" % tok)
                continue
            looks_path = ("/" in tok and not tok.endswith("/") or tok.endswith(EXTS)) and re.match(r"^[\w./@+-]+$", tok)
            if looks_path and not os.path.exists(os.path.join(root, tok)):
                errs.append("path does not exist: %s" % tok)
    return errs


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    root = ROOT
    if "--root" in argv:
        i = argv.index("--root")
        root = argv[i + 1]
        del argv[i:i + 2]
    runbook = argv[0] if argv else os.path.join(root, "RUNBOOK.md")
    if not os.path.isabs(runbook) and not os.path.exists(runbook):
        runbook = os.path.join(root, runbook)
    if not os.path.isfile(runbook):
        print("runbook check: %s not found" % runbook, file=sys.stderr)
        return 1
    errs = check(runbook, root)
    for e in errs:
        print("RUNBOOK FAIL " + e, file=sys.stderr)
    print("runbook check: %d problems" % len(errs))
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
