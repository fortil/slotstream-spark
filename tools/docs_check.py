#!/usr/bin/env python3
"""Check the public documents against the rules of the project.

  docs_check.py [--root DIR] [--terms-file FILE]

Over README.md, RUNBOOK.md, AGENTS.md, CONTRIBUTING.md, SECURITY.md, CHANGELOG.md, THIRD_PARTY_NOTICES.md,
docs/*.md, db/DB.md and .github/**/*.md (the ones that exist):
1. every relative link resolves to a file or folder, and a `#anchor` names a heading of the target (GitHub slugs);
2. none of the characters em dash, middle dot, arrow, check mark or section sign appears in prose (fenced code blocks
   are exempt: db/DB.md shows the record format, whose Verdict line uses an em dash);
3. no local absolute path under a person's home directory and no ssh login appears;
4. with --terms-file, no line of the file (one regex per line, case-insensitive, # comments) matches any line.
Prints one line per problem and `docs check: N problems`; exit 1 when N is not 0. Standard library only.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["README.md", "RUNBOOK.md", "AGENTS.md", "CONTRIBUTING.md", "SECURITY.md", "CHANGELOG.md",
         "THIRD_PARTY_NOTICES.md", "db/DB.md"]
FORBIDDEN = {"—": "em dash", "·": "middle dot", "→": "arrow", "✓": "check mark",
             "§": "section sign"}
LOCAL = [(re.compile(r"/Users/[^/\s<>$]+/"), "local path"), (re.compile(r"/home/[^/\s<>$]+/"), "local path"),
         (re.compile(r"\bssh\s+[\w.-]+@[\w.-]+"), "ssh login")]
LINK = re.compile(r"(?<!\!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"^(```|~~~).*?^\1[ \t]*$", re.M | re.S)


def slug(heading):
    text = re.sub(r"[`*_]", "", heading).strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(path, cache={}):
    if path not in cache:
        text = FENCE.sub("", path.read_text(encoding="utf-8"))
        seen, out = {}, set()
        for m in re.finditer(r"^#{1,6}\s+(.*?)\s*#*\s*$", text, re.M):
            s = slug(m.group(1))
            n = seen.get(s, 0)
            seen[s] = n + 1
            out.add(s if n == 0 else "%s-%d" % (s, n))
        cache[path] = out
    return cache[path]


def documents(root):
    found = [root / f for f in FILES if (root / f).is_file()]
    found += sorted((root / "docs").glob("*.md")) if (root / "docs").is_dir() else []
    found += sorted((root / ".github").rglob("*.md")) if (root / ".github").is_dir() else []
    return found


def check(root, terms=()):
    problems = []
    for doc in documents(root):
        rel = doc.relative_to(root)
        text = doc.read_text(encoding="utf-8")
        fenced = False
        for no, line in enumerate(text.splitlines(), 1):
            if re.match(r"^(```|~~~)", line):
                fenced = not fenced
                continue
            for ch, name in FORBIDDEN.items():
                if ch in line and not fenced:
                    problems.append("%s:%d: %s" % (rel, no, name))
            for rx, name in LOCAL:
                if rx.search(line):
                    problems.append("%s:%d: %s" % (rel, no, name))
            for rx in terms:
                if rx.search(line):
                    problems.append("%s:%d: private term" % (rel, no))
        for target in LINK.findall(FENCE.sub("", text)):
            if re.match(r"^(?:[a-z][a-z0-9+.-]*:|/)", target, re.I):
                continue
            path, _, frag = target.partition("#")
            dest = (doc.parent / path).resolve() if path else doc
            if not dest.exists():
                problems.append("%s: broken link %s" % (rel, target))
            elif frag and dest.suffix == ".md" and dest.is_file() and frag.lower() not in anchors(dest):
                problems.append("%s: no heading for anchor %s" % (rel, target))
    return problems


def load_terms(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(re.compile(line, re.I))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--terms-file")
    a = ap.parse_args(argv)
    problems = check(Path(a.root).resolve(), load_terms(a.terms_file) if a.terms_file else ())
    for p in problems:
        print(p)
    print("docs check: %d problems" % len(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
