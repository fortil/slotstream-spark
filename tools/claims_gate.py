#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Port of Tools/claims_gate.py from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""Every public number has a claim record; every claim's needle is on its surfaces and in its records.

Reads db/records/claims/claim-*.md. Port of third_party/slotstream/Tools/claims_gate.py with one extension:

- a `current` claim's `needle` must appear verbatim in every file of `surfaces`;
- it must also appear verbatim in every record of `supported_by`, so a number can only be published as the
  measurement wrote it (a rounded or edited figure is not found there);
- a `withdrawn` claim's needle must appear in none of its surfaces;
- every number in the README sections "Speed", "Quality", "Real workload" and "Memory and coexistence" must appear in the needle or the body of some current claim.

Matching ignores line wrapping: runs of whitespace collapse to one space. Standard library only.

  claims_gate.py [--root DIR]
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
JUDGED_HEADINGS = ("## Speed", "## Quality", "## Real workload", "## Memory and coexistence")


def squash(text):
    return " ".join(text.split())


def read_record(path):
    """(frontmatter scalars, body). `key: value` lines only."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if not text.startswith("---\n"):
        raise ValueError("no frontmatter")
    end = text.find("\n---\n", 4)
    if end < 0:
        raise ValueError("unterminated frontmatter")
    fm = {}
    for line in text[4:end].split("\n"):
        m = re.match(r"^([A-Za-z0-9_-]+):(.*)$", line)
        if m and m.group(2).strip():
            fm[m.group(1)] = m.group(2).strip()
    return fm, text[end + len("\n---\n"):]


def split_list(value):
    return [s.strip() for s in value.split(",") if s.strip()]


def judged_numbers(readme_text):
    """Numbers found under each judged heading, as {heading: [numbers]}."""
    found = {}
    for heading in JUDGED_HEADINGS:
        m = re.search(r"^%s[ \t]*$" % re.escape(heading), readme_text, re.M)
        if not m:
            continue
        rest = readme_text[m.end():]
        nxt = re.search(r"^## ", rest, re.M)
        section = rest[:nxt.start()] if nxt else rest
        section = re.sub(r"\]\([^)]*\)", "]", section)      # link targets are paths, not numbers
        section = re.sub(r"`[^`]*`", " ", section)           # code spans: paths, flags, commands
        found[heading] = sorted({n.rstrip(",") for n in NUMBER.findall(section)})
    return found


def run(root):
    claims_dir = os.path.join(root, "db", "records", "claims")
    if not os.path.isdir(claims_dir):
        print("no claims folder at %s" % claims_dir, file=sys.stderr)
        return 1
    failures, checked, texts = [], 0, []
    for name in sorted(os.listdir(claims_dir)):
        if not name.startswith("claim-") or not name.endswith(".md"):
            continue
        try:
            fm, body = read_record(os.path.join(claims_dir, name))
        except ValueError as e:
            failures.append("%s: %s" % (name, e))
            continue
        needle = fm.get("needle", "")
        status = fm.get("status", "")
        surfaces = split_list(fm.get("surfaces", ""))
        supported = split_list(fm.get("supported_by", ""))
        if fm.get("type") != "claim" or not fm.get("id") or not fm.get("summary") or fm.get("basis") != "measured":
            failures.append("%s: needs type claim, id, summary and basis measured" % name)
        if not needle or not surfaces or status not in ("current", "withdrawn"):
            failures.append("%s: needs needle, surfaces, and a current/withdrawn status" % name)
            continue
        if status == "current":
            if not supported:
                failures.append("%s: a current claim needs supported_by" % name)
            texts.append(needle + " " + body)
        for surface in surfaces:
            path = os.path.join(root, surface)
            if not os.path.isfile(path):
                failures.append("%s: surface %s does not exist" % (name, surface))
                continue
            with open(path, encoding="utf-8") as fh:
                present = squash(needle) in squash(fh.read())
            checked += 1
            if status == "current" and not present:
                failures.append("%s: current claim needle %r missing from %s" % (name, needle, surface))
            if status == "withdrawn" and present:
                failures.append("%s: withdrawn claim needle %r still on %s" % (name, needle, surface))
        if status == "current":
            for rec in supported:
                path = os.path.join(root, rec)
                if not os.path.isfile(path):
                    failures.append("%s: supporting record %s does not exist" % (name, rec))
                    continue
                with open(path, encoding="utf-8") as fh:
                    present = squash(needle) in squash(fh.read())
                checked += 1
                if not present:
                    failures.append("%s: needle %r missing from supporting record %s" % (name, needle, rec))
    readme = os.path.join(root, "README.md")
    if os.path.isfile(readme):
        with open(readme, encoding="utf-8") as fh:
            judged = judged_numbers(fh.read())
        known = squash(" ".join(texts))
        for heading, nums in judged.items():
            for n in nums:
                checked += 1
                if n not in known:
                    failures.append("README %s: number %s has no claim" % (heading, n))
    for f in failures:
        print("CLAIM FAIL " + f, file=sys.stderr)
    print("claims gate: %d needle checks, %d failures" % (checked, len(failures)))
    return 1 if failures else 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    root = ROOT
    if argv[:1] == ["--root"] and len(argv) >= 2:
        root = argv[1]
    return run(root)


if __name__ == "__main__":
    sys.exit(main())
