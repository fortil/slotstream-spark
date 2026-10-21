#!/usr/bin/env bash
# Compares the vendored fla_v052 files with the fla-core 0.5.2 sdist from PyPI. Exit 0 only when the
# difference equals expected.diff byte for byte. `diff.sh --update` rewrites expected.diff.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
mode=${1:-check}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
sum() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
url=$(awk '$1=="sdist"{print $2}' "$here/UPSTREAM.txt")
want=$(awk '$1=="sdist_sha256"{print $2}' "$here/UPSTREAM.txt")
curl -sfL "$url" -o "$tmp/sdist.tgz" || { echo "diff.sh: cannot fetch the sdist" >&2; exit 2; }
got=$(sum "$tmp/sdist.tgz")
[ "$got" = "$want" ] || { echo "diff.sh: sdist sha256 $got != $want" >&2; exit 3; }
tar xzf "$tmp/sdist.tgz" -C "$tmp"
root=$(ls -d "$tmp"/fla_core-*)
: > "$tmp/actual.diff"
while read -r tag path sha; do
  [ "$tag" = file ] || continue
  got=$(sum "$root/$path")
  [ "$got" = "$sha" ] || { echo "diff.sh: upstream $path sha256 $got != $sha" >&2; exit 3; }
  rel=${path#fla/}
  diff -u --label "upstream/$path" --label "vendored/fla_v052/$rel" "$root/$path" "$here/fla_v052/$rel" >> "$tmp/actual.diff" || true
done < "$here/UPSTREAM.txt"
if [ "$mode" = "--update" ]; then
  cp "$tmp/actual.diff" "$here/expected.diff"
  echo "expected.diff updated ($(wc -l < "$here/expected.diff" | tr -d ' ') lines)"
  exit 0
fi
if cmp -s "$tmp/actual.diff" "$here/expected.diff"; then
  echo "fla-0.5.2: vendored files match expected.diff"
else
  diff -u "$here/expected.diff" "$tmp/actual.diff" | head -40 >&2 || true
  echo "diff.sh: vendored files differ from expected.diff" >&2
  exit 1
fi
