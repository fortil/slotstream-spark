#!/usr/bin/env bash
# Compares the vendored vLLM files with upstream at the pinned sha. Exit 0 only when the
# difference equals expected.diff byte for byte. `diff.sh --update` rewrites expected.diff.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
mode=${1:-check}
sha=$(awk '$1=="sha"{print $2}' "$here/UPSTREAM.txt")
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
sum() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi; }
: > "$tmp/actual.diff"
while read -r tag path name want; do
  [ "$tag" = file ] || continue
  curl -sfL "https://raw.githubusercontent.com/vllm-project/vllm/$sha/$path" -o "$tmp/$name" \
    || { echo "diff.sh: cannot fetch $path at $sha" >&2; exit 2; }
  got=$(sum "$tmp/$name")
  [ "$got" = "$want" ] || { echo "diff.sh: upstream $name sha256 $got != $want" >&2; exit 3; }
  diff -u --label "upstream/$name" --label "vendored/$name" "$tmp/$name" "$here/qwen4_exp_ops/$name" >> "$tmp/actual.diff" || true
done < "$here/UPSTREAM.txt"
if [ "$mode" = "--update" ]; then
  cp "$tmp/actual.diff" "$here/expected.diff"
  echo "expected.diff updated ($(wc -l < "$here/expected.diff" | tr -d ' ') lines)"
  exit 0
fi
if cmp -s "$tmp/actual.diff" "$here/expected.diff"; then
  echo "vllm-187a0eb9: vendored files match expected.diff"
else
  diff -u "$here/expected.diff" "$tmp/actual.diff" | head -40 >&2 || true
  echo "diff.sh: vendored files differ from expected.diff" >&2
  exit 1
fi
