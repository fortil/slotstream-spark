#!/bin/bash
# Spark: download the wikitext-2-raw-v1 test split once into $R/state/wikitext/ and pin its sha256.
#   fetch_wikitext.sh            download when absent, verify the pin otherwise
#   fetch_wikitext.sh --check    verify only (exit 1 when absent or when the sha256 differs from the pin)
# The rows come from the Hugging Face datasets server (stdlib urllib, 100 rows per request, 4,358 rows), exactly as the
# dataset stores them (a text line keeps its trailing newline, a blank line is the empty string). They are written as a
# JSON list to test_rows.json; the sha256 of that file is the pin (test_rows.sha256). A later run that finds a different
# sha256 stops: the perplexity windows must be the same text every time.
set -eu
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$HERE/tools/spark/env.sh"
on_spark || { echo "fetch_wikitext.sh runs on the Spark (tools/spark/run.sh -- bash tools/quality/fetch_wikitext.sh)" >&2; exit 2; }
D=$R/state/wikitext
mkdir -p "$D"
ROWS=$D/test_rows.json
PIN=$D/test_rows.sha256
CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

sha() { sha256sum "$1" | cut -d' ' -f1; }

if [ -s "$ROWS" ] && [ -s "$PIN" ]; then
  if [ "$(sha "$ROWS")" = "$(cat "$PIN")" ]; then
    echo "wikitext test split: pinned $(cat "$PIN")"
    exit 0
  fi
  echo "wikitext test split: sha256 $(sha "$ROWS") differs from the pin $(cat "$PIN")" >&2
  exit 1
fi
[ "$CHECK" = 1 ] && { echo "wikitext test split: not downloaded" >&2; exit 1; }

python3 - "$ROWS.tmp" <<'PY'
import json, sys, urllib.request
URL = ("https://datasets-server.huggingface.co/rows?dataset=Salesforce/wikitext&config=wikitext-2-raw-v1&split=test"
       "&offset={o}&length=100")
rows, off, total = [], 0, None
while total is None or off < total:
    with urllib.request.urlopen(URL.format(o=off), timeout=60) as r:
        d = json.load(r)
    total = int(d["num_rows_total"])
    got = [x["row"]["text"] for x in d["rows"]]
    if not got:
        raise SystemExit(f"no rows at offset {off}")
    rows += got
    off += len(got)
if len(rows) != 4358 or len(rows) != total:
    raise SystemExit(f"expected 4358 test rows, got {len(rows)} (server says {total})")
json.dump(rows, open(sys.argv[1], "w"), ensure_ascii=False)
PY
mv "$ROWS.tmp" "$ROWS"
sha "$ROWS" > "$PIN"
echo "wikitext test split: downloaded, pinned $(cat "$PIN")"
