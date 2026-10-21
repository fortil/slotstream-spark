#!/bin/bash
# Spark: fetch the optional router-tap correction sidecar next to the weights (plan step 73).
#   tools/spark/run.sh -- bash tools/fetch_tap_correction.sh
# The file is optional, like slotstream `pull` (DOWNLOAD-FORMAT.md:24-31): exit 0 for both outcomes, and exactly one line
# in $R/model/tap-correction.status says which:
#   tap_correction: present 37540708 <sha256>
#   tap_correction: absent (<reason>)       reason: hf_fetch exit N | size X | sha mismatch <sha> | no $MODEL
# Exit 75 only when $MODEL does not exist or tools/spark/hf_fetch.sh is missing. Idempotent: a file with the right size and
# sha256 skips the download. On absent the partial file is deleted, so a file at $MODEL/lookahead/ always has the pinned sha.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$HERE/tools/spark/env.sh"

REPO=carloslfu/Qwen3.8-Flash-Next-MLX-4bit-Slotpack      # the Slotpack repo carries the file; the raw mirror does not
REV=8c1f9c34e4567e83d46cebe1af432e8eba4f3ea8
REL=lookahead/tap-correction-attention-rank128-v1.safetensors
SIZE=37540708
SHA=37b00d3a32d1e1889a1794bbb8e97905a157a77c0508db620c1a11f2a895f7f5
FILE=$MODEL/$REL
STATUS=$R/model/tap-correction.status
FETCH=$HERE/tools/spark/hf_fetch.sh

say() { [ -d "$R/model" ] && echo "$1" > "$STATUS"; echo "$1"; }
absent() {
  rm -f "$FILE"
  rmdir "$MODEL/lookahead" 2>/dev/null
  say "tap_correction: absent ($1)"
  exit 0
}

if [ ! -d "$MODEL" ]; then say 'tap_correction: absent (no $MODEL)'; exit 75; fi
if [ ! -f "$FETCH" ]; then echo "hf_fetch.sh is missing: $FETCH" >&2; exit 75; fi

check() {   # -> 0 if $FILE has the pinned size and sha256
  [ -f "$FILE" ] || return 1
  [ "$(stat -c %s "$FILE")" = "$SIZE" ] || return 1
  [ "$(sha256sum "$FILE" | cut -d' ' -f1)" = "$SHA" ]
}

if check; then say "tap_correction: present $SIZE $SHA"; exit 0; fi
rm -f "$FILE"                                           # a wrong file must not survive the download attempt
SLOTSTREAM_LABEL=tap-correction bash "$FETCH" "$REPO" "$REV" "$MODEL" "$REL"
rc=$?
[ "$rc" -eq 0 ] || absent "hf_fetch exit $rc"
[ -f "$FILE" ] || absent "hf_fetch exit 0 but no file"
got=$(stat -c %s "$FILE")
[ "$got" = "$SIZE" ] || absent "size $got"
sha=$(sha256sum "$FILE" | cut -d' ' -f1)
[ "$sha" = "$SHA" ] || absent "sha mismatch $sha"
say "tap_correction: present $SIZE $SHA"
exit 0
