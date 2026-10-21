#!/bin/bash
# Spark: verify the downloaded and built artifacts. Writes $CAP/artifacts.json (or $R/tmp/artifacts.json).
# Exit non-zero on any failure.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$HERE/tools/spark/env.sh"
export LC_ALL=C

OUT=${CAP:-$R/tmp}
mkdir -p "$OUT"
fail=0
bad() { echo "FAIL: $*"; fail=1; }

LLAMA_COMMIT=23b0202a189c44a54625aadcb37a946dd1d6278d
MTP_SHA=c80b58faae46eeacb94dea49dd3453566ee05597fbd28c7c647eccb2862ab744

# checkpoint
V=$R/model/verify-ckpt.txt
ckpt_ok=$(grep -c ': OK$' "$V" 2>/dev/null); ckpt_ok=${ckpt_ok:-0}
ckpt_failed=$(grep -c 'FAILED' "$V" 2>/dev/null); ckpt_failed=${ckpt_failed:-0}
[ "$ckpt_ok" = 24 ] || bad "checkpoint: $ckpt_ok OK lines (want 24)"
[ "$ckpt_failed" = 0 ] || bad "checkpoint: $ckpt_failed FAILED lines"
ckpt_bytes=$(cd "$MODEL" && awk '{print $2}' "$R/model/SHA256SUMS.model" | sed 's/^\*//' | xargs stat -c %s | paste -sd+ | bc)

# hf cache checks must be empty
hf_nonempty=0
for f in "$R"/model/hfcache-check-*.txt "$LB"/hfcache-check-*.txt; do
  [ -e "$f" ] || continue
  if [ -s "$f" ]; then bad "$f is not empty"; hf_nonempty=$((hf_nonempty + 1)); fi
done
hf_files=$(ls "$R"/model/hfcache-check-*.txt "$LB"/hfcache-check-*.txt 2>/dev/null | wc -l)
[ "$hf_files" -gt 0 ] || bad "no hfcache-check files found"

# llama.cpp
commit=$(git -C "$LB/llama.cpp" rev-parse HEAD 2>/dev/null)
[ "$commit" = "$LLAMA_COMMIT" ] || bad "llama.cpp HEAD $commit"
ver=$("$LB/llama.cpp/build/bin/llama-server" --version 2>&1 | grep -m1 '^version:')
case "$ver" in *23b0202a*) ;; *) bad "llama-server --version: $ver" ;; esac

# GGUF sets
q41_ok=$(grep -c ': OK$' "$LB/gguf/verify-q41.txt" 2>/dev/null); q41_ok=${q41_ok:-0}
iq3_ok=$(grep -c ': OK$' "$LB/gguf/verify-iq3xxs.txt" 2>/dev/null); iq3_ok=${iq3_ok:-0}
[ "$q41_ok" = 3 ] || bad "q41: $q41_ok OK (want 3)"
[ "$iq3_ok" = 2 ] || bad "iq3xxs: $iq3_ok OK (want 2)"
grep -q FAILED "$LB/gguf/verify-q41.txt" "$LB/gguf/verify-iq3xxs.txt" 2>/dev/null && bad "a GGUF verification has FAILED lines"

# MTP
mtp_sha=$(sha256sum "$MODEL/mtp.safetensors" 2>/dev/null | cut -d' ' -f1)
[ "$mtp_sha" = "$MTP_SHA" ] || bad "mtp.safetensors sha256 $mtp_sha"
mirror=no; names_equal=null; dtypes_equal=null
if [ -f "$MODEL/mtp-compare.json" ]; then
  mirror=yes
  read -r names_equal dtypes_equal < <(python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));print(str(d.get("names_equal")).lower(),str(d.get("dtypes_shapes_equal")).lower())' "$MODEL/mtp-compare.json")
  [ "$names_equal" = true ] && [ "$dtypes_equal" = true ] || bad "mtp-compare: names_equal=$names_equal dtypes_shapes_equal=$dtypes_equal"
  echo "mtp: mirror copy installed; converted differs"
fi

rm -f "$OUT/artifacts.json"
python3 - "$OUT/artifacts.json" <<PY
import json, sys
d = {
  "checkpoint": {"ok": $ckpt_ok, "failed": $ckpt_failed, "bytes": ${ckpt_bytes:-0}},
  "hfcache_nonempty": $hf_nonempty, "hfcache_files": $hf_files,
  "llama": {"commit": "$commit", "version": "$ver"},
  "gguf": {"q41_ok": $q41_ok, "iq3xxs_ok": $iq3_ok},
  "mtp": {"sha256": "$mtp_sha", "mirror_installed": "$mirror", "names_equal": "$names_equal", "dtypes_shapes_equal": "$dtypes_equal"},
  "pass": $([ $fail = 0 ] && echo True || echo False),
}
json.dump(d, open(sys.argv[1], "w"), indent=1)
PY
[ -s "$OUT/artifacts.json" ] || bad "artifacts.json not written"
[ $fail = 0 ] && echo "artifacts: all checks passed" || echo "artifacts: FAILED"
exit $fail
