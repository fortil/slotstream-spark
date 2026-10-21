#!/bin/bash
# Spark: chained repack job (experts, MTP experts, trunks, n-gram rows) and the R0/E1 verification.
# Runs under tools/capture.sh (CAP set); resumable: each step is skipped when its output and manifest exist
# and the manifest's model identity matches the current checkpoint index.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$HERE/tools/spark/env.sh"
: "${CAP:?repack_all.sh must run under capture.sh}"
cd "$HERE"
PY=tools/spark/py.sh
mkdir -p "$PACKED"

IDX=$(sha256sum "$MODEL/model.safetensors.index.json" | cut -d' ' -f1)
MTP_HDR=$($PY - "$MODEL/mtp.safetensors" <<'PYE'
import hashlib, struct, sys
with open(sys.argv[1], "rb") as f:
    (n,) = struct.unpack("<Q", f.read(8))
    f.seek(0)
    print(hashlib.sha256(f.read(8 + n)).hexdigest())
PYE
)

# manifest_ok FILE KEY SHA: true when FILE exists and JSON key KEY equals SHA
manifest_ok() {
  [ -f "$1" ] || return 1
  $PY - "$1" "$2" "$3" <<'PYE'
import json, sys
sys.exit(0 if str(json.load(open(sys.argv[1])).get(sys.argv[2])) == sys.argv[3] else 1)
PYE
}
# trunk_ok FILE SHA: true when the safetensors metadata names the source index sha
trunk_ok() {
  [ -f "$1" ] || return 1
  $PY - "$1" "$2" <<'PYE'
import json, struct, sys
with open(sys.argv[1], "rb") as f:
    (n,) = struct.unpack("<Q", f.read(8))
    m = json.loads(f.read(n)).get("__metadata__", {})
sys.exit(0 if m.get("source_index_sha256") == sys.argv[2] else 1)
PYE
}

# (1) routed experts
if manifest_ok "$PACKED/experts.manifest.json" model_identity "$IDX" && [ -f "$PACKED/experts.bin" ]; then
  echo "skip experts"
else
  $PY tools/repack_experts.py --model "$MODEL" --out "$PACKED"
fi
# (2) MTP experts
if manifest_ok "$PACKED/mtp_experts.manifest.json" model_identity "$MTP_HDR" && [ -f "$PACKED/mtp_experts.bin" ]; then
  echo "skip mtp experts"
else
  $PY tools/repack_experts.py --model "$MODEL" --out "$PACKED" --mtp
fi
# (3) trunk
if trunk_ok "$PACKED/trunk.safetensors" "$IDX"; then
  echo "skip trunk"
else
  $PY tools/extract_trunk.py --model "$MODEL" --out "$PACKED/trunk.safetensors"
fi
# (4) MTP trunk
if trunk_ok "$PACKED/mtp_trunk.safetensors" "$MTP_HDR"; then
  echo "skip mtp trunk"
else
  $PY tools/extract_trunk.py --model "$MODEL" --mtp --out "$PACKED/mtp_trunk.safetensors"
fi
# (5) n-gram rows, row format decided by the platform measurements
ROWB=$(python3 -c 'import tomllib;print(tomllib.load(open("slotstream_spark/config/platform.toml","rb"))["ngram"]["row_bytes"])')
if manifest_ok "$PACKED/ngram.manifest.json" row_bytes "$ROWB" 2>/dev/null && [ -f "$PACKED/ngram.bin" ] \
   && $PY - "$PACKED/ngram.manifest.json" "$IDX" <<'PYE'
import json, sys
sys.exit(0 if json.load(open(sys.argv[1]))["source"]["index_sha256"] == sys.argv[2] else 1)
PYE
then
  echo "skip ngram"
else
  $PY tools/repack_ngram.py --model "$MODEL" --out "$PACKED" --row-bytes "$ROWB"
fi
# (6) R0 + E1
$PY tools/verify_r0.py --model "$MODEL" --packed "$PACKED" --e1 1000 --json "$CAP/r0.json"
