#!/bin/bash
# Spark: produce $MODEL/mtp.safetensors (resumable).
# 1. convert from the official checkpoint with MLX when the 4b preflight passed;
# 2. accept it if its sha256 equals model.lock.json mtp.sha256;
# 3. otherwise use the pinned mirror, compare, and keep the converted provenance aside.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$HERE/tools/spark/env.sh"
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
MLXPY=${SLOTSTREAM_TOOLS_VENV:-$R/venv-tools}/bin/python      # tools/spark/setup_tools_env.sh makes the venv
LOCK=$HERE/model.lock.json
WANT=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["mtp"]["sha256"])' "$LOCK")
MIRROR_REPO=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["mirror"]["repo"])' "$LOCK")
MIRROR_REV=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["mirror"]["revision"])' "$LOCK")

# (1) the converter writes under ~/.slotstream/...: point HOME at our own tree
export HOME=$R/tmp/mtp-home HF_HOME=$R/hf TMPDIR=$R/tmp MLX_PTX_CACHE_DIR=$R/tmp/mlx-ptx-cache
mkdir -p "$HOME" "$TMPDIR" "$MLX_PTX_CACHE_DIR" "$MODEL"
CONV_DIR=$HOME/.slotstream/models/qwen38-flash-next-mlx-4bit
CONV=$CONV_DIR/mtp.safetensors
log() { echo "$(date -Is) mtp_build: $*"; }

sha() { sha256sum "$1" | cut -d' ' -f1; }

if [ -f "$MODEL/mtp.safetensors" ] && [ "$(sha "$MODEL/mtp.safetensors")" = "$WANT" ]; then
  log "already installed"
  sha256sum "$MODEL/mtp.safetensors" > "$R/model/verify-mtp.txt"; exit 0
fi

PASS4B=0
grep -qs '^Verdict: PASS' "$SRC/db/records/measurements/4b-mlx.md" && PASS4B=1

# (2)
if [ $PASS4B = 1 ] && [ ! -f "$CONV" ]; then
  if "$MLXPY" -c 'import mlx' >/dev/null 2>&1; then
    log "converting with MLX"
    "$MLXPY" "$SRC/third_party/slotstream/Tools/mtp_convert.py" || log "conversion failed ($?)"
  else
    log "mlx is not importable in $MLXPY: skipping the conversion, using the pinned mirror"
  fi
fi

# (3)
if [ -f "$CONV" ] && [ "$(sha "$CONV")" = "$WANT" ]; then
  log "converted file matches the lock"
  cp "$CONV" "$MODEL/mtp.safetensors" && cp "$CONV_DIR/mtp.provenance.json" "$MODEL/mtp.provenance.json"
else
  # (4)
  log "using the mirror"
  MIR=$R/tmp/mtp-mirror
  bash "$HERE/tools/spark/hf_fetch.sh" "$MIRROR_REPO" "$MIRROR_REV" "$MIR" mtp.safetensors mtp.provenance.json || exit $?
  [ "$(sha "$MIR/mtp.safetensors")" = "$WANT" ] || { log "mirror sha256 differs from the lock"; exit 1; }
  if [ -f "$CONV" ]; then
    "$MLXPY" "$HERE/tools/mtp_compare.py" "$CONV" "$MIR/mtp.safetensors" > "$MODEL/mtp-compare.json"
    [ -f "$CONV_DIR/mtp.provenance.json" ] && cp "$CONV_DIR/mtp.provenance.json" "$MODEL/mtp.provenance.converted.json"
  fi
  cp "$MIR/mtp.safetensors" "$MODEL/mtp.safetensors" && cp "$MIR/mtp.provenance.json" "$MODEL/mtp.provenance.json"
fi

# (5)
sha256sum "$MODEL/mtp.safetensors" > "$R/model/verify-mtp.txt"
log "done: $(cut -c1-16 "$R/model/verify-mtp.txt")"
exit 0
