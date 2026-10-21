#!/bin/bash
# Spark: create the small tools venv that hf_fetch.sh, mtp_build.sh and setup_env.sh use.
#   tools/spark/setup_tools_env.sh
# Location: $SLOTSTREAM_TOOLS_VENV, default $R/venv-tools. It holds the Hugging Face CLI and, when a wheel exists
# for this platform, mlx (only needed to convert the MTP weights; without it mtp_build.sh uses the pinned mirror).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$HERE/tools/spark/env.sh"
VENV=${SLOTSTREAM_TOOLS_VENV:-$R/venv-tools}
log() { echo "$(date -Is) setup_tools_env: $*"; }

if [ ! -x "$VENV/bin/python" ]; then
  mkdir -p "$(dirname "$VENV")"
  python3 -m venv "$VENV" || { log "python3 -m venv failed"; exit 1; }
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip || log "pip upgrade failed, continuing"
"$VENV/bin/python" -m pip install --quiet "huggingface_hub[cli]" || { log "huggingface_hub install failed"; exit 1; }
if "$VENV/bin/python" -m pip install --quiet mlx; then
  log "mlx: installed"
else
  log "mlx: unavailable"
fi
test -x "$VENV/bin/hf" || { log "hf not found in $VENV/bin"; exit 1; }
log "done: $VENV"
