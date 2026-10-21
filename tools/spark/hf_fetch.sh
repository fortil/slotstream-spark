#!/bin/bash
# Spark: download from Hugging Face into a private HF_HOME.
#   hf_fetch.sh REPO REV DEST [INCLUDE...]
# Exit 4 if anything under ~/.cache/huggingface became newer than the start marker.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$HERE/tools/spark/env.sh"

REPO=${1:?usage: hf_fetch.sh REPO REV DEST [INCLUDE...]}
REV=${2:?}
DEST=${3:?}
shift 3
LABEL=${SLOTSTREAM_LABEL:-hf-$$}
export HF_HOME=$R/hf HF_XET_CACHE=$R/hf/xet HF_HUB_DISABLE_TELEMETRY=1
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
mkdir -p "$HF_HOME" "$HF_XET_CACHE" "$DEST" "$R/run"
MARK=$R/run/hfmark-$LABEL
touch "$MARK"
# HOME may be redirected by the caller (mtp_build.sh): use the account home.
UHOME=$(getent passwd "$(id -u)" | cut -d: -f6)
HF=${SLOTSTREAM_HF_CLI:-${SLOTSTREAM_TOOLS_VENV:-$R/venv-tools}/bin/hf}      # tools/spark/setup_tools_env.sh makes the venv
CGCTL=/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/cgroup.controllers

if [ -x "$HF" ] && [ -r "$CGCTL" ] && tr ' ' '\n' < "$CGCTL" | grep -qx memory; then
  inc=()
  for i in "$@"; do inc+=(--include "$i"); done
  systemd-run --user --scope -p MemoryHigh=3G -p MemoryMax=4G \
    "$HF" download "$REPO" --revision "$REV" --local-dir "$DEST" "${inc[@]}" --max-workers 8
  rc=$?
else
  # The only container allowed network.
  inc=$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1:]))' "$@")
  docker run --rm --name "slotstream-spark-$LABEL" --memory 4g --oom-score-adj 1000 --user "$(id -u):$(id -g)" \
    -e HF_HOME=/w/hf -e HF_XET_CACHE=/w/hf/xet -e HOME=/w/tmp -v "$R:/w" -v "$DEST:/dest" \
    --entrypoint python3 vllm/vllm-openai:v0.28.0 -c "
import json
from huggingface_hub import snapshot_download
inc = json.loads('''$inc''') or None
snapshot_download('$REPO', revision='$REV', local_dir='/dest', allow_patterns=inc, max_workers=8)"
  rc=$?
fi

CHK=$(dirname "$DEST")/hfcache-check-$LABEL.txt
if [ -d "$UHOME/.cache/huggingface" ]; then
  find "$UHOME/.cache/huggingface" -newer "$MARK" \( -path '*xet*' -o -path '*hub*' \) > "$CHK" 2>/dev/null
else
  : > "$CHK"
fi
if [ -s "$CHK" ]; then echo "hf_fetch: files written under ~/.cache/huggingface (see $CHK)" >&2; exit 4; fi
exit $rc
