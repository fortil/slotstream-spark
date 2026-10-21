#!/bin/bash
# Spark: run the project Python with the right environment. py.sh [--online] ARGS...
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$HERE/tools/spark/env.sh"

OFFLINE=1
if [ "${1:-}" = "--online" ]; then OFFLINE=0; shift; fi

ENV_KIND=venv PYTHON=python3 IMAGE=vllm/vllm-openai:v0.28.0 PYLIB=$R/pylib
[ -f "$R/ENV" ] && . "$R/ENV"
# ENV written for venv/standalone has an empty IMAGE; the container kind (also forced by
# SLOTSTREAM_ENV_KIND) still needs one.
IMAGE=${IMAGE:-vllm/vllm-openai:v0.28.0}
ENV_KIND=${SLOTSTREAM_ENV_KIND:-$ENV_KIND}
S=${SLOTSTREAM_SRC:-$SRC}

export TRITON_CACHE_DIR=$R/tmp/triton HF_HOME=$R/hf TMPDIR=$R/tmp \
       PIP_CACHE_DIR=$R/tmp/pip-cache XDG_CACHE_HOME=$R/tmp/xdg-cache CUDA_CACHE_PATH=$R/tmp/nv-cache \
       SLOTSTREAM_MODEL=$MODEL SLOTSTREAM_PACKED=$PACKED SLOTSTREAM_R=$R
if [ $OFFLINE = 1 ]; then export HF_HUB_OFFLINE=1; fi
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"

if [ "$ENV_KIND" != container ]; then
  export PYTHONPATH=$S:$S/build/slotio
  cd "$S" && exec "$PYTHON" "$@"
fi

# Container kind: translate $R/... to /w/...
tr_path() { case "$1" in "$R"/*) echo "/w/${1#"$R"/}" ;; "$R") echo /w ;; *) echo "$1" ;; esac; }
args=()
for a in "$@"; do args+=("$(tr_path "$a")"); done
SW=/w/src
case "$S" in "$R"/*) SW=$(tr_path "$S") ;; esac
CAPW=${CAP:+$(tr_path "$CAP")}
offline_env=()
[ $OFFLINE = 1 ] && offline_env=(-e HF_HUB_OFFLINE=1)
exec docker run --rm --name "slotstream-spark-${SLOTSTREAM_LABEL:-py-$$}" --gpus all --network none --oom-score-adj 1000 \
  --user "$(id -u):$(id -g)" ${SLOTSTREAM_DOCKER_ARGS:-} \
  -e HOME=/w/tmp -e TMPDIR=/w/tmp -e TRITON_CACHE_DIR=/w/tmp/triton \
  -e PYTHONPATH="$SW:$SW/build/slotio:/w/pylib" -e HF_HOME=/w/hf "${offline_env[@]}" \
  -e SLOTSTREAM_MODEL=/w/model/qwen38-flash-next-mlx-4bit -e SLOTSTREAM_PACKED=/w/model/packed -e SLOTSTREAM_R=/w \
  ${CAPW:+-e CAP="$CAPW"} \
  -v "$R:/w" -w "$SW" --entrypoint python3 "$IMAGE" "${args[@]}"
