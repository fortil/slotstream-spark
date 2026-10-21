#!/bin/bash
# Spark: build the project Python environment. Idempotent.
# Kinds: venv (host python has Python.h), standalone (python-build-standalone), container (vllm image + --target pylib).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$HERE/tools/spark/env.sh"
export PIP_CACHE_DIR=$R/tmp/pip-cache XDG_CACHE_HOME=$R/tmp/xdg-cache CUDA_CACHE_PATH=$R/tmp/nv-cache TMPDIR=$R/tmp
mkdir -p "$R/tmp" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
IMG=vllm/vllm-openai:v0.28.0
PYPKGS="triton==3.7.1 transformers==5.19.0 flash-linear-attention==0.5.2 safetensors tokenizers jinja2 numpy nanobind pytest hypothesis nvidia-ml-py starlette uvicorn h11 httpx"
CPKGS="transformers==5.19.0 flash-linear-attention==0.5.2 nanobind hypothesis nvidia-ml-py starlette uvicorn h11 httpx safetensors tokenizers jinja2 pytest"

log() { echo "$(date -Is) setup_env: $*"; }

verify_venv() { # python
  "$1" - <<'PY'
import sys
import torch, triton, transformers
assert torch.__version__.startswith("2.13.0"), torch.__version__
assert str(torch.version.cuda).startswith("13"), torch.version.cuda
assert triton.__version__ == "3.7.1", triton.__version__
assert transformers.__version__ == "5.19.0", transformers.__version__
print("torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__, "transformers", transformers.__version__)
PY
}

if [ -f "$R/ENV" ]; then
  . "$R/ENV"
  if [ "$ENV_KIND" != container ] && [ -x "${PYTHON:-/nonexistent}" ] && verify_venv "$PYTHON" >/dev/null 2>&1; then
    log "already set up ($ENV_KIND)"; exit 0
  fi
  if [ "$ENV_KIND" = container ] && [ -d "${PYLIB:-/nonexistent}" ]; then
    out=$(docker run --rm --network none --oom-score-adj 1000 --user "$(id -u):$(id -g)" -e PYTHONPATH=/w/pylib -v "$R:/w" \
      --entrypoint python3 "$IMG" -c 'import transformers;print(transformers.__version__)' 2>/dev/null)
    [ "$out" = 5.19.0 ] && { log "already set up (container)"; exit 0; }
  fi
  log "ENV exists but fails its checks; rebuilding"
fi

KIND=""; PYTHON=""; PYTHON_H=False
HAS_H=$(python3 -c "import sysconfig,os;print(os.path.exists(os.path.join(sysconfig.get_paths()['include'],'Python.h')))")

install_venv() { # pythonbin
  "$1" -m venv "$R/venv" || return 1
  "$R/venv/bin/pip" install --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0 || return 1
  "$R/venv/bin/pip" install $PYPKGS || return 1
  verify_venv "$R/venv/bin/python"
}

if [ "$HAS_H" = True ]; then
  KIND=venv; PYTHON_H=True
  log "kind=venv"
  install_venv python3 || KIND=""
else
  log "Python.h absent; trying python-build-standalone"
  api=$(curl -fsSL https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest) || api=""
  if [ -n "$api" ]; then
    url=$(printf '%s' "$api" | python3 -c '
import json,re,sys
r=json.load(sys.stdin)
pat=re.compile(r"^cpython-3\.12\.([0-9]+)\+[0-9]+-aarch64-unknown-linux-gnu-install_only\.tar\.gz$")
best=None
for a in r.get("assets",[]):
    m=pat.match(a["name"])
    if m and (best is None or int(m.group(1))>best[0]): best=(int(m.group(1)),a["name"],a["browser_download_url"])
sums=[a["browser_download_url"] for a in r.get("assets",[]) if a["name"]=="SHA256SUMS"]
if best and sums: print(best[1],best[2],sums[0])')
    if [ -n "$url" ]; then
      set -- $url
      name=$1 asset=$2 sums=$3
      curl -fsSL -o "$R/tmp/$name" "$asset" && curl -fsSL -o "$R/tmp/SHA256SUMS.pbs" "$sums" \
        && (cd "$R/tmp" && sed "s#  *\*\?#  #" SHA256SUMS.pbs > SHA256SUMS.pbs.n && sha256sum -c --ignore-missing SHA256SUMS.pbs.n) \
        && mkdir -p "$R/python.tmp" && tar -xzf "$R/tmp/$name" -C "$R/python.tmp" \
        && rm -rf "$R/python" && mv "$R/python.tmp/python" "$R/python" && rm -rf "$R/python.tmp" \
        && rm -rf "$R/venv" \
        && "$R/python/bin/python3" -c "import sysconfig,os;assert os.path.exists(os.path.join(sysconfig.get_paths()['include'],'Python.h'))" \
        && KIND=standalone PYTHON_H=True && log "kind=standalone" && { install_venv "$R/python/bin/python3" || KIND=""; }
    fi
  fi
fi

if [ -n "$KIND" ]; then
  PYTHON=$R/venv/bin/python
  {
    echo "ENV_KIND=$KIND"; echo "PYTHON=$PYTHON"; echo "IMAGE="; echo "PYLIB="; echo "PYTHON_H=$PYTHON_H"
    "$PYTHON" -c 'import torch,triton,transformers;print("TORCH="+torch.__version__);print("TRITON="+triton.__version__);print("TRANSFORMERS="+transformers.__version__)'
  } > "$R/ENV"
  "$R/venv/bin/pip" freeze > "$R/requirements.lock"
  log "done kind=$KIND"; exit 0
fi

# ---- container fallback
log "falling back to kind=container"
rm -rf "$R/venv"
PYV=$(docker run --rm --network none --entrypoint python3 "$IMG" -c 'import sys;print("%d.%d"%sys.version_info[:2])') || { log "cannot read the template python"; exit 1; }
mkdir -p "$R/pylib"
MLXPY=${SLOTSTREAM_TOOLS_VENV:-$R/venv-tools}/bin/python      # tools/spark/setup_tools_env.sh makes the venv
pipt() { "$MLXPY" -m pip install --target "$R/pylib" --no-deps --only-binary=:all: --python-version "$PYV" --platform manylinux2014_aarch64 "$@"; }
pipt $CPKGS || { log "pip install --target failed"; exit 1; }
for i in $(seq 1 10); do
  chk=$(docker run --rm --network none --oom-score-adj 1000 --user "$(id -u):$(id -g)" -e PYTHONPATH=/w/pylib -v "$R:/w" \
        --entrypoint python3 "$IMG" -m pip check 2>&1)
  miss=$(printf '%s\n' "$chk" | sed -n 's/.* requires \([A-Za-z0-9_.-]*\),.*which is not installed.*/\1/p' | sort -u)
  [ -z "$miss" ] && break
  log "pip check loop $i: adding $miss"
  pipt $miss || break
done
ver=$(docker run --rm --network none --oom-score-adj 1000 --user "$(id -u):$(id -g)" -e PYTHONPATH=/w/pylib -v "$R:/w" \
      --entrypoint python3 "$IMG" -c 'import transformers;print(transformers.__version__)' 2>/dev/null)
[ "$ver" = 5.19.0 ] || { log "transformers $ver != 5.19.0"; exit 1; }
tv=$(docker run --rm --network none --oom-score-adj 1000 --user "$(id -u):$(id -g)" -e PYTHONPATH=/w/pylib -v "$R:/w" \
     --entrypoint python3 "$IMG" -c 'import torch,triton;print(torch.__version__,triton.__version__)' 2>/dev/null)
{
  echo "ENV_KIND=container"; echo "PYTHON=python3"; echo "IMAGE=$IMG"; echo "PYLIB=$R/pylib"; echo "PYTHON_H=False"
  echo "TORCH=${tv%% *}"; echo "TRITON=${tv##* }"; echo "TRANSFORMERS=5.19.0"
} > "$R/ENV"
docker run --rm --network none --oom-score-adj 1000 --user "$(id -u):$(id -g)" -e PYTHONPATH=/w/pylib -v "$R:/w" \
  --entrypoint python3 "$IMG" -m pip freeze > "$R/requirements.lock" 2>/dev/null
log "done kind=container"
exit 0
