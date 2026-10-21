# Sourced on both machines (bash 3.2-safe). Defines the Spark client environment.
# Host-specific values (login, model router URL, container count, NVMe device) live in the
# untracked private/env.local.sh; see tools/spark/env.local.sh.example for every knob.
# SLOTSTREAM_SKIP_LOCAL=1 ignores the local files.

_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

on_spark() { [ "$(uname -s)" = "Linux" ]; }

if [ "${SLOTSTREAM_SKIP_LOCAL:-}" != 1 ]; then
  if [ -f "$_ENV_DIR/../../private/env.local.sh" ]; then
    . "$_ENV_DIR/../../private/env.local.sh"
  elif [ -f "$_ENV_DIR/env.local.sh" ]; then
    echo "env.sh: tools/spark/env.local.sh is deprecated, move it to private/env.local.sh" >&2
    . "$_ENV_DIR/env.local.sh"
  fi
fi

SPARK_HOST=${SPARK_HOST:-spark}
[ -n "${SLOTSTREAM_RUNNING_URL+x}" ] && export SLOTSTREAM_RUNNING_URL
[ -n "${EXPECTED_CONTAINERS:-}" ] && export EXPECTED_CONTAINERS
[ -n "${SLOTSTREAM_NVME_DEVICE:-}" ] && export SLOTSTREAM_NVME_DEVICE
[ -n "${SLOTSTREAM_HWMON_INPUT:-}" ] && export SLOTSTREAM_HWMON_INPUT
[ -n "${SLOTSTREAM_TOOLS_VENV:-}" ] && export SLOTSTREAM_TOOLS_VENV
SPARK_SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=15"

if on_spark; then
  R=${SLOTSTREAM_R_REMOTE:-$HOME/slotstream-spark}
else
  # Kept unexpanded: it must start a remote word, unquoted, so the remote shell expands it.
  R='~/slotstream-spark'
  [ -n "${SLOTSTREAM_R_REMOTE:-}" ] && R=$SLOTSTREAM_R_REMOTE
fi
SRC=$R/src
MODEL=$R/model/qwen38-flash-next-mlx-4bit
PACKED=$R/model/packed
LB=$R/lb

spark_ssh() { ssh $SPARK_SSH_OPTS "$SPARK_HOST" "$@"; }

# Mac: short commit id, plus a hash of the diff and untracked list when dirty.
sync_id() {
  local id st h
  id=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
  st=$(git status --porcelain 2>/dev/null)
  if [ -n "$st" ]; then
    h=$( { git diff HEAD 2>/dev/null; git ls-files --others --exclude-standard 2>/dev/null; } | shasum -a 256 | cut -c1-8)
    id="$id-dirty-$h"
  fi
  echo "$id"
}

# Pause mode (bash twin of slotstream_spark/guard/paused.py). Reads $R/state/PAUSED, or FILE when given, or
# $SLOTSTREAM_PAUSED_FILE. Sets PAUSED_ACTIVE (0|1), PAUSED_NO_WINDOWS (0|1) and PAUSED_EXPECTED (integer or empty).
# Absent file: all 0/empty, rc 0. Malformed file (empty, no "=", unknown key, bad value): message on stderr, rc 2.
# Only meaningful on the Spark (on the Mac $R is an unexpanded remote path).
paused_load() {
  local f=${1:-${SLOTSTREAM_PAUSED_FILE:-$R/state/PAUSED}} line k v n=0
  PAUSED_ACTIVE=0 PAUSED_NO_WINDOWS=0 PAUSED_EXPECTED=""
  [ -e "$f" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line=${line#"${line%%[![:space:]]*}"}; line=${line%"${line##*[![:space:]]}"}
    [ -z "$line" ] && continue
    case "$line" in "#"*) continue ;; esac
    k=${line%%=*}; v=${line#*=}
    if [ "$k" = "$line" ]; then echo "paused_load: $f: malformed line '$line'" >&2; return 2; fi
    case "$k" in
      NO_WINDOWS)
        case "$v" in 0|1) PAUSED_NO_WINDOWS=$v ;; *) echo "paused_load: $f: NO_WINDOWS must be 0 or 1" >&2; return 2 ;; esac ;;
      EXPECTED_CONTAINERS)
        case "$v" in ''|*[!0-9]*) echo "paused_load: $f: EXPECTED_CONTAINERS must be an integer" >&2; return 2 ;; esac
        PAUSED_EXPECTED=$v ;;
      *) echo "paused_load: $f: unknown key '$k'" >&2; return 2 ;;
    esac
    n=$((n + 1))
  done < "$f"
  if [ "$n" = 0 ]; then echo "paused_load: $f: no key=value line" >&2; return 2; fi
  PAUSED_ACTIVE=1
}
