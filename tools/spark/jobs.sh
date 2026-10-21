#!/bin/bash
# Long resumable jobs under systemd --user.
#   jobs.sh start NAME FOOTPRINT DUR [--memcg S] [--container] [--nolock] [--exclusive] [--allow-truncate] -- CMD...
#   jobs.sh start NAME --heavy LABEL FOOTPRINT DUR [flags] -- CMD...   (alias; LABEL ignored)
#   jobs.sh status NAME | wait NAME [--timeout S] | log NAME [N] | restart NAME
# On the Mac every subcommand forwards itself to the Spark through run.sh.
# wait exits 0 only for "done 0", 1 for failed/cut, 124 on timeout.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$HERE/tools/spark/env.sh"

if ! on_spark; then
  cd "$HERE"
  if [ "${1:-}" = "wait" ]; then
    s=${SLOTSTREAM_WAIT_S:-1800}
    args=("$@")
    i=0
    while [ $i -lt ${#args[@]} ]; do
      if [ "${args[$i]}" = "--timeout" ]; then s=${args[$((i + 1))]}; fi
      i=$((i + 1))
    done
    case "$s" in ''|*[!0-9]*) s=1800 ;; esac
    has_to=0; for a in "$@"; do [ "$a" = "--timeout" ] && has_to=1; done
    if [ $has_to = 0 ]; then set -- "$@" --timeout "$s"; fi
    exec tools/spark/run.sh --timeout $((s + 100)) -- tools/spark/jobs.sh "$@"
  fi
  exec tools/spark/run.sh -- tools/spark/jobs.sh "$@"
fi

JD=$R/run/jobs
UNIT_PREFIX=slotstream-spark-job-
mkdir -p "$JD"
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}

iso() { date -Iseconds; }
status_line() { cat "$JD/$1.status" 2>/dev/null; }
set_status() { echo "$2 $3 $(iso)" > "$JD/$1.status"; }
unit_active() {
  local st
  st=$(systemctl --user is-active "$UNIT_PREFIX$1.service" 2>/dev/null)
  [ "$st" = active ] || [ "$st" = activating ]
}

# Rewrites a stale "running" line (unit gone) to "failed 70".
refresh_status() {
  local l
  l=$(status_line "$1")
  case "$l" in
    "running "*)
      if ! unit_active "$1"; then set_status "$1" failed 70; fi ;;
  esac
}

cmd_start() {
  local name=${1:-} footprint dur
  [ -n "$name" ] || { echo "usage: jobs.sh start NAME FOOTPRINT DUR [flags] -- CMD..." >&2; return 2; }
  shift
  if [ "${1:-}" = "--heavy" ]; then
    # alias: --heavy LABEL FOOTPRINT DUR
    footprint=$3 dur=$4
    shift 4
  else
    footprint=$1 dur=$2
    shift 2
  fi
  local flags=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --memcg) flags+=("$1" "$2"); shift 2 ;;
      --container|--nolock|--exclusive|--allow-truncate) flags+=("$1"); shift ;;
      --) shift; break ;;
      *) echo "jobs.sh start: unexpected argument: $1" >&2; return 2 ;;
    esac
  done
  [ $# -gt 0 ] || { echo "jobs.sh start: missing command after --" >&2; return 2; }

  refresh_status "$name"
  case "$(status_line "$name")" in
    "running "*|"queued "*) echo "job $name is already running" >&2; return 2 ;;
  esac

  {
    printf 'JOB_FOOTPRINT=%q\nJOB_DUR=%q\n' "$footprint" "$dur"
    printf 'JOB_FLAGS=('; [ ${#flags[@]} -gt 0 ] && printf '%q ' "${flags[@]}"; printf ')\n'
    printf 'JOB_ARGV=('; printf '%q ' "$@"; printf ')\n'
  } > "$JD/$name.cmd"

  if ! "$SRC/deploy/guard.sh" precheck "${flags[@]}" "$footprint" "$dur"; then
    set_status "$name" failed 75
    return 75
  fi

  : > "$JD/$name.log"
  set_status "$name" queued -
  systemd-run --user --unit="$UNIT_PREFIX$name" --collect \
    -p OOMScoreAdjust=1000 \
    --working-directory="$SRC" \
    --setenv=SLOTSTREAM_MODE="${SLOTSTREAM_MODE:-D5}" \
    --setenv=PATH="$PATH" --setenv=HOME="$HOME" \
    -- "$SRC/tools/spark/jobs.sh" _exec "$name" >> "$JD/$name.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    set_status "$name" failed 71
    echo "systemd-run failed ($rc); see $JD/$name.log" >&2
    return $rc
  fi
  echo "started $name"
}

cmd_exec() {
  local name=$1
  # shellcheck disable=SC1090
  . "$JD/$name.cmd"
  set_status "$name" running -
  "$SRC/deploy/guard.sh" run "${JOB_FLAGS[@]}" "$name" "$JOB_FOOTPRINT" "$JOB_DUR" -- "${JOB_ARGV[@]}" >> "$JD/$name.log" 2>&1
  local rc=$?
  case $rc in
    0) set_status "$name" done 0 ;;
    76) set_status "$name" cut 76 ;;
    *) set_status "$name" failed $rc ;;
  esac
  return $rc
}

cmd_status() {
  local name=${1:?usage: jobs.sh status NAME}
  refresh_status "$name"
  local l; l=$(status_line "$name")
  if [ -z "$l" ]; then echo "none - -"; return 1; fi
  echo "$l"
}

cmd_wait() {
  local name=${1:?usage: jobs.sh wait NAME [--timeout S]} t=${SLOTSTREAM_WAIT_S:-1800}
  shift
  while [ $# -gt 0 ]; do
    case "$1" in --timeout) t=$2; shift 2 ;; *) shift ;; esac
  done
  local end=$(( $(date +%s) + t )) l
  while :; do
    refresh_status "$name"
    l=$(status_line "$name")
    case "$l" in
      "done 0 "*) echo "$l"; return 0 ;;
      "done "*|"failed "*|"cut "*)
        echo "$l"; echo "restart with: tools/spark/jobs.sh restart $name"; return 1 ;;
      "")
        echo "no such job: $name"; return 1 ;;
    esac
    [ "$(date +%s)" -ge $end ] && { echo "timeout: $l"; return 124; }
    sleep 15
  done
}

cmd_log() { tail -n "${2:-50}" "$JD/${1:?usage: jobs.sh log NAME [N]}.log"; }

cmd_restart() {
  local name=${1:?usage: jobs.sh restart NAME}
  [ -f "$JD/$name.cmd" ] || { echo "no saved command for $name" >&2; return 2; }
  # shellcheck disable=SC1090
  . "$JD/$name.cmd"
  cmd_start "$name" "$JOB_FOOTPRINT" "$JOB_DUR" "${JOB_FLAGS[@]}" -- "${JOB_ARGV[@]}"
}

sub=${1:-}; [ $# -gt 0 ] && shift
case "$sub" in
  start) cmd_start "$@" ;;
  _exec) cmd_exec "$@" ;;
  status) cmd_status "$@" ;;
  wait) cmd_wait "$@" ;;
  log) cmd_log "$@" ;;
  restart) cmd_restart "$@" ;;
  *) echo "usage: jobs.sh start|status|wait|log|restart ..." >&2; exit 2 ;;
esac
exit $?
