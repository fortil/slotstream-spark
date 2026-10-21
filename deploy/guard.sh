#!/bin/bash
# Supervisor for every heavy run on the Spark: precheck | run | postcheck.
#   guard.sh precheck [flags] FOOTPRINT DUR
#   guard.sh run [flags] LABEL FOOTPRINT DUR -- CMD...
#   guard.sh postcheck LABEL
# flags: --nolock --exclusive --allow-truncate --memcg SCOPE --container --mode MODE
# FOOTPRINT is a MiB figure or the literal "auto".
# Exit: 0 ok | 75 refused | 76 cut | 77 postcheck failed | else CMD's rc.
set -u

# SLOTSTREAM_RUNNING_URL is the model router's /running endpoint; empty or unset = no router to watch. A value the
# caller set wins over private/env.local.sh (tests and drills point it at a file).
_URL_SET=${SLOTSTREAM_RUNNING_URL+x}; _URL_CALLER=${SLOTSTREAM_RUNNING_URL:-}
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../tools/spark/env.sh"
[ -n "$_URL_SET" ] && SLOTSTREAM_RUNNING_URL=$_URL_CALLER

GUARD_ROOT=${GUARD_ROOT:-$R}
GUARD_MEMINFO=${GUARD_MEMINFO:-/proc/meminfo}
RUNNING_URL=${SLOTSTREAM_RUNNING_URL:-}
GUARD_DOCKER=${GUARD_DOCKER:-docker}
# Pause mode: $GUARD_ROOT/state/PAUSED (see paused_load in env.sh). Absent = windows as always and the container count only if EXPECTED_CONTAINERS is set;
# malformed = the guard refuses to run at all. An explicit EXPECTED_CONTAINERS in the environment wins over the file.
paused_load "$GUARD_ROOT/state/PAUSED" || { echo "guard: malformed pause file, refusing" >&2; exit 2; }
NO_WINDOWS=$PAUSED_NO_WINDOWS
# Empty = the production container count is not checked (generic host).
EXPECTED_CONTAINERS=${EXPECTED_CONTAINERS:-${PAUSED_EXPECTED:-}}
MEMAVAILABLE_MIN_MIB=${MEMAVAILABLE_MIN_MIB:-10240}
MEMFREE_MIN_MIB=${MEMFREE_MIN_MIB:-10240}
GUARD_DONTNEED=${GUARD_DONTNEED:-$GUARD_ROOT/lb/gguf/*.gguf:$GUARD_ROOT/lb/gguf/*/*.gguf}
GUARD_KLOG=${GUARD_KLOG:-journalctl -k --no-pager}
GUARD_NVSMI=${GUARD_NVSMI:-nvidia-smi}
GUARD_CGCTL=${GUARD_CGCTL:-/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/cgroup.controllers}

MODE_CEILING_MIB_D1=19866
MODE_NEED_MIB_D1=62644
MODE_CEILING_MIB_D2=26112
MODE_NEED_MIB_D2=56413
MODE_CEILING_MIB_D5=65126
MODE_NEED_MIB_D5=16896

MODE=${SLOTSTREAM_MODE:-D5}

now() { if [ -n "${GUARD_NOW:-}" ]; then echo "$GUARD_NOW"; else date +%s; fi; }
jstr() { local s=${1//\\/\\\\}; s=${s//\"/\\\"}; printf '%s' "$s"; }
meminfo_mib() { awk -v k="$1:" '$1==k {print int($2/1024)}' "$GUARD_MEMINFO"; }
running_raw() { if [ -z "$RUNNING_URL" ]; then echo '{"running":[]}'; else curl -s --max-time 5 "$RUNNING_URL" 2>/dev/null; fi; }
running_field() { if [ -z "$RUNNING_URL" ]; then printf 'off'; else jstr "$(running_raw | tr -d '[:space:]')"; fi; }
group_alive() { [ -n "${1:-}" ] && [ -n "$(pgrep -g "$1" 2>/dev/null)" ]; }
containers_all() { $GUARD_DOCKER ps --format '{{.Names}}' 2>/dev/null; }

# ---------------------------------------------------------------- flags
F_NOLOCK=0 F_EXCL=0 F_TRUNC=0 F_MEMCG="" F_CTR=0
LABEL="" FOOTPRINT="" DUR="" ARGV_REST=()

parse_flags() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --nolock) F_NOLOCK=1; shift ;;
      --exclusive) F_EXCL=1; shift ;;
      --allow-truncate) F_TRUNC=1; shift ;;
      --memcg) F_MEMCG=$2; shift 2 ;;
      --container) F_CTR=1; shift ;;
      --mode) MODE=$2; shift 2 ;;
      *) break ;;
    esac
  done
  REST=("$@")
}

flags_string() {
  local s=""
  [ $F_NOLOCK = 1 ] && s="$s --nolock"
  [ $F_EXCL = 1 ] && s="$s --exclusive"
  [ $F_TRUNC = 1 ] && s="$s --allow-truncate"
  [ -n "$F_MEMCG" ] && s="$s --memcg $F_MEMCG"
  [ $F_CTR = 1 ] && s="$s --container"
  echo "${s# }"
}

# ---------------------------------------------------------------- precheck
PRECHECK_JSON=""
CEILING=0 NEED=0 DEADLINE=0

emit_json() { # ok
  PRECHECK_JSON=$(printf '{"mode":"%s","footprint_mib":%s,"memavailable_mib":%s,"memfree_mib":%s,"running":"%s","containers":%s,"ok":%s}' \
    "$MODE" "${FOOTPRINT:-0}" "${MA:-0}" "${MF:-0}" "$(jstr "${RUN:-}")" "${NC:-0}" "$1")
  [ -n "${QUIET:-}" ] || echo "$PRECHECK_JSON"
}

refuse() { emit_json false; echo "REFUSED: $1"; return 75; }

today_limit() { # epoch of today HH:MM relative to now()
  local n day; n=$(now); day=$(date -d "@$n" +%Y-%m-%d); date -d "$day $1" +%s
}

do_precheck() { # uses FOOTPRINT DUR flags; OWN_LOCK skips the lock check
  local n h lim ids c o ctr pgid f
  MA=$(meminfo_mib MemAvailable); MF=$(meminfo_mib MemFree); RUN=""; [ -z "$RUNNING_URL" ] && RUN=off; NC=0
  MA=${MA:-0}; MF=${MF:-0}

  case "$MODE" in
    D1|D2|D5) ;;
    X) FOOTPRINT=${FOOTPRINT:-0}; refuse "X windows are not authorized (user decision 2026-10-11)"; return $? ;;
    *) refuse "unknown mode $MODE"; return $? ;;
  esac
  eval "CEILING=\$MODE_CEILING_MIB_$MODE NEED=\$MODE_NEED_MIB_$MODE"

  if [ "$FOOTPRINT" = auto ]; then
    FOOTPRINT=$(( MA - NEED )); [ $FOOTPRINT -gt $CEILING ] && FOOTPRINT=$CEILING
    FOOTPRINT=$(( FOOTPRINT - 256 ))
    if [ $FOOTPRINT -lt 16384 ]; then refuse "auto footprint $FOOTPRINT MiB is below 16384"; return $?; fi
  fi

  n=$(now)
  # (1) window (skipped while the pause file says NO_WINDOWS=1)
  if [ "$NO_WINDOWS" != 1 ]; then
    h=$((10#$(date -d "@$n" +%H)))
    if [ $h -lt 4 ]; then refuse "00:00-04:00"; return $?; fi
    if [ $F_TRUNC = 1 ]; then
      lim=$(today_limit 23:40)
      if [ $n -ge $lim ]; then refuse "after 23:40"; return $?; fi
    else
      lim=$(today_limit 23:45)
      if [ $(( n + DUR )) -gt $lim ]; then refuse "run would end after 23:45"; return $?; fi
    fi
  fi

  # (2) model router idle (skipped when SLOTSTREAM_RUNNING_URL is empty)
  if [ -n "$RUNNING_URL" ]; then
    RUN=$(running_raw | tr -d '[:space:]')
    if [ "$RUN" != '{"running":[]}' ]; then refuse "model router /running is not empty or unreadable"; return $?; fi
  fi

  # (3) memory
  if [ "$FOOTPRINT" -gt "$CEILING" ]; then refuse "footprint $FOOTPRINT exceeds the $MODE ceiling $CEILING"; return $?; fi
  if [ $(( MA - FOOTPRINT )) -lt "$NEED" ]; then refuse "MemAvailable $MA - footprint $FOOTPRINT < need $NEED"; return $?; fi

  # (4) production containers
  ids=$(containers_all)
  NC=$(printf '%s\n' "$ids" | grep -v '^slotstream-spark-' | grep -c . )
  if [ -n "$EXPECTED_CONTAINERS" ] && [ "$NC" != "$EXPECTED_CONTAINERS" ]; then refuse "production containers $NC != $EXPECTED_CONTAINERS"; return $?; fi

  # (5) own containers only if a live .run owns them
  for c in $(printf '%s\n' "$ids" | grep '^slotstream-spark-' || true); do
    o=0
    for f in "$GUARD_ROOT"/run/*.run; do
      [ -f "$f" ] || continue
      ctr=$(sed -n 's/^ctr=//p' "$f"); pgid=$(sed -n 's/^pgid=//p' "$f")
      if [ "$ctr" = "$c" ] && group_alive "$pgid"; then o=1; fi
    done
    [ $o = 1 ] || { refuse "own container $c running without a live run"; return $?; }
  done

  # (6) model lock
  if [ $F_NOLOCK = 0 ] && [ -z "${OWN_LOCK:-}" ]; then
    mkdir -p "$GUARD_ROOT"
    if ! flock -n "$GUARD_ROOT/.model.lock" true; then refuse "model lock held"; return $?; fi
  fi

  # (7) exclusive
  if [ $F_EXCL = 1 ]; then
    for f in "$GUARD_ROOT"/run/*.run; do
      [ -f "$f" ] || continue
      if group_alive "$(sed -n 's/^pgid=//p' "$f")"; then refuse "another run is alive ($(basename "$f"))"; return $?; fi
    done
  fi

  # (8) memcg
  if [ -n "$F_MEMCG" ]; then
    if ! { [ -r "$GUARD_CGCTL" ] && tr ' ' '\n' < "$GUARD_CGCTL" | grep -qx memory; }; then
      refuse "memory controller is not delegated"; return $?
    fi
  fi

  emit_json true
  return 0
}

# ---------------------------------------------------------------- run helpers
cached_kib() { awk '/^Cached:/ {print $2}' /proc/meminfo; }

release_page_cache() {
  local before after pat f files=()
  before=$(cached_kib)
  local IFS=:
  for pat in $GUARD_DONTNEED; do
    for f in $pat; do [ -f "$f" ] && files+=("$f"); done
  done
  unset IFS
  if [ ${#files[@]} -gt 0 ]; then
    python3 -c 'import os,sys
for p in sys.argv[1:]:
    fd=os.open(p,os.O_RDONLY)
    os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
    os.close(fd)' "${files[@]}" 2>/dev/null || true
  fi
  after=$(cached_kib)
  printf '{"t":%s,"reason":"dontneed","cached_kib_before":%s,"cached_kib_after":%s}\n' \
    "$(date +%s)" "${before:-0}" "${after:-0}" >> "$CAP/guard-events.jsonl"
}

log_event() { # reason
  printf '{"t":%s,"reason":"%s","memavailable_mib":%s,"memfree_mib":%s,"running":"%s"}\n' \
    "$(date +%s)" "$1" "$(meminfo_mib MemAvailable)" "$(meminfo_mib MemFree)" \
    "$(running_field)" >> "$CAP/guard-events.jsonl"
}

cut_run() { # PGID CTR
  local pgid=$1 ctr=$2 e=10 i
  [ -n "$F_MEMCG" ] && e=2
  [ -n "$ctr" ] && $GUARD_DOCKER stop -t $e "$ctr" >/dev/null 2>&1
  kill -TERM -- "-$pgid" 2>/dev/null
  sleep $e
  [ -n "$ctr" ] && $GUARD_DOCKER kill "$ctr" >/dev/null 2>&1
  kill -KILL -- "-$pgid" 2>/dev/null
  if [ -n "$F_MEMCG" ]; then
    for i in $(seq 1 30); do
      if ! group_alive "$pgid" && [ -z "$(scope_cgroup)" ]; then break; fi
      sleep 1
    done
    release_page_cache
  fi
}

scope_cgroup() {
  [ -n "$F_MEMCG" ] || return 0
  systemctl --user show -p ControlGroup --value "$F_MEMCG.scope" 2>/dev/null
}

watchdog() { # PGID CTR DEADLINE FOOTPRINT
  local pgid=$1 ctr=$2 deadline=$3 footprint=$4 cg reason ma mf cur nv pid cpid
  while group_alive "$pgid"; do
    reason=""
    ma=$(meminfo_mib MemAvailable); mf=$(meminfo_mib MemFree)
    if [ "${ma:-0}" -lt "$MEMAVAILABLE_MIN_MIB" ]; then reason=memavailable; fi
    if [ -z "$reason" ] && [ -n "$RUNNING_URL" ] && [ "$(running_raw | tr -d '[:space:]')" != '{"running":[]}' ]; then reason=running; fi
    if [ -z "$reason" ] && [ "$(date +%s)" -ge "$deadline" ]; then reason=deadline; fi
    if [ -z "$reason" ] && [ -e "$GUARD_ROOT/run/STOP" ]; then reason=stop; fi
    if [ -z "$reason" ] && [ -n "$F_MEMCG" ]; then
      cg=$(scope_cgroup)
      if [ -n "$cg" ] && [ -d "/sys/fs/cgroup$cg" ]; then
        cur=$(( $(cat "/sys/fs/cgroup$cg/memory.current" 2>/dev/null || echo 0) / 1048576 ))
        nv=0
        while read -r pid; do
          nv=$(( nv + $($GUARD_NVSMI --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
                | awk -F', *' -v p="$pid" '$1==p {s+=$2} END{print s+0}') ))
        done < "/sys/fs/cgroup$cg/cgroup.procs"
        if [ $(( cur + nv )) -gt "$footprint" ]; then reason=memcg_over
        elif [ "${mf:-0}" -lt "$MEMFREE_MIN_MIB" ]; then reason=memfree; fi
      fi
    fi
    if [ -z "$reason" ] && [ -n "$ctr" ]; then
      cpid=$($GUARD_DOCKER inspect -f '{{.State.Pid}}' "$ctr" 2>/dev/null || true)
      if [ -n "$cpid" ] && [ "$cpid" != 0 ] && [ -r "/proc/$cpid/oom_score_adj" ] \
         && [ "$(cat "/proc/$cpid/oom_score_adj")" != 1000 ]; then reason=ctr_oom_adj; fi
    fi
    if [ -n "$reason" ]; then
      echo "$reason" > "$CAP/cut"
      log_event "$reason"
      cut_run "$pgid" "$ctr"
      return 0
    fi
    sleep 1
  done
}

postcheck_write() { # rc cut start pgid ctr label
  local rc=$1 cut=$2 start=$3 pgid=$4 ctr=$5 alive=false own=0 up klog=clean kl=0 nv=0 out
  group_alive "$pgid" && alive=true
  if [ -n "$ctr" ] && containers_all | grep -qx "$ctr"; then own=1; fi
  up=$(containers_all | grep -v '^slotstream-spark-' | grep -c . )
  if out=$($GUARD_KLOG --since "@$start" 2>/dev/null); then
    kl=$(printf '%s\n' "$out" | grep -Eic 'oom-kill|out of memory|NV_ERR_NO_MEMORY')
    [ "$kl" -gt 0 ] && klog=oom
  else
    klog=unreadable; echo "WARNING: kernel log unreadable" >&2
  fi
  if [ "$alive" = true ]; then
    nv=$($GUARD_NVSMI --query-compute-apps=pid --format=csv,noheader 2>/dev/null | while read -r p; do
         [ "$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')" = "$pgid" ] && echo x; done | grep -c x)
  fi
  local cutj=null; [ -n "$cut" ] && cutj="\"$cut\""
  printf '{"rc":%s,"cut":%s,"group_alive":%s,"own_containers":%s,"containers_up":%s,"running":"%s","memavailable_mib":%s,"memfree_mib":%s,"klog":"%s","klog_lines":%s,"nvsmi_own_pids":%s,"expected_containers":%s,"paused":%s}\n' \
    "$rc" "$cutj" "$alive" "$own" "$up" "$(running_field)" \
    "$(meminfo_mib MemAvailable)" "$(meminfo_mib MemFree)" "$klog" "$kl" "${nv:-0}" "${EXPECTED_CONTAINERS:-null}" "$([ "$PAUSED_ACTIVE" = 1 ] && echo true || echo false)" > "$CAP/guard-post.json"
  POST_ALIVE=$alive; POST_OWN=$own; POST_KLOG=$klog
}

# ---------------------------------------------------------------- commands
cmd_precheck() {
  parse_flags "$@"; set -- "${REST[@]}"
  [ $# -eq 2 ] || { echo "usage: guard.sh precheck [flags] FOOTPRINT DUR" >&2; return 2; }
  FOOTPRINT=$1 DUR=$2
  do_precheck
}

cmd_run() {
  parse_flags "$@"; set -- "${REST[@]}"
  [ $# -ge 5 ] && [ "$4" = "--" ] || { echo "usage: guard.sh run [flags] LABEL FOOTPRINT DUR -- CMD..." >&2; return 2; }
  LABEL=$1 FOOTPRINT=$2 DUR=$3; shift 4

  trap '' HUP
  if [ $F_NOLOCK = 0 ]; then
    mkdir -p "$GUARD_ROOT"
    exec 9>"$GUARD_ROOT/.model.lock"
    flock -n 9 || { echo "REFUSED: model lock held"; return 75; }
    OWN_LOCK=1
  fi
  QUIET=1 do_precheck || { echo "REFUSED: precheck failed at run" >&2; return 75; }

  local ts yd cap n lim seg deadline pid pgid ctr="" wd rc cut=""
  ts=$(date +%Y%m%dT%H%M%S); yd=$(date +%Y/%m)
  CAP="$GUARD_ROOT/runs/$yd/$ts-$LABEL"
  mkdir -p "$CAP" "$GUARD_ROOT/run"
  echo "$CAP" > "$GUARD_ROOT/runs/LAST-$LABEL"
  [ $F_CTR = 1 ] && ctr="slotstream-spark-$LABEL"
  echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true

  seg=$DUR
  if [ "$NO_WINDOWS" != 1 ]; then
    n=$(now); lim=$(today_limit 23:45)
    [ $(( lim - n )) -lt $seg ] && seg=$(( lim - n )); [ $seg -lt 1 ] && seg=1
  fi
  deadline=$(( $(date +%s) + seg ))
  export CAP SLOTSTREAM_LABEL=$LABEL SLOTSTREAM_MODE=$MODE SLOTSTREAM_FOOTPRINT_MIB=$FOOTPRINT \
         SLOTSTREAM_DEADLINE=$deadline SLOTSTREAM_R=$GUARD_ROOT

  setsid timeout --signal=TERM --kill-after=30 $(( seg + 15 )) "$@" &
  pid=$!
  pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' '); pgid=${pgid:-$pid}
  local start; start=$(date +%s)
  printf '{"label":"%s","flags":"%s","precheck":%s,"start":%s,"pgid":%s,"cap":"%s"}\n' \
    "$LABEL" "$(flags_string)" "$PRECHECK_JSON" "$start" "$pgid" "$(jstr "$CAP")" > "$CAP/guard.json"
  printf 'pgid=%s\nctr=%s\nlabel=%s\nstart=%s\n' "$pgid" "$ctr" "$LABEL" "$start" > "$GUARD_ROOT/run/$LABEL.run"

  watchdog "$pgid" "$ctr" "$deadline" "$FOOTPRINT" &
  wd=$!
  wait "$pid"; rc=$?
  if [ -e "$CAP/cut" ]; then wait "$wd" 2>/dev/null; else kill "$wd" 2>/dev/null; wait "$wd" 2>/dev/null; fi
  [ -e "$CAP/cut" ] && cut=$(cat "$CAP/cut")

  if [ -z "$cut" ] && [ -n "$F_MEMCG" ]; then release_page_cache; fi
  postcheck_write "$rc" "$cut" "$start" "$pgid" "$ctr"
  if [ "$POST_ALIVE" = false ] && [ "$POST_OWN" = 0 ]; then rm -f "$GUARD_ROOT/run/$LABEL.run"; fi

  [ -n "$cut" ] && return 76
  if [ "$POST_ALIVE" = true ] || [ "$POST_OWN" != 0 ] || [ "$POST_KLOG" = oom ]; then return 77; fi
  return $rc
}

cmd_postcheck() {
  local label=${1:-} f pgid ctr
  f="$GUARD_ROOT/run/$label.run"
  [ -f "$f" ] || return 0
  pgid=$(sed -n 's/^pgid=//p' "$f"); ctr=$(sed -n 's/^ctr=//p' "$f")
  if group_alive "$pgid" || { [ -n "$ctr" ] && containers_all | grep -qx "$ctr"; }; then
    echo "postcheck: $label still has processes or a container"
    return 77
  fi
  rm -f "$f"
  return 0
}

sub=${1:-}; [ $# -gt 0 ] && shift
case "$sub" in
  precheck) cmd_precheck "$@" ;;
  run) cmd_run "$@" ;;
  postcheck) cmd_postcheck "$@" ;;
  *) echo "usage: guard.sh precheck|run|postcheck ..." >&2; exit 2 ;;
esac
exit $?
