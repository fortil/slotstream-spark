#!/bin/bash
# Spark: launch the pinned llama.cpp server (LB yardstick) inside a memory-limited user scope.
#   llamacpp.sh --weights q41|iq3|q41exact --arm A|B|R --mode D1|D2|D5
#               [--threads 16] [--port 8091] [--unit slotstream-spark-lb]
#               [--idle-exit S] [--print-footprint] [--dry-run] [--memfree-mib N]
#               [--memfree-stop MIB] [--memfree-log FILE] [--print-flags]
# Arms: A = experts on CPU (-cmoe), B = A plus llama.cpp's expert cache, R = resident (no -cmoe).
# Envelope: FOOTPRINT is the memory the run may use; MemoryMax caps the page cache of the mmap'd GGUF.
# CUDA reserve = ${LB_CUDA_MIB:-8704} MiB (step m-lb4 may override it).
# MemFree clamp: MemoryMax = min(table, MemFree_MiB - 10240 - CUDA - 4096). A real launch reads
# MemFree from /proc/meminfo right before starting; --memfree-mib N gives it explicitly (tests, and
# yardstick_session.sh so that the number it records is the one used). --print-footprint and --dry-run without
# --memfree-mib never read the host: they print the table value.
# Cache clamp (arm B, user answer 2026-10-13): CACHE = min(table, MemFree_MiB - 26945), so the full
# 43520 MiB cache needs MemFree >= 70465 MiB (LB2 started at 72693 and bottomed 2228 MiB over the soft
# floor). Below 2048 MiB of cache it exits 4. --print-footprint with --memfree-mib appends
# CACHE_TABLE_MIB=<table> CACHE_FULL=true|false on cache arms.
# Soft floor: memfree_watch.sh ends the server cleanly when MemFree < --memfree-stop (default 14336,
# 0 disables); --memfree-log is its 1 Hz CSV, and LOG.stop marks a stop. --print-flags prints
# OP_OFFLOAD=<bool> MEMFREE_STOP_MIB=<n> and exits without reading the host.
# Exit: 0 ok, 4 clamp leaves less than 2048 MiB, 5 soft floor stop, 2 bad combination or usage,
# 3 memcg not delegated, else the server's code.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$HERE/tools/spark/env.sh"

WEIGHTS="" ARM="" MODE="" THREADS=16 PORT=8091 UNIT=slotstream-spark-lb IDLE=0 PRINT=0 DRY=0 MEMFREE= MEMFREE_STOP=14336 MEMFREE_LOG=/dev/null FLAGS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --weights) WEIGHTS=$2; shift 2 ;;
    --arm) ARM=$2; shift 2 ;;
    --mode) MODE=$2; shift 2 ;;
    --threads) THREADS=$2; shift 2 ;;
    --port) PORT=$2; shift 2 ;;
    --unit) UNIT=$2; shift 2 ;;
    --idle-exit) IDLE=$2; shift 2 ;;
    --print-footprint) PRINT=1; shift ;;
    --dry-run) DRY=1; shift ;;
    --memfree-mib) MEMFREE=$2; shift 2 ;;
    --memfree-stop) MEMFREE_STOP=$2; shift 2 ;;
    --memfree-log) MEMFREE_LOG=$2; shift 2 ;;
    --print-flags) FLAGS=1; shift ;;
    *) echo "llamacpp.sh: unexpected argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$WEIGHTS" ] && [ -n "$ARM" ] && [ -n "$MODE" ] || { echo "llamacpp.sh: --weights, --arm and --mode are required" >&2; exit 2; }

CUDA=${LB_CUDA_MIB:-8704}
CACHE="" UB=2048 CMOE="-cmoe"
case "$WEIGHTS:$ARM:$MODE" in
  q41:A:D1)      FOOTPRINT=18944; MAXM=$((FOOTPRINT - CUDA)) ;;
  q41:A:D2)      FOOTPRINT=25088; MAXM=$((FOOTPRINT - CUDA)) ;;
  q41:A:D5|q41exact:A:D5)
                 FOOTPRINT=57344; MAXM=$((FOOTPRINT - CUDA - 1024)) ;;
  q41:B:D1)      FOOTPRINT=18944; MAXM=4096; CACHE=$((FOOTPRINT - CUDA - 4096)) ;;
  q41:B:D5)      FOOTPRINT=57344; MAXM=4096; CACHE=$((FOOTPRINT - CUDA - 4096 - 1024)) ;;
  iq3:A:D1)      FOOTPRINT=18954; MAXM=$((FOOTPRINT - CUDA - 266)) ;;
  iq3:R:D5)      FOOTPRINT=57344; MAXM=8192; CMOE=""; UB=512 ;;
  *) echo "llamacpp.sh: unsupported combination $WEIGHTS/$ARM/$MODE" >&2; exit 2 ;;
esac
if [ "$FLAGS" = 1 ]; then
  if [ -n "$CMOE" ]; then echo "OP_OFFLOAD=false MEMFREE_STOP_MIB=$MEMFREE_STOP"
  else echo "OP_OFFLOAD=true MEMFREE_STOP_MIB=$MEMFREE_STOP"; fi
  exit 0
fi
TABLE=$MAXM MEMFREE_SRC=""
if [ -n "$MEMFREE" ]; then
  MEMFREE_SRC=arg
elif [ "$PRINT" = 0 ] && [ "$DRY" = 0 ]; then
  MEMFREE=$(awk '/^MemFree:/{print int($2/1024)}' /proc/meminfo)
  MEMFREE_SRC=meminfo
fi
if [ -n "$MEMFREE" ]; then
  LIM=$((MEMFREE - 10240 - CUDA - 4096))
  [ "$LIM" -lt "$MAXM" ] && MAXM=$LIM
  if [ "$MAXM" -lt 2048 ]; then
    echo "llamacpp.sh: MemFree $MEMFREE MiB leaves MemoryMax $MAXM MiB (< 2048)" >&2
    exit 4
  fi
fi
CACHE_TABLE=$CACHE CACHE_FULL=""
if [ -n "$MEMFREE" ] && [ -n "$CACHE" ]; then
  CACHE_FULL_MEMFREE_MIB=70465 CACHE_OFFSET_MIB=26945     # 70465 = 43520 + 26945
  LIMC=$((MEMFREE - CACHE_OFFSET_MIB))
  [ "$LIMC" -lt "$CACHE" ] && CACHE=$LIMC
  if [ "$CACHE" -lt 2048 ]; then
    echo "llamacpp.sh: MemFree $MEMFREE MiB leaves cache $CACHE MiB (< 2048)" >&2
    exit 4
  fi
  if [ "$CACHE" = "$CACHE_TABLE" ]; then CACHE_FULL=true; else CACHE_FULL=false; fi
fi
HIGH=$((MAXM - 512))

if [ "$PRINT" = 1 ]; then
  out="FOOTPRINT_MIB=$FOOTPRINT MEMORYMAX_MIB=$MAXM"
  [ -n "$CACHE" ] && out="$out CACHE_MIB=$CACHE"
  [ -n "$MEMFREE" ] && out="$out MEMFREE_MIB=$MEMFREE TABLE_MIB=$TABLE"
  [ -n "$CACHE_FULL" ] && out="$out CACHE_TABLE_MIB=$CACHE_TABLE CACHE_FULL=$CACHE_FULL"
  echo "$out"
  exit 0
fi

case "$WEIGHTS" in
  q41)      GDIR=$LB/gguf/Qwen3.8-Flash-Next-Q4_1; PAT='*-00001-of-*.gguf' ;;
  iq3)      GDIR=$LB/gguf/IQ3_XXS; PAT='*-00001-of-*.gguf' ;;
  q41exact) GDIR=$R/model/q4_1-exact; PAT='model.gguf' ;;   # repack_q4_1.py writes one file
esac
# $R may start with '~' (Mac side); the glob needs it expanded on the Spark.
GDIR=${GDIR/#\~/$HOME}
FIRST=$(ls "$GDIR"/$PAT 2>/dev/null | head -1)
[ -n "$FIRST" ] || FIRST="$GDIR/$PAT"
SERVER=${LLAMA_SERVER:-${LB/#\~/$HOME}/llama.cpp/build/bin/llama-server}

ARGS=(-m "$FIRST" -lm mmap -lzm on)
[ -n "$CMOE" ] && ARGS+=("$CMOE")
ARGS+=(--no-repack -ngl 99 -fit off -c 32768 -np 1 -b 2048 -ub "$UB" -fa on -t "$THREADS"
       --cache-ram 0 -ctxcp 4 --host 127.0.0.1 --port "$PORT" --metrics -lv 4)
# With -cmoe the default --op-offload runs batch >= 32 MoE matmuls on the GPU from the host-mapped
# GGUF; the kernel charges those page-cache pages to the root memcg, outside MemoryMax (DIAG_LB2.md).
[ -n "$CMOE" ] && ARGS+=(--no-op-offload)
[ -n "$CACHE" ] && ARGS+=(--moe-cache-mib "$CACHE")

CMD=(systemd-run --user --scope "--unit=$UNIT" -p "MemoryMax=${MAXM}M" -p "MemoryHigh=${HIGH}M" -p MemorySwapMax=0
     nice -n 10 "$SERVER" "${ARGS[@]}")

if [ "$DRY" = 1 ]; then
  printf '%s ' "${CMD[@]}"; echo
  exit 0
fi

echo "llamacpp.sh: MemoryMax=${MAXM}M (table ${TABLE}M, MemFree ${MEMFREE}M from ${MEMFREE_SRC})${CACHE:+ cache=${CACHE}M (table ${CACHE_TABLE}M, full needs MemFree >= 70465M)}" >&2
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}
if ! grep -qw memory "/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/cgroup.controllers" 2>/dev/null; then
  echo "llamacpp.sh: the memory controller is not delegated to the user manager" >&2
  exit 3
fi

dontneed() {
  python3 - "$GDIR" <<'PY'
import glob, os, sys
for p in glob.glob(os.path.join(sys.argv[1], "*.gguf")):
    fd = os.open(p, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(fd)
PY
}

"${CMD[@]}" &
SRV=$!
rm -f "$MEMFREE_LOG.stop"
bash "$HERE/deploy/lb/memfree_watch.sh" "$SRV" "$MEMFREE_STOP" "$MEMFREE_LOG" &
WPID=$!
trap 'kill -TERM "$SRV" "$WPID" 2>/dev/null; wait "$SRV" 2>/dev/null; dontneed' EXIT
trap 'exit 143' TERM INT

if [ "$IDLE" -gt 0 ]; then
  (
    last=""; since=$(date +%s)
    while kill -0 "$SRV" 2>/dev/null; do
      sleep 60
      m=$(curl -s --max-time 5 "http://127.0.0.1:$PORT/metrics") || continue
      sig=$(echo "$m" | awk '/^llamacpp:(prompt_tokens_total|tokens_predicted_total) /{s=s $2 ","} END{print s}')
      busy=$(echo "$m" | awk '/^llamacpp:requests_processing /{print $2}')
      now=$(date +%s)
      if [ "$sig" != "$last" ] || [ "${busy:-0}" != 0 ]; then last=$sig; since=$now; fi
      if [ $((now - since)) -ge "$IDLE" ]; then kill -TERM "$SRV" 2>/dev/null; break; fi
    done
  ) &
fi

wait "$SRV"
rc=$?
kill -TERM "$WPID" 2>/dev/null; wait "$WPID" 2>/dev/null
trap - EXIT
dontneed
if [ -f "$MEMFREE_LOG.stop" ]; then
  read -r _ _ mfs _ < "$MEMFREE_LOG.stop"
  echo "llamacpp.sh: memfree soft floor: server stopped at MemFree $mfs MiB (< $MEMFREE_STOP)" >&2
  exit 5
fi
exit $rc
