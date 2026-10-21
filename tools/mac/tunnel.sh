#!/bin/bash
# Mac: the SSH tunnel to the Spark front (127.0.0.1:18434 on both ends).
# tunnel.sh up|status|down
#   up      sources tools/spark/env.sh and opens the forward; idempotent (nothing to do while the port answers).
#   status  exit 0 when /api/version answers 2xx or 503 (503 is the front saying the engine is loading).
#   down    kills only the ssh that carries this forward.
# SLOTSTREAM_URL overrides the target of `status` and `up` (a stub in tests); SLOTSTREAM_TUNNEL_PORT the port.
set -u
cd "$(dirname "$0")/../.."
. tools/spark/env.sh
PORT=${SLOTSTREAM_TUNNEL_PORT:-18434}
URL=${SLOTSTREAM_URL:-http://127.0.0.1:$PORT}
FORWARD="$PORT:127.0.0.1:$PORT"

answers() {
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$URL/api/version" 2>/dev/null) || return 1
  case "$code" in 2??|503) return 0 ;; *) return 1 ;; esac
}

# pids of the ssh processes started with exactly this forward (and nothing else)
tunnel_pids() {
  ps -axo pid=,command= | awk -v f="-L $FORWARD" '$2 ~ /(^|\/)ssh$/ && index($0, f) > 0 && index($0, "-fN") > 0 {print $1}'
}

case "${1:-}" in
  status)
    if answers; then echo "tunnel: $URL answers"; exit 0; fi
    echo "tunnel: $URL does not answer" >&2
    exit 1
    ;;
  up)
    if answers; then echo "tunnel: already up ($URL)"; exit 0; fi
    if [ -n "$(tunnel_pids)" ]; then
      echo "tunnel: an ssh with this forward exists but $URL does not answer; run 'tunnel.sh down' first" >&2
      exit 1
    fi
    [ -z "${SLOTSTREAM_URL:-}" ] || { echo "tunnel: SLOTSTREAM_URL is set to $URL and does not answer; not opening ssh" >&2; exit 1; }
    ssh $SPARK_SSH_OPTS -fN -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
        -L "$FORWARD" "$SPARK_HOST" || { echo "tunnel: ssh failed" >&2; exit 1; }
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      if answers; then echo "tunnel: up ($URL)"; exit 0; fi
      sleep 1
    done
    echo "tunnel: ssh is running but $URL does not answer" >&2
    exit 1
    ;;
  down)
    pids=$(tunnel_pids)
    if [ -z "$pids" ]; then echo "tunnel: not running"; exit 0; fi
    kill $pids 2>/dev/null
    echo "tunnel: stopped"
    exit 0
    ;;
  *)
    echo "usage: tunnel.sh up|status|down" >&2
    exit 2
    ;;
esac
