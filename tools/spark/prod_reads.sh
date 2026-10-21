#!/bin/bash
# Spark, read-only: production reads v1, a1, m-lb5 and b1 (Pendientes section D).
#   prod_reads.sh                           raw reads into $CAP/raw/ (needs CAP)
#   prod_reads.sh --window START END --out FILE
#                                           b1 metrics restricted to the window (epoch seconds)
# Prometheus publishes no host port: it is reached through the container's bridge IP;
# `docker exec ... wget` is only the fallback and is recorded as prom_access="exec".
# Container, job and model names are matched by patterns that come from the environment (set them in
# private/env.local.sh); none has a default, so no production name lives in this repository:
#   PROM_PATTERN      container name of the metrics server
#   SWAP_PATTERN      container name of the model router
#   VLLM_JOB_RE       metrics job names of the model servers
#   LOAD_MODEL_RE     model names whose load lines are kept from the router log
#   V1_IMAGE_PATTERN  image name of the first-generation server
# Exit 2 when any of them is unset.
set -u
PROM_PATTERN=${PROM_PATTERN:-}
SWAP_PATTERN=${SWAP_PATTERN:-}
VLLM_JOB_RE=${VLLM_JOB_RE:-}
LOAD_MODEL_RE=${LOAD_MODEL_RE:-}
V1_IMAGE_PATTERN=${V1_IMAGE_PATTERN:-}
unset_vars=""
for v in PROM_PATTERN SWAP_PATTERN VLLM_JOB_RE LOAD_MODEL_RE V1_IMAGE_PATTERN; do
  [ -n "${!v}" ] || unset_vars="$unset_vars $v"
done
if [ -n "$unset_vars" ]; then
  echo "prod_reads.sh: set these variables first (see tools/spark/env.local.sh.example):$unset_vars" >&2
  exit 2
fi

WINDOW=0 WS="" WE="" OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --window) WINDOW=1; WS=$2; WE=$3; shift 3 ;;
    --out) OUT=$2; shift 2 ;;
    *) echo "prod_reads.sh: unexpected argument: $1" >&2; exit 2 ;;
  esac
done

PROM_CTR=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -i -- "$PROM_PATTERN" | head -1)
[ -n "$PROM_CTR" ] || { echo "prod_reads.sh: no running container matches '$PROM_PATTERN'" >&2; exit 1; }
PROM_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$PROM_CTR" 2>/dev/null)
export PROM_CTR PROM_IP

# prom_get PATH?QUERY (already URL-encoded): curl through the bridge IP, fallback to exec.
# Prints the body; sets PROM_ACCESS via the file $PROM_ACCESS_FILE.
PROM_ACCESS_FILE=$(mktemp)
echo curl > "$PROM_ACCESS_FILE"
export PROM_ACCESS_FILE

PY=$(cat <<'PYEOF'
import json, os, subprocess, sys, urllib.parse

CTR = os.environ["PROM_CTR"]
IP = os.environ.get("PROM_IP", "")
ACCESS_FILE = os.environ["PROM_ACCESS_FILE"]


def get(path, params=None):
    q = ("?" + urllib.parse.urlencode(params)) if params else ""
    url = f"/api/v1/{path}{q}"
    if IP:
        r = subprocess.run(["curl", "-s", "--max-time", "20", f"http://{IP}:9090{url}"],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip().startswith("{"):
            return json.loads(r.stdout)
    r = subprocess.run(["docker", "exec", CTR, "wget", "-qO-", f"http://localhost:9090{url}"],
                       capture_output=True, text=True)
    open(ACCESS_FILE, "w").write("exec")
    return json.loads(r.stdout) if r.stdout.strip().startswith("{") else {"status": "error", "error": r.stderr[:200]}


def jobs(regex):
    import re
    t = get("targets", {"state": "active"})
    names = sorted({x["labels"].get("job", "") for x in t.get("data", {}).get("activeTargets", [])})
    return t, [n for n in names if re.search(regex, n)]


METRICS = {
    "ttft_p50": ("histogram_quantile(0.5, sum by (le)(increase(vllm:time_to_first_token_seconds_bucket{{job=\"{job}\"}}[{w}])))"),
    "ttft_p99": ("histogram_quantile(0.99, sum by (le)(increase(vllm:time_to_first_token_seconds_bucket{{job=\"{job}\"}}[{w}])))"),
    "e2e_p99": ("histogram_quantile(0.99, sum by (le)(increase(vllm:e2e_request_latency_seconds_bucket{{job=\"{job}\"}}[{w}])))"),
    "queue_p99": ("histogram_quantile(0.99, sum by (le)(increase(vllm:request_queue_time_seconds_bucket{{job=\"{job}\"}}[{w}])))"),
    "rph": ("sum(increase(vllm:time_to_first_token_seconds_count{{job=\"{job}\"}}[{w}]))"),
}


def first_value(resp):
    try:
        r = resp["data"]["result"]
        if r:
            v = float(r[0]["value"][1])
            return None if v != v else v
    except Exception:
        pass
    return None


def window_mode(ws, we, out, job_re):
    ws, we = int(ws), int(we)
    w = max(we - ws, 60)
    _, js = jobs(job_re)
    res = {"start": ws, "end": we, "window_s": w, "jobs": {}}
    for job in js:
        m = {}
        for k, tpl in METRICS.items():
            m[k] = first_value(get("query", {"query": tpl.format(job=job, w=f"{w}s"), "time": we}))
        m["requests"] = m.pop("rph")
        res["jobs"][job] = m
    res["prom_access"] = open(ACCESS_FILE).read().strip()
    open(out, "w").write(json.dumps(res, indent=1) + "\n")


def raw_mode(capdir, job_re, load_re, log_lines):
    import re
    raw = os.path.join(capdir, "raw")
    os.makedirs(raw, exist_ok=True)
    t, js = jobs(job_re)
    json.dump(t, open(os.path.join(raw, "targets.json"), "w"))
    json.dump(get("status/config"), open(os.path.join(raw, "prom-config.json"), "w"))
    json.dump(js, open(os.path.join(raw, "b1-jobs.json"), "w"))

    # a1 + m-lb5: loads from the llama-swap log (health check passed = ready; the
    # load start is the first connection-refused proxy error in the 600 s before).
    import datetime
    def ts(line):
        s = line.split(" ", 1)[0]
        s = s[:26] + "+00:00" if s.endswith("Z") else s
        return datetime.datetime.fromisoformat(s[:19] + "+00:00").timestamp()
    lines = [l for l in log_lines if l.strip()]
    ready = sorted({ts(l) for l in lines if re.search(load_re, l) and "Health check passed" in l})
    refused = sorted(ts(l) for l in lines if "connection refused" in l)
    dedup = []
    for r in ready:
        if not dedup or r - dedup[-1] >= 120:
            dedup.append(r)
    loads = []
    for r in dedup[-15:]:
        prior = [x for x in refused if r - 600 <= x <= r]
        start = prior[0] if prior else r
        loads.append({"T": int(start), "ready": int(r)})
    json.dump(loads, open(os.path.join(raw, "loads.json"), "w"))
    # llamaswap-a.txt keeps only the load lines and the refused lines around the kept loads.
    keep = [l for l in lines
            if (re.search(load_re, l) and "Health check passed" in l)
            or any(ld["T"] - 5 <= ts(l) <= ld["ready"] for ld in loads)]
    open(os.path.join(raw, "llamaswap-a.txt"), "w").write("\n".join(keep) + "\n")
    for ld in loads:
        T = ld["T"]
        for name, metric in (("used", "llamaswap_memory_used_bytes"), ("free", "llamaswap_memory_free_bytes")):
            json.dump(get("query_range", {"query": metric, "start": T - 120, "end": T + 900, "step": "1s"}),
                      open(os.path.join(raw, f"a1-{T}-{name}.json"), "w"))

    # b1: 15 days at step 3600 per vllm job.
    import time
    end = int(time.time())
    start = end - 15 * 86400
    for job in js:
        for k, tpl in METRICS.items():
            q = tpl.format(job=job, w="1h")
            json.dump(get("query_range", {"query": q, "start": start, "end": end, "step": "3600"}),
                      open(os.path.join(raw, f"b1-{job}-{k}.json"), "w"))
    open(os.path.join(raw, "prom_access.txt"), "w").write(open(ACCESS_FILE).read().strip() + "\n")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "window":
        window_mode(sys.argv[2], sys.argv[3], sys.argv[4], os.environ["VLLM_JOB_RE"])
    else:
        raw_mode(sys.argv[2], os.environ["VLLM_JOB_RE"], os.environ["LOAD_MODEL_RE"],
                 sys.stdin.read().splitlines())
PYEOF
)
export VLLM_JOB_RE LOAD_MODEL_RE

if [ "$WINDOW" = 1 ]; then
  [ -n "$OUT" ] || { echo "prod_reads.sh: --window needs --out" >&2; exit 2; }
  python3 -c "$PY" window "$WS" "$WE" "$OUT"
  rc=$?
  rm -f "$PROM_ACCESS_FILE"
  exit $rc
fi

[ -n "${CAP:-}" ] || { echo "prod_reads.sh: CAP is not set (run under capture.sh or the guard)" >&2; exit 2; }
RAW="$CAP/raw"
mkdir -p "$RAW"

# v1: grep the vllm package inside a throwaway container from the image (no network).
IMG=$(docker images --format '{{.Repository}}:{{.Tag}}' | grep -- "$V1_IMAGE_PATTERN" | head -1)
if [ -n "$IMG" ]; then
  docker run --rm --name "slotstream-spark-${SLOTSTREAM_LABEL:-prod-reads}-v1" --network none --oom-score-adj 1000 \
    --entrypoint bash "$IMG" -c '
      d=$(python3 -c "import os,vllm;print(os.path.dirname(vllm.__file__))" 2>/dev/null)
      echo "image-vllm-dir: $d"
      grep -rn -E "psutil\.virtual_memory|mem_get_info|is_integrated|less than desired GPU memory utilization" "$d" 2>/dev/null
      python3 -c "import vllm;print(vllm.__version__)"' > "$RAW/v1-grep.txt" 2>&1 || true
else
  echo "no image matches $V1_IMAGE_PATTERN" > "$RAW/v1-grep.txt"
fi

# a1 + m-lb5: load events from the llama-swap log (read-only).
SWAP_CTR=$(docker ps --format '{{.Names}}' | grep -i -- "$SWAP_PATTERN" | head -1)
if [ -n "$SWAP_CTR" ]; then
  docker logs --since 360h --timestamps "$SWAP_CTR" 2>&1 | grep -E "$LOAD_MODEL_RE|connection refused" > "$RAW/llamaswap-full.tmp" || true
fi
python3 -c "$PY" raw "$CAP" < "$RAW/llamaswap-full.tmp"
rc=$?
rm -f "$RAW/llamaswap-full.tmp"
rm -f "$PROM_ACCESS_FILE"
exit $rc
