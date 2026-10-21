#!/bin/bash
# Spark: m8, 20 sustained O_DIRECT runs at QD12 with the NVMe temperature before each.
# Run under tools/capture.sh (CAP set). Writes $CAP/temps.csv, $CAP/m8.jsonl, $CAP/summary.json.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
. "$HERE/tools/spark/env.sh"
: "${CAP:?m8.sh must run under capture.sh}"
cd "$HERE"

HW=/sys/class/hwmon/hwmon1/temp1_input
F=("$MODEL"/model-000{01..10}.safetensors)
for f in "${F[@]}"; do [ -f "$f" ] || { echo "missing $f" >&2; exit 2; }; done
: > "$CAP/m8.jsonl"
echo "i,temp_mc" > "$CAP/temps.csv"
for i in $(seq 1 20); do
  echo "$i,$(cat $HW 2>/dev/null || echo -1)" >> "$CAP/temps.csv"
  python3 tools/probes/iobench.py run --label "m8-qd12-$i" --bs 2764800 --qd 12 --dur 30 --direct --repeat "${F[@]}" >> "$CAP/m8.jsonl"
done
echo "end,$(cat $HW 2>/dev/null || echo -1)" >> "$CAP/temps.csv"

# Rule fixed before measuring: nvme_temp_limit_mc = temperature at the first run whose GBps is
# below 0.9 x the median of runs 1-3, else the maximum observed (no_throttle=true).
# Runs with dur_s < 29 are flagged invalid.
python3 - "$CAP" <<'PY'
import json, statistics, sys
cap = sys.argv[1]
runs = [json.loads(l) for l in open(cap + "/m8.jsonl") if l.strip().startswith("{")]
temps = {}
for l in open(cap + "/temps.csv").read().splitlines()[1:]:
    k, v = l.split(",")
    temps[k] = int(v)
rows = []
for i, r in enumerate(runs, 1):
    g = r.get("GBps", r.get("gbps"))
    rows.append({"i": i, "gbps": g, "temp_mc": temps.get(str(i)), "dur_s": r.get("dur_s"), "invalid": (r.get("dur_s") or 0) < 29})
base = statistics.median([x["gbps"] for x in rows[:3]]) if len(rows) >= 3 else None
limit = None
for x in rows:
    if base and x["gbps"] is not None and x["gbps"] < 0.9 * base:
        limit = x["temp_mc"]
        break
no_throttle = limit is None
if no_throttle:
    limit = max([v for v in temps.values()] or [0])
json.dump({"runs": rows, "baseline_gbps": base, "nvme_temp_limit_mc": limit, "no_throttle": no_throttle,
           "runs_n": len(rows)}, open(cap + "/summary.json", "w"), indent=1)
print(json.dumps({"nvme_temp_limit_mc": limit, "no_throttle": no_throttle, "runs_n": len(rows)}))
PY
