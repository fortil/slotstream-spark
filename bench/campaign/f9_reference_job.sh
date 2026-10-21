#!/bin/bash
# Spark: the F9 llama.cpp differential (plan step 96). Start it with
#   tools/spark/jobs.sh start f9-ref --heavy f9-ref 57344 12600 --memcg slotstream-spark-f9-ref -- bash bench/campaign/f9_reference_job.sh
# Same wrapping as f9_oracle_job.sh (frozen sources from snap.sh, tools/capture.sh, under the guard). Phases, one model process
# at a time; every stop is a result line, not a crash, and the job exits 1 on it:
#   0   preflight: free space >= 150 GB at $R; finish by 23:45 unless $R/state/PAUSED has NO_WINDOWS=1; the helper is built
#   1   repack_q4_1.py --sweep: the job stops only when `bad` is non-empty; the tolerated list must equal q4_1_exceptions.json
#   1b  repack_q4_1.py --check-only: a tensor under the 0.95 floor or unmappable stops with a DECISION line that lists every such tensor
#   2   repack_q4_1.py build -> $R/model/q4_1-exact/{model.gguf,manifest.json}; exceptions == sweep, overrides == ple_conv1d only,
#       the 194 hyper-connection matrices written as dequantized BF16 (DIAG_F9c.md), no other tensor
#   3   llama-server on the exact GGUF through deploy/lb/llamacpp.sh (arm A, D5, --no-op-offload, MemFree clamp, soft floor)
#       answers a one-prompt smoke
#   4   llamacpp_ppl.py: the first 32 windows of the engine's ids inside the f9-ref memcg
# F9_REF_SMOKE=1 runs phases 0, 3 and 4 only, on the step-12 template GGUF and 1 window (a check of the tooling, not a result).
# $CAP/result.json holds everything decide.py f9-diff judges. F9_ENGINE_CAP defaults to the capture in $R/runs/LAST-f9-engine.
if [ -z "${SLOTSTREAM_SRC:-}" ]; then
  exec bash tools/spark/snap.sh f9-ref bash bench/campaign/f9_reference_job.sh "$@"
fi
if [ -z "${F9_CAPTURED:-}" ]; then
  export F9_CAPTURED=1
  exec bash tools/capture.sh f9-ref -- bash bench/campaign/f9_reference_job.sh "$@"
fi
set -u
. tools/spark/env.sh
PY=tools/spark/py.sh
: "${CAP:?CAP is not set (the guard exports it)}"
R=${SLOTSTREAM_R:-$R}
LB=${LB/#\~/$HOME}
MODEL=${MODEL/#\~/$HOME}
OUTD=$R/model/q4_1-exact
TEMPLATE=$LB/gguf/Qwen3.8-Flash-Next-Q4_1
UNIT=slotstream-spark-f9-ref
SMOKE=${F9_REF_SMOKE:-0}
WINDOWS=32
[ "$SMOKE" = 1 ] && WINDOWS=1 && OUTD=$TEMPLATE
ENGINE_CAP=${F9_ENGINE_CAP:-$(cat "$R/runs/LAST-f9-engine")}
mkdir -p "$CAP/f9-ref"
stop() { echo "f9-ref: $1"; echo "$1" > "$CAP/stop.txt"; exit "${2:-1}"; }

# ---- 0 preflight
avail=$(df -BG --output=avail "$R" | tail -1 | tr -dc 0-9)
[ "$SMOKE" = 1 ] || [ "${avail:-0}" -ge 150 ] || stop "preflight: ${avail:-?} GB free at $R, 150 GB needed"
paused_load || stop "preflight: $R/state/PAUSED is malformed"
if [ "$PAUSED_NO_WINDOWS" != 1 ]; then
  [ "$(date +%H%M)" -lt 2345 ] || stop "preflight: after 23:45"
fi
bash tools/quality/fetch_wikitext.sh || stop "preflight: wikitext fetch failed"
bash tools/quality/build_ppl_ids.sh || stop "preflight: the ppl-ids helper did not build"
python3 - "$CAP/f9_job.json" "$ENGINE_CAP" "$SMOKE" <<'PY'
import json, sys
json.dump({"engine_capture": sys.argv[2], "smoke": sys.argv[3] == "1"}, open(sys.argv[1], "w"), indent=1, sort_keys=True)
PY

if [ "$SMOKE" != 1 ]; then
# ---- 1 exactness sweep
$PY tools/repack_q4_1.py --model "$MODEL" --sweep --json "$CAP/repack_q4_1_sweep.json" 2>&1 | tee "$CAP/sweep.log"
src=${PIPESTATUS[0]}
[ "$src" = 0 ] || stop "sweep: tensors not exact or not tolerated (rc $src): $(grep -m3 'NOT EXACT' "$CAP/sweep.log" | tr '\n' ' ')"
python3 - "$CAP/repack_q4_1_sweep.json" slotstream_spark/config/q4_1_exceptions.json <<'PY' || stop "sweep: the tolerated tensors differ from q4_1_exceptions.json" 
import json, sys
s, e = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
got = sorted(t["name"] for t in s["tolerated"])
want = sorted(e["tensors"])
if s["bad"] or got != want or any(t["max_abs_err"] > e["abs_tol"] for t in s["tolerated"]):
    print("sweep mismatch:", s["bad"][:5], got, want); sys.exit(1)
print("sweep ok:", s["tensors"], "tensors,", len(got), "tolerated")
PY

# ---- 1b mapping check
$PY tools/repack_q4_1.py --model "$MODEL" --template "$TEMPLATE" --out "$CAP/check" --check-only 2>&1 | tee "$CAP/check.log"
crc=${PIPESTATUS[0]}
if [ "$crc" != 0 ]; then
  nb=$(grep -c '^BELOW FLOOR:' "$CAP/check.log" || true)
  if [ "${nb:-0}" -gt 0 ]; then
    lst=$(grep '^BELOW FLOOR:' "$CAP/check.log" | sed 's/^BELOW FLOOR: //' | paste -sd ';' - | sed 's/;/; /g')
    echo "DECISION: Q4_1 mapping: $nb tensors under the 0.95 floor or unmappable: $lst — investigate the mapping or decide a floor"
  else
    echo "DECISION: Q4_1 mapping: tensors under the 0.95 floor or unmappable: $(grep -E 'Error|error' "$CAP/check.log" | tail -1) — investigate the mapping or decide a floor"
  fi
  stop "mapping check failed (rc $crc)"
fi

# ---- 2 build
$PY tools/repack_q4_1.py --model "$MODEL" --template "$TEMPLATE" --out "$OUTD" 2>&1 | tee "$CAP/build.log"
brc=${PIPESTATUS[0]}
[ "$brc" = 0 ] || stop "build failed (rc $brc)"
python3 - "$OUTD/manifest.json" "$CAP/repack_q4_1_sweep.json" <<'PY' || stop "build: the manifest's exceptions or type overrides differ from the plan"
import json, sys
m, s = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
want = sorted((t["name"], t["count"]) for t in s["tolerated"])
got = sorted((t["name"], t["count"]) for t in m["exceptions"])
hc = [t["name"] for t in m["tensors"] if t["name"].endswith(("hc_attn_down.weight", "hc_attn_up.weight", "hc_ffn_down.weight", "hc_ffn_up.weight",
                                                                "output_hc_down.weight", "output_hc_up.weight"))]
if got != want or m["type_overrides"] != ["blk.1.ple_conv1d.weight"] or sorted(m["dequantized_bf16"]) != sorted(hc) or len(hc) != 194:
    print("needs a decision: exceptions", got, "vs sweep", want, "overrides", m["type_overrides"], "bf16", len(m["dequantized_bf16"]), "vs hc", len(hc)); sys.exit(1)
print("manifest ok:", len(m["tensors"]), "tensors,", m["file_bytes"], "bytes")
PY
fi

# ---- 3 smoke
W=q41exact; [ "$SMOKE" = 1 ] && W=q41
PORT=8091
mf=$(awk '/^MemFree:/{print int($2/1024)}' /proc/meminfo)
bash deploy/lb/llamacpp.sh --weights "$W" --arm A --mode D5 --memfree-mib "$mf" --memfree-stop 14336 --memfree-log "$CAP/smoke-memfree.csv" \
  --port $PORT --unit $UNIT > "$CAP/smoke-server.log" 2>&1 &
LPID=$!
code=000
for i in $(seq 1 1800); do
  kill -0 $LPID 2>/dev/null || break
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null || echo 000)
  [ "$code" = 200 ] && break
  sleep 1
done
smoke_ok=0
if [ "$code" = 200 ]; then
  curl -s --max-time 300 "http://127.0.0.1:$PORT/completion" -d '{"prompt":"The capital of France is","n_predict":8,"temperature":0}' > "$CAP/smoke.json"
  python3 - "$CAP/smoke.json" <<'PY' && smoke_ok=1
import json, sys
d = json.load(open(sys.argv[1]))
print("smoke:", repr(d.get("content")))
sys.exit(0 if str(d.get("content", "")).strip() else 1)
PY
fi
kill -TERM $LPID 2>/dev/null
wait $LPID 2>/dev/null
for i in $(seq 1 120); do
  [ "$(systemctl --user is-active $UNIT.scope 2>/dev/null)" = active ] || break
  sleep 1
done
[ -f "$CAP/smoke-memfree.csv.stop" ] && stop "smoke: MemFree soft stop: $(cat "$CAP/smoke-memfree.csv.stop")" 5
[ "$smoke_ok" = 1 ] || stop "smoke: the llama-server did not answer (health $code); see smoke-server.log"

# ---- 4 differential
$PY tools/quality/llamacpp_ppl.py --engine-ppl "$ENGINE_CAP/quality/ppl.json" --windows $WINDOWS --weights $W \
  --out "$CAP/f9-ref" --memfree-log "$CAP/ppl-memfree.csv" --unit $UNIT 2>&1 | tee "$CAP/llamacpp_ppl.log"
prc=${PIPESTATUS[0]}

# ---- result.json
python3 - "$CAP" "$OUTD" "$SMOKE" "$prc" <<'PY'
import json, sys
from pathlib import Path
cap, outd, smoke, prc = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "1", int(sys.argv[4])
def load(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None
res = {"smoke_run": smoke, "ppl_rc": prc, "sweep": load(cap / "repack_q4_1_sweep.json"), "smoke": load(cap / "smoke.json"),
       "ppl": load(cap / "f9-ref" / "result.json")}
chk = load(cap / "check" / "check.json")
if chk:
    res["below_099"] = chk["below_099"]
    res["corr_floor"] = chk["corr_floor"]
    res["failures"] = chk["failures"]
    res["check_tensors"] = len(chk["tensors"])
man = load(outd / "manifest.json")
if man:
    res["manifest"] = {"tensors": len(man["tensors"]), "file_bytes": man["file_bytes"], "exceptions": man["exceptions"],
                       "type_overrides": man["type_overrides"], "dequantized_bf16": man.get("dequantized_bf16", [])}
(cap / "result.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
PY
exit "$prc"
