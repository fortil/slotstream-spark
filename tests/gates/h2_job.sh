#!/bin/bash
# H2 job body. Start it with:
#   tools/spark/jobs.sh start h2 --heavy h2 auto 14400 -- bash tests/gates/h2_job.sh
# It runs under the guard (CAP, SLOTSTREAM_FOOTPRINT_MIB are exported) and re-executes itself from a frozen copy of
# the source (tools/spark/snap.sh), so a later sync cannot change a running job.
#   $CAP/started.json       written first
#   $CAP/h2_model.json, h2_gdn.json, h2_oracle.json   one per part
#   $CAP/result.json        written last: the model part at top level plus {"gdn", "oracle", "exit"}
set -u
if [ "${1:-}" != inner ]; then
  : "${CAP:?h2_job.sh must run under the guard}"
  printf '{"label":"h2","started":"%s","footprint_mib":"%s","synced":"%s"}\n' "$(date -Iseconds)" \
    "${SLOTSTREAM_FOOTPRINT_MIB:-}" "$(cat "$(dirname "$0")/../../.synced" 2>/dev/null || echo unknown)" > "$CAP/started.json"
  exec bash tools/spark/snap.sh h2 bash tests/gates/h2_job.sh inner
fi

PY=tools/spark/py.sh
$PY -m pytest -q -m gate -p no:cacheprovider tests/gates/test_h2_model.py tests/gates/test_gdn_fused_vs_ref.py
rc1=$?
$PY tests/gates/h2_oracle_compare.py
rc2=$?
$PY - "$CAP" "$rc1" "$rc2" <<'PYE'
import json, os, sys
cap, rc1, rc2 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

def load(name):
    p = os.path.join(cap, name)
    return json.load(open(p)) if os.path.exists(p) else None

res = load("h2_model.json") or {}
res["gdn"] = load("h2_gdn.json")
res["oracle"] = load("h2_oracle.json")
res["exit"] = {"pytest": rc1, "oracle": rc2}
with open(os.path.join(cap, "result.json"), "w") as f:
    json.dump(res, f, indent=1, sort_keys=True)
PYE
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]
