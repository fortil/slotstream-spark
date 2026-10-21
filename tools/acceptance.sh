#!/bin/bash
# Final acceptance (plan step 99). Runs on the Spark inside the guard:
#   tools/spark/run.sh --heavy acceptance auto 10800 -- bash tools/acceptance.sh final
#   tools/acceptance.sh --list      the names of the checks, one per line (no Spark needed)
#   tools/acceptance.sh --emit-py   the Python of the record checks (tests compile and run it on the Mac)
#
# Checks, in order (each logs to $CAP/<name>.log; a failing check does not stop the next one):
#   claims-gate, runbook-check, docs-check     tools/claims_gate.py, tools/runbook_check.py, tools/docs_check.py
#   required-records                           every phase record has a PASS verdict and its rule lines; h5-agents binds to the current tree
#   oracle-parity31-calibration                present with Verdict: INFO (INFO by design, never PASS)
#   parity31-limits-rebind                     tests/gates/parity31_limits.json against its source capture and the gate capture
#   lb-yardstick-consistency                   the arm B yardstick, the stop/go record and the campaign justification agree
#   contract-rebind                            numeric_contract.toml against its calibration capture
#   t0-suite, t0-cuda-subset                   the unit tests (not cuda, not gate) and the cuda subset
#   gpu-fused-moe, gpu-attention-oracle        GPU unit gates
#   sweep-check, prefix-exact, decode-overlap, mtp-check, token-graph, persistent-restart   the reduced weight gates
#   live-acceptance                            serve_gates.sh suite: cold 8K TTFT, decode floor, API robustness, tools, images, probe
#   ppl-16-windows                             the first 16 wikitext windows, NLL bit-identical to the F9 record
#   self-impact                                MemAvailable, kernel log and production PSI over this run
# ACCEPTANCE_ONLY="name ..." runs just those checks and does not write the tracked record.
# A check that does not apply (the token graph is off by decision) exits 77 and shows SKIP. Exit 0 only when no check FAILed.
# Writes $CAP/acceptance.md and $SRC/db/records/measurements/acceptance.md (tools/spark/fetch.sh brings it back).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

CHECKS="claims-gate runbook-check docs-check required-records oracle-parity31-calibration parity31-limits-rebind lb-yardstick-consistency contract-rebind t0-suite t0-cuda-subset gpu-fused-moe gpu-attention-oracle sweep-check prefix-exact decode-overlap mtp-check token-graph persistent-restart live-acceptance ppl-16-windows self-impact"

emit_py() {
  cat <<'PYSCRIPT'
#!/usr/bin/env python3
"""Checks of tools/acceptance.sh that read records and captures (stdlib, Python >= 3.11).

  accept_checks.py CMD [--root DIR] [ARGS]
  CMD: records | info | parity31 | lb | contract | ppl16 NEW_PPL_JSON

Every command prints its findings and ends with one `RESULT: PASS ...` or `RESULT: FAIL ...` line; exit 0 only on PASS.
"""
import argparse
import hashlib
import json
import math
import re
import sys
import tomllib
from pathlib import Path

M = "db/records/measurements/"
D = "db/records/decisions/"
ROUTING_LINE = "Routing rule: seq_rule (factor 2.0, ceiling c_k/2), design decision 16"
LOGITS_LINE = ("Logits rule: seq KL (factor 3.0, floor = calibrated limit, ceilings 0.05/0.5/1.5 nats; "
               "top-1 calibrated), design decision 18")
LAYER_SEQ_RULE = ("L_seq = min(max(L_k, seq_factor*d_k + margin), c_k/seq_ceiling_ratio); d_k = metric(oracle bf16, "
                  "oracle fp32) on the judged sequence; asserted iff separable[k]")
LOGITS_SEQ_RULE = ("kl_X_seq = min(max(kl_X_max_nats, kl_factor*p_X), kl_X_ceiling_nats) for X in {mean, p99, max}; "
                   "p_X = statistic X of KL(oracle bf16 -> oracle fp32) on the judged sequence, same token ids as the "
                   "engine dump; top1_min stays calibrated; judgeable iff every p_X <= kl_X_ceiling_nats")
PASS_MEASUREMENTS = ["guard-drill", "r0", "goldens", "ppl-ablate", "m0prime", "h1", "h2", "h3", "h4-drills",
                     "f5-sweep-check", "f5-prefill", "f6-decode-d5", "f9-quality", "f9-campaign-d5", "f9-impact",
                     "f9-reference", "h5-agents", "oracle-calibration", "oracle-parity31"]
DECISIONS = ["h0", "stopgo-f3", "f6-lookahead-d5", "f6-mtp-d5", "f8-entry"]
RULE_LINES = {"h2": [ROUTING_LINE], "f5-sweep-check": [LOGITS_LINE], "f9-quality": [ROUTING_LINE, LOGITS_LINE]}


class Report:
    def __init__(self):
        self.fails, self.notes = [], []

    def need(self, cond, msg):
        if not cond:
            self.fails.append(msg)
            print("FAIL " + msg)
        return bool(cond)

    def note(self, msg):
        self.notes.append(msg)
        print("ok   " + msg)

    def finish(self, what):
        if self.fails:
            print("RESULT: FAIL %s: %d problems, first: %s" % (what, len(self.fails), self.fails[0]))
            return 1
        print("RESULT: PASS %s" % what)
        return 0


def lines_of(path):
    return Path(path).read_text().splitlines()


def field(lines, name):
    for ln in lines:
        if ln.startswith(name + ":"):
            return ln.split(":", 1)[1].strip()
    return None


def verdicts(lines):
    return [ln for ln in lines if ln.startswith("Verdict:")]


def load_db_check(root):
    sys.path.insert(0, str(Path(root) / "tools"))
    import db_check
    return db_check


def tree_sha(root):
    sys.path.insert(0, str(Path(root) / "tools"))
    import webui_sequence
    return webui_sequence.tree_sha256(Path(root))


def cmd_records(root, rep, extra):
    dbc = load_db_check(root)
    root = Path(root)
    for rid in PASS_MEASUREMENTS:
        p = root / (M + rid + ".md")
        if not rep.need(p.is_file(), "record missing: " + M + rid + ".md"):
            continue
        ls = lines_of(p)
        # Capture directories must exist; the line format of older records (no Capture line, a colon after PASS) is
        # not part of this gate, only the verdict word is.
        errs = [e for e in dbc.check_measurement(str(p), str(root)) if e.startswith("capture")]
        rep.need(not errs, "%s: %s" % (rid, errs))
        v = verdicts(ls)
        rep.need(len(v) == 1 and re.match(r"^Verdict: PASS\b", v[0]) is not None, "%s: Verdict is not PASS (%s)" % (rid, v))
        for want in RULE_LINES.get(rid, []):
            rep.need(any(want in ln for ln in ls), "%s: missing line %r" % (rid, want))
    for rid in DECISIONS:
        p = root / (D + rid + ".md")
        if not rep.need(p.is_file(), "record missing: " + D + rid + ".md"):
            continue
        ls = lines_of(p)
        dec = field(ls, "Decision")
        rep.need(bool(dec), "%s: no Decision" % rid)
        v = verdicts(ls)
        if v:
            rep.need(re.match(r"^Verdict: PASS\b", v[0]) is not None, "%s: Verdict is not PASS (%s)" % (rid, v))
    h0 = field(lines_of(root / (D + "h0.md")), "Decision") or ""
    rep.need(h0.startswith("H0_CLOSED"), "h0: Decision is not H0_CLOSED")
    sg = field(lines_of(root / (D + "stopgo-f3.md")), "Decision") or ""
    rep.need(sg.split()[0:1] in (["GO"], ["GO_ON_TTFT"]) if sg else False, "stopgo-f3: Decision is not GO or GO_ON_TTFT")
    if sg.startswith("GO_ON_TTFT"):
        rep.need("user-answers" in sg or "answer" in sg, "stopgo-f3: GO_ON_TTFT without an answer reference")
    entry = field(lines_of(root / (D + "f8-entry.md")), "Decision") or ""
    if entry.strip() == "enter":
        p = root / (D + "f8-token-graph.md")
        if rep.need(p.is_file(), "f8-entry says enter but f8-token-graph.md is missing"):
            v = verdicts(lines_of(p))
            rep.need(len(v) == 1 and re.match(r"^Verdict: PASS\b", v[0]) is not None, "f8-token-graph: Verdict is not PASS")
    # f9-quality: the two rule lines and two Capture lines
    q = lines_of(root / (M + "f9-quality.md"))
    rep.need(any(ln.startswith("Top-1 rule: pooled over the 8 reserved prompts") for ln in q), "f9-quality: no 'Top-1 rule:' line")
    rep.need(any(ln.startswith("PPL rule: engine vs transformers oracle") for ln in q), "f9-quality: no 'PPL rule:' line")
    caps = [ln for ln in q if ln.startswith("Capture:")]
    rep.need(len(caps) == 2, "f9-quality: expected two Capture lines, found %d" % len(caps))
    rep.need(not dbc.check_measurement(str(root / (M + "f9-quality.md")), str(root)), "f9-quality is not db_check clean")
    # f9-reference: the exceptions and the type overrides
    ref = (root / (M + "f9-reference.md")).read_text()
    rep.need("exact except the 7 named exceptions" in ref, "f9-reference: does not name the 7 exceptions")
    rep.need("type_overrides" in ref, "f9-reference: no type_overrides")
    # h5-agents: the server tree it judged is the current one
    h5 = (root / (M + "h5-agents.md")).read_text()
    m = re.search(r"server_tree_sha256: ([0-9a-f]{64})", h5)
    cur = tree_sha(root)
    rep.need(bool(m) and m.group(1) == cur, "h5-agents: server_tree_sha256 %s is not the current tree %s"
             % (m.group(1) if m else None, cur))
    rep.note("%d measurement records and %d decision records checked" % (len(PASS_MEASUREMENTS), len(DECISIONS)))
    return rep.finish("required-records")


def cmd_info(root, rep, extra):
    dbc = load_db_check(root)
    p = Path(root) / (M + "oracle-parity31-calibration.md")
    if rep.need(p.is_file(), "record missing: oracle-parity31-calibration.md"):
        rep.need(not dbc.check_file(str(p), str(root)), "oracle-parity31-calibration is not db_check clean")
        v = verdicts(lines_of(p))
        rep.need(len(v) == 1 and v[0].startswith("Verdict: INFO"), "oracle-parity31-calibration: Verdict is not INFO (%s)" % v)
    return rep.finish("oracle-parity31-calibration present with INFO")


def cmd_parity31(root, rep, extra):
    root = Path(root)
    dbc = load_db_check(root)
    path = root / "tests/gates/parity31_limits.json"
    lim = json.loads(path.read_text())
    src = root / lim["source_capture"]
    res = json.loads((src / "result.json").read_text())
    by_layer = {x["layer"]: x for x in res["layers"]}
    rep.need(len(lim["layers"]) == 4 and len(by_layer) == 4, "parity31: expected 4 layers in the limits and in the capture")
    for x in lim["layers"]:
        r = by_layer.get(x["layer"])
        if not rep.need(r is not None, "layer %s missing from %s" % (x["layer"], src)):
            continue
        c = min(r["control_noexp"], r["control_swap"])
        rep.need(x["spread"] == r["spread_bf16_fp32"], "layer %d: spread differs from the capture" % x["layer"])
        rep.need(x["control"] == c, "layer %d: control differs from min(noexp, swap)" % x["layer"])
        rep.need(abs(x["limit"] - (2 * x["spread"] + 0.01)) < 1e-12, "layer %d: limit is not 2*spread + 0.01" % x["layer"])
        rep.need(x["separable"] is True, "layer %d is not separable" % x["layer"])
    cap_line = field(lines_of(root / (M + "oracle-parity31.md")), "Capture")
    latest = dbc.latest("oracle-p31", root)
    rep.need(latest is not None, "no oracle-p31 capture under db/sources/runs")
    if latest is not None:
        rel = str(latest.relative_to(root))
        rep.need(rel == (cap_line or "").rstrip("/"), "latest oracle-p31 capture %s is not the one named in oracle-parity31.md (%s)" % (rel, cap_line))
        gate = json.loads((latest / "result.json").read_text())
        rep.need(gate.get("pass") is True, "oracle-p31 result.json: pass is not true")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        rep.need(gate.get("limits_sha256") == sha, "oracle-p31 limits_sha256 %s != sha256 of the tracked limits %s" % (gate.get("limits_sha256"), sha))
    print("parity31-limits-rebind | layers %d | source %s" % (len(lim["layers"]), lim["source_capture"]))
    return rep.finish("parity31-limits-rebind")


def cmd_lb(root, rep, extra):
    root = Path(root)
    sys.path.insert(0, str(root / "tools"))
    import yardstick
    # table and record names are part of the recorded decision
    p = str(root / "db/records/decisions/vara-d5.md")
    try:
        y = yardstick.yardstick(p)
        blocks = yardstick.load_record(p)
    except Exception as e:                      # noqa: BLE001
        rep.need(False, "vara-d5.md: %s" % e)
        return rep.finish("lb-yardstick-consistency")
    rep.need(y.get("arm") == "q41-b-d5", "yardstick arm is %r, not q41-b-d5" % y.get("arm"))
    rep.need(y.get("op_offload") is False, "yardstick op_offload is not false")
    rep.need(y.get("prefill_device") == "cpu", "yardstick prefill_device is not cpu")
    rep.need(bool(y.get("decode_tps_median")) and bool(y.get("prefill_tps_8k")), "yardstick lacks decode_tps_median or prefill_tps_8k")
    v = blocks.get("vara", {})
    rep.need(v.get("op_offload") is False and v.get("prefill_device") == "cpu", "[vara] block lost op_offload=false or prefill_device=cpu")
    sg = (root / (D + "stopgo-f3.md")).read_text()
    dec = field(sg.splitlines(), "Decision") or ""
    rep.need(dec.startswith("GO_ON_TTFT") or dec.split()[0:1] == ["GO"], "stopgo-f3: Decision is not GO or GO_ON_TTFT")
    rep.need("q41-b-d5" in sg, "stopgo-f3 does not name q41-b-d5")
    cq = field(sg.splitlines(), "Compared quantity") or ""
    rep.need(cq.lower().startswith("decode"), "stopgo-f3: the compared quantity is not decode")
    camp = (root / (M + "f9-campaign-d5.md")).read_text().splitlines()
    just = field(camp, "Justification")
    rep.need(bool(just) and just.split()[0] != "none", "f9-campaign-d5: no Justification other than none")
    if just and re.match(r"^ttft\b", just):
        rep.need(dec.startswith("GO_ON_TTFT"), "f9-campaign-d5 justifies on ttft but stopgo-f3 is not GO_ON_TTFT")
    ref = (root / (M + "f9-reference.md")).read_text()
    rep.need("prefill_device" in ref, "f9-reference does not name prefill_device")
    print("lb-yardstick-consistency | arm %s | decode %s | justification %s" % (y.get("arm"), y.get("decode_tps_median"), just))
    return rep.finish("lb-yardstick-consistency")


def close(a, b):
    return abs(a - b) <= 1e-12


def cmd_contract(root, rep, extra):
    root = Path(root)
    sys.path.insert(0, str(root / "tools"))
    import calibrate_contract as cc
    cfg = tomllib.loads((root / "slotstream_spark/config/numeric_contract.toml").read_text())
    cap_line = field(lines_of(root / (M + "oracle-calibration.md")), "Capture")
    calib_path = root / (cap_line or "missing") / "calibration.json"
    if not rep.need(calib_path.is_file(), "calibration.json missing at %s" % calib_path):
        return rep.finish("contract-rebind")
    calib = json.loads(calib_path.read_text())
    arrays = cc.derive(calib, cfg)
    for table in ("layer", "routing"):
        for key in cc.ARRAYS:
            got, want = cfg[table][key], arrays[table][key]
            if not rep.need(len(got) == len(want), "[%s] %s: length %d != %d" % (table, key, len(got), len(want))):
                continue
            if key == "separable":
                rep.need(list(got) == list(want), "[%s] separable differs from the recomputation" % table)
            else:
                bad = [i for i, (g, w) in enumerate(zip(got, want)) if not close(g, w)]
                rep.need(not bad, "[%s] %s differs at layers %s" % (table, key, bad[:8]))
    lg = calib["logits"]
    limits, bad = cc.rule_logits(lg, cfg["logits"])
    rep.need(not bad, "logits limits outside the validity range: %s" % bad)
    for key, val in limits.items():
        rep.need(close(cfg["logits"][key], val), "[logits] %s %r != recomputed %r" % (key, cfg["logits"][key], val))
    for key, src in (("o_top1", "top1"), ("o_kl_mean_nats", "kl_mean"), ("o_kl_p99_nats", "kl_p99"),
                     ("o_kl_max_nats", "kl_max"), ("o_positions", "positions")):
        rep.need(cfg["logits"][key] == lg[src], "[logits] %s != capture %s" % (key, src))
    rep.need(close(cfg["fixture"]["kl_max_nats"], cfg["logits"]["kl_max_nats"]), "[fixture] kl_max_nats != [logits] kl_max_nats")
    for table in ("layer", "routing"):
        rep.need(cfg[table]["seq_factor"] == 2.0 and cfg[table]["seq_ceiling_ratio"] == 2.0, "[%s] seq_factor/seq_ceiling_ratio changed" % table)
        rep.need(cfg[table]["seq_rule"] == LAYER_SEQ_RULE, "[%s] seq_rule string changed" % table)
    rep.need(cfg["logits"]["seq_rule"] == LOGITS_SEQ_RULE, "[logits] seq_rule string changed")
    print("contract-rebind | capture %s | separable layers %d/%d" % (cap_line, sum(cfg["layer"]["separable"]), len(cfg["layer"]["separable"])))
    return rep.finish("contract-rebind")


def cmd_ppl16(root, rep, extra):
    root = Path(root)
    new = json.loads(Path(extra[0]).read_text())
    q = lines_of(root / (M + "f9-quality.md"))
    stored = None
    for ln in q:
        if ln.startswith("Capture:"):
            p = root / ln.split(":", 1)[1].strip() / "quality/ppl.json"
            if p.is_file():
                stored, src = json.loads(p.read_text()), p
                break
    if not rep.need(stored is not None, "no f9-quality Capture holds quality/ppl.json"):
        return rep.finish("ppl-16-windows")
    a, b = new["per_window"][:16], stored["per_window"][:16]
    rep.need(len(a) == 16, "the smoke run scored %d windows, not 16" % len(a))
    rep.need(new.get("ids_sha256") == stored.get("ids_sha256"), "ids_sha256 differs from the F9 record")
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x["nll"] != y["nll"] or x["start"] != y["start"]]
    rep.need(not diff, "NLL differs from %s at windows %s" % (src, diff[:8]))
    print("ppl-16-windows | windows %d | bit-identical %s" % (len(a), not diff))
    return rep.finish("ppl-16-windows")


CMDS = {"records": cmd_records, "info": cmd_info, "parity31": cmd_parity31, "lb": cmd_lb, "contract": cmd_contract,
        "ppl16": cmd_ppl16}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=sorted(CMDS))
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("extra", nargs="*")
    a = ap.parse_intermixed_args(argv)
    rep = Report()
    try:
        return CMDS[a.cmd](a.root, rep, a.extra)
    except Exception as e:                      # noqa: BLE001 - an unreadable input is a failed check, not a crash
        rep.need(False, "%s: %s: %s" % (a.cmd, type(e).__name__, e))
        return rep.finish(a.cmd)


if __name__ == "__main__":
    sys.exit(main())
PYSCRIPT
}

case "${1:-}" in
  --list) for c in $CHECKS; do echo "$c"; done; exit 0 ;;
  --emit-py) emit_py; exit 0 ;;
  final) ;;
  *) echo "usage: acceptance.sh {final|--list|--emit-py}" >&2; exit 2 ;;
esac

PARTIAL=0
if [ -n "${ACCEPTANCE_ONLY:-}" ]; then      # a re-run of some checks while a failure is fixed: never writes the tracked record
  for c in $ACCEPTANCE_ONLY; do
    case " $CHECKS " in *" $c "*) ;; *) echo "acceptance: unknown check '$c'" >&2; exit 2 ;; esac
  done
  CHECKS=$ACCEPTANCE_ONLY; PARTIAL=1
fi
. tools/spark/env.sh
PY=tools/spark/py.sh
START=$(date +%s)
BUDGET_S=${ACCEPTANCE_BUDGET_S:-10500}      # no new check starts after this; the guard cuts the run at 10800 s
OUT=${CAP:-}
if [ -z "$OUT" ]; then OUT=$R/runs/$(date +%Y/%m)/$(date +%Y%m%dT%H%M%S)-acceptance; fi
mkdir -p "$OUT"
export CAP=$OUT
emit_py > "$OUT/accept_checks.py"
TABLE=$OUT/table.tsv
: > "$TABLE"

# ---- samplers for self-impact: 1 s MemAvailable, 2 s production cgroup pressure (read-only)
paused_load || true
SAMPLERS=()
(
  echo "epoch,MemAvailable_kB" > "$OUT/accept-meminfo.csv"
  while :; do
    awk -v e="$(date +%s)" '$1=="MemAvailable:"{print e","$2}' /proc/meminfo >> "$OUT/accept-meminfo.csv"
    sleep 1
  done
) & SAMPLERS+=($!)
PSI_CONTAINERS=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -v '^slotstream-spark' | tr '\n' ' ')
(
  echo "epoch,container,io_some_avg10,mem_some_avg10" > "$OUT/accept-psi.csv"
  while :; do
    for c in $PSI_CONTAINERS; do
      id=$(docker inspect --format '{{.Id}}' "$c" 2>/dev/null) || continue
      d=/sys/fs/cgroup/system.slice/docker-$id.scope
      io=$(sed -n 's/^some .*avg10=\([0-9.]*\).*/\1/p' "$d/io.pressure" 2>/dev/null)
      mem=$(sed -n 's/^some .*avg10=\([0-9.]*\).*/\1/p' "$d/memory.pressure" 2>/dev/null)
      echo "$(date +%s),$c,${io:-},${mem:-}" >> "$OUT/accept-psi.csv"
    done
    sleep 2
  done
) & SAMPLERS+=($!)
stop_samplers() {
  local p
  for p in "${SAMPLERS[@]}"; do pkill -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; done
  for p in "${SAMPLERS[@]}"; do wait "$p" 2>/dev/null; done
  SAMPLERS=()
} 2>/dev/null
trap stop_samplers EXIT

# ---- the checks
gate_py() { $PY "$OUT/accept_checks.py" "$@" --root "$ROOT"; }

c_claims_gate() { $PY tools/claims_gate.py; }
c_runbook_check() { $PY tools/runbook_check.py RUNBOOK.md; }
c_docs_check() { $PY tools/docs_check.py; }
c_required_records() { gate_py records; }
c_oracle_parity31_calibration() { gate_py info; }
c_parity31_limits_rebind() { gate_py parity31; }
c_lb_yardstick_consistency() { gate_py lb; }
c_contract_rebind() { gate_py contract; }
# T0 on the Spark. SLOTSTREAM_PAUSED_FILE points at a file that does not exist: the unit tests expect production not
# paused, and the Spark holds the real $R/state/PAUSED while the user has production paused. Three tests need inputs
# that only the Mac has and are deselected by name (they run in the Mac suite of the same tree):
#   test_extract_trunk.py::test_main_trunk, test_repack_experts.py::test_resume_gives_identical_file   numpy bfloat16 files
#       through safetensors (the Spark venv has no ml_dtypes) and a resume on the Spark filesystem
#   test_repack_q4_1.py::test_the_tracked_exceptions_file   reads third_party/qwen38-flash-next-metadata/tensors.csv
T0_DESELECT="--deselect tests/t0/test_extract_trunk.py::test_main_trunk --deselect tests/t0/test_repack_experts.py::test_resume_gives_identical_file --deselect tests/t0/test_repack_q4_1.py::test_the_tracked_exceptions_file"
c_t0_suite() {
  echo "deselected (Mac-only inputs): $T0_DESELECT"
  SLOTSTREAM_PAUSED_FILE=$OUT/no-pause-file $PY -m pytest -q -m "not cuda and not gate" tests $T0_DESELECT
}
c_t0_cuda_subset() { SLOTSTREAM_PAUSED_FILE=$OUT/no-pause-file $PY -m pytest -q -m "cuda and not gate" tests/t0; }
c_gpu_fused_moe() { $PY -m pytest -q -m "cuda or gate" tests/gates/test_fused_moe_affine.py; }
c_gpu_attention_oracle() { $PY -m pytest -q -m "cuda or gate" tests/gates/test_prefill_attention_oracle.py; }
c_sweep_check() { $PY -m pytest -q -m "cuda or gate" tests/gates/test_sweep_check.py; }
c_prefix_exact() { $PY -m pytest -q -m "cuda or gate" tests/t0/test_prefill_driver_control.py tests/gates/test_prefix_exact.py; }
c_decode_overlap() { $PY -m pytest -q -m "cuda or gate" tests/gates/test_decode_overlap.py; }
c_mtp_check() {
  if ! grep -q '^Decision: ADOPT_RESIDENT' db/records/decisions/f6-mtp-d5.md; then echo "MTP is not enabled by the f6-mtp-d5 decision"; return 77; fi
  $PY -m pytest -q -m "cuda or gate" tests/gates/test_mtp_check.py
}
c_token_graph() {
  if ! grep -q '^Decision: *\(on\|enable\|adopt\)' db/records/decisions/f8-token-graph.md; then
    echo "the token graph is off by decision ($(grep -m1 '^Decision:' db/records/decisions/f8-token-graph.md))"; return 77
  fi
  $PY -m pytest -q -m "cuda or gate" tests/gates/test_token_graph.py
}
c_persistent_restart() { $PY -m pytest -q -m "cuda or gate" tests/gates/test_persistent_restart.py; }
c_live_acceptance() { bash tools/serve_gates.sh "${ACCEPTANCE_PORT:-18439}" live-acceptance; }
c_ppl_16_windows() {
  mkdir -p "$OUT/ppl16"
  $PY tools/quality/ppl_wikitext.py --mode D5 --windows 16 --out "$OUT/ppl16" || return 1
  gate_py ppl16 "$OUT/ppl16/ppl.json"
}
c_self_impact() {
  stop_samplers
  local rc=0 min psi n
  min=$(awk -F, 'NR>1 && $2!="" {if (m=="" || $2<m) m=$2} END{print m}' "$OUT/accept-meminfo.csv")
  echo "MemAvailable minimum over the run: ${min:-none} kB ($(( ${min:-0} / 1024 )) MiB), samples $(($(wc -l < "$OUT/accept-meminfo.csv") - 1))"
  if [ -z "$min" ] || [ "$min" -lt 8388608 ]; then echo "FAIL MemAvailable fell under 8 GiB"; rc=1; fi
  n=$(journalctl -k --since "@$START" --no-pager 2>/dev/null | grep -Eci 'out of memory|oom-kill|oom_kill|NV_ERR_NO_MEMORY')
  echo "oom-kill or NV_ERR_NO_MEMORY lines in the kernel log since the start: ${n:-0}"
  [ "${n:-0}" = 0 ] || { echo "FAIL kernel log reports memory kills"; rc=1; }
  if [ "$PAUSED_ACTIVE" = 1 ]; then
    echo "production PSI: not_applicable: production paused by the user"
  else
    psi=$(awk -F, 'NR>1 {for (i=3;i<=4;i++) if ($i!="" && $i+0>m) m=$i+0} END{print m+0}' "$OUT/accept-psi.csv")
    echo "production cgroup PSI some avg10 maximum: $psi"
    awk -v p="$psi" 'BEGIN{exit !(p < 10)}' || { echo "FAIL production PSI reached 10"; rc=1; }
  fi
  return $rc
}

ROWS_FAIL=0
run_check() {
  local name=$1 fn rc t0 t1 status detail
  fn=c_${name//-/_}
  t0=$(date +%s)
  if [ $((t0 - START)) -gt "$BUDGET_S" ]; then
    echo "not started: the time budget of $BUDGET_S s is spent" > "$OUT/$name.log"; rc=124
  else
    echo "=== $name: $(date -Is)"
    $fn > "$OUT/$name.log" 2>&1
    rc=$?
  fi
  t1=$(date +%s)
  case $rc in 0) status=PASS ;; 77) status=SKIP ;; *) status=FAIL; ROWS_FAIL=$((ROWS_FAIL + 1)) ;; esac
  detail=$(grep -h '^RESULT:' "$OUT/$name.log" | tail -n 1)
  [ -n "$detail" ] || detail=$(grep -hE '^(=+ )?[0-9]+ (passed|failed)|^[0-9]+ passed|passed in|failed,' "$OUT/$name.log" | tail -n 1)
  [ -n "$detail" ] || detail=$(grep -hE '^serve_gates: |: passed [0-9]+|rc=' "$OUT/$name.log" | tail -n 1)
  [ -n "$detail" ] || detail=$(grep -v '^[[:space:]]*$' "$OUT/$name.log" | tail -n 1)
  # no local paths in a tracked record
  detail=$(printf '%s' "$detail" | sed -e "s|$R|<R>|g" -e 's|/home/[^ )]*|<path>|g' | tr '|\t' '  ' | cut -c1-200)
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$status" "$rc" "$((t1 - t0))" "$detail" >> "$TABLE"
  echo "    $status rc=$rc $((t1 - t0)) s  $detail"
}

for name in $CHECKS; do run_check "$name"; done
stop_samplers

echo
echo "acceptance: result table"
echo "| check | status | rc | seconds | detail |"
echo "|---|---|---|---|---|"
awk -F'\t' '{printf "| %s | %s | %s | %s | %s |\n", $1, $2, $3, $4, $5}' "$TABLE"

# ---- the record
CAPREL=${OUT#"$R/runs/"}
REC=$ROOT/db/records/measurements/acceptance.md
total=$(( $(date +%s) - START ))
if [ "$ROWS_FAIL" = 0 ]; then VERDICT="PASS — every acceptance check passed or was skipped by decision (see the table)"
else VERDICT="FAIL — $ROWS_FAIL acceptance checks failed (see the table)"; fi
{
  echo "# acceptance: final acceptance on the Spark"
  echo "Date: $(date +%F)"
  echo "Capture: db/sources/runs/$CAPREL/"
  echo "Command: tools/spark/run.sh --heavy acceptance auto 10800 -- bash tools/acceptance.sh final"
  echo
  echo "| check | status | rc | seconds | detail |"
  echo "|---|---|---|---|---|"
  awk -F'\t' '{printf "| %s | %s | %s | %s | %s |\n", $1, $2, $3, $4, $5}' "$TABLE"
  echo
  echo "- total $total s; skipped by decision: $(awk -F'\t' '$2=="SKIP"{printf "%s ", $1}' "$TABLE")"
  echo "Verdict: $VERDICT"
} > "$OUT/acceptance.md"
if [ "$PARTIAL" = 0 ]; then
  mkdir -p "$(dirname "$REC")"
  cp "$OUT/acceptance.md" "$REC"
  echo "record: $REC"
else
  echo "partial run (ACCEPTANCE_ONLY): record kept in $OUT/acceptance.md only"
fi
[ "$ROWS_FAIL" = 0 ]
