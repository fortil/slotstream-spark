#!/usr/bin/env python3
"""Gate judgements and records for F2-F4, written on the Mac from the Spark's result.json.

  records.py m0prime RESULT_JSON [--note TEXT]... [--write db/records/measurements/m0prime.md]
  records.py h1      RESULT_JSON [--note TEXT]... [--write db/records/measurements/h1.md]
  records.py h2      RESULT_JSON [--note TEXT]... [--write db/records/measurements/h2.md]
  records.py sweep-check SWEEP_CHECK_JSON [--note TEXT]... [--write db/records/measurements/f5-sweep-check.md]
  records.py partition RESULT_JSON [--note TEXT]... [--write db/records/measurements/partition-d5.md]
  records.py soak    RESULT_JSON [--note TEXT]... [--write db/records/measurements/soak-safe-path.md]
  records.py h3      RESULT_JSON [--note TEXT]... [--write db/records/measurements/h3.md]
  records.py stopgo  RESULT_JSON [--write db/records/decisions/stopgo-f3.md] [--user-decision continue-on-ttft|stop --answer REF]
  records.py h4      CAPTURE_DIR... [--note TEXT]... [--write db/records/measurements/h4-drills.md]

Exit 0 = every gate passed, 1 = a gate failed (a `DECISION:` line names the value and what to revert), 3 = unreadable
input. The record is written in both cases (Verdict PASS or FAIL) so the evidence is kept.
Numeric limits come from slotstream_spark/config/numeric_contract.toml; the GEMV, tau and footprint thresholds come
from bench/m0/check_gates.py. No literal numeric limit lives here except the plan's own gate numbers below.

Rules amended on 2026-10-12 for M0' (user answer "Aceptar y seguir", BRIEF.md "User answers"):
  - the GEMV 2560x2560 4-bit row is informational at M0'; the hard gate (>= 121.5 GB/s, cold-L2 graph) is judged at
    step 55 on the tuned engine kernel;
  - the runtime footprint ceiling at M0' is M0PRIME_MAX_FOOTPRINT (measured + 10%); the design budget 2.5 GB stays
    informational and the ledger reads platform.toml measured.runtime_footprint_mib.
Rule amended on 2026-10-12 for H1 (user answer "Contar cada byte una vez"): the (d) peak is footprint.peak() over the
per-phase samples, the pool subtracted once per counter where it shows (like the M0' footprint); once_bytes is printed.
Rule for H2 (design decision 16, user approval 2026-10-12): the oracle comparison on the engine's own sequences uses the
per-sequence limits of tests/gates/contract.py (seq_limit) computed by h2_oracle_compare.py; the peak is count-once and
gated against the run's cap; the ledger envelope is gated against the D5 design ceiling, never against the cap.
"""
import json
import sys
import tomllib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "bench" / "m0", ROOT / "tests" / "gates"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from slotstream_spark.config import modes  # noqa: E402
from slotstream_spark.memory import footprint  # noqa: E402
from slotstream_spark.memory.ledger import Ledger  # noqa: E402
import check_gates as cg  # noqa: E402
import perf_model as pm  # noqa: E402

CONTRACT = ROOT / "slotstream_spark" / "config" / "numeric_contract.toml"
DECISION_DOC = "the technology decision"
PROJECTED_MIN_TPS = 17.0          # plan: D5 at h = 0.98, phi = 0 (H3 design gate)
FREE_FLOOR_S = 2.0                # 47 x chunk free p99 + 0.25 s
FREE_EXTRA_S = 0.25
H_D5 = 0.98
H1_PEAK_LIMIT_MIB = 4096
H1_READ_MAX_S = 2.0
H2_RULE_LINE = "Routing rule: seq_rule (factor 2.0, ceiling c_k/2), design decision 16"
H2_GDN_DECISION = ("DECISION: fused GDN fails the contract at full model; reference/chunk recurrence costs "
                   f"~4.3 ms/token ({DECISION_DOC})")
LAYERS = 48
# M0' rule of 2026-10-12. The design's 2.5e9 B was a budget set before a CUDA 13 + torch + Triton engine process
# existed; the measured M0' value is 2.64e9 B (a fixed per-process cost, the pool is subtracted per counter). The
# ceiling is measured + 10%: below one more CUDA context (1,267 MiB, m0-host.md), far below one leaked pool chunk
# (1 GiB) or a pool double count, and above the run-to-run jitter of the runtime terms.
M0PRIME_MAX_FOOTPRINT = 2.9e9
DESIGN_FOOTPRINT = 2.5e9


def load_contract():
    with open(CONTRACT, "rb") as f:
        return tomllib.load(f)


def project(res, h, phi):
    """tok/s with the measured t_comp_48 (replaces the formula t_comp of the plan model)."""
    t_memcpy = res["fill"]["t_memcpy_mean_ms"]
    t = (res["timings"]["t_comp_48_ms"] + pm.t_host_ms(res["tau"]["p50_us"])
         + (1.0 - phi) * (pm.t_io_ms(h) + pm.t_copy_ms(h, t_memcpy)))
    return 1000.0 / t


def evaluate_m0prime(res, contract):
    """-> (rows [(name, value, limit, ok)], decisions [str], info [str])."""
    rows, decisions, info = [], [], []
    g = res["gemv"]["2560x2560_b4"]["gbs_cold_graph"]
    rows.append(("GEMV 2560x2560 4-bit, cold-L2 graph (GB/s) — informational at M0', hard gate "
                 f">= {cg.GEMV_MIN_GBS} at step 55", g, f"info (>= {cg.GEMV_MIN_GBS} at step 55)", True))
    info.append(f"GEMV engine kernel {g:.2f} GB/s at M0' against the {cg.GEMV_MIN_GBS} GB/s gate judged at step 55 "
                "(step-23 bench kernel 193.75 GB/s, m0-gemv.md); untuned engine config at this capture")
    tau = res["tau"]["p50_us"]
    ok = tau <= cg.HOST_MAX_TAU_US
    rows.append(("tau p50 (us)", tau, f"<= {cg.HOST_MAX_TAU_US:.0f}", ok))
    info.append(f"tau p99 {res['tau']['p99_us']} us; p50 parts: wait {res['tau']['wait_p50_us']}, ensure "
                f"{res['tau']['ensure_p50_us']}, launch {res['tau']['launch_p50_us']} us")
    if not ok:
        decisions.append(f"DECISION: tau p50 {tau:.1f} us > {cg.HOST_MAX_TAU_US:.0f} us; revert per {DECISION_DOC} "
                         "(layer loop to C++ with cudaGraphLaunch inside slotio; bring F8 forward)")
    foot = res["runtime"]["footprint_bytes"]
    ok = foot <= M0PRIME_MAX_FOOTPRINT
    rows.append(("runtime footprint (B)", foot,
                 f"<= {M0PRIME_MAX_FOOTPRINT:.3g} B (M0' rule of 2026-10-12; design budget "
                 f"{DESIGN_FOOTPRINT:.3g} informational)", ok))
    info.append(f"runtime footprint {foot / 1e9:.3f} GB vs the design budget {DESIGN_FOOTPRINT / 1e9:.1f} GB "
                f"({DECISION_DOC}): accepted on 2026-10-12; the ledger uses platform.toml "
                "measured.runtime_footprint_mib")
    if not ok:
        decisions.append(f"DECISION: runtime footprint {foot / 1e9:.2f} GB > {M0PRIME_MAX_FOOTPRINT / 1e9:.1f} GB; "
                         f"revert per {DECISION_DOC} (CUDA_MODULE_LOADING=LAZY, no cuBLAS workspaces, allocator cap; "
                         "else C++ decode)")
    lay, sep, lim = contract["layer"], contract["layer"]["separable"], contract["layer"]["limits"]
    per = res["gdn_ab"]["rel_l2_by_layer"]
    gdn_ok = True
    for k in (0, 1, 2):
        v = per[str(k)]
        if sep[k]:
            ok = v <= lim[k]
            gdn_ok = gdn_ok and ok
            rows.append((f"GDN fused vs ref rel_l2, layer {k}", v, f"<= {lim[k]:.5g}", ok))
        else:
            rows.append((f"GDN fused vs ref rel_l2, layer {k} (not separable: reported)", v, f"{lim[k]:.5g}", True))
    info.append(f"layer 3 (QSA) rel_l2 of the same A/B, reported: {per['3']:.5g}")
    if not gdn_ok:
        decisions.append(f"DECISION: fused GDN misses the numeric contract against the reference recurrence over "
                         f"{res['gdn_ab']['positions']} positions; revert per {DECISION_DOC} (decode GDN with the "
                         "reference or chunked form, t_comp recomputed)")
    klm = contract["logits"]["kl_mean_max_nats"]
    kl = res["gdn_ab"]["kl_mean"]
    ok = kl <= klm and res["gdn_ab"]["finite"]
    rows.append(("KL mean, truncated logits fused vs ref (nats)", kl, f"<= {klm:.5g}", ok))
    info.append(f"truncated logits fused vs ref, printed not asserted: top-1 {res['gdn_ab']['top1']:.4f}, "
                f"KL p99 {res['gdn_ab']['kl_p99']:.5g}, max {res['gdn_ab']['kl_max']:.5g}")
    if not ok:
        decisions.append(f"DECISION: KL mean {kl:.5g} nats (finite={res['gdn_ab']['finite']}) > {klm:.5g}; revert per "
                         f"{DECISION_DOC} (GDN decode with the reference or chunked form)")
    proj = project(res, H_D5, 0.0)
    ok = proj >= PROJECTED_MIN_TPS
    rows.append((f"projected D5 decode, h = {H_D5}, phi = 0 (tok/s)", proj, f">= {PROJECTED_MIN_TPS}", ok))
    if not ok:
        decisions.append(f"DECISION: projected D5 decode {proj:.1f} tok/s < {PROJECTED_MIN_TPS} at h = {H_D5}, phi = 0; "
                         "stop and ask the user (H3 design gate)")
    p99 = res["chunk_free"]["p99_ns"]
    floor = 47 * p99 * 1e-9 + FREE_EXTRA_S
    ok = floor <= FREE_FLOOR_S
    rows.append(("47 x chunk free p99 + 0.25 s (s)", floor, f"<= {FREE_FLOOR_S}", ok))
    if not ok:
        decisions.append(f"DECISION: floor time {floor:.2f} s > {FREE_FLOOR_S} s (chunk free p99 {p99 / 1e6:.2f} ms); "
                         "stop and ask the user (governor floor)")
    qsa = res["timings"]["layer_steps"]["3"]
    if qsa["mode"] != "graph":
        info.append(f"t_QSA was measured eagerly (graph capture failed: {qsa['note']}); it includes host launch time")
    return rows, decisions, info


def table_3_8(res):
    out = ["| mode | h | no lookahead tok/s | lookahead tok/s (phi 0.3 at low h, 0.5 at high h) |", "|---|---|---|---|"]
    for mode, (lo, hi) in pm.MODES.items():
        nl = (project(res, lo, 0.0), project(res, hi, 0.0))
        la = (project(res, lo, 0.3), project(res, hi, 0.5))
        out.append(f"| {mode} | {lo:.2f}-{hi:.2f} | {nl[0]:.1f}-{nl[1]:.1f} | {la[0]:.1f}-{la[1]:.1f} |")
    return out


def capture_rel(result_path):
    p = Path(result_path).resolve()
    try:
        return str(p.parent.relative_to(ROOT)) + "/"
    except ValueError:
        return str(p.parent) + "/"


def fmt(v):
    return f"{v:.5g}" if isinstance(v, float) else str(v)


def write_m0prime(res, rows, decisions, info, result_path, dest, notes=()):
    ok = all(r[3] for r in rows)
    f, t = res["fill"], res["timings"]
    lines = ["# m0prime: four real layers, design triggers and the recalibrated performance table",
             f"Date: {date.today().isoformat()}",
             f"Capture: {capture_rel(result_path)}",
             "Command: tools/spark/run.sh --heavy m0prime 10240 2700 -- tools/spark/py.sh bench/m0prime/run.py "
             f"(torch {res['versions']['torch']})"]
    lines += list(notes)
    lines += ["", "| gate | value | limit | ok |", "|---|---|---|---|"]
    lines += [f"| {n} | {fmt(v)} | {lim} | {'yes' if o else 'NO'} |" for n, v, lim, o in rows]
    lines += ["", "Measured values:", "",
              "| item | value |", "|---|---|",
              f"| fill | {f['records']} records, {f['bytes']} B in {f['seconds']} s; IO {f['io_gbs']} GB/s over the "
              f"ensure read walls, {f['wall_gbs']} GB/s end to end; {f['lanes']} reader lanes on CPUs {f['lane_cpus']} |",
              f"| t_memcpy per record | mean {f['t_memcpy_mean_ms']} ms; p50 {f['t_memcpy_p50_ms']} ms, p99 "
              f"{f['t_memcpy_p99_ms']} ms (log2 buckets of the reader histogram) |",
              f"| chunk free | {res['chunk_free']['samples']} samples, p50 {res['chunk_free']['p50_ns']} ns, p99 "
              f"{res['chunk_free']['p99_ns']} ns, max {res['chunk_free']['max_ns']} ns |",
              f"| layer step (full layer, L2 flushed per replay) | GDN {t['t_gdn_us']} us (mean of layers 0 and 2), "
              f"QSA {t['t_qsa_us']} us |",
              f"| t_embed / t_lm_head | {t['t_embed_us']} us / {t['t_lm_head_us']} us |",
              f"| t_comp_48 = 36 t_GDN + 12 t_QSA + t_embed + t_lm_head | {t['t_comp_48_ms']} ms |",
              f"| tau | p50 {res['tau']['p50_us']} us, p99 {res['tau']['p99_us']} us over {res['tau']['steps']} layer steps "
              f"after {res['tau']['warmup']} warm-up |",
              f"| runtime footprint | {res['runtime']['footprint_bytes']} B = VmRSS {res['runtime']['vmrss']} + NVML "
              f"{res['runtime']['nvml']} ({res['runtime']['nvml_source']}) - pool {res['runtime']['pool_bytes']} "
              f"(subtracted per counter: RSS {res['runtime']['pool_in_rss']}, NVML {res['runtime']['pool_in_nvml']}) - weights {res['runtime']['weights_bytes']} - state "
              f"{res['runtime']['state_bytes']} |", ""]
    lines += ["GEMV on every shape (cold-L2 graph; all informational at M0', 2560x2560 4-bit is the step-55 gate shape)", "",
              "| shape | bits | us | GB/s |", "|---|---|---|---|"]
    for k, v in res["gemv"].items():
        lines.append(f"| {v['n']}x{v['k']} | {v['bits']} | {v['cold_graph_us']} | {v['gbs_cold_graph']} |")
    lines += [""] + [f"- {i}" for i in info]
    lines += ["", "Recalibrated section 3.8 table (measured t_comp_48, tau p50, t_memcpy mean):", ""] + table_3_8(res)
    lines += [""] + [d for d in decisions]
    if ok:
        lines.append("Verdict: PASS — every M0' design trigger is within its limit (GEMV informational at M0', "
                     "judged at step 55; footprint <= 2.9e9 B per the 2026-10-12 rule)")
    else:
        failed = "; ".join(r[0] for r in rows if not r[3])
        lines.append(f"Verdict: FAIL — {failed}")
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text("\n".join(lines) + "\n")
    return ok


def evaluate_h1(res):
    rows, decisions = [], []
    a = res["a"]
    ok = a["ok"] and a["separable"]
    rows.append((f"(a) layer 0 rel_l2 over {a['positions']} tokens (limits[0] from the contract)", a["rel_l2"],
                 f"<= {a['limit']:.5g}, separable", ok))
    b = res["b"]
    rows.append((f"(b) bitwise equal outputs with pools {b['pools']}", b["bitwise_equal"], "True", bool(b["bitwise_equal"])))
    c = res["c"]
    rows.append(("(c) ensures: errors", c["errors"], "0", c["errors"] == 0))
    rows.append(("(c) bytes equal an independent O_DIRECT read", c["bytes_equal"], "all", c["bytes_equal"] == c["ensures"]))
    rows.append(("(c) GPU checksum correct (incl. after overwriting a slot)", c["gpu_checksum_ok"], "True",
                 bool(c["gpu_checksum_ok"])))
    rows.append(("(c) max read (s)", c["max_read_s"], f"< {H1_READ_MAX_S}", c["max_read_s"] < H1_READ_MAX_S))
    d = res["d"]
    p = footprint.peak([footprint.sample_from_dict(x) for x in res["samples"]],
                       footprint.flags_from_dict(res["flags"]))
    if p.nonpool_bytes != d["peak_bytes"]:
        raise ValueError(f"h1: recorded peak_bytes {d['peak_bytes']} != recomputed {p.nonpool_bytes}")
    peak = p.nonpool_bytes / (1 << 20)
    rows.append(("(d) peak VmHWM + NVML + pinned, pool subtracted once per counter where it shows (MiB)", peak, f"< {H1_PEAK_LIMIT_MIB}", peak < H1_PEAK_LIMIT_MIB))
    for n, v, lim, o in rows:
        if not o:
            decisions.append(f"DECISION: H1 gate failed: {n} = {fmt(v)} (limit {lim})")
    return rows, decisions


def write_h1(res, rows, decisions, result_path, dest, notes=()):
    ok = all(r[3] for r in rows)
    a = res["a"]
    lines = ["# h1: layer 0 against the oracle, pool sizes, 1,000 reads, coherence and peak",
             f"Date: {date.today().isoformat()}",
             f"Capture: {capture_rel(result_path)}",
             "Command: tools/spark/run.sh --heavy h1 4096 3600 -- tools/spark/py.sh -m pytest -q -m gate "
             f"tests/gates/test_layer0.py (torch {res.get('torch', '?')})"]
    lines += list(notes)
    lines += ["", "| gate | value | limit | ok |", "|---|---|---|---|"]
    lines += [f"| {n} | {fmt(v)} | {lim} | {'yes' if o else 'NO'} |" for n, v, lim, o in rows]
    d = res["d"]
    smp = [footprint.sample_from_dict(x) for x in res["samples"]]
    fl = footprint.flags_from_dict(res["flags"])
    per = " ".join(f"{s.tag}={footprint.nonpool_bytes(s, fl) / (1 << 20):.0f}" for s in smp)
    lines += ["", f"- layer 0 spread s_0 = {fmt(a['spread'])}, limit L_0 = {fmt(a['limit'])}, control c_0 = "
              f"{fmt(a['control'])}; decode path (one token per step) rel_l2 {fmt(a['decode_rel_l2'])}",
              f"- layer 0 routing disagreement {fmt(a['routing_disagreement'])} (limit {fmt(a['routing_limit'])}, "
              f"separable {a['routing_separable']}: layer 0 routing is informational); n-gram ids exact: {a['ngram_ok']}",
              f"- prompt dev0 has {a['positions']} tokens (the first 64 are used; the prompt is shorter)",
              f"- pool sizes {res['b']['pools']}: output SHA-256 {res['b']['sha256']}",
              f"- reads: {res['c']['ensures']} ensures over layers 0-47, slots {res['c']['pool_slots']}, hits "
              f"{res['c']['hits']}, misses {res['c']['misses']}",
              f"- peak parts: VmHWM {d['vmhwm']} B, NVML max {d['nvml_max']} B ({d['nvml_source']}), pinned "
              f"{d['pinned']} B; pool high-water {d['unified_hwm']['pool']} B shows in VmRSS {fl['pool'][0]}, NVML "
              f"{fl['pool'][1]}; peak {d['peak_bytes']} B set by sample '{d['argmax_tag']}' (VmHWM bound "
              f"{d['hwm_bound_bytes']} B); samples (MiB): {per}",
              f"- each byte once (informational): {d['once_bytes']} B = peak + pool high-water; the 4,096 MiB budget "
              "excludes the pool, which is the ledger's own term"]
    lines += [""] + decisions
    lines.append("Verdict: PASS — layer 0 within its limit; pools bitwise equal; reads exact; peak under 4 GiB with the pool subtracted once per counter (2026-10-12 rule)" if ok
                 else "Verdict: FAIL — " + "; ".join(r[0] for r in rows if not r[3]))
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text("\n".join(lines) + "\n")
    return ok


def evaluate_h2(res, capture_dir):
    """-> (rows [(name, value, limit, ok)], decisions [str], info [str]); ValueError/KeyError -> exit 3."""
    rows, decisions, info = [], [], []
    o = res["oracle"]
    if o is None or not all("routing_pair" in L and "limit_src" in L
                            for p in o["prompts"].values() for L in p["layers"]):
        raise ValueError("h2: oracle result predates the per-sequence rule")
    sz = res["sizes"]
    sizes = sorted(sz["per_size"])
    rows.append((f"(a) tokens and logits equal across the {len(sizes)} pool sizes {', '.join(sizes)}",
                 f"tokens {sz['tokens_equal']}, logits {sz['logits_equal']}", "True, True",
                 bool(sz["tokens_equal"]) and bool(sz["logits_equal"]) and len(sizes) == 5))
    rows.append(("(b) resize sequence identical", res["resize"]["all_identical"], "True",
                 bool(res["resize"]["all_identical"])))
    f = res["fault"]
    for key in ("fired", "map_unchanged", "pins_restored", "retry_identical"):
        rows.append((f"(c) fault.{key}", f[key], "True", bool(f[key])))
    g = res["gdn"]
    gok = g["failing_layers"] == [] and bool(g["logits"]["ok"])
    rows.append(("(d) fused GDN vs reference: failing layers, logits", f"{g['failing_layers']}, {g['logits']['ok']}",
                 "[], True", gok))
    if not gok:
        decisions.append(H2_GDN_DECISION)
    rows.append(("(e) oracle compare ok", o["ok"], "True", bool(o["ok"])))
    for name in sorted(o["prompts"]):
        p = o["prompts"][name]
        rows.append((f"(e) {name}: ngram exact, failing layers, logits", f"{p['ngram_ok']}, {p['failing_layers']}, "
                     f"{p['logits']['ok']}", "True, [], True",
                     bool(p["ngram_ok"]) and p["failing_layers"] == [] and bool(p["logits"]["ok"])))
    smp = [footprint.sample_from_dict(x) for x in res["samples"]]
    pk = footprint.peak(smp, footprint.flags_from_dict(res["flags"]))
    if pk.once_bytes != res["peak"]["once_bytes"]:
        raise ValueError(f"h2: recorded once_bytes {res['peak']['once_bytes']} != recomputed {pk.once_bytes}")
    rows.append(("(f) peak, each byte once (B)", pk.once_bytes, f"<= cap {res['cap_bytes']} B",
                 pk.once_bytes <= res["cap_bytes"]))
    ceiling = modes.get("D5").ceiling_mib << 20
    env = Ledger("D5", staging_groups=2).expected_peak_bytes(res["chunks_max"])
    rows.append((f"(g) ledger D5 envelope at {res['chunks_max']} chunks, step-51 ledger (2x32 staging; H2 predates the sweep) (B)", env, f"<= D5 ceiling {ceiling} B",
                 env <= ceiling))
    ex = res["exit"]
    rows.append(("(h) exit codes pytest, oracle", f"{ex['pytest']}, {ex['oracle']}", "0, 0",
                 ex["pytest"] == 0 and ex["oracle"] == 0))
    gp = json.loads((Path(capture_dir) / "guard-post.json").read_text())
    exp_ctr = gp.get("expected_containers", 14)      # 14 unless the guard ran under the pause file
    rows.append(("(i) guard post: cut, containers up", f"{gp['cut']}, {gp['containers_up']}", f"None, {exp_ctr}",
                 gp["cut"] is None and gp["containers_up"] == exp_ctr))
    for n, v, lim, ok in rows:
        if not ok:
            decisions.append(f"DECISION: H2 gate failed: {n} = {fmt(v)} (limit {lim})")
    info.append(f"current ledger (4x32 staging) at {res['chunks_max']} chunks: "
                f"{Ledger('D5').expected_peak_bytes(res['chunks_max'])} B (informational)")
    info.append(f"nonpool_bytes {pk.nonpool_bytes} B beside Ledger('D5').fixed_bytes() {Ledger('D5').fixed_bytes()} B "
                "(informational)")
    for name in sorted(o["prompts"]):
        p = o["prompts"][name]
        l43 = p["layers"][43]
        info.append(f"{name}: worst routing ratio {fmt(p['worst_routing_ratio'])} at layer {p['worst_routing_layer']} "
                    f"({p['worst_routing_src']}); bound layers {p['bound_layers']}; layer 43 routing "
                    f"{fmt(l43['routing'])} vs pair {fmt(l43['routing_pair'])}, limit {fmt(l43['routing_limit'])} "
                    f"({l43['routing_limit_src']})")
    return rows, decisions, info


def write_h2(res, rows, decisions, info, result_path, dest, notes=()):
    ok = all(r[3] for r in rows)
    lines = ["# h2: 48-layer model over pool sizes, resize, read fault, fused GDN, oracle and peak",
             f"Date: {date.today().isoformat()}",
             f"Capture: {capture_rel(result_path)}",
             "Command: tools/spark/jobs.sh restart h2 (saved step-50 line: --heavy h2 auto 14400 -- bash "
             "tests/gates/h2_job.sh)"]
    lines += list(notes)
    lines += ["", "| gate | value | limit | ok |", "|---|---|---|---|"]
    lines += [f"| {n} | {fmt(v)} | {lim} | {'yes' if o else 'NO'} |" for n, v, lim, o in rows]
    lines += [""] + [f"- {i}" for i in info]
    lines += ["", H2_RULE_LINE, ""] + decisions
    lines.append("Verdict: PASS — pool sizes, resize and fault identical; fused GDN and oracle within the contract "
                 "(per-sequence rule); peak under the cap, each byte once; ledger envelope under the D5 ceiling" if ok
                 else "Verdict: FAIL — " + "; ".join(r[0] for r in rows if not r[3]))
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text("\n".join(lines) + "\n")
    return ok


LOGITS_RULE_LINE = "Logits rule: seq KL (factor 3.0, floor = calibrated limit, ceilings 0.05/0.5/1.5 nats; top-1 calibrated), design decision 18"
SWEEP_NOTE_STALE = "sweep-check: result predates the per-sequence rule"


def evaluate_sweep_check(res, capture_dir):
    """Step 70a: re-judge the stored F5 sweep-check capture with the per-sequence logits rule (recomputed, never trusted)."""
    import contract as ct
    lg = res.get("logits") or {}
    if "pair" not in lg or not all("limit_src" in L for L in res["layers"]):
        raise ValueError(SWEEP_NOTE_STALE)
    c = ct.load()
    j = ct.logits_judge_seq(lg, lg["pair"], c)
    rows, decisions, info = [], [], []
    rows.append(("(a) failing_layers", res["failing_layers"], "[]", res["failing_layers"] == []))
    good = sum(1 for L in res["layers"] if L["ok"] and L["routing_ok"] and L["sweep_vs_f2_ok"])
    rows.append(("(b) layers ok, routing_ok and sweep_vs_f2_ok", f"{good} of {len(res['layers'])}", "48 of 48",
                 good == len(res["layers"]) == LAYERS))
    sm = res["summary"]
    rows.append(("(c) summary prompts, hash and 32 ids", sorted(sm), "['2k', '8k', '9k']",
                 sorted(sm) == ["2k", "8k", "9k"] and all(sm[k].get("hidden_sha256") and len(sm[k]["ids"]) == 32 for k in sm)))
    lim, src = j["limits_seq"], j["limit_src"]
    rows.append(("(d) logits.top1", j["top1"], f">= {fmt(lim['top1_min'])} (calibrated)", j["top1"] >= lim["top1_min"]))
    for n, (x, key) in zip("efg", (("kl_mean", "kl_mean_max_nats"), ("kl_p99", "kl_p99_max_nats"), ("kl_max", "kl_max_nats"))):
        rows.append((f"({n}) logits.{x}", j[x], f"<= {fmt(lim[key])} ({src[key]})", j[x] <= lim[key]))
    rows.append(("(h) logits.judgeable (pair under every ceiling)", j["judgeable"], "True", j["judgeable"]))
    gp = json.loads((Path(capture_dir) / "guard-post.json").read_text())
    exp_ctr = gp.get("expected_containers", 14)      # 14 unless the guard ran under the pause file
    rows.append(("(i) guard post: cut, containers up", f"{gp['cut']}, {gp['containers_up']}", f"None, {exp_ctr}",
                 gp["cut"] is None and gp["containers_up"] == exp_ctr))
    ceil = {x: c.logits[ck] for x, _, ck in ct.LOGITS_SEQ}
    if not j["judgeable"]:
        over = [x for x in ceil if j["pair"][x] > ceil[x]]
        for x in over:
            decisions.append(f"DESIGN GATE: logits pair {x} {fmt(j['pair'][x])} > ceiling {fmt(ceil[x])} on this sequence "
                             f"— not judgeable (engine {fmt(j[x])}, fixed limit {fmt(c.logits[dict((a, b) for a, b, _ in ct.LOGITS_SEQ)[x]])})")
    for x, key, _ in ct.LOGITS_SEQ:
        if j["judgeable"] and j[x] > lim[key]:
            decisions.append(f"DECISION: sweep prefill logits {x} {fmt(j[x])} > per-sequence limit {fmt(lim[key])} ({src[key]}) "
                             f"- pair {fmt(j['pair'][x])}, ceiling {fmt(ceil[x])}, fixed {fmt(c.logits[key])}")
    if j["judgeable"] and j["top1"] < lim["top1_min"]:
        decisions.append(f"DECISION: sweep prefill logits top1 {fmt(j['top1'])} < top1_min {fmt(lim['top1_min'])} "
                         f"- pair {fmt(j['pair']['top1'])}")
    for n, v, l, ok in rows:
        if not ok and not any(n.startswith(p) for p in ("(d)", "(e)", "(f)", "(g)", "(h)")):
            decisions.append(f"DECISION: F5 sweep-check failed: {n} = {fmt(v)} (limit {l})")
    info.append(f"fixed_limits_ok {j['fixed_limits_ok']} (engine against the calibrated limits; false is expected on 2,048 positions)")
    info.append(f"stored bridge (not used): ok {lg.get('ok')}, limits_seq.kl_max {fmt((lg.get('limits_seq') or {}).get('kl_max'))}")
    info.append("pair: " + ", ".join(f"{k} {fmt(v)}" for k, v in j["pair"].items()))
    info.append("ceilings: " + ", ".join(f"{x} {fmt(v)}" for x, v in ceil.items()))
    return rows, decisions, info


def write_sweep_check(res, rows, decisions, info, result_path, dest, notes=()):
    ok = all(r[3] for r in rows)
    lines = ["# f5-sweep-check: sweep prefill vs pool states, read scope and the oracle on 2K ids",
             f"Date: {date.today().isoformat()}",
             f"Capture: {capture_rel(result_path)}",
             "Command: tools/spark/run.sh --heavy f5-sweep-check auto 2700 -- tools/spark/py.sh -m pytest -q -m gate "
             "tests/gates/test_sweep_check.py (run as a jobs.sh job from a snap.sh copy, 21.5 min)"]
    lines += list(notes)
    lines += ["", "| gate | value | limit | ok |", "|---|---|---|---|"]
    lines += [f"| {n} | {fmt(v)} | {lim} | {'yes' if o else 'NO'} |" for n, v, lim, o in rows]
    lines += [""] + [f"- {i}" for i in info]
    lines += ["", LOGITS_RULE_LINE, ""] + decisions
    lines.append("Verdict: PASS — sweep prefill identical across pool states and scopes; layers, routing and logits within "
                 "the per-sequence rule" if ok else "Verdict: FAIL — " + "; ".join(r[0] for r in rows if not r[3]))
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text("\n".join(lines) + "\n")
    return ok


PARTITION_TAU_LIMIT_US = 150.0
PARTITION_MIN_TOKENS = 16


def _partition_module():
    d = str(ROOT / "bench")
    if d not in sys.path:
        sys.path.insert(0, d)
    import partition
    return partition


def evaluate_partition(res):
    """Step 57: the 100% wall-time partition of `cli bench decode --profile` (bench/partition.py arithmetic)."""
    pt = _partition_module()
    rows, decisions, info = [], [], []
    rows.append(("profiled run (--profile, result ok)", bool(res.get("profile")) and bool(res.get("ok")),
                 "true", bool(res.get("profile")) and bool(res.get("ok"))))
    for k, e in res["prompts"].items():
        c = pt.check(e)
        share = 100.0 * c["other_share"]
        rows.append((f"{k}: named phases + other = wall (100%)", f"{c['named_ns'] + c['other_ns']} of {c['wall_ns']} ns",
                     "equal and the recorded other matches", c["named_ns"] + c["other_ns"] == c["wall_ns"]
                     and c["recorded_other_matches"]))
        rows.append((f"{k}: phases do not overlap (other >= 0)", c["other_ns"], ">= 0", not c["overlap"]))
        rows.append((f"{k}: other share of wall (%)", round(share, 3), f"<= {100 * pt.OTHER_MAX:g}",
                     0 <= c["other_share"] <= pt.OTHER_MAX))
        rows.append((f"{k}: t_copy separate from IO (miss-read + copy-tail = read wall)",
                     e["phases_ns"]["expert_miss_read"] + e["phases_ns"]["expert_copy_tail"],
                     f"= {e['read_wall_ns']}", c["read_split_exact"] and c["copy_separate"]))
        rows.append((f"{k}: profiled tokens", c["tokens"], f">= {PARTITION_MIN_TOKENS}", c["tokens"] >= PARTITION_MIN_TOKENS))
        rows.append((f"{k}: decode tok/s, (N-1)/sum of inter-token intervals (informational, H3 gates it)", e["tok_s"],
                     "info", True))
        rows.append((f"{k}: pool hit rate h over the decode (informational)", e["hit_rate"], "info", True))
        info.append(f"{k}: {e['prompt_tokens']} prompt tokens, prefill {e['prefill_s']} s, {e['new_tokens']} new tokens, "
                    f"{e['hits']} hits / {e['misses']} misses; NVTX ranges {'on' if e.get('nvtx') else 'off'}")
    tau = res.get("tau")
    if tau is None:
        rows.append(("tau p50 (us)", "not measured", f"<= {PARTITION_TAU_LIMIT_US:g}", False))
    else:
        ok = tau["p50_us"] <= PARTITION_TAU_LIMIT_US
        rows.append(("tau p50 (us), ensure + launch of all-hit layer steps", tau["p50_us"],
                     f"<= {PARTITION_TAU_LIMIT_US:g}", ok))
        if not ok:
            decisions.append(f"DECISION: tau={tau['p50_us']} us > {PARTITION_TAU_LIMIT_US:g} us -> move the layer loop to "
                             "C++ / bring F8 forward (the technology decision)")
    return rows, decisions, info


def write_partition(res, rows, decisions, info, result_path, dest, notes=()):
    pt = _partition_module()
    ok = all(r[3] for r in rows)
    lines = ["# partition-d5: decode wall-time partition in D5",
             f"Date: {date.today().isoformat()}",
             f"Capture: {capture_rel(result_path)}",
             "Command: tools/spark/run.sh --heavy partition auto 3600 -- tools/spark/py.sh -m slotstream_spark.cli bench "
             "decode --mode D5 --prompts bench/lb/prompts.json --prompt-keys short,2k --tokens 256 --warmup 1 --profile "
             f"(torch {res['versions']['torch']}, pool {res['slots']} slots)"]
    lines += list(notes)
    lines += ["", "| gate | value | limit | ok |", "|---|---|---|---|"]
    lines += [f"| {n} | {fmt(v)} | {lim} | {'yes' if o else 'NO'} |" for n, v, lim, o in rows]
    for k, e in res["prompts"].items():
        lines += ["", f"Partition of the decode wall time, prompt `{k}` ({e['tokens_profiled']} tokens, "
                      f"{e['wall_ns'] / 1e6:.1f} ms of wall):", "", "| phase | ms | share of wall |", "|---|---|---|"]
        lines += [f"| {n} | {ns / 1e6:.3f} | {100 * sh:.2f} % |" for n, ns, sh in pt.table(e)]
    lines += [""] + [f"- {i}" for i in info]
    lines += [""] + decisions
    lines.append("Verdict: PASS — the partition adds up to 100% of the wall time with other <= 10%, t_copy separate "
                 "from IO, tau within its limit" if ok
                 else "Verdict: FAIL — " + "; ".join(r[0] for r in rows if not r[3]))
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text("\n".join(lines) + "\n")
    return ok


SOAK_MIN_READS = 1_000_000
SOAK_MAX_READ_MS = 2000.0


def evaluate_soak(res, capture_dir):
    """Safe-path soak gates (step 61). A missing guard-post.json (cut unknown) is exit 3, never a pass."""
    post = json.loads((Path(capture_dir) / "guard-post.json").read_text())
    cut = bool(post.get("cut")) or post.get("rc") not in (0, None)
    rows, decisions = [], []
    rows.append(("reads", res["reads"], f">= {SOAK_MIN_READS}", res["reads"] >= SOAK_MIN_READS))
    rows.append(("read errors", res["errors"], "0", res["errors"] == 0))
    rows.append(("timeouts (batch not joined in 2 s)", res["timeouts"], "0", res["timeouts"] == 0))
    rows.append(("checksum mismatches", res["checksum_mismatches"], "0", res["checksum_mismatches"] == 0))
    rows.append(("max read (ms)", res["max_read_ms"], f"< {SOAK_MAX_READ_MS:g}", res["max_read_ms"] < SOAK_MAX_READ_MS))
    rows.append(("guard cut or non-zero rc", cut, "False", not cut))
    rows.append(("terminated by signal", bool(res.get("terminated")), "False", not res.get("terminated")))
    for n, v, lim, o in rows:
        if not o:
            decisions.append(f"DECISION: soak gate failed: {n} = {fmt(v)} (limit {lim})")
    return rows, decisions


def write_soak(res, rows, decisions, result_path, dest, notes=()):
    ok = all(r[3] for r in rows)
    lines = ["# soak-safe-path: 10^6 reads through the real pool, ensurer and GPU checksum",
             f"Date: {date.today().isoformat()}",
             f"Capture: {capture_rel(result_path)}",
             "Command: tools/spark/jobs.sh start soak --heavy soak 4096 7200 -- bash tools/spark/snap.sh soak "
             "tools/spark/py.sh tools/soak_safe_path.py --reads 1000000"]
    lines += list(notes)
    lines += ["", "| gate | value | limit | ok |", "|---|---|---|---|"]
    lines += [f"| {n} | {fmt(v)} | {lim} | {'yes' if o else 'NO'} |" for n, v, lim, o in rows]
    lines += ["", f"- read latency per batch: p50 {fmt(res['p50_read_ms'])} ms, p99 {fmt(res['p99_read_ms'])} ms, "
                  f"max {fmt(res['max_read_ms'])} ms over {res['batches']} batches",
              f"- {fmt(res['bytes'] / 1e9)} GB read in {fmt(res['seconds'])} s = {fmt(res['gbps'])} GB/s "
              f"(limit {fmt(res['rate_gbps_limit'])}); demand reads {res['demand_reads']}, background reads "
              f"{res['spec_reads']}, ensure hits {res['hits']}",
              f"- CRC checks against an independent O_DIRECT read: {res['crc_checks']}; GPU checksum checks: "
              f"{res['gpu_checks']} (gpu {res['gpu']})",
              f"- NVMe hwmon pauses (limit {res['temp_limit_mc']} mC): {res['pauses']}"]
    lines += [""] + decisions
    lines.append("Verdict: PASS — every read exact, no error, no hang, no guard cut" if ok
                 else "Verdict: FAIL — " + "; ".join(r[0] for r in rows if not r[3]))
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text("\n".join(lines) + "\n")
    return ok


# ---- step 65: H3 and the F3 stop/go against the arm B yardstick -------------------------------------------------
H3_MIN_TPS = 17.0            # unchanged absolute floor for the engine's D5 decode
H3_MIN_CLEAN = 5
D1_TARGET_TPS = 9.0          # informative
GO_MEDIAN, GO_POSITIVE, HALF = 1.00, 0.80, 0.50
H3_RECORD = "db/records/measurements/h3.md"
AA_PROMPTS = ("short", "2k")


def _median(v):
    return _stats().true_median(v)


def _stats():
    sys.path.insert(0, str(ROOT / "bench"))
    import stats  # noqa: F401
    return stats


def aa_summary(res):
    """Clean D5 A/A pairs of the session: engine tok/s per run, per-pair |ratio - 1|, NVMe temperature, exclusions."""
    pairs = res["phases"]["aa"]["pairs"]
    clean = [p for p in pairs if p.get("eligible")]
    runs = [p[side][k] for p in clean for side in ("a", "b") for k in AA_PROMPTS if p[side].get(k) is not None]
    dev = [abs(sum(p["ratio"].values()) / len(p["ratio"]) - 1) for p in clean]
    reasons = {}
    for p in pairs:
        for r in p.get("reasons") or ([] if p.get("eligible") else ["unspecified"]):
            reasons[r] = reasons.get(r, 0) + 1
    temps = [p["snapshot"]["nvme_temp_mc"] / 1000 for p in pairs if p.get("snapshot", {}).get("nvme_temp_mc") is not None]
    st = _stats()
    return {"pairs": len(pairs), "clean": len(clean), "runs": runs, "dev": dev, "reasons": reasons,
            "median_tps": st.true_median(runs) if runs else None,
            "median_abs_dev": st.true_median(dev) if dev else None, "max_abs_dev": max(dev) if dev else None,
            "temp_min": min(temps) if temps else None, "temp_max": max(temps) if temps else None}


def _record_verdict(rel):
    p = ROOT / rel
    if not p.exists():
        return None, ""
    txt = p.read_text()
    m = [ln for ln in txt.splitlines() if ln.startswith("Verdict:")]
    return (m[-1] if m else ""), txt


def evaluate_h3(res):
    aa = aa_summary(res)
    rows, decisions = [], []
    rows.append(("clean A/A pairs", aa["clean"], f">= {H3_MIN_CLEAN}", aa["clean"] >= H3_MIN_CLEAN))
    med = aa["median_tps"]
    ok_tps = med is not None and med >= H3_MIN_TPS
    rows.append(("engine D5 decode median over clean A/A runs (tok/s)", med, f">= {H3_MIN_TPS:g}", ok_tps))
    pv, ptxt = _record_verdict("db/records/measurements/partition-d5.md")
    rows.append(("partition record, t_copy separate", pv or "missing", "PASS with t_copy separate",
                 bool(pv) and "PASS" in pv and "t_copy separate" in ptxt))
    sv, _ = _record_verdict("db/records/measurements/soak-safe-path.md")
    rows.append(("soak record", sv or "missing", "PASS", bool(sv) and "PASS" in sv))
    d1 = res["phases"]["d1"]["runs"]
    d1v = [t for r in d1 if not r.get("error") for t in r["tps"].values()]
    d1med = _median(d1v) if d1v else None
    rows.append(("D1 decode recorded (informative; target >= %g)" % D1_TARGET_TPS, d1med, "recorded", d1med is not None))
    if not ok_tps:
        decisions.append(f"DECISION: D5 decode {fmt(med)} < {H3_MIN_TPS:g} tok/s at H3 \u2014 stop and ask")
    return rows, decisions, aa, d1med


def write_h3(res, rows, decisions, aa, d1med, result_path, dest, notes=()):
    ok = all(r[3] for r in rows)
    lines = ["# h3: engine D5 decode, A/A noise, pool-size invariance",
             f"Date: {date.today().isoformat()}", f"Capture: {capture_rel(result_path)}",
             "Command: tools/spark/run.sh --heavy h3 auto 5400 -- tools/spark/py.sh -m pytest -q -m gate tests/gates/test_h3.py "
             "(pools 13 and 48 per layer and the D5 maximum: bitwise-equal tokens and logits, 3-row verify = three 1-row steps)"]
    lines += list(notes)
    lines += ["", "| gate | value | limit | ok |", "|---|---|---|---|"]
    lines += [f"| {n} | {fmt(v)} | {lim} | {'yes' if o else 'NO'} |" for n, v, lim, o in rows]
    lines += ["", "A/A noise over the clean D5 pairs (per-pair |mean ratio - 1|), read by tools/decide.py through "
              "`tools.yardstick.load_record(\"db/records/measurements/h3.md\")[\"aa\"]`:", "", "```toml", "[aa]",
              f"pairs = {aa['clean']}", f"median_abs_dev = {aa['median_abs_dev']:.6f}" if aa["dev"] else "median_abs_dev = 0.0",
              f"max_abs_dev = {aa['max_abs_dev']:.6f}" if aa["dev"] else "max_abs_dev = 0.0",
              f"nvme_temp_c_min = {aa['temp_min']}", f"nvme_temp_c_max = {aa['temp_max']}",
              "excluded_reasons = {" + ", ".join(f'"{k}" = {v}' for k, v in sorted(aa["reasons"].items())) + "}", "```", ""]
    lines += [f"- A/A pairs run: {aa['pairs']}, clean: {aa['clean']}; D1 decode median {fmt(d1med)} tok/s (informative)"]
    lines += [""] + decisions
    lines.append("Verdict: PASS \u2014 engine D5 decode above the floor, pool-size checks run by the gate test" if ok
                 else "Verdict: FAIL \u2014 " + "; ".join(r[0] for r in rows if not r[3]))
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text("\n".join(lines) + "\n")
    return ok


def _yardstick_mod():
    sys.path.insert(0, str(ROOT / "tools"))
    import yardstick
    return yardstick


STOPGO_NEW_KEYS = ("cache_mib", "memfree_start_mib", "memavailable_start_mib", "snapshot")


def stopgo(res, user_decision=None, answer=None):
    """-> (exit_code, lines for stdout, dict for the record). Pure over the result, the yardstick and h3.md."""
    out = {}
    try:
        y = _yardstick_mod().yardstick()
    except (ValueError, OSError) as e:
        return 1, [f"stopgo: yardstick block invalid: {e}"], None
    if y.get("arm") != "q41-b-d5" or y.get("op_offload") is not False or y.get("prefill_device") != "cpu":
        return 1, ["stopgo: yardstick block invalid: need arm q41-b-d5, op_offload false, prefill_device cpu"], None
    lb = res["lb"]
    if "cache_rule" not in lb:
        return 3, ["stopgo: result schema predates the 2026-10-13 replan (lb.cache_rule)"], None
    for p in lb["pairs"]:
        for k in STOPGO_NEW_KEYS:
            if k not in p:
                return 3, [f"stopgo: result schema predates the 2026-10-13 replan ({k})"], None
    if lb.get("arm") != y["arm"] or lb.get("prefill_device") != y["prefill_device"]:
        return 1, [f"stopgo: the LB pairs carry arm {lb.get('arm')} / {lb.get('prefill_device')}, not the yardstick"], None
    st = _stats()
    clean = [p for p in lb["pairs"] if p.get("clean")]
    excluded = [(p["pair"], p.get("reason") or "unspecified") for p in lb["pairs"] if not p.get("clean")]
    ratios = [p["ratio"] for p in clean]
    full = float(lb["cache_rule"]["full_mib"])
    reduced = [p for p in clean if p["cache_mib"] < full]
    eng = [sum(p["engine_tps"].values()) / len(p["engine_tps"]) for p in clean]
    armb = [sum(p["lb_tps"].values()) / len(p["lb_tps"]) for p in clean]
    out.update(clean=clean, excluded=excluded, reduced=reduced, ratios=ratios, yardstick=y, full_mib=int(full),
               memfree_full=lb["cache_rule"]["memfree_full_mib"])
    lines = []
    if len(clean) < H3_MIN_CLEAN:
        out["decision"] = f"DECISION: {len(clean)} clean pairs < {H3_MIN_CLEAN} \u2014 the F3 session cannot decide stop/go; re-run it"
        return 1, [out["decision"]], out
    med, pos = st.true_median(ratios), st.positive_fraction(ratios)
    e_med, b_med = st.true_median(eng), st.true_median(armb)
    out.update(median=med, positive=pos, engine_tps=e_med, armb_tps=b_med)
    aa = aa_summary(res)
    abs_e = aa["median_tps"] if aa["median_tps"] is not None else e_med
    out["abs_e"] = abs_e
    suffix = ""
    if reduced:
        mf = [p["memfree_start_mib"] for p in clean]
        suffix = (f" [arm B reduced cache in {len(reduced)}/{len(clean)} clean pairs: min {min(p['cache_mib'] for p in reduced)} MiB "
                  f"(MemFree at start {min(mf)}-{max(mf)} MiB); full {int(full):,} needs MemFree \u2265 {lb['cache_rule']['memfree_full_mib']:,}]")
    out["suffix"] = suffix
    verdict, _ = _record_verdict(H3_RECORD)
    h3_pass = bool(verdict) and "PASS" in verdict
    ydec = y["decode_tps_median"]
    if user_decision is not None and (not answer or med >= GO_MEDIAN and pos >= GO_POSITIVE):
        return 2, ["stopgo: --user-decision needs --answer and is refused when the median is >= 1.00 with >= 80% positive pairs"], out
    if med >= GO_MEDIAN and pos >= GO_POSITIVE:
        out["decision"] = f"Decision: GO \u2014 engine beats llama.cpp arm B on decode (ratio {med:.3f}, {len(clean)} clean pairs){suffix}"
        return 0, [out["decision"]], out
    if h3_pass and med >= HALF:
        text = (f"DECISION: engine D5 decode {e_med:.1f} tok/s = {med * 100:.0f}% of llama.cpp arm B ({ydec:g} tok/s, GPU expert "
                f"cache 43,520 MiB inside D5); the engine does not beat the yardstick on decode. The TTFT axis (arm B cold 8k "
                f"prefill {y['prefill_tps_8k']:g} tok/s on the CPU) is measured at F5 (step 72) and F9 (step 95). "
                f"Answer: continue-on-ttft | stop{suffix}")
        out["decode_ratio_text"] = text
        if user_decision == "continue-on-ttft":
            out["decision"] = (f"Decision: GO_ON_TTFT \u2014 decode ratio {med:.3f} ({med * 100:.0f}% of arm B); the user chose to continue "
                               f"on the TTFT axis ({answer}); the engine must win axis T at F5 (step 72) and F9 (step 95), rule of "
                               f"PLAN.md design decision 11{suffix}")
            return 0, [out["decision"]], out
        if user_decision == "stop":
            out["decision"] = f"Decision: STOP \u2014 the user chose to stop ({answer}){suffix}"
            return 1, [out["decision"]], out
        out["decision"] = "Decision: PENDING \u2014 user answer required (continue-on-ttft | stop)"
        return 1, [text], out
    if user_decision is not None:
        return 2, ["stopgo: --user-decision is refused: the median is below the half-yardstick stop line or h3.md is not PASS"], out
    text = ("DECISION: engine does not reach half of the arm B yardstick in D5 (the technology decision, arm B by the user "
            "answer of 2026-10-12) \u2014 stop before F4/F5 and ask" + suffix)
    out["decision"] = "Decision: STOP \u2014 below half of arm B"
    return 1, [text], out


def write_stopgo(out, res, result_path, dest, lines_out):
    y = out["yardstick"]
    lines = ["# stopgo-f3: F3 stop/go against llama.cpp arm B",
             f"Date: {date.today().isoformat()}", f"Capture: {capture_rel(result_path)}",
             f"Compared quantity: decode tok/s, (N-1)/sum; engine D5 vs unmodified llama.cpp arm {y['arm']} (prefill_device="
             f"{y['prefill_device']}, op_offload={y['op_offload']}); the LB side's CPU prefill is excluded. IQ3_XXS and arm A are not compared.",
             # table names ([vara_b]) are part of the recorded decision
             f"Yardstick: [vara_b] decode {y['decode_tps_median']:g} tok/s, cold 8k prefill {y['prefill_tps_8k']:g} tok/s",
             "", "| pair | memfree_start_mib | memavailable_start_mib | cache_mib | memfree_min_mib | ratio | nvme_temp_mc | nvme_limit_mc |",
             "|---|---|---|---|---|---|---|---|"]
    for p in out["clean"]:
        s = p["snapshot"]
        lines.append(f"| {p['pair']} | {p['memfree_start_mib']} | {p['memavailable_start_mib']} | {p['cache_mib']} | "
                     f"{p.get('memfree_min_mib')} | {p['ratio']:.4f} | {s.get('nvme_temp_mc')} | {s.get('nvme_temp_limit_mc')} |")
    lines += ["", f"Clean pairs: {len(out['clean'])}; excluded: " + (", ".join(f"pair {a} ({b})" for a, b in out["excluded"]) or "none"),
              f"cache_full_pairs / clean pairs: {len(out['clean']) - len(out['reduced'])} / {len(out['clean'])}"]
    if "median" in out:
        lines += [f"Median ratio {out['median']:.4f}, positive fraction {out['positive']:.2f}; engine {out['engine_tps']:.2f} tok/s "
                  f"vs arm B {out['armb_tps']:.2f} tok/s in the pairs (medians of per-pair means over short and 2k).",
                  f"Arm B cold prefill tok/s observed in the pairs: " + ", ".join(
                      f"{k} {_median([p['prefill_tps_lb'][k] for p in out['clean']]):.1f}" for k in out['clean'][0]['prefill_tps_lb'])
                  + f" (corroboration of [vara_b] prefill_tps_2k {y.get('prefill_tps_2k')})."]
        if out["reduced"]:
            lines.append(f"Absolute reference: engine D5 decode median {out['abs_e']:.1f} tok/s = "
                         f"{out['abs_e'] / y['decode_tps_median'] * 100:.0f} % of [vara_b] {y['decode_tps_median']:g} tok/s (LB2, full cache; "
                         "informative here, the gate input at F6/F9 through tools.yardstick.yardstick())")
    lines += [""] + [out.get("decision", "")] + [ln for ln in lines_out if ln.startswith("DECISION")]
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text("\n".join(lines) + "\n")


def main_stopgo(argv):
    def opt(name):
        return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None
    ud, ans, dest = opt("--user-decision"), opt("--answer"), opt("--write")
    if ud not in (None, "continue-on-ttft", "stop"):
        print("stopgo: --user-decision is continue-on-ttft or stop", file=sys.stderr)
        return 2
    try:
        res = json.loads(Path(argv[1]).read_text())
        rc, lines, out = stopgo(res, ud, ans)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as e:
        print(f"records: cannot evaluate {argv[1]}: {e!r}", file=sys.stderr)
        return 3
    for ln in lines:
        print(ln)
    if out is not None and rc in (0, 1) and "median" in out:
        print(f"median ratio {out['median']:.4f}, positive fraction {out['positive']:.2f}, clean pairs {len(out['clean'])}, "
              f"engine {out['engine_tps']:.2f} tok/s, arm B {out['armb_tps']:.2f} tok/s, "
              f"arm B cold prefill {out['yardstick']['prefill_tps_8k']:g} tok/s (TTFT axis, measured at F5/F9)")
    if dest and out is not None and rc in (0, 1) and "clean" in out:
        write_stopgo(out, res, argv[1], dest, lines)
        print(f"wrote {dest}")
    return rc


# ---- step 66: H4, the F4 drills ----------------------------------------------------------------------------------
H4_FLOOR_S, H4_EXIT_S, H4_D5_EXIT_S, H4_RELEASE_S = 2.0, 2.0, 15.0, 2.0
H4_RISE_BYTES, H4_AVAIL_MIN_BYTES, H4_GROW_GAP_S, H4_STOP_S = int(0.9 * (1 << 30)), 8 << 30, 59.9, 15.0
H4_REQUIRED = ("running-stub-d5", "running-stub-d1-a", "running-stub-d1-b", "running-stub-d2-a", "hog-d5", "hog-d1",
               "restart", "stop")


def load_h4(dirs):
    """{name: dict} from every `h4-<name>.json` under the capture dirs (a later dir wins on a repeated name)."""
    out = {}
    for d in dirs:
        for f in sorted(Path(d).glob("h4-*.json")):
            out[f.stem[3:]] = json.loads(f.read_text())
    return out


def evaluate_h4(recs):
    rows, decisions, info = [], [], []

    def need(name):
        r = recs.get(name)
        if r is None:
            rows.append((f"{name} record", "missing", "present", False))
        return r

    def le(label, v, lim):
        rows.append((label, v, f"<= {lim:g}", v is not None and v <= lim))

    r = need("running-stub-d5")
    if r:
        le("D5 floor after the flip (s)", r["floor_s"], H4_FLOOR_S)
        le("D5 exit while /running stays non-empty (s)", r["exit_s"], H4_D5_EXIT_S)
    r = need("running-stub-d1-a")
    if r:
        le("D1+A floor (s)", r["floor_s"], H4_FLOOR_S)
        gaps = r["grow_gaps_s"]
        ok = r["grows"] >= 2 and all(x == 1 for x in r["grow_steps_chunks"]) and all(g >= H4_GROW_GAP_S for g in gaps)
        rows.append(("D1+A ready: regrow one chunk per >= 60 s (0.1 s observation tolerance)", f"{r['grows']} grows, gaps {gaps}",
                     f">= 2 grows, +1 chunk, gaps >= {H4_GROW_GAP_S:g} s (60 s less the 0.1 s tolerance)", ok))
    r = need("running-stub-d1-b")
    if r:
        le("D1+B exit (s)", r["exit_s"], H4_EXIT_S)
    r = need("running-stub-d2-a")
    if r:
        le("D2+A floor (s)", r["floor_s"], H4_FLOOR_S)
    for mode in ("d5", "d1"):
        r = need(f"hog-{mode}")
        if not r:
            continue
        ev = r["releases"]
        delays = [e["delay_s"] for e in ev if e["delay_s"] is not None]
        rises = [e["avail_rise_bytes"] for e in ev if e["avail_rise_bytes"] is not None]
        U = mode.upper()
        rows.append((f"{U} hog: releases observed", len(ev), ">= 1", len(ev) >= 1))
        rows.append((f"{U} hog: slowest release after the hog step (s)", max(delays) if delays else None,
                     f"<= {H4_RELEASE_S:g}", bool(delays) and len(delays) == len(ev) and max(delays) <= H4_RELEASE_S))
        rows.append((f"{U} hog: smallest MemAvailable rise at a release (GiB)", round(min(rises) / (1 << 30), 2) if rises else None,
                     ">= 0.9", bool(rises) and len(rises) == len(ev) and min(rises) >= H4_RISE_BYTES))
        am = r["avail_min_bytes"]
        rows.append((f"{U} hog: MemAvailable minimum (GiB)", None if am is None else round(am / (1 << 30), 2), ">= 8",
                     am is not None and am >= H4_AVAIL_MIN_BYTES))
        info.append(f"{U} hog stopped by {r['hog_stop_reason']} at {r['hog_size_mib']} MiB after {r['hog_steps']} steps; "
                    f"engine cap {r['engine_cap_mib']} MiB, chunks at start {r['chunks_start']}; engine peak (footprint.once_bytes over "
                    f"the 20 ms series, informational) {fmt((r['engine_peak_once_bytes'] or 0) / (1 << 30))} GiB")
    r = need("restart")
    if r:
        a, b, c = r["first_start"], r["second_start"], r["stub_nonempty"]
        ok = (a["first_status"], a["retry_after"]) == (503, "10") and a["seconds_to_200"] is not None
        rows.append(("restart: first request 503 + Retry-After, then 200", f"{a['first_status']}/{a['retry_after']}, 200 after {a['seconds_to_200']} s", "503/10 then 200", ok))
        rows.append(("restart: engine gone after the idle exit", f"{r['idle_exit_s']} s, procs {r['engine_procs_after_idle']}", "none", r["idle_exit_s"] is not None and not r["engine_procs_after_idle"]))
        ok = (b["first_status"], b["retry_after"]) == (503, "10") and b["seconds_to_200"] is not None
        rows.append(("restart: request after the idle exit 503 + Retry-After, then 200", f"{b['first_status']}/{b['retry_after']}, 200 after {b['seconds_to_200']} s", "503/10 then 200", ok))
        rows.append(("restart: /running non-empty -> 503 and no engine process", f"{c['status']}, procs {c['engine_procs']}, unit {c['engine_unit']}",
                     "503, none", c["status"] == 503 and not c["engine_procs"] and c["engine_unit"] not in ("active", "activating")))
    r = need("stop")
    if r:
        le("stop unit: socket, front and engine inactive (s)", r["stop_s"], H4_STOP_S)
        rows.append(("stop unit: timers", f"{r['start_timer']} | {r['stop_timer']}", "04:00:00 and 23:59:00",
                     "04:00:00" in r["start_timer"] and "23:59:00" in r["stop_timer"]))
        mk = r["may_start"]
        rows.append(("may_start 03:59 / 04:00 / 23:44 / 23:45", f"{mk['03:59'][0]} / {mk['04:00'][0]} / {mk['23:44'][0]} / {mk['23:45'][0]}",
                     "False / True / True / False", [mk[k][0] for k in ("03:59", "04:00", "23:44", "23:45")] == [False, True, True, False]))
        rows.append(("socket restored by the 04:00 service; no [Install]", f"{r['socket_restored']}, install {r['socket_has_install_section']}",
                     "active, no", r["socket_restored"] == "active" and not r["socket_has_install_section"]))
    ooms = [ln for k, v in recs.items() if isinstance(v, dict) for ln in v.get("oom_lines", [])]
    rows.append(("oom-kill or NV_ERR_NO_MEMORY lines in journalctl -k", len(ooms), "0", not ooms))
    arb = [v for k, v in recs.items() if k.startswith("arbiter-")]
    rows.append(("Arbiter throttle_c named (device temp1_max - 5 C)", ", ".join(sorted({fmt(a["throttle_c"]) for a in arb})) or "missing",
                 "recorded", bool(arb)))
    for n, v, lim, o in rows:
        if not o:
            decisions.append(f"DECISION: H4 gate failed: {n} = {fmt(v)} (limit {lim})")
    return rows, decisions, info


def write_h4(rows, decisions, info, dirs, dest, notes=()):
    ok = all(r[3] for r in rows)
    lines = ["# h4-drills: /running stub, bounded hog, on-demand restart and the 23:59 stop",
             f"Date: {date.today().isoformat()}", "Captures: " + ", ".join(capture_rel(Path(d) / "x") for d in dirs),
             "Command: tools/spark/run.sh --heavy drills auto 5400 -- tools/spark/py.sh -m pytest -q -m gate "
             "tests/gates/drills/test_running_stub.py tests/gates/drills/test_hog.py; "
             "tools/spark/run.sh --timeout 1800 -- tools/capture.sh drills-light -- tools/spark/py.sh -m pytest -q -m gate "
             "tests/gates/drills/test_restart.py tests/gates/drills/test_stop.py"]
    lines += list(notes)
    lines += ["", "| gate | value | limit | ok |", "|---|---|---|---|"]
    lines += [f"| {n} | {fmt(v)} | {lim} | {'yes' if o else 'NO'} |" for n, v, lim, o in rows]
    lines += [""] + [f"- {i}" for i in info]
    lines += [""] + decisions
    lines.append("Verdict: PASS \u2014 every release and floor inside its limit, MemAvailable never below 8 GiB, no oom, restart and stop drills pass"
                 if ok else "Verdict: FAIL \u2014 " + "; ".join(r[0] for r in rows if not r[3]))
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text("\n".join(lines) + "\n")
    return ok


def main_h4(argv):
    dirs = []
    i = 1
    while i < len(argv) and not argv[i].startswith("--"):
        dirs.append(argv[i])
        i += 1
    notes = [argv[j + 1] for j, a in enumerate(argv) if a == "--note" and j + 1 < len(argv)]
    dest = argv[argv.index("--write") + 1] if "--write" in argv else None
    try:
        if not dirs:
            raise ValueError("no capture directory")
        recs = load_h4(dirs)
        rows, decisions, info = evaluate_h4(recs)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as e:
        print(f"records: cannot evaluate h4: {e!r}", file=sys.stderr)
        return 3
    for n, v, lim, o in rows:
        print(f"{'ok  ' if o else 'FAIL'} {n}: {fmt(v)} (limit {lim})")
    for x in info:
        print("info:", x)
    for d in decisions:
        print(d)
    if dest:
        write_h4(rows, decisions, info, dirs, dest, notes)
        print(f"wrote {dest}")
    return 0 if all(r[3] for r in rows) else 1


def main(argv):
    if len(argv) < 2 or argv[0] not in ("m0prime", "h1", "h2", "sweep-check", "partition", "soak", "h3", "stopgo", "h4"):
        print(__doc__)
        return 3
    if argv[0] == "h4":
        return main_h4(argv)
    if argv[0] == "stopgo":
        return main_stopgo(argv)
    dest = None
    notes = [argv[i + 1] for i, a in enumerate(argv) if a == "--note" and i + 1 < len(argv)]
    if "--write" in argv:
        dest = argv[argv.index("--write") + 1]
    try:
        res = json.loads(Path(argv[1]).read_text())
        if argv[0] == "m0prime":
            contract = load_contract()
            rows, decisions, info = evaluate_m0prime(res, contract)
            ok = all(r[3] for r in rows)
        elif argv[0] == "h2":
            rows, decisions, info = evaluate_h2(res, Path(argv[1]).resolve().parent)
            ok = all(r[3] for r in rows)
        elif argv[0] == "sweep-check":
            rows, decisions, info = evaluate_sweep_check(res, Path(argv[1]).resolve().parent)
            ok = all(r[3] for r in rows)
        elif argv[0] == "partition":
            rows, decisions, info = evaluate_partition(res)
            ok = all(r[3] for r in rows)
        elif argv[0] == "soak":
            rows, decisions = evaluate_soak(res, Path(argv[1]).resolve().parent)
            ok = all(r[3] for r in rows)
        elif argv[0] == "h3":
            rows, decisions, aa, d1med = evaluate_h3(res)
            ok = all(r[3] for r in rows)
        else:
            rows, decisions = evaluate_h1(res)
            ok = all(r[3] for r in rows)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as e:
        print(f"records: cannot evaluate {argv[1]}: {e!r}", file=sys.stderr)
        return 3
    for n, v, lim, o in rows:
        print(f"{'ok  ' if o else 'FAIL'} {n}: {fmt(v)} (limit {lim})")
    if argv[0] in ("m0prime", "h2", "partition", "sweep-check"):
        for i in info:
            print("info:", i)
    if argv[0] == "h2":
        print(H2_RULE_LINE)
    if argv[0] == "sweep-check":
        print(LOGITS_RULE_LINE)
    for d in decisions:
        print(d)
    if dest:
        if argv[0] == "m0prime":
            write_m0prime(res, rows, decisions, info, argv[1], dest, notes)
        elif argv[0] == "h2":
            write_h2(res, rows, decisions, info, argv[1], dest, notes)
        elif argv[0] == "sweep-check":
            write_sweep_check(res, rows, decisions, info, argv[1], dest, notes)
        elif argv[0] == "partition":
            write_partition(res, rows, decisions, info, argv[1], dest, notes)
        elif argv[0] == "soak":
            write_soak(res, rows, decisions, argv[1], dest, notes)
        elif argv[0] == "h3":
            write_h3(res, rows, decisions, aa, d1med, argv[1], dest, notes)
        else:
            write_h1(res, rows, decisions, argv[1], dest, notes)
        print(f"wrote {dest}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
