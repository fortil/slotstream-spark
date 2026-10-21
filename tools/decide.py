#!/usr/bin/env python3
"""Decisions of F5-F9, judged on the Mac from a fetched capture (plan step 72 onward).

  decide.py f5-prefill (--latest LABEL | CAPTURE_DIR) [--write [PATH]] [--note TEXT]...
  decide.py f6-lookahead (--latest LABEL | CAPTURE_DIR) [--write]
  decide.py f6-mtp (--latest LABEL | CAPTURE_DIR) --mode D5 [--write]
  decide.py f6-target (--latest LABEL | CAPTURE_DIR) [--min 23.0] [--no-write] [--note TEXT]...
  decide.py f8-entry (--latest LABEL | CAPTURE_DIR) [--write]
  decide.py f8-token-graph (--latest f8-ab | CAPTURE_DIR) (--soak-latest f8-ab-soak | --soak CAPTURE_DIR) [--write]
  decide.py f9-quality (--latest f9-engine | CAPTURE_DIR) (--oracle-latest f9-oracle | --oracle DIR) [--write]
  decide.py f9-campaign (--latest f9-engine | CAPTURE_DIR) [--write]
  decide.py f9-diff (--latest f9-ref | CAPTURE_DIR) [--engine-latest f9-engine | --engine DIR] [--oracle-latest f9-oracle | --oracle DIR] [--write]
  decide.py w131k-campaign (--latest w131k-engine | CAPTURE_DIR) [--write]
  decide.py w131k-envelope (--latest w131k-engine | CAPTURE_DIR) [--write]

Exit 0 = the gate passed, 1 = it failed (a `F5 gate failed:` or `DECISION:` line says what), 3 = unreadable or
inconsistent input (missing samples, a recorded peak that the samples do not reproduce, a failed child).
Medians come from bench/stats.py (`true_median`), the yardstick through tools/yardstick.py, the peak through
slotstream_spark/memory/footprint.py recomputed from the capture's own samples, the envelope from the ledger.
The bars are the plan's own (section 2.2): cold 8K TTFT median <= 30 s (target 15), follow-up median <= 12 s
(target 8), peak <= Ledger("D5", window=W).expected_peak_bytes(chunks) with W the window the capture ran at (its
`window` field; the default when absent).

F6 (plan steps 77-78) judges the captures of bench/f6_ab.sh (bench/ab.py result.json, one row per pair). Exit codes: 0 decided
(adopted or kept off; with --write the record and the default in the config are written), 1 the A/B cannot decide or the
D5 target is missed (a `DECISION:` line or the `D5 decode X < 23 tok/s at end of F6: design decision required` line says
what), 2 correctness failure (the two arms produced different outputs), 3 unreadable or inconsistent input (including
`decide: pairs without snapshot`). Before either rule is applied, the adoption margin (0.02 lookahead, 0.05 MTP) must be at
least the F3 A/A noise `[aa] median_abs_dev` of db/records/measurements/h3.md. Both rules need at least 5 clean pairs and
at least 80% of the clean pairs positive; the median is the true median of the paired ratios (bench/stats.py `adoption`).
"""
import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import db_check                                          # noqa: E402
import yardstick                                           # noqa: E402
from bench.stats import adoption, true_median            # noqa: E402

COLD_BAR_S, COLD_TARGET_S = 30.0, 15.0
FOLLOWUP_BAR_S, FOLLOWUP_TARGET_S = 12.0, 8.0
AXIS_T_FACTOR = 4
STOPGO = "db/records/decisions/stopgo-f3.md"
DEFAULT_RECORD = "db/records/measurements/f5-prefill.md"


class BadInput(Exception):
    pass


def _fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def stopgo_decision(path=STOPGO):
    p = Path(path) if Path(path).is_absolute() else ROOT / path
    m = re.search(r"^Decision: (\w+)", p.read_text(), re.M)
    if not m:
        raise BadInput(f"{path}: no Decision: line")
    return m.group(1)


def _children(res, prefix):
    return [(k, v) for k, v in sorted(res["children"].items()) if k.startswith(prefix)]


def recompute_peaks(res):
    """-> list of (child, once_bytes, nonpool_bytes, drop_bytes, gate_bytes, ledger_bytes, chunks); BadInput when a
    child has no samples or its recorded peak is not what the samples give."""
    from slotstream_spark.memory import footprint as fp
    from slotstream_spark.memory.ledger import DEFAULT_WINDOW, Ledger
    out = []
    for name, c in sorted(res["children"].items()):
        if not c.get("samples") or not c.get("flags") or "peak" not in c:
            raise BadInput(f"{name}: no samples/flags/peak in the capture")
        samples = [fp.sample_from_dict(d) for d in c["samples"]]
        pk = fp.peak(samples, fp.flags_from_dict(c["flags"]))
        rec = c["peak"]
        if rec.get("once_bytes") != pk.once_bytes or rec.get("nonpool_bytes") != pk.nonpool_bytes:
            raise BadInput(f"{name}: recorded peak once {rec.get('once_bytes')} / nonpool {rec.get('nonpool_bytes')} "
                           f"differs from the samples' {pk.once_bytes} / {pk.nonpool_bytes}")
        drop = int(c["memavailable_drop_bytes"])
        chunks = int(c["chunks"])
        out.append((name, pk.once_bytes, pk.nonpool_bytes, drop, max(pk.once_bytes, drop),
                    Ledger("D5", window=int(c.get("window") or res.get("window") or DEFAULT_WINDOW)).expected_peak_bytes(chunks),
                    chunks))
    return out


def f5_prefill(res, yard=None, stopgo=None):
    """-> (rows, decisions, info, code). rows are (name, value, limit, ok)."""
    if res.get("failures"):
        raise BadInput(f"the capture reports failed children: {res['failures']}")
    cold_children = _children(res, "cold8k-")
    if not cold_children or len(cold_children) != int(res.get("samples", 0)):
        raise BadInput(f"expected {res.get('samples')} cold 8K children, found {len(cold_children)}")
    yard = yard or yardstick.yardstick()
    stopgo = stopgo or stopgo_decision()
    rows, decisions, info = [], [], []
    cold = [c["cold"] for _, c in cold_children]
    follow = [c["followup"] for _, c in cold_children]
    for name, c in cold_children:
        if c["cold"]["prompt_tokens"] != 8192:
            raise BadInput(f"{name}: cold prompt has {c['cold']['prompt_tokens']} tokens, not 8192")
    bad = [name for (name, _), s in zip(cold_children, cold) if s["reused_tokens"] > 0]
    if bad:
        decisions.append(f"F5 gate failed: cold sample(s) {', '.join(bad)} reused tokens from a prefix (invalid, plan "
                         f"design decision 11); the axis needs fresh processes with no prefix-cache entry")
        rows.append(("cold samples with reused_tokens == 0", f"{len(cold) - len(bad)} of {len(cold)}", "all", False))
        return rows, decisions, info, 1
    rows.append(("cold samples with reused_tokens == 0", f"{len(cold)} of {len(cold)}", "all", True))
    med = {k: true_median([s[k] for s in cold]) for k in ("ttft_s", "queue_s", "prefill_s", "first_sample_s", "store_s")}
    ok_cold = med["ttft_s"] <= COLD_BAR_S
    rows.append(("cold 8K TTFT median (s)", round(med["ttft_s"], 3), f"<= {COLD_BAR_S:g} (target {COLD_TARGET_S:g})",
                 ok_cold))
    fmed = true_median([s["ttft_s"] for s in follow])
    ok_follow = fmed <= FOLLOWUP_BAR_S
    rows.append(("follow-up TTFT median (s)", round(fmed, 3), f"<= {FOLLOWUP_BAR_S:g} (target {FOLLOWUP_TARGET_S:g})",
                 ok_follow))
    peaks = recompute_peaks(res)
    worst = max(peaks, key=lambda p: p[4] / p[5])
    ok_peak = all(p[4] <= p[5] for p in peaks)
    rows.append((f"peak bytes, max(once_bytes, MemAvailable drop), worst child {worst[0]}", worst[4],
                 f"<= {worst[5]} (ledger, {worst[6]} chunks)", ok_peak))
    tps = true_median([s["prompt_tokens"] / s["prefill_s"] for s in cold])
    ratio = tps / yard["prefill_tps_8k"]
    axis_win = ok_cold and tps >= AXIS_T_FACTOR * yard["prefill_tps_8k"]
    info.append(f"cold 8K split (medians): queue {med['queue_s']:.3f} s, prefill {med['prefill_s']:.3f} s, "
                f"first sample {med['first_sample_s']:.3f} s, store {med['store_s']:.3f} s, TTFT {med['ttft_s']:.3f} s")
    info.append("cold 8K TTFT samples: " + ", ".join(f"{s['ttft_s']:.2f}" for s in cold))
    info.append("follow-up: " + ", ".join(f"{s['ttft_s']:.2f} s (reused {s['reused_tokens']}, "
                                          f"read {s['prompt_tokens'] - s['reused_tokens']})" for s in follow))
    for name, once, nonpool, drop, gate, led, chunks in peaks:
        info.append(f"peak {name}: once {once:,} B, nonpool {nonpool:,} B, MemAvailable drop {drop:,} B, ledger {led:,} B "
                    f"({chunks} chunks)")
    for tag, key in (("2K once", "once2k"), ("32K once", "once32k")):
        c = res["children"].get(key)
        if c:
            s = c["cold"]
            info.append(f"{tag}: TTFT {s['ttft_s']:.2f} s, prefill {s['prefill_s']:.2f} s "
                        f"({s['prompt_tokens'] / s['prefill_s']:.1f} tok/s), bytes requested {s.get('bytes_requested', 0):,}")
    s16 = res["children"].get("scope16k")
    if s16:
        for r in s16["runs"]:
            info.append(f"16K pass {r['prefill_pass']} scope {r['read_scope']}: scopes {r.get('scopes')}, bytes requested "
                        f"{r.get('bytes_requested', 0):,}, records read {r.get('records_read')}, prefill "
                        f"{r['prefill_s']:.2f} s ({r['prompt_tokens'] / r['prefill_s']:.1f} tok/s)")
    info.append(f"Axis T: {'WIN' if axis_win else 'LOSE'} vs llama.cpp arm B cold 8k prefill {yard['prefill_tps_8k']:g} "
                f"tok/s ({yard.get('prefill_cold_8k_s', 346.5):g} s on {yard.get('prefill_cold_8k_tokens', 11124)} tokens, CPU)")
    info.append(f"engine cold 8K prefill {tps:.1f} tok/s, ratio {ratio:.2f} to arm B "
                f"(WIN needs TTFT <= {COLD_BAR_S:g} s and >= {AXIS_T_FACTOR} x {yard['prefill_tps_8k']:g} = "
                f"{AXIS_T_FACTOR * yard['prefill_tps_8k']:.1f} tok/s)")
    code = 0
    for name, v, lim, ok in rows:
        if not ok:
            decisions.append(f"F5 gate failed: {name} {_fmt(v)} vs {lim} (plan §2.2), design decision required")
            code = 1
    if stopgo == "GO_ON_TTFT" and not axis_win:
        decisions.append(f"DECISION: engine loses on both axes (decode 61% of arm B 32.9 tok/s at F3; cold 8K prefill "
                         f"{tps:.1f} tok/s < {AXIS_T_FACTOR} x {yard['prefill_tps_8k']:g}) — stop and ask")
        code = 1
    return rows, decisions, info, code


def capture_rel(p):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT)) + "/"
    except ValueError:
        return str(p) + "/"


def write_f5(rows, decisions, info, capture, dest, code, notes=()):
    lines = ["# f5-prefill: prefill and TTFT in D5", f"Date: {date.today().isoformat()}", f"Capture: {capture}",
             "Command: tools/spark/run.sh --heavy f5-ttft auto 3600 -- tools/capture.sh f5-ttft -- tools/spark/py.sh "
             "bench/ttft.py --mode D5 --samples 5"]
    lines += list(notes)
    lines += ["", "| gate | value | limit | ok |", "|---|---|---|---|"]
    lines += [f"| {n} | {_fmt(v)} | {lim} | {'yes' if o else 'NO'} |" for n, v, lim, o in rows]
    lines += [""] + [f"- {i}" for i in info] + [""] + decisions
    if code == 0:
        lines.append("Verdict: PASS — cold 8K TTFT, follow-up TTFT and peak within the F5 bars; axis T recorded")
    else:
        lines.append("Verdict: FAIL — " + "; ".join(d.split(" (plan")[0].split(" — ")[0] for d in decisions))
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text("\n".join(lines) + "\n")


def cmd_f5(a):
    d = db_check.latest(a.latest) if a.latest else Path(a.capture)
    if d is None or not Path(d).is_dir():
        print(f"no capture for {a.latest or a.capture}", file=sys.stderr)
        return 3
    rp = Path(d) / "result.json"
    if not rp.exists():
        print(f"{rp} missing", file=sys.stderr)
        return 3
    res = json.loads(rp.read_text())
    try:
        rows, decisions, info, code = f5_prefill(res)
    except (BadInput, KeyError, ValueError) as e:
        print(f"decide f5-prefill: unreadable input: {e}", file=sys.stderr)
        return 3
    for n, v, lim, o in rows:
        print(f"{'ok  ' if o else 'FAIL'} {n}: {_fmt(v)} (limit {lim})")
    for i in info:
        print(f"  {i}")
    for dline in decisions:
        print(dline)
    if a.write is not None:
        dest = a.write or DEFAULT_RECORD
        write_f5(rows, decisions, info, capture_rel(d), dest, code, a.note)
        print(f"wrote {dest}")
    print("f5-prefill: PASS" if code == 0 else "f5-prefill: FAIL")
    return code


# ----------------------------------------------------------------------------------------------------------- F6 (steps 77-78)
H3 = "db/records/measurements/h3.md"
LA_MARGIN, MTP_MARGIN = 0.02, 0.05
MIN_CLEAN, MIN_POSITIVE = 5, 0.80
D5_TARGET = 23.0
LA_RECORD = "db/records/decisions/f6-lookahead-d5.md"
MTP_RECORD = "db/records/decisions/f6-mtp-d5.md"
TARGET_RECORD = "db/records/measurements/f6-decode-d5.md"
POLICY_TOML = "slotstream_spark/config/policy.toml"
PLATFORM_TOML = "slotstream_spark/config/platform.toml"


class NoSnapshot(Exception):
    pass


class OutputsDiffer(Exception):
    pass


def _abs(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_ab(d):
    rp = Path(d) / "result.json"
    if not rp.exists():
        raise BadInput(f"{rp} missing")
    res = json.loads(rp.read_text())
    if "pairs" not in res:
        raise BadInput(f"{rp}: no pairs")
    return res


def check_snapshots(res):
    bad = [r["pair"] for r in res["pairs"] if not r.get("snapshot")]
    if bad:
        raise NoSnapshot(f"pairs {bad}")


def check_outputs(res):
    """Different outputs in the two arms of a pair (both ran) = correctness failure."""
    bad = [r["pair"] for r in res["pairs"] if r.get("a") is not None and r.get("b") is not None
           and (r.get("a_ident") is None or r.get("a_ident") != r.get("b_ident"))]
    if bad:
        raise OutputsDiffer(f"pairs {bad}")


def clean_pairs(res):
    return [r for r in res["pairs"] if r.get("eligible") and r.get("ratio") is not None]


def aa_noise(path=H3):
    return float(yardstick.load_record(path)["aa"]["median_abs_dev"])


def arm_files(d, rows, arm):
    out = []
    for r in rows:
        f = Path(d) / "arms" / f"p{int(r['pair']):02d}-{arm}.json"
        if f.exists():
            out.append(json.loads(f.read_text()))
    return out


def acceptance(arms):
    """Sum of the MTP counters of the arm outputs -> {depth1: (accepted, attempts), depth2: ..., rounds, emitted}."""
    acc, att, rounds, emitted, n = [0, 0], [0, 0], 0, 0, 0
    for a in arms:
        m = a.get("mtp")
        if not m:
            continue
        n += 1
        for d in (0, 1):
            acc[d] += m["accepted_by_depth"][d]
            att[d] += m["attempts_by_depth"][d]
        rounds += m["rounds"]
        emitted += sum(m["emitted_by_round"])
    return {"arms_with_mtp": n, "accepted": acc, "attempts": att, "rounds": rounds, "emitted": emitted}


def lookahead_counters(arms):
    issued = adopted = discarded = failed = n = 0
    for a in arms:
        la = a.get("lookahead")
        if not la:
            continue
        n += 1
        issued += la["pool"]["issued"]
        adopted += la["pool"]["adopted"]
        discarded += la["pool"]["discarded"]
        failed += la["pool"]["failed"]
    return {"arms_with_lookahead": n, "issued": issued, "adopted": adopted, "discarded": discarded, "failed": failed}


def f6_judge(res, margin, noise):
    """-> dict(code, lines, adopted, median, positive, clean, why). code 1 = cannot decide (a DECISION: line), 2 = outputs
    differ, 0 = decided. Raises NoSnapshot before anything else."""
    check_snapshots(res)
    out = {"code": 0, "lines": [], "adopted": False, "median": None, "positive": None, "clean": 0, "why": "", "noise": noise,
           "margin": margin}
    if margin < noise:
        out["code"] = 1
        out["lines"].append(f"DECISION: F3 A/A noise {noise:.6f} ≥ adoption margin {margin:g} — the A/B cannot decide at 16 "
                            f"pairs; stop and ask")
        return out
    try:
        check_outputs(res)
    except OutputsDiffer as e:
        out["code"] = 2
        out["lines"].append(f"decide: correctness failure, the two arms produced different outputs in {e}")
        return out
    rows = clean_pairs(res)
    out["clean"] = len(rows)
    if len(rows) < MIN_CLEAN and res.get("stopped") == "futility":
        # bench/ab.py stops a pair series when even all-positive remaining pairs could not reach the positive bar: the
        # variant cannot be adopted, so "kept off" is decided without the 5 clean pairs adoption needs
        ratios = [r["ratio"] for r in rows]
        out.update(adopted=False, median=true_median(ratios) if ratios else None,
                   positive=(sum(1 for x in ratios if x > 1.0) / len(ratios)) if ratios else None,
                   why=f"stopped for futility after {len(rows)} clean pairs: the {MIN_POSITIVE:.0%} positive bar could not be "
                       f"reached; not adoptable")
        return out
    if len(rows) < MIN_CLEAN:
        out["code"] = 1
        out["lines"].append(f"DECISION: {len(rows)} clean pairs < {MIN_CLEAN} — the A/B cannot decide; stop and ask")
        return out
    ok, med, pos, why = adoption([r["ratio"] for r in rows], 1.0 + margin, MIN_CLEAN, MIN_POSITIVE)
    out.update(adopted=ok, median=med, positive=pos, why=why)
    return out


def _excluded(res):
    ex = [r for r in res["pairs"] if not (r.get("eligible") and r.get("ratio") is not None)]
    return "; ".join(f"pair {r['pair']}: {','.join(r.get('reasons') or ['no ratio'])}" for r in ex) or "none"


def _table(res):
    lines = ["| pair | order | A (tok/s) | B (tok/s) | ratio | clean |", "|---|---|---|---|---|---|"]
    for r in res["pairs"]:
        a = "-" if r.get("a") is None else f"{r['a']:.3f}"
        b = "-" if r.get("b") is None else f"{r['b']:.3f}"
        ra = "-" if r.get("ratio") is None else f"{r['ratio']:.4f}"
        lines.append(f"| {r['pair']} | {r['order']} | {a} | {b} | {ra} | {'yes' if r.get('eligible') and r.get('ratio') is not None else 'no'} |")
    return lines


def _set_line(path, section, key, value, comment):
    """Replace `key = ...` inside [section] of a toml file, keeping everything else byte for byte."""
    p = _abs(path)
    text = p.read_text().splitlines()
    cur, done = None, False
    for i, line in enumerate(text):
        m = re.match(r"^\[(\w+)\]", line)
        if m:
            cur = m.group(1)
        elif cur == section and re.match(rf"^{key}\s*=", line):
            pad = line.split("#", 1)[0]
            width = max(len(pad.rstrip()) + 1, 32)
            text[i] = f"{key} = {value}".ljust(width) + f"# {comment}"
            done = True
            break
    if not done:
        raise BadInput(f"{path}: [{section}] {key} not found")
    p.write_text("\n".join(text) + "\n")


def write_f6_decision(kind, res, j, d, dest, variant, arms_b, fields, reversal):
    cap = capture_rel(d)
    decision = fields["decision"]
    lines = [f"# f6-{kind}-d5", f"Date: {date.today().isoformat()}", f"Capture: {cap}",
             f"Decision: {decision}", f"Evidence: {cap}result.json, {H3}", f"Reversal: {reversal}", ""]
    lines += [f"- A arm `{res['arms']['A']}`, B arm `{res['arms']['B']}`; D5, 16 planned pairs, {res['pairs_run']} run, "
              f"{j['clean']} clean; excluded: {_excluded(res)}",
              f"- paired median {j['median']:.4f} (adoption bar {1.0 + j['margin']:.2f}), positive fraction {j['positive']:.3f} "
              f"(bar {MIN_POSITIVE:.2f}), clean pairs {j['clean']} (bar {MIN_CLEAN}); {j['why']}",
              f"- F3 A/A noise [aa] median_abs_dev {j['noise']:.6f} read from {H3}; adoption margin {j['margin']:g} ≥ noise",
              "- outputs identical across the two arms in every pair that ran"]
    lines += [f"- {x}" for x in fields.get("extra", [])] + [""] + _table(res) + [""]
    lines.append(f"Verdict: PASS — {decision}: paired median {j['median']:.4f}, {j['clean']} clean pairs")
    _abs(dest).parent.mkdir(parents=True, exist_ok=True)
    _abs(dest).write_text("\n".join(lines) + "\n")


def cmd_f6_lookahead(a):
    d = db_check.latest(a.latest) if a.latest else Path(a.capture)
    if d is None or not Path(d).is_dir():
        print(f"no capture for {a.latest or a.capture}", file=sys.stderr)
        return 3
    try:
        res = load_ab(d)
        j = f6_judge(res, LA_MARGIN, aa_noise())
    except NoSnapshot as e:
        print(f"decide: pairs without snapshot ({e})", file=sys.stderr)
        return 3
    except (BadInput, KeyError, ValueError, OSError) as e:
        print(f"decide f6-lookahead: unreadable input: {e}", file=sys.stderr)
        return 3
    for line in j["lines"]:
        print(line)
    if j["code"]:
        return j["code"]
    variant = res["arms"]["B"]
    if variant not in ("attention", "attention-corrected", "boundary2"):
        print(f"decide f6-lookahead: B arm {variant!r} is not a lookahead mode", file=sys.stderr)
        return 3
    mode = variant if j["adopted"] else "off"
    rows = clean_pairs(res)
    c = lookahead_counters(arm_files(d, rows, "B"))
    print(f"lookahead {variant}: median {j['median']:.4f}, positive {j['positive']:.3f}, clean pairs {j['clean']} -> "
          f"{'ADOPT' if j['adopted'] else 'KEEP OFF'} ({j['why']})")
    if a.write:
        extra = [f"B arm tickets over the clean pairs: issued {c['issued']}, adopted {c['adopted']}, discarded {c['discarded']}, "
                 f"failed {c['failed']} ({c['arms_with_lookahead']} arm outputs)"]
        write_f6_decision("lookahead", res, j, d, LA_RECORD, variant, None,
                          {"decision": f"ADOPT_{mode.upper().replace('-', '_')}" if j["adopted"] else "KEEP_OFF", "extra": extra},
                          f"`[lookahead] mode` in platform.toml is the default; SLOTSTREAM_LOOKAHEAD=off turns it off per "
                          f"process. Rule: median >= 1.02, >= 80% positive, >= 5 clean pairs, identical outputs.")
        _set_line(PLATFORM_TOML, "lookahead", "mode", f'"{mode}"',
                  f"step 78: db/records/decisions/f6-lookahead-d5.md ({'adopted' if j['adopted'] else 'kept off'})")
        print(f"wrote {LA_RECORD}; platform.toml [lookahead] mode = {mode}")
    return 0


def cmd_f6_mtp(a):
    if a.mode != "D5":
        print("decide f6-mtp: only D5 was measured (D1/D2: off (not measured))", file=sys.stderr)
        return 3
    d = db_check.latest(a.latest) if a.latest else Path(a.capture)
    if d is None or not Path(d).is_dir():
        print(f"no capture for {a.latest or a.capture}", file=sys.stderr)
        return 3
    try:
        res = load_ab(d)
        j = f6_judge(res, MTP_MARGIN, aa_noise())
    except NoSnapshot as e:
        print(f"decide: pairs without snapshot ({e})", file=sys.stderr)
        return 3
    except (BadInput, KeyError, ValueError, OSError) as e:
        print(f"decide f6-mtp: unreadable input: {e}", file=sys.stderr)
        return 3
    for line in j["lines"]:
        print(line)
    if j["code"]:
        return j["code"]
    rows = clean_pairs(res)
    ac = acceptance(arm_files(d, rows, "B"))
    rate = [ac["accepted"][k] / ac["attempts"][k] if ac["attempts"][k] else 0.0 for k in (0, 1)]
    print(f"mtp resident: median {j['median']:.4f}, positive {j['positive']:.3f}, clean pairs {j['clean']} -> "
          f"{'ADOPT' if j['adopted'] else 'KEEP OFF'} ({j['why']}); acceptance depth 1 {ac['accepted'][0]}/{ac['attempts'][0]} = "
          f"{rate[0]:.3f}, depth 2 {ac['accepted'][1]}/{ac['attempts'][1]} = {rate[1]:.3f}")
    if a.write:
        m = re.search(r"lookahead-(\S+)$", res["arms"]["A"])
        extra = [f"acceptance per depth over the clean pairs (B arm): depth 1 {ac['accepted'][0]}/{ac['attempts'][0]} = {rate[0]:.3f}, "
                 f"depth 2 {ac['accepted'][1]}/{ac['attempts'][1]} = {rate[1]:.3f}; {ac['rounds']} verify rounds, "
                 f"{ac['emitted'] / ac['rounds']:.2f} tokens per round" if ac["rounds"] else
                 "acceptance per depth: no MTP counters in the arm outputs",
                 f"lookahead in both arms: {m.group(1) if m else 'unknown'}; cohort B0 (test split)"]
        write_f6_decision("mtp", res, j, d, MTP_RECORD, "resident", None,
                          {"decision": "ADOPT_RESIDENT" if j["adopted"] else "KEEP_OFF", "extra": extra},
                          "`[mtp] D5` in slotstream_spark/config/policy.toml is the default; SLOTSTREAM_MTP=off|resident "
                          "overrides it per process. Rule: median >= 1.05, >= 80% positive, >= 5 clean pairs, identical outputs.")
        pol = _abs(POLICY_TOML)
        pol.write_text(
            "# Engine policy per memory mode. Values: \"off\" | \"resident\". \"streaming\" is not implemented in v1 and raises.\n"
            "# D5 is set by step 78 from db/records/decisions/f6-mtp-d5.md; D1 and D2 were not measured and stay off.\n\n"
            "[mtp]\n"
            "D1 = \"off\"                      # off (not measured)\n"
            "D2 = \"off\"                      # off (not measured)\n"
            f"D5 = \"{'resident' if j['adopted'] else 'off'}\"                  # step 78: f6-mtp-d5.md "
            f"({'adopted' if j['adopted'] else 'kept off'})\n")
        print(f"wrote {MTP_RECORD}; policy.toml [mtp] D5 = {'resident' if j['adopted'] else 'off'}")
    return 0


def lookahead_decision(path=LA_RECORD):
    m = re.search(r"^Decision: (\w+)", _abs(path).read_text(), re.M)
    if not m:
        raise BadInput(f"{path}: no Decision: line")
    return m.group(1)


def stopgo_ratio(path=STOPGO):
    try:
        m = re.search(r"^Median ratio ([0-9.]+)", _abs(path).read_text(), re.M)
    except OSError:
        return None                           # informative only
    return float(m.group(1)) if m else None


def is_mtp_ab(res):
    """The MTP A/B's arm A is MTP off with the decided lookahead: it is the arm in force whatever the lookahead decision."""
    return str(res.get("label", "")).startswith("f6-ab-mtp")


def f6_target(res, adopted, yard, minimum):
    """-> (median tok/s of the arm in force, clean pairs, axis_d_ratio, lines, code)."""
    check_snapshots(res)
    rows = clean_pairs(res)
    arm = "a" if is_mtp_ab(res) else ("b" if adopted else "a")
    vals = [r[arm] for r in rows if r.get(arm) is not None]
    if not vals:
        raise BadInput("no clean pair with a decode value")
    med = true_median(vals)
    ratio = med / float(yard["decode_tps_median"])
    lines, code = [], 0
    if med < minimum:
        lines.append(f"D5 decode {med:.2f} < {minimum:g} tok/s at end of F6: design decision required")
        code = 1
    return med, len(vals), ratio, lines, code


def cmd_f6_target(a):
    d = db_check.latest(a.latest) if a.latest else Path(a.capture)
    if d is None or not Path(d).is_dir():
        print(f"no capture for {a.latest or a.capture}", file=sys.stderr)
        return 3
    try:
        res = load_ab(d)
        yard = yardstick.yardstick()
        decision = lookahead_decision()
        adopted = decision.startswith("ADOPT")
        med, n, ratio, lines, code = f6_target(res, adopted, yard, a.min)
    except NoSnapshot as e:
        print(f"decide: pairs without snapshot ({e})", file=sys.stderr)
        return 3
    except (BadInput, KeyError, ValueError, OSError) as e:
        print(f"decide f6-target: unreadable input: {e}", file=sys.stderr)
        return 3
    yd = float(yard["decode_tps_median"])
    axis = "WIN" if ratio >= 1.0 else "LOSE"
    same_day = stopgo_ratio()
    axis_line = (f"vs llama.cpp arm B ({yd:g} tok/s, full 43,520 MiB cache, LB2): {ratio * 100:.1f}% — Axis D: {axis}")
    mtp_ab = is_mtp_ab(res)
    if mtp_ab:
        arm_txt = f"MTP A/B arm A: lookahead {decision}, MTP off, cohort B0"
    else:
        arm_txt = f"{'lookahead arm B' if adopted else 'lookahead off arm A'}, MTP off"
    print(f"D5 decode median {med:.3f} tok/s over {n} clean pairs ({arm_txt})")
    print(axis_line)
    for line in lines:
        print(line)
    if not a.no_write:
        cap = capture_rel(d)
        out = ["# f6-decode-d5: engine D5 decode at the end of F6", f"Date: {date.today().isoformat()}", f"Capture: {cap}",
               f"Command: tools/spark/jobs.sh start f6-ab --heavy f6-ab auto 14400 -- bash bench/f6_ab.sh (A/B label {res.get('label')})",
               "", (f"- arm in force: MTP A/B arm A ({res['arms']['A']}), lookahead {decision}, MTP off; metric (N-1)/sum(inter-token) per pair"
                    if mtp_ab else
                    f"- arm in force: {'lookahead ' + str(res['arms']['B']) if adopted else 'lookahead off'} (decision {decision}), "
                    "MTP off; metric (N-1)/sum(inter-token) per pair"),
               f"- D5 decode median {med:.3f} tok/s over {n} clean pairs (target >= {a.min:g})",
               f"- {axis_line}", f"- axis_d_ratio = {ratio:.4f}",
               "- informative at F6: the F3 answer in db/records/decisions/stopgo-f3.md governs and F9 decides" +
               (f"; the F3 same-day paired ratio was {same_day:.4f} (arm B ran with a reduced cache)" if same_day else ""),
               ""]
        out += [f"- note: {t}" for t in a.note]
        out += lines
        out.append("Verdict: PASS — D5 decode median at or above the F6 target" if code == 0 else
                   f"Verdict: FAIL — D5 decode {med:.2f} tok/s below the {a.min:g} tok/s target")
        _abs(TARGET_RECORD).parent.mkdir(parents=True, exist_ok=True)
        _abs(TARGET_RECORD).write_text("\n".join(out) + "\n")
        print(f"wrote {TARGET_RECORD}")
    return code


F8_G2 = "db/records/measurements/g2-g3.md"
F8_RECORD = "db/records/decisions/f8-entry.md"
F8_THRESHOLD = 0.10
F8_MIN_TOKENS = 16
# The host-and-wait share F8 can remove: host bookkeeping that runs with the GPU idle (sample, n-gram hash and
# embed at the token boundary, the wait for the n-gram rows, graph launches, the hit-path ensure, and what the
# partition leaves as `other`). gpu_wait is the host waiting for a busy GPU; the miss read and its copy tail wait
# for the NVMe, which a token graph with host-memory waits still waits for.
F8_HOST_PHASES = ("sample", "ngram_hash", "embed_read", "ngram_wait", "launch", "ensure_hits")
F8_IO_PHASES = ("expert_miss_read", "expert_copy_tail")


def f8_fraction(res):
    """Pooled and per-prompt host-and-wait share of the token time from a `bench decode --profile` result.
    Raises BadInput for a result that is not a profile or whose partition does not add up."""
    if not res.get("profile") or not res.get("prompts"):
        raise BadInput("not a --profile result (no `profile` flag or no prompts)")
    per, host_tot, io_tot, wall_tot = {}, 0, 0, 0
    for k, e in sorted(res["prompts"].items()):
        for key in ("phases_ns", "wall_ns", "other_ns", "tokens_profiled"):
            if key not in e:
                raise BadInput(f"prompt {k!r}: no {key}")
        if e["tokens_profiled"] < F8_MIN_TOKENS:
            raise BadInput(f"prompt {k!r}: {e['tokens_profiled']} profiled tokens < {F8_MIN_TOKENS}")
        ph = e["phases_ns"]
        if sum(ph.values()) + e["other_ns"] != e["wall_ns"]:
            raise BadInput(f"prompt {k!r}: phases + other do not add up to the wall time")
        host = sum(ph[p] for p in F8_HOST_PHASES) + e["other_ns"]
        io = sum(ph[p] for p in F8_IO_PHASES)
        per[k] = {"tokens": e["tokens_profiled"], "wall_ms": e["wall_ns"] / 1e6, "host_ms": host / 1e6,
                  "io_ms": io / 1e6, "gpu_wait_ms": ph["gpu_wait"] / 1e6, "fraction": host / e["wall_ns"],
                  "tok_s": e.get("tok_s")}
        host_tot, io_tot, wall_tot = host_tot + host, io_tot + io, wall_tot + e["wall_ns"]
    return host_tot / wall_tot, io_tot / wall_tot, per


def f8_g2(path=F8_G2):
    p = _abs(path)
    if not p.exists():
        raise BadInput(f"{path}: missing")
    m = re.search(r"^Verdict: (PASS|FAIL|INFO)", p.read_text(), re.M)
    if not m:
        raise BadInput(f"{path}: no Verdict: line")
    return m.group(1)


def cmd_f8_entry(a):
    d = db_check.latest(a.latest) if a.latest else Path(a.capture)
    if d is None or not (Path(d) / "result.json").exists():
        print("decide: no result.json in the capture", file=sys.stderr)
        return 3
    try:
        res = json.loads((Path(d) / "result.json").read_text())
        fraction, io_fraction, per = f8_fraction(res)
        g2 = f8_g2()
    except (BadInput, ValueError) as e:
        print(f"decide: {e}", file=sys.stderr)
        return 3
    enter = fraction >= F8_THRESHOLD and g2 == "PASS"
    decision = "enter" if enter else "skip"
    cap = capture_rel(d)
    why = (f"host+waits {fraction * 100:.1f}% of the token {'>=' if fraction >= F8_THRESHOLD else '<'} "
           f"{F8_THRESHOLD * 100:.0f}% and g2 {g2}")
    print(f"F8 {decision}: {why}")
    if a.write:
        lines = ["# f8-entry", f"Date: {date.today().isoformat()}", f"Capture: {cap}", f"Decision: {decision}",
                 f"Evidence: {cap}result.json, {F8_G2}", "Reversal: re-measure the partition after a change to the "
                 "per-layer host path; enter when the host-and-wait share reaches 10% with the g2 verdict pass", "",
                 f"- capture id: {Path(d).name}",
                 f"- fraction = host-and-wait share = {fraction:.4f} ({fraction * 100:.2f}% of the token time, pooled over "
                 f"{len(per)} prompts); enter iff fraction >= {F8_THRESHOLD:g} and the g2 verdict is pass",
                 f"- g2 verdict (memop in graph, {F8_G2}): {g2}",
                 f"- counted as host-and-wait: {', '.join(F8_HOST_PHASES)} and `other` (the host work left over by the partition); "
                 "counted as neither: gpu_wait (the GPU is busy) and the expert miss read plus copy tail "
                 f"({io_fraction * 100:.2f}% of the token, waiting on the NVMe, which a token graph still waits for)",
                 "- `[f8] token_graph = false` in slotstream_spark/config/platform.toml until a token graph is adopted (step 92)",
                 "", "| prompt | profiled tokens | wall ms | host+waits ms | share | miss read ms | gpu_wait ms | tok/s |",
                 "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {k} | {v['tokens']} | {v['wall_ms']:.1f} | {v['host_ms']:.1f} | {v['fraction'] * 100:.2f}% | "
                  f"{v['io_ms']:.1f} | {v['gpu_wait_ms']:.1f} | {v['tok_s']} |" for k, v in per.items()]
        lines += ["", f"Verdict: PASS — F8 {decision}: {why}"]
        _abs(F8_RECORD).parent.mkdir(parents=True, exist_ok=True)
        _abs(F8_RECORD).write_text("\n".join(lines) + "\n")
        _set_line("slotstream_spark/config/platform.toml", "f8", "token_graph", "false",
                  f"step 90: db/records/decisions/f8-entry.md ({decision}); true only after step 92 adopts the token graph")
        print(f"wrote {F8_RECORD}")
    return 0


F8_TG_RECORD = "db/records/decisions/f8-token-graph.md"
F8_TG_MARGIN = 0.05                 # adoption bar 1.05
F8_SOAK_MIN_S = 8 * 3600


def f8_tg_judge(res, soak, noise):
    """-> dict like f6_judge plus `reasons` (every criterion that failed) and `soak`. The token graph is adopted iff the paired
    median >= 1.05 (margin checked against the F3 A/A noise), >= 80% positive pairs, identical outputs, and an 8 h soak with
    no failed arm, no differing spot bit-check and 0 CUDA context losses."""
    j = f6_judge(res, F8_TG_MARGIN, noise)
    j["reasons"], j["soak"] = [], soak
    if j["code"]:
        return j
    if not j["adopted"]:
        j["reasons"].append(f"A/B: {j['why']}")
    s = soak or {}
    if not s or s.get("ab_rc") not in (0, 1):
        j["reasons"].append("no soak record")
    elif s["soak_seconds"] < F8_SOAK_MIN_S:
        j["reasons"].append(f"soak {s['soak_seconds']} s < {F8_SOAK_MIN_S} s (8 h)" if j["adopted"] else
                            "soak skipped because the A/B did not pass")
    else:
        if s["soak_arms_failed"]:
            j["reasons"].append(f"{s['soak_arms_failed']} soak arms failed")
        if s["spot_checks_differing"]:
            j["reasons"].append(f"{s['spot_checks_differing']} spot bit-checks differ")
        if s["cuda_context_losses"]:
            j["reasons"].append(f"{s['cuda_context_losses']} CUDA context losses / GPU errors in the kernel log")
        if s["spot_checks"] < 1:
            j["reasons"].append("no spot bit-check ran")
        if s.get("early_stop"):
            j["reasons"].append(f"soak stopped early: {s['early_stop']}")
    if soak and soak.get("token_graph_tokens", 0) <= 0:
        j["reasons"].append("the arms never ran on the token graph (no counters)")
    j["adopted"] = j["adopted"] and not j["reasons"]
    return j


def cmd_f8_token_graph(a):
    d = db_check.latest(a.latest) if a.latest else Path(a.capture)
    sd = db_check.latest(a.soak_latest) if a.soak_latest else Path(a.soak)
    if d is None or not Path(d).is_dir() or sd is None or not Path(sd).is_dir():
        print("decide: no capture for the A/B or the soak", file=sys.stderr)
        return 3
    try:
        res = load_ab(d)
        soak = json.loads((Path(sd) / "f8_soak.json").read_text())
        j = f8_tg_judge(res, soak, aa_noise())
    except NoSnapshot as e:
        print(f"decide: pairs without snapshot ({e})", file=sys.stderr)
        return 3
    except (BadInput, KeyError, ValueError, OSError) as e:
        print(f"decide f8-token-graph: unreadable input: {e}", file=sys.stderr)
        return 3
    for line in j["lines"]:
        print(line)
    if j["code"]:
        return j["code"]
    decision = "ADOPT" if j["adopted"] else "KEEP_OFF"
    why = "; ".join(j["reasons"]) or "every criterion met"
    print(f"F8 token graph: median {j['median']:.4f}, positive {j['positive']:.3f}, clean pairs {j['clean']} -> {decision} ({why})")
    if a.write:
        cap, scap = capture_rel(d), capture_rel(sd)
        lines = ["# f8-token-graph", f"Date: {date.today().isoformat()}", f"Capture: {cap}", f"Decision: {decision}",
                 f"Evidence: {cap}result.json, {scap}f8_soak.json, {H3}",
                 "Reversal: `[f8] token_graph` in slotstream_spark/config/platform.toml; SLOTSTREAM_TOKEN_GRAPH=on|off overrides it "
                 "per process. Rule: paired median >= 1.05, >= 80% positive pairs, identical outputs, an 8 h soak with 0 CUDA "
                 "context losses.", "",
                 f"- A arm `{res['arms']['A']}`, B arm `{res['arms']['B']}`; D5, {res['pairs_planned']} planned pairs, {res['pairs_run']} run, "
                 f"{j['clean']} clean; excluded: {_excluded(res)}",
                 f"- paired median {j['median']:.4f} (adoption bar {1.0 + j['margin']:.2f}), positive fraction {j['positive']:.3f} "
                 f"(bar {MIN_POSITIVE:.2f}); F3 A/A noise [aa] median_abs_dev {j['noise']:.6f} from {H3}",
                 f"- soak: {soak['soak_seconds']} s of {soak['soak_seconds_wanted']} s wanted, {soak['soak_arms']} arms "
                 f"({soak['soak_arms_failed']} failed), {soak['spot_checks']} spot bit-checks ({soak['spot_checks_differing']} differ), "
                 f"{soak['cuda_context_losses']} CUDA context losses",
                 f"- token graph counters over the arms that used it: {soak['token_graph_tokens']} tokens on the graph, "
                 f"{soak['token_graph_segmented_tokens']} on the F3 path, {soak['token_graph_fallbacks']} wait fallbacks",
                 f"- reasons: {why}", ""] + _table(res) + ["",
                 f"Verdict: PASS — {decision}: paired median {j['median']:.4f}, {j['clean']} clean pairs, soak {soak['soak_seconds']} s"]
        _abs(F8_TG_RECORD).parent.mkdir(parents=True, exist_ok=True)
        _abs(F8_TG_RECORD).write_text("\n".join(lines) + "\n")
        _set_line(PLATFORM_TOML, "f8", "token_graph", "true" if j["adopted"] else "false",
                  f"step 92: db/records/decisions/f8-token-graph.md ({decision})")
        print(f"wrote {F8_TG_RECORD}; [f8] token_graph = {'true' if j['adopted'] else 'false'}")
    return 0


# ----------------------------------------------------------------------------------------------------------- F9 (step 95)
F9_QUALITY = "db/records/measurements/f9-quality.md"
F9_D5 = "db/records/measurements/f9-campaign-d5.md"
F9_D1 = "db/records/measurements/f9-campaign-d1.md"
F9_REF = "db/records/measurements/f9-reference.md"
F9_SWEEP_TENSORS = 3215         # the checkpoint arrays of the exactness sweep
# One mapping-correlation floor for every family (BRIEF 2026-10-15 01:25). F9_CHECK_TENSORS counts the template tensors the
# mapping check and the build walk (len(specs) in tools/repack_q4_1.py, one check.json entry per spec): a different
# population from F9_SWEEP_TENSORS; the two counts are never mixed in one row.
Q41_CORR_FLOOR = 0.95
F9_CHECK_TENSORS = 1224
Q41_EXCEPTIONS = "slotstream_spark/config/q4_1_exceptions.json"
F9_DECODE_BAR = 23.0
PROBE_ITEMS = 15


def _contract():
    gates = str(ROOT / "tests" / "gates")
    if gates not in sys.path:
        sys.path.insert(0, gates)
    import contract
    return contract


def _json_file(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError) as e:
        raise BadInput(f"{p}: {e}")


def rule_lines(c):
    """The four literal rule lines of the record, built from the contract's own parameters."""
    L, P = c.logits, c.raw["perplexity"]
    ceil = "/".join(f"{float(L[k]):g}" for k in ("kl_mean_ceiling_nats", "kl_p99_ceiling_nats", "kl_max_ceiling_nats"))
    logits = (f"Logits rule: seq KL (factor {float(L['kl_factor']):.1f}, floor = calibrated limit, ceilings {ceil} nats; "
              f"top-1 calibrated), design decision 18")
    routing = (f"Routing rule: seq_rule (factor {float(c.routing['seq_factor']):.1f}, ceiling c_k/"
               f"{float(c.routing['seq_ceiling_ratio']):g}), design decision 16")
    top1 = (f"Top-1 rule: pooled over the 8 reserved prompts (sum of agreeing positions / sum of positions >= top1_min "
            f"{float(L['top1_min']):.5f}, calibrated); per-prompt top-1 informative; design decision 21, user decision 2026-10-14")
    ppl = (f"PPL rule: engine vs transformers oracle on the same {int(P['oracle_windows'])} wikitext windows, "
           f"|PPL_engine/PPL_oracle - 1| <= {float(P['oracle_rel_tol']) * 100:g}%; README reference {P['wikitext2_ref']} "
           f"+-{float(P['rel_tol']) * 100:g}% informative; design decision 21, user decision 2026-10-14")
    return logits, routing, top1, ppl


def ppl_ratio(engine_rows, other_rows):
    """Engine PPL against another scorer's over the same windows -> dict; BadInput when the windows do not line up."""
    import math
    if len(engine_rows) != len(other_rows) or not engine_rows:
        raise BadInput(f"window counts differ: {len(engine_rows)} vs {len(other_rows)}")
    per = []
    for a, b in zip(engine_rows, other_rows):
        if int(a["start"]) != int(b["start"]) or int(a["tokens"]) != int(b["tokens"]):
            raise BadInput(f"window {a.get('window')}: start/tokens differ ({a['start']}/{a['tokens']} vs {b['start']}/{b['tokens']})")
        per.append({"window": a.get("window"), "start": int(a["start"]), "nll_a": float(a["nll"]), "nll_b": float(b["nll"]),
                    "ratio_minus_1": math.exp((float(a["nll"]) - float(b["nll"])) / int(a["tokens"])) - 1.0})
    tokens = sum(int(r["tokens"]) for r in engine_rows)
    na, nb = math.fsum(r["nll_a"] for r in per), math.fsum(r["nll_b"] for r in per)
    pa, pb = math.exp(na / tokens), math.exp(nb / tokens)
    return {"ppl_a": pa, "ppl_b": pb, "ratio_minus_1": pa / pb - 1.0, "per_window": per, "tokens": tokens}


def f9_ppl(ppl, oracle, c):
    """-> gate dict. Validates the engine's 145-window file and the oracle's file, judges the engine against the oracle on the
    first `oracle_windows` windows; the 145-window figure against the README reference is informative only."""
    import math
    cfg = c.raw["perplexity"]
    ref, windows, wt = float(cfg["wikitext2_ref"]), int(cfg["windows"]), int(cfg["window_tokens"])
    nw, otol = int(cfg["oracle_windows"]), float(cfg["oracle_rel_tol"])
    per = ppl.get("per_window") or []
    if len(per) != windows:
        raise BadInput(f"ppl.json has {len(per)} windows, expected {windows}")
    tokens = sum(int(r["tokens"]) for r in per)
    if tokens != windows * (wt - 1) or any(int(r["tokens"]) != wt - 1 for r in per):
        raise BadInput(f"ppl.json scored {tokens} tokens, expected {windows * (wt - 1)} ({windows} x {wt - 1})")
    nll = math.fsum(float(r["nll"]) for r in per)
    p = math.exp(nll / tokens)
    if abs(p - float(ppl["ppl"])) > 1e-9 * p:
        raise BadInput(f"stored ppl {ppl['ppl']} differs from the per-window NLL's {p}")
    orows = oracle.get("per_window") or []
    if len(orows) != nw:
        raise BadInput(f"oracle_ppl.json has {len(orows)} windows, expected {nw}")
    for k, r in enumerate(orows):
        if int(r.get("window", -1)) != k or int(r["start"]) != k * wt or int(r["tokens"]) != wt - 1:
            raise BadInput(f"oracle window {k}: window/start/tokens are {r.get('window')}/{r.get('start')}/{r.get('tokens')}")
    for key in ("ids_sha256", "rows_sha256"):
        if not oracle.get(key) or oracle.get(key) != ppl.get(key):
            raise BadInput(f"oracle_ppl.json {key} {oracle.get(key)} differs from the engine's {ppl.get(key)}")
    if not oracle.get("dtype"):
        raise BadInput("oracle_ppl.json has no dtype")
    r = ppl_ratio(per[:nw], orows)
    worst = max(r["per_window"], key=lambda x: abs(x["ratio_minus_1"]))
    return {"ppl_a": r["ppl_a"], "ppl_b": r["ppl_b"], "ratio_minus_1": r["ratio_minus_1"], "tol": otol,
            "ok": bool(abs(r["ratio_minus_1"]) <= otol), "windows": nw, "tokens": r["tokens"], "dtype": oracle["dtype"],
            "per_window": r["per_window"], "worst_window": worst["window"], "worst_ratio": worst["ratio_minus_1"],
            "engine_above": sum(1 for x in r["per_window"] if x["nll_a"] > x["nll_b"]),
            "ppl145": p, "ref": ref, "rel_diff": p / ref - 1.0, "tokens145": tokens, "windows145": windows,
            "text_sha": ppl.get("rows_sha256"), "ids_sha": ppl.get("ids_sha256")}


def f9_prompt(pid, r, c):
    """Recomputes one reserved prompt's verdict from the stored metrics and pairs (never from a stored `ok`).
    -> dict; raises BadInput for a record without the pair data."""
    lg = r.get("logits") or {}
    if "pair" not in lg:
        raise BadInput(f"{pid}: teacher_forced.json has no logits.pair")
    metrics = {k: lg[k] for k in ("top1", "kl_mean", "kl_p99", "kl_max")}
    tokens = int(r.get("tokens", 0))
    if "top1_agree" in lg and "positions" in lg:
        agree, positions = int(lg["top1_agree"]), int(lg["positions"])
        if positions != tokens or abs(float(lg["top1"]) * positions - agree) > 1e-6:
            raise BadInput(f"{pid}: top1_agree {agree}/{positions} is inconsistent with top1 {lg['top1']} x {tokens} tokens")
    else:
        exact = float(lg["top1"]) * tokens
        agree, positions = round(exact), tokens
        if abs(exact - agree) >= 1e-6:
            raise BadInput(f"{pid}: top1 {lg['top1']} x {tokens} tokens = {exact} is not a whole number of positions")
    judged = c_judge(metrics, lg["pair"], c)
    layers = r.get("layers")
    if not layers:
        raise BadInput(f"{pid}: teacher_forced.json has no per-layer entries")
    bad_layers, worst, bound = [], (0.0, -1, ""), []
    for ly in layers:
        for key in ("routing_pair", "routing_limit_src", "rel_l2_pair", "limit_src"):
            if key not in ly:
                raise BadInput(f"{pid}: layer {ly.get('layer')} has no {key}")
        k = int(ly["layer"])
        ct = _contract()
        lim_l, src_l = ct.seq_limit("layer", k, ly["rel_l2_pair"], c)
        lim_r, src_r = ct.seq_limit("routing", k, ly["routing_pair"], c)
        sep_l, sep_r = bool(c.layer["separable"][k]), bool(c.routing["separable"][k])
        ok_l = (float(ly["rel_l2"]) <= lim_l) if sep_l else True
        ok_r = (float(ly["routing"]) <= lim_r) if sep_r else True
        if not (ok_l and ok_r):
            bad_layers.append(k)
        if sep_r and float(ly["routing"]) / lim_r > worst[0]:
            worst = (float(ly["routing"]) / lim_r, k, src_r)
        if src_l != "calibrated" or src_r != "calibrated":
            bound.append(k)
    kl_ok = _contract().logits_kl_ok(judged)
    return {"id": pid, "tokens": tokens, "agree": agree, "positions": positions, "kl_ok": kl_ok, "ngram_ok": r.get("ngram_ok") is True, "judged": judged,
            "failing_layers": bad_layers, "worst_routing_ratio": worst[0], "worst_routing_layer": worst[1],
            "worst_routing_src": worst[2], "bound_layers": bound,
            "ok": bool(kl_ok and r.get("ngram_ok") is True and not bad_layers)}


def c_judge(metrics, pair, c):
    return _contract().logits_judge_seq(metrics, pair, c)


def f9_quality(qdir, odir):
    """-> (rows [(name, value, limit, ok)], info lines, decisions, code, contract, ppl dict). `ok` None = informative row."""
    qdir, odir = Path(qdir), Path(odir)
    c = _contract().load()
    ppl = f9_ppl(_json_file(qdir / "ppl.json"), _json_file(odir / "oracle_ppl.json"), c)
    tf = _json_file(qdir / "teacher_forced.json")
    prompts = [f9_prompt(pid, r, c) for pid, r in sorted(tf["prompts"].items())]
    if len(prompts) != 8:
        raise BadInput(f"teacher_forced.json has {len(prompts)} prompts, expected the 8 reserved ones")
    try:
        probe = (qdir / "probe.txt").read_text()
    except OSError as e:
        raise BadInput(f"probe.txt: {e}")
    m = re.search(r"quality probe: passed (\d+), failed (\d+)", probe)
    if not m:
        raise BadInput("probe.txt has no `quality probe: passed N, failed M` line")
    passed, failed = int(m.group(1)), int(m.group(2))
    ct = _contract()
    try:
        pooled = ct.top1_pooled([(p["id"], p["agree"], p["positions"]) for p in prompts], c)
    except ValueError as e:
        raise BadInput(str(e))
    pair_agree = sum(float(p["judged"]["pair"]["top1"]) * p["positions"] for p in prompts)
    pair_pooled = pair_agree / pooled["positions"]
    rows, info, decisions, code = [], [], [], 0
    P = c.raw["perplexity"]
    rows.append((f"wikitext-2 PPL engine vs transformers oracle over {ppl['windows']} windows / {ppl['tokens']:,} tokens",
                 f"engine {ppl['ppl_a']:.4f}, oracle {ppl['ppl_b']:.4f}, ratio-1 {ppl['ratio_minus_1'] * 100:+.2f}%",
                 f"|ratio-1| <= {ppl['tol'] * 100:g}% (oracle dtype {ppl['dtype']})", ppl["ok"]))
    rows.append(("top-1 pooled over 8 reserved prompts", f"{pooled['top1']:.4f} ({pooled['agree']}/{pooled['positions']})",
                 f">= {pooled['top1_min']:.5f} (calibrated; pooled pair {pair_pooled:.4f})", pooled["ok"]))
    for p in prompts:
        pt = p["agree"] / p["positions"]
        rows.append((f"{p['id']} top-1", f"{pt:.4f} ({p['agree']}/{p['positions']})",
                     f"informative (pair {float(p['judged']['pair']['top1']):.4f})", None))
    for p in prompts:
        j = p["judged"]
        lim = j["limits_seq"]
        if not j["judgeable"]:
            over = [k for k in ("kl_mean", "kl_p99", "kl_max") if j["pair"][k] > float(c.logits[{"kl_mean": "kl_mean_ceiling_nats", "kl_p99": "kl_p99_ceiling_nats", "kl_max": "kl_max_ceiling_nats"}[k]])]
            decisions.append(f"DESIGN GATE: {p['id']}: the oracle's own bf16-vs-fp32 pair exceeds a [logits] ceiling on this sequence "
                             f"({', '.join(over)}; pair {json.dumps(j['pair'])}) — the per-sequence rule cannot judge it, design decision required")
            code = 1
        for key, lk in (("kl_mean", "kl_mean_max_nats"), ("kl_p99", "kl_p99_max_nats"), ("kl_max", "kl_max_nats")):
            rows.append((f"{p['id']} {key}", f"{j[key]:.5f}", f"<= {lim[lk]:.5f} ({j['limit_src'][lk]}; pair {j['pair'][key]:.5f})",
                         j[key] <= lim[lk]))
        rows.append((f"{p['id']} n-gram ids exact", "yes" if p["ngram_ok"] else "NO", "exact", p["ngram_ok"]))
        rows.append((f"{p['id']} routing and separable layers", f"{len(p['failing_layers'])} failing", "0 failing within seq_rule",
                     not p["failing_layers"]))
        info.append(f"{p['id']}: {p['tokens']} positions; worst routing ratio {p['worst_routing_ratio']:.3f} at layer "
                    f"{p['worst_routing_layer']} ({p['worst_routing_src']}); bound layers {p['bound_layers']}; "
                    f"fixed limits {'held' if j['fixed_limits_ok'] else 'not held'} (informative)")
    rows.append(("quality probe", f"{passed}/{passed + failed}", f"{PROBE_ITEMS}/{PROBE_ITEMS}", passed == PROBE_ITEMS and failed == 0))
    for name, v, lim, ok in rows:
        if ok is False:
            decisions.append(f"F9 quality gate failed: {name} {v} vs {lim}, needs a decision")
            code = 1
    info.append(f"wikitext-2 PPL over {ppl['windows145']} windows / {ppl['tokens145']:,} tokens: {ppl['ppl145']:.4f}; README reference "
                f"{ppl['ref']} ({ppl['rel_diff'] * 100:+.2f}%), informative: the reference's text/window protocol is not published (DIAG_F9.md)")
    info.append(f"oracle ratio per window: max |ratio-1| {abs(ppl['worst_ratio']) * 100:.2f}% at window {ppl['worst_window']}; "
                f"engine NLL above the oracle's on {ppl['engine_above']} of {ppl['windows']} windows")
    info.append(f"wikitext rows sha256 {ppl['text_sha']}, token ids sha256 {ppl['ids_sha']}")
    return rows, info, decisions, code, c, ppl


def write_f9_quality(rows, info, decisions, code, c, capture, oracle_capture, dest, ppl=None):
    rl = list(rule_lines(c))
    lines = ["# f9-quality: perplexity, teacher-forced agreement and behaviour probe of the engine", f"Date: {date.today().isoformat()}",
             f"Capture: {capture}", f"Capture: {oracle_capture}",
             "Command: tools/spark/jobs.sh start f9-engine --heavy f9-engine auto 16200 -- bash bench/campaign/f9_engine_job.sh "
             "(quality phase: tools/quality/ppl_wikitext.py, teacher_forced.py, quality_probe.sh); "
             "tools/spark/jobs.sh start f9-oracle --heavy f9-oracle 16384 9000 -- bash bench/campaign/f9_oracle_job.sh "
             "(tools/quality/oracle_ppl.py)"]
    lines += rl + ["", "| gate | value | limit | ok |", "|---|---|---|---|"]
    lines += [f"| {n} | {v} | {lim} | {'info' if o is None else ('yes' if o else 'NO')} |" for n, v, lim, o in rows]
    if ppl:
        lines += ["", "| window | start | engine NLL | oracle NLL | ratio-1 |", "|---|---|---|---|---|"]
        lines += [f"| {w['window']} | {w['start']} | {w['nll_a']:.2f} | {w['nll_b']:.2f} | {w['ratio_minus_1'] * 100:+.2f}% |"
                  for w in ppl["per_window"]]
    lines += [""] + [f"- {i}" for i in info] + [""] + decisions
    if code == 0:
        P = c.raw["perplexity"]
        lines.append(f"Verdict: PASS — PPL within +-{float(P['oracle_rel_tol']) * 100:g}% of the transformers oracle on "
                     f"{int(P['oracle_windows'])} windows, top-1 pooled >= {float(c.logits['top1_min']):.5f}, KL, routing and layers "
                     "within the per-sequence rules on the 8 reserved prompts, n-gram ids exact, probe 15/15")
    else:
        lines.append("Verdict: FAIL — " + "; ".join(d.split(", needs a decision")[0].split(" — ")[0] for d in decisions))
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text("\n".join(lines) + "\n")


def cmd_f9_quality(a):
    d = db_check.latest(a.latest) if a.latest else Path(a.capture)
    if d is None or not Path(d).is_dir():
        print(f"no capture for {a.latest or a.capture}", file=sys.stderr)
        return 3
    if not (a.oracle_latest or a.oracle):
        print("decide f9-quality: oracle PPL capture required (step 95d)", file=sys.stderr)
        return 3
    od = db_check.latest(a.oracle_latest) if a.oracle_latest else Path(a.oracle)
    if od is None or not Path(od).is_dir():
        print(f"no oracle capture for {a.oracle_latest or a.oracle}", file=sys.stderr)
        return 3
    try:
        rows, info, decisions, code, c, ppl = f9_quality(Path(d) / "quality", Path(od) / "quality")
    except (BadInput, KeyError, ValueError, OSError) as e:
        print(f"decide f9-quality: unreadable input: {e}", file=sys.stderr)
        return 3
    for n, v, lim, o in rows:
        print(f"{'info' if o is None else ('ok  ' if o else 'FAIL')} {n}: {v} (limit {lim})")
    for i in info:
        print(f"  {i}")
    for dl in decisions:
        print(dl)
    if a.write:
        write_f9_quality(rows, info, decisions, code, c, capture_rel(d), capture_rel(od), str(ROOT / F9_QUALITY), ppl)
        print(f"wrote {F9_QUALITY}")
    print("f9-quality: PASS" if code == 0 else "f9-quality: FAIL")
    return code


def load_cells(cap, mode):
    out = {}
    for r in sorted(Path(cap).glob(f"{mode.lower()}-*/result.json")):
        out[r.parent.name[len(mode) + 1:]] = _json_file(r)
    return out


def cell_peak(cell, mode):
    """Recomputes the peak of a cell from its stored samples; BadInput when it differs from the recorded one.
    -> (once_bytes, nonpool_bytes, drop_bytes, ledger_bytes, chunks)."""
    from slotstream_spark.memory import footprint as fp
    from slotstream_spark.memory.ledger import DEFAULT_WINDOW, Ledger
    if not cell.get("footprint_samples") or not cell.get("flags") or "peak" not in cell:
        raise BadInput(f"cell {cell.get('cell')}: no samples/flags/peak")
    pk = fp.peak([fp.sample_from_dict(d) for d in cell["footprint_samples"]], fp.flags_from_dict(cell["flags"]))
    rec = cell["peak"]
    if rec.get("once_bytes") != pk.once_bytes or rec.get("nonpool_bytes") != pk.nonpool_bytes:
        raise BadInput(f"cell {cell['cell']}: recorded peak once {rec.get('once_bytes')} / nonpool {rec.get('nonpool_bytes')} "
                       f"differs from the samples' {pk.once_bytes} / {pk.nonpool_bytes}")
    chunks = int(cell["chunks"])
    window = int(cell.get("window") or DEFAULT_WINDOW)
    return (pk.once_bytes, pk.nonpool_bytes, int(cell["memavailable_drop_bytes"]),
            Ledger(cell["mode"], window=window).expected_peak_bytes(chunks), chunks)


def f9_d5(cells, yard, stopgo):
    """-> (rows, info, decisions, justification line, code)."""
    need = ("decode-8k", "ttft-cold-2k", "ttft-cold-8k", "ttft-cold-32k")
    for n in need:
        if n not in cells:
            raise BadInput(f"the capture has no D5 cell {n}")
        if not cells[n].get("ok"):
            raise BadInput(f"D5 cell {n} did not finish ok: {cells[n].get('failures')}")
    dec, cold = cells["decode-8k"], cells["ttft-cold-8k"]
    rows, info, decisions, code = [], [], [], 0
    bad = [f"{n} sample {s['sample']}" for n in need[1:] for s in cells[n]["samples"] if s["cold"]["reused_tokens"] > 0]
    if bad:
        decisions.append(f"F9 campaign gate failed: cold sample(s) {', '.join(bad)} reused tokens from a prefix (invalid), needs a decision")
        rows.append(("cold samples with reused_tokens == 0", f"{len(bad)} violations", "0", False))
        return rows, info, decisions, None, 1
    rows.append(("cold samples with reused_tokens == 0", "all", "all", True))
    for s in cold["samples"]:
        if s["cold"]["prompt_tokens"] != 8192:
            raise BadInput(f"cold 8K sample {s['sample']} has {s['cold']['prompt_tokens']} prompt tokens")
    tps = [s["decode_tps"] for s in dec["samples"]]
    dmed = true_median(tps)
    rows.append(("D5 decode median (tok/s), decode-8k cell", round(dmed, 3), f">= {F9_DECODE_BAR:g}", dmed >= F9_DECODE_BAR))
    tt = [s["cold"]["ttft_s"] for s in cold["samples"]]
    tmed = true_median(tt)
    rows.append(("cold 8K TTFT median (s)", round(tmed, 3), f"<= {COLD_BAR_S:g}", tmed <= COLD_BAR_S))
    fm = true_median([s["followup"]["ttft_s"] for s in cold["samples"]])
    rows.append(("follow-up TTFT median (s)", round(fm, 3), f"<= {FOLLOWUP_BAR_S:g}", fm <= FOLLOWUP_BAR_S))
    ptps = true_median([s["cold"]["prefill_tps"] for s in cold["samples"]])
    info.append("decode-8k samples (tok/s): " + ", ".join(f"{x:.2f}" for x in tps) + f" (MTP policy {dec['samples'][0]['mtp_policy']}, "
                f"pool {dec['samples'][0]['pool_prefill'] and 'pre-filled' or 'cold'})")
    info.append("cold 8K TTFT samples (s): " + ", ".join(f"{x:.2f}" for x in tt))
    info.append("follow-up: " + ", ".join(f"{s['followup']['ttft_s']:.2f} s (reused {s['followup']['reused_tokens']}, new "
                                          f"{s['followup']['new_tokens']})" for s in cold["samples"]))
    for n in ("ttft-cold-2k", "ttft-cold-32k"):
        c = cells[n]
        info.append(f"{n}: {c.get('prompt_tokens') or c['tokens']} tokens, TTFT median {c['medians']['ttft_s']:.2f} s, prefill {c['medians']['prefill_tps']:.1f} tok/s "
                    f"(published, not gated)")
    for n in need:
        once, nonpool, drop, led, chunks = cell_peak(cells[n], "D5")
        info.append(f"peak {n}: once {once:,} B, nonpool {nonpool:,} B, MemAvailable drop {drop:,} B, ledger {led:,} B ({chunks} chunks) "
                    f"(informational; the hard memory criteria are the impact ones)")
    for name, v, lim, ok in rows:
        if not ok:
            decisions.append(f"F9 campaign gate failed: {name} {_fmt(v)} vs {lim} (absolute bar), needs a decision")
            code = 1
    yd, yp = float(yard["decode_tps_median"]), float(yard["prefill_tps_8k"])
    axis_d = dmed >= yd
    axis_t = tmed <= COLD_BAR_S and ptps >= AXIS_T_FACTOR * yp
    just = "decode+ttft" if axis_d and axis_t else ("decode" if axis_d else ("ttft" if axis_t else "none"))
    jline = (f"Justification: {just} (decode {dmed:.2f} tok/s = {dmed / yd * 100:.1f}% of arm B {yd:g}; cold 8K prefill {ptps:.1f} tok/s = "
             f"{ptps / yp:.1f}x arm B {yp:g})")
    if just == "none":
        decisions.append("F9: engine beats llama.cpp arm B on neither axis — needs a decision")
        code = 1
    elif just == "ttft" and stopgo != "GO_ON_TTFT":
        decisions.append(f"DECISION: engine justified on TTFT only (decode {dmed:.2f} tok/s = {dmed / yd * 100:.1f}% of arm B {yd:g}; cold 8K "
                         f"prefill {ptps:.1f} tok/s = {ptps / yp:.1f}x arm B {yp:g}) — accept as the D5 result or stop")
        code = 1
    return rows, info, decisions, jline, code


def f9_d1(cells):
    lines = []
    for n in sorted(cells):
        c = cells[n]
        if c.get("kind") == "decode":
            med = c.get("decode_tps_median")
            lines.append(f"{n}: {'ok' if c.get('ok') else 'FAILED'}, decode median {med:.2f} tok/s over {len(c['samples'])} prompts" if med else
                         f"{n}: FAILED {c.get('failures')}")
        else:
            m = c.get("medians")
            lines.append(f"{n}: {'ok' if c.get('ok') else 'FAILED'}" + (f", TTFT median {m['ttft_s']:.2f} s, prefill {m['prefill_tps']:.1f} tok/s"
                                                                         + (f", follow-up {m['followup_ttft_s']:.2f} s" if 'followup_ttft_s' in m else "") if m else f" {c.get('failures')}"))
    return lines


def write_f9_campaign(path, title, capture, rows, info, decisions, jline, code, d5=True):
    lines = [f"# {title}", f"Date: {date.today().isoformat()}", f"Capture: {capture}",
             "Command: tools/spark/jobs.sh start f9-engine --heavy f9-engine auto 16200 -- bash bench/campaign/f9_engine_job.sh "
             f"(bench/campaign/run_campaign.py --mode {'D5 --samples 5' if d5 else 'D1 --samples 3'})"]
    if d5:
        lines += [jline or "Justification: none (the campaign did not reach the justification)", "", "| gate | value | limit | ok |", "|---|---|---|---|"]
        lines += [f"| {n} | {_fmt(v)} | {lim} | {'yes' if o else 'NO'} |" for n, v, lim, o in rows]
    lines += [""] + [f"- {i}" for i in info] + [""] + decisions
    if not d5:
        lines.append("Verdict: INFO — D1 recorded only; no gate")
    elif code == 0:
        lines.append("Verdict: PASS — D5 decode, cold 8K TTFT and follow-up within the absolute bars; justified against llama.cpp arm B")
    else:
        lines.append("Verdict: FAIL — " + "; ".join(d.split(", needs a decision")[0].split(" — ")[0] for d in decisions))
    return "\n".join(lines) + "\n"


def cmd_f9_campaign(a):
    d = db_check.latest(a.latest) if a.latest else Path(a.capture)
    if d is None or not Path(d).is_dir():
        print(f"no capture for {a.latest or a.capture}", file=sys.stderr)
        return 3
    try:
        yard = yardstick.yardstick()
        stopgo = stopgo_decision()
        d5 = load_cells(d, "D5")
        rows, info, decisions, jline, code = f9_d5(d5, yard, stopgo)
        d1 = load_cells(d, "D1")
    except (BadInput, KeyError, ValueError, OSError) as e:
        print(f"decide f9-campaign: unreadable input: {e}", file=sys.stderr)
        return 3
    for n, v, lim, o in rows:
        print(f"{'ok  ' if o else 'FAIL'} {n}: {_fmt(v)} (limit {lim})")
    for i in info:
        print(f"  {i}")
    if jline:
        print(jline)
    for dl in decisions:
        print(dl)
    d1_info = f9_d1(d1) if d1 else ["no D1 cell in the capture"]
    for i in d1_info:
        print(f"  D1 {i}")
    if a.write:
        cap = capture_rel(d)
        Path(ROOT / F9_D5).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / F9_D5).write_text(write_f9_campaign(F9_D5, "f9-campaign-d5: engine D5 campaign (N=5)", cap, rows, info, decisions, jline, code))
        (ROOT / F9_D1).write_text(write_f9_campaign(F9_D1, "f9-campaign-d1: engine D1 campaign (N=3), informative", cap, [], d1_info, [], None, 0, d5=False))
        print(f"wrote {F9_D5}, {F9_D1}")
    print("f9-campaign: PASS" if code == 0 else "f9-campaign: FAIL")
    return code


# ----------------------------------------------------------------------------------------- context window 131072
W131K_D5 = "db/records/measurements/w131k-campaign-d5.md"
W131K_ENVELOPE = "db/records/measurements/w131k-envelope.md"
W131K_WINDOW = 131072
W131K_COLD_TOKENS = 78000
W131K_REUSE_MIN = 70_000                # the 78K follow-up must resume at least this many tokens (the snapshot sits near 76,800)
W131K_CELLS = ("decode-8k", "ttft-cold-8k", "ttft-cold-78k")
W131K_COMMAND = ("tools/spark/jobs.sh start w131k-engine --heavy w131k-engine auto 10800 -- bash bench/campaign/w131k_engine_job.sh "
                 "(bench/campaign/run_campaign.py --mode D5 --samples 5 --window 131072 --cells decode-8k,ttft-cold-8k,ttft-cold-78k)")


def w131k_d5(cells):
    """The D5 campaign at a 131,072-token window: the f9 bars for decode, cold 8K and its follow-up, the same follow-up
    bar for the 78K conversation plus a reuse floor, and every cell's peak against the ledger at the window.
    -> (rows, info, decisions, code)."""
    for n in W131K_CELLS:
        if n not in cells:
            raise BadInput(f"the capture has no D5 cell {n}")
        if not cells[n].get("ok"):
            raise BadInput(f"D5 cell {n} did not finish ok: {cells[n].get('failures')}")
    rows, info, decisions, code = [], [], [], 0
    windows = {n: int(cells[n].get("window") or 0) for n in W131K_CELLS}
    same = len(set(windows.values())) == 1
    rows.append(("window of every cell", str(next(iter(windows.values()))) if same else ", ".join(f"{n} {w}" for n, w in windows.items()),
                 str(W131K_WINDOW), all(w == W131K_WINDOW for w in windows.values())))
    bad = [f"{n} sample {s['sample']}" for n in W131K_CELLS[1:] for s in cells[n]["samples"] if s["cold"]["reused_tokens"] > 0]
    if bad:
        decisions.append(f"W131K campaign gate failed: cold sample(s) {', '.join(bad)} reused tokens from a prefix (invalid), needs a decision")
        rows.append(("cold samples with reused_tokens == 0", f"{len(bad)} violations", "0", False))
        return rows, info, decisions, 1
    rows.append(("cold samples with reused_tokens == 0", "all", "all", True))
    dec, cold8, cold78 = cells["decode-8k"], cells["ttft-cold-8k"], cells["ttft-cold-78k"]
    for s in cold8["samples"]:
        if s["cold"]["prompt_tokens"] != 8192:
            raise BadInput(f"cold 8K sample {s['sample']} has {s['cold']['prompt_tokens']} prompt tokens")
    if not all("followup" in s for s in cold8["samples"] + cold78["samples"]):
        raise BadInput("a follow-up is missing from the 8K or the 78K cell")
    tps = [s["decode_tps"] for s in dec["samples"]]
    dmed = true_median(tps)
    rows.append(("D5 decode median (tok/s), decode-8k cell at the window", round(dmed, 3), f">= {F9_DECODE_BAR:g}", dmed >= F9_DECODE_BAR))
    t8 = [s["cold"]["ttft_s"] for s in cold8["samples"]]
    t8m = true_median(t8)
    rows.append(("cold 8K TTFT median (s)", round(t8m, 3), f"<= {COLD_BAR_S:g}", t8m <= COLD_BAR_S))
    f8m = true_median([s["followup"]["ttft_s"] for s in cold8["samples"]])
    rows.append(("follow-up TTFT median (s), 8K", round(f8m, 3), f"<= {FOLLOWUP_BAR_S:g}", f8m <= FOLLOWUP_BAR_S))
    f78m = true_median([s["followup"]["ttft_s"] for s in cold78["samples"]])
    rows.append(("follow-up TTFT median (s), 78K", round(f78m, 3), f"<= {FOLLOWUP_BAR_S:g}", f78m <= FOLLOWUP_BAR_S))
    reuse_min = min(int(s["followup"]["reused_tokens"]) for s in cold78["samples"])
    rows.append(("follow-up reused tokens, 78K, least sample", reuse_min, f">= {W131K_REUSE_MIN}", reuse_min >= W131K_REUSE_MIN))
    ptoks = int(cold78.get("prompt_tokens") or cold78["tokens"])
    rows.append(("cold 78K prompt tokens", ptoks, f">= {W131K_COLD_TOKENS - 256} (at most {W131K_COLD_TOKENS}, cut exactly)",
                 W131K_COLD_TOKENS - 256 <= ptoks <= W131K_COLD_TOKENS))
    worst = None
    peaks = {}
    for n in W131K_CELLS:
        once, nonpool, drop, led, chunks = cell_peak(cells[n], "D5")
        peaks[n] = (once, nonpool, drop, led, chunks)
        m = led - max(once, drop)
        if worst is None or m < worst[1]:
            worst = (n, m, max(once, drop), led, chunks)
    rows.append((f"peak, max(once_bytes, MemAvailable drop), worst cell {worst[0]} (B)", worst[2],
                 f"<= {worst[3]} (ledger at {W131K_WINDOW} tokens, {worst[4]} chunks)", worst[1] >= 0))
    info.append("decode-8k samples (tok/s): " + ", ".join(f"{x:.2f}" for x in tps) + f" (MTP policy {dec['samples'][0]['mtp_policy']}, "
                f"pool {dec['samples'][0]['pool_prefill'] and 'pre-filled' or 'cold'}, {peaks['decode-8k'][4]} chunks)")
    info.append("cold 8K TTFT samples (s): " + ", ".join(f"{x:.2f}" for x in t8))
    info.append("follow-up 8K: " + ", ".join(f"{s['followup']['ttft_s']:.2f} s (reused {s['followup']['reused_tokens']}, new "
                                             f"{s['followup']['new_tokens']})" for s in cold8["samples"]))
    m78 = cold78["medians"]
    info.append(f"ttft-cold-78k: {ptoks} tokens, TTFT median {m78['ttft_s']:.2f} s, prefill {m78['prefill_tps']:.1f} tok/s "
                f"(published, not gated)")
    info.append("cold 78K TTFT samples (s): " + ", ".join(f"{s['cold']['ttft_s']:.2f}" for s in cold78["samples"]))
    info.append("follow-up 78K: " + ", ".join(f"{s['followup']['ttft_s']:.2f} s (reused {s['followup']['reused_tokens']}, new "
                                              f"{s['followup']['new_tokens']})" for s in cold78["samples"]))
    for n in W131K_CELLS:
        once, nonpool, drop, led, chunks = peaks[n]
        info.append(f"peak {n}: once {once:,} B, nonpool {nonpool:,} B, MemAvailable drop {drop:,} B, ledger {led:,} B "
                    f"({chunks} chunks at a {W131K_WINDOW}-token window)")
    for name, v, lim, ok in rows:
        if not ok:
            decisions.append(f"W131K campaign gate failed: {name} {_fmt(v)} vs {lim} (absolute bar), needs a decision")
            code = 1
    return rows, info, decisions, code


def write_w131k_campaign(capture, rows, info, decisions, code):
    lines = [f"# w131k-campaign-d5: engine D5 campaign at a 131,072-token window (N=5)", f"Date: {date.today().isoformat()}",
             f"Capture: {capture}", f"Command: {W131K_COMMAND}", "", "| gate | value | limit | ok |", "|---|---|---|---|"]
    lines += [f"| {n} | {_fmt(v)} | {lim} | {'yes' if o else 'NO'} |" for n, v, lim, o in rows]
    lines += [""] + [f"- {i}" for i in info] + [""] + decisions
    if code == 0:
        lines.append("Verdict: PASS — decode, cold 8K, both follow-ups, the 78K reuse floor and every peak within the bars at a "
                     "131,072-token window")
    else:
        lines.append("Verdict: FAIL — " + "; ".join(d.split(", needs a decision")[0].split(" — ")[0] for d in decisions))
    return "\n".join(lines) + "\n"


def cmd_w131k_campaign(a):
    d = db_check.latest(a.latest) if a.latest else Path(a.capture)
    if d is None or not Path(d).is_dir():
        print(f"no capture for {a.latest or a.capture}", file=sys.stderr)
        return 3
    try:
        rows, info, decisions, code = w131k_d5(load_cells(d, "D5"))
    except (BadInput, KeyError, ValueError, OSError) as e:
        print(f"decide w131k-campaign: unreadable input: {e}", file=sys.stderr)
        return 3
    for n, v, lim, o in rows:
        print(f"{'ok  ' if o else 'FAIL'} {n}: {_fmt(v)} (limit {lim})")
    for i in info:
        print(f"  {i}")
    for dl in decisions:
        print(dl)
    if a.write:
        Path(ROOT / W131K_D5).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / W131K_D5).write_text(write_w131k_campaign(capture_rel(d), rows, info, decisions, code))
        print(f"wrote {W131K_D5}")
    print("w131k-campaign: PASS" if code == 0 else "w131k-campaign: FAIL")
    return code


def w131k_envelope(res, stored=None):
    """INFO: the prefill envelope cells of a grid beyond the stored one. -> (rows, info). Nothing is gated: the stored
    envelope (`qk_product`, the pass schedule, the disk-tier identity) is never rewritten by this path."""
    cells = res.get("cells") or []
    if not cells:
        raise BadInput("the envelope result has no cells")
    budget, wd = int(res["budget_bytes"]), float(res["watchdog_ms"])
    rows = []
    for c in sorted(cells, key=lambda c: (int(c["context"]), int(c["pass"]))):
        name = f"pass {c['pass']} x context {c['context']} (product {int(c['product']):,})"
        lim = f"transient <= {budget:,} B and attention <= {wd:g} ms"
        if c.get("error"):
            rows.append((name, "attention out of device memory", lim, False))
        else:
            rows.append((name, f"transient {int(c['transient_bytes']):,} B, attention {float(c['attn_ms']):.1f} ms", lim, bool(c["fits"])))
    info = [f"largest product that fits on this grid: {int(res['qk_product']):,}",
            f"contexts {list(res.get('contexts', []))}, passes {list(res.get('passes', []))}",
            f"longest launch {float(res['launch_ms_max']):.1f} ms, all launches within the watchdog: {bool(res['all_launches_within_watchdog'])}"]
    if stored:
        info.append(f"stored envelope qk_product {int(stored['qk_product']):,} (f5-prefill-envelope.json, not rewritten)")
    return rows, info


def write_w131k_envelope(capture, rows, info):
    lines = ["# w131k-envelope: prefill transient and launch time at 65,536 and 131,072 tokens", f"Date: {date.today().isoformat()}",
             f"Capture: {capture}", "Command: tools/prefill_envelope.py --contexts 65536,131072 (phase `envelope` of bench/campaign/w131k_engine_job.sh)",
             "", "| cell | value | fits when | fits |", "|---|---|---|---|"]
    lines += [f"| {n} | {v} | {lim} | {'yes' if o else 'no'} |" for n, v, lim, o in rows]
    lines += [""] + [f"- {i}" for i in info] + [""]
    lines.append("Verdict: INFO — grid beyond 32,768 tokens recorded; the stored envelope was not rewritten")
    return "\n".join(lines) + "\n"


def cmd_w131k_envelope(a):
    d = db_check.latest(a.latest) if a.latest else Path(a.capture)
    if d is None or not Path(d).is_dir():
        print(f"no capture for {a.latest or a.capture}", file=sys.stderr)
        return 3
    stored_path = ROOT / "db" / "records" / "measurements" / "f5-prefill-envelope.json"
    try:
        res = _json_file(Path(d) / "envelope" / "result.json")
        stored = _json_file(stored_path) if stored_path.is_file() else None
        rows, info = w131k_envelope(res, stored)
    except (BadInput, KeyError, ValueError, OSError) as e:
        print(f"decide w131k-envelope: unreadable input: {e}", file=sys.stderr)
        return 3
    for n, v, lim, o in rows:
        print(f"{'fits' if o else 'no  '} {n}: {v} ({lim})")
    for i in info:
        print(f"  {i}")
    if a.write:
        (ROOT / W131K_ENVELOPE).write_text(write_w131k_envelope(capture_rel(d), rows, info))
        print(f"wrote {W131K_ENVELOPE}")
    return 0


def f9_diff(ref, engine_ppl, oracle, c, exceptions):
    """Judges the llama.cpp differential of step 96 from the reference job's result.json (`ref`), the engine's ppl.json and
    (informative) the oracle's oracle_ppl.json. -> rows [(name, value, limit, ok)], info, decisions, code, ratio dict.
    Raises BadInput for a result that does not line up with the engine's windows or ids."""
    cfg = c.raw["perplexity"]
    nw, tol, wt = int(cfg["oracle_windows"]), float(cfg["oracle_rel_tol"]), int(cfg["window_tokens"])
    if ref.get("smoke_run"):
        raise BadInput("the capture is a smoke run of the tooling, not the reference job")
    pp = ref.get("ppl")
    if not pp or not pp.get("per_window"):
        raise BadInput(f"the job left no perplexity result (ppl_rc {ref.get('ppl_rc')})")
    eper = engine_ppl.get("per_window") or []
    for key in ("ids_sha256", "rows_sha256"):
        if not pp.get(key) or pp.get(key) != engine_ppl.get(key):
            raise BadInput(f"llama.cpp result {key} {pp.get(key)} differs from the engine's {engine_ppl.get(key)}")
    lrows = pp["per_window"]
    if len(lrows) != nw or len(eper) < nw:
        raise BadInput(f"llama.cpp has {len(lrows)} windows, the engine {len(eper)}; the contract asks for {nw}")
    for k, r in enumerate(lrows):
        if int(r.get("window", -1)) != k or int(r["start"]) != k * wt or int(r["tokens"]) != wt - 1:
            raise BadInput(f"llama.cpp window {k}: window/start/tokens are {r.get('window')}/{r.get('start')}/{r.get('tokens')}")
    r = ppl_ratio(eper[:nw], lrows)
    rows, info, decisions, code = [], [], [], 0

    sw = ref.get("sweep") or {}
    want = sorted(exceptions["tensors"])
    got = sorted(t["name"] for t in sw.get("tolerated", []))
    abs_tol = float(exceptions["abs_tol"])
    sweep_ok = bool(sw) and not sw.get("bad") and got == want and sw.get("tensors") == F9_SWEEP_TENSORS \
        and all(float(t["max_abs_err"]) <= abs_tol for t in sw.get("tolerated", []))
    rows.append(("exactness sweep", f"{sw.get('tensors')} tensors, {len(got)} named exceptions, {len(sw.get('bad', []))} not exact",
                 f"{F9_SWEEP_TENSORS} tensors, exactly the {len(want)} names of q4_1_exceptions.json, abs error <= 2^-24", sweep_ok))
    band = ref.get("below_099") or []
    failures = ref.get("failures") or []
    chk_ok = ref.get("check_tensors") == F9_CHECK_TENSORS and ref.get("corr_floor") == Q41_CORR_FLOOR and not failures
    chk_got = f"{ref.get('check_tensors')} tensors, floor {ref.get('corr_floor')}, {len(band)} in [0.95, 0.99), {len(failures)} under the floor"
    rows.append(("mapping check", chk_got, f"{F9_CHECK_TENSORS} tensors, floor {Q41_CORR_FLOOR}, none under it", chk_ok))
    stopped = bool(pp.get("memfree_stop"))
    rows.append(("memfree soft stop", "yes" if stopped else "no", "no", not stopped))
    dev_ok = pp.get("prefill_device") == "cpu" and pp.get("op_offload") is False
    rows.append(("prefill device", f"{pp.get('prefill_device')}, op_offload={str(pp.get('op_offload')).lower()}", "cpu, op_offload=false", dev_ok))
    rows.append((f"|PPL_engine({nw})/PPL_llama({nw}) - 1|", f"{abs(r['ratio_minus_1']) * 100:.2f}% (engine {r['ppl_a']:.4f}, llama.cpp {r['ppl_b']:.4f})",
                 f"<= {tol * 100:g}%", bool(abs(r["ratio_minus_1"]) <= tol)))
    for name, v, lim, ok in rows:
        if ok is False:
            code = 1
    if stopped:
        decisions.append(f"F9 differential failed: memfree soft stop at {pp.get('memfree_stop_mib')} MiB (floor {pp.get('memfree_floor_mib')}), needs a decision")
    if not sweep_ok:
        decisions.append(f"F9 differential failed: exactness sweep is not 3,215 tensors exact except the named exceptions "
                         f"({F9_SWEEP_TENSORS} expected, got {sw.get('tensors')}; bad {len(sw.get('bad', []))}; tolerated {got} vs {want}), needs a decision")
    if not chk_ok:
        decisions.append(f"F9 differential failed: the mapping check is not {F9_CHECK_TENSORS:,} template tensors at floor {Q41_CORR_FLOOR} "
                         f"with none under it (got {chk_got}), needs a decision")
    if not dev_ok:
        decisions.append("F9 differential failed: the llama.cpp prefill did not run on the CPU with --no-op-offload, needs a decision")
    if abs(r["ratio_minus_1"]) > tol:
        decisions.append(f"engine and exact-Q4_1 llama.cpp disagree by {r['ratio_minus_1'] * 100:+.2f}%: numerics investigation needed")
    worst = max(r["per_window"], key=lambda x: abs(x["ratio_minus_1"]))
    orc = f"{oracle['ppl']:.4f}" if oracle and oracle.get("ppl") else "n/a"
    info.append(f"PPL({nw}): engine {r['ppl_a']:.4f}, transformers oracle {orc} (f9-quality), llama.cpp exact-Q4_1 {r['ppl_b']:.4f}")
    info.append(f"ratio per window: max |ratio-1| {abs(worst['ratio_minus_1']) * 100:.2f}% at window {worst['window']}; "
                f"engine NLL above llama.cpp's on {sum(1 for x in r['per_window'] if x['nll_a'] > x['nll_b'])} of {nw} windows")
    info.append("sweep: " + f"{sw.get('tensors')} tensors, exact except the {len(got)} named exceptions (abs error <= 2^-24): "
                + "; ".join(f"{t['name']} {t['count']} {float(t['max_abs_err']):.3e}" for t in sw.get("tolerated", [])))
    man = ref.get("manifest") or {}
    info.append("type_overrides (written F32): " + (", ".join(man.get("type_overrides", [])) or "none"))
    info.append(f"hyper-connection matrices written as dequantized BF16 instead of Q4_1 (DIAG_F9c): {len(man.get('dequantized_bf16', []))} tensors")
    info.append(f"correlation floor {Q41_CORR_FLOOR} for every family (replan 2026-10-15); tensors with correlation in [0.95, 0.99): {len(band)}")
    for x in sorted(band, key=lambda x: x["name"]):
        info.append(f"[0.95, 0.99) {x['name']} {x['corr']:.4f} ({x['family']})")
    info.append(f"prefill_device={pp.get('prefill_device')} (--no-op-offload: the CPU Q4_1xQ8 dot and the CUDA MMQ path differ by op order only)")
    info.append(f"llama.cpp prompt processing {pp.get('llama_tok_s')} tok/s; memcg peak {pp.get('memcg_peak_mib')} MiB; MemoryMax {pp.get('memorymax_mib')} MiB "
                f"(table {pp.get('memorymax_table_mib')} MiB, MemFree at start {pp.get('memfree_at_start_mib')} MiB); memfree_min {pp.get('memfree_min_mib')} MiB; "
                f"memfree_stop={str(stopped).lower()}")
    info.append(f"token ids sha256 {pp['ids_sha256']}, wikitext rows sha256 {pp['rows_sha256']}")
    return rows, info, decisions, code, r


def write_f9_diff(rows, info, decisions, code, c, capture, dest, ratio):
    P = c.raw["perplexity"]
    lines = ["# f9-reference: engine against exact-Q4_1 llama.cpp, perplexity on the same wikitext windows", f"Date: {date.today().isoformat()}",
             f"Capture: {capture}",
             "Command: tools/spark/jobs.sh start f9-ref --heavy f9-ref 57344 12600 --memcg slotstream-spark-f9-ref -- bash bench/campaign/f9_reference_job.sh "
             "(repack_q4_1.py --sweep, --check-only and build, llamacpp.sh smoke, tools/quality/llamacpp_ppl.py)",
             f"Rule: |PPL_engine({int(P['oracle_windows'])})/PPL_llama({int(P['oracle_windows'])}) - 1| <= {float(P['oracle_rel_tol']) * 100:g}% "
             f"on the first {int(P['oracle_windows'])} windows (same ids_sha256, same starts); design decision 21 · mapping: correlation >= 0.95 for every family",
             "", "| gate | value | limit | ok |", "|---|---|---|---|"]
    lines += [f"| {n} | {v} | {lim} | {'yes' if o else 'NO'} |" for n, v, lim, o in rows]
    lines += ["", "| window | start | engine NLL | llama.cpp NLL | ratio-1 |", "|---|---|---|---|---|"]
    lines += [f"| {w['window']} | {w['start']} | {w['nll_a']:.2f} | {w['nll_b']:.2f} | {w['ratio_minus_1'] * 100:+.2f}% |" for w in ratio["per_window"]]
    lines += [""] + [f"- {i}" for i in info] + [""] + decisions
    if code == 0:
        lines.append(f"Verdict: PASS: engine and exact-Q4_1 llama.cpp agree within {float(P['oracle_rel_tol']) * 100:g}% "
                     f"({ratio['ratio_minus_1'] * 100:+.2f}%) on {int(P['oracle_windows'])} windows; sweep exact except the named exceptions; "
                     f"mapping floor 0.95, {sum(1 for i in info if i.startswith('[0.95, 0.99) '))} tensors in [0.95, 0.99) listed; "
                     "prefill_device=cpu; no MemFree soft stop")
    else:
        lines.append("Verdict: FAIL: " + "; ".join(d.split(", needs a decision")[0] for d in decisions))
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    Path(dest).write_text("\n".join(lines) + "\n")


def cmd_f9_diff(a):
    d = db_check.latest(a.latest) if a.latest else Path(a.capture)
    if d is None or not Path(d).is_dir():
        print(f"no capture for {a.latest or a.capture}", file=sys.stderr)
        return 3
    ed = db_check.latest(a.engine_latest or "f9-engine") if not a.engine else Path(a.engine)
    if ed is None or not Path(ed).is_dir():
        print(f"no engine capture for {a.engine_latest or a.engine or 'f9-engine'}", file=sys.stderr)
        return 3
    od = db_check.latest(a.oracle_latest) if a.oracle_latest else (Path(a.oracle) if a.oracle else db_check.latest("f9-oracle"))
    try:
        ref = _json_file(Path(d) / "result.json")
        engine = _json_file(Path(ed) / "quality" / "ppl.json")
        oracle = _json_file(Path(od) / "quality" / "oracle_ppl.json") if od and Path(od).is_dir() else None
        exceptions = _json_file(ROOT / Q41_EXCEPTIONS)
        rows, info, decisions, code, ratio = f9_diff(ref, engine, oracle, _contract().load(), exceptions)
    except (BadInput, KeyError, ValueError, OSError) as e:
        print(f"decide f9-diff: unreadable input: {e}", file=sys.stderr)
        return 3
    for n, v, lim, o in rows:
        print(f"{'ok  ' if o else 'FAIL'} {n}: {v} (limit {lim})")
    for i in info:
        print(f"  {i}")
    for dl in decisions:
        print(dl)
    if a.write:
        write_f9_diff(rows, info, decisions, code, _contract().load(), capture_rel(d), str(ROOT / F9_REF), ratio)
        print(f"wrote {F9_REF}")
    print("f9-diff: PASS" if code == 0 else "f9-diff: FAIL")
    return code


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("f5-prefill")
    g = f.add_mutually_exclusive_group(required=True)
    g.add_argument("--latest")
    g.add_argument("capture", nargs="?")
    f.add_argument("--write", nargs="?", const="", default=None)
    f.add_argument("--note", action="append", default=[])
    for name in ("f6-lookahead", "f6-mtp", "f6-target"):
        q = sub.add_parser(name)
        g = q.add_mutually_exclusive_group(required=True)
        g.add_argument("--latest")
        g.add_argument("capture", nargs="?")
        if name == "f6-target":
            q.add_argument("--min", type=float, default=D5_TARGET)
            q.add_argument("--no-write", action="store_true")
            q.add_argument("--note", action="append", default=[])
        else:
            q.add_argument("--write", action="store_true")
        if name == "f6-mtp":
            q.add_argument("--mode", default="D5")
    q = sub.add_parser("f8-entry")
    g = q.add_mutually_exclusive_group(required=True)
    g.add_argument("--latest")
    g.add_argument("capture", nargs="?")
    q.add_argument("--write", action="store_true")
    q = sub.add_parser("f8-token-graph")
    g = q.add_mutually_exclusive_group(required=True)
    g.add_argument("--latest")
    g.add_argument("capture", nargs="?")
    g = q.add_mutually_exclusive_group(required=True)
    g.add_argument("--soak-latest")
    g.add_argument("--soak")
    q.add_argument("--write", action="store_true")
    for name in ("f9-quality", "f9-campaign", "w131k-campaign", "w131k-envelope"):
        q = sub.add_parser(name)
        g = q.add_mutually_exclusive_group(required=True)
        g.add_argument("--latest")
        g.add_argument("capture", nargs="?")
        q.add_argument("--write", action="store_true")
        if name == "f9-quality":
            g = q.add_mutually_exclusive_group()
            g.add_argument("--oracle-latest")
            g.add_argument("--oracle")
    q = sub.add_parser("f9-diff")
    g = q.add_mutually_exclusive_group(required=True)
    g.add_argument("--latest")
    g.add_argument("capture", nargs="?")
    g = q.add_mutually_exclusive_group()
    g.add_argument("--engine-latest")
    g.add_argument("--engine")
    g = q.add_mutually_exclusive_group()
    g.add_argument("--oracle-latest")
    g.add_argument("--oracle")
    q.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    return {"f5-prefill": cmd_f5, "f6-lookahead": cmd_f6_lookahead, "f6-mtp": cmd_f6_mtp,
            "f6-target": cmd_f6_target, "f8-entry": cmd_f8_entry,
            "f8-token-graph": cmd_f8_token_graph, "f9-quality": cmd_f9_quality, "f9-campaign": cmd_f9_campaign,
            "f9-diff": cmd_f9_diff, "w131k-campaign": cmd_w131k_campaign, "w131k-envelope": cmd_w131k_envelope}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
