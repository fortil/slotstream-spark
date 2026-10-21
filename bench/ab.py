"""A/B harness: fresh-process pairs, balanced order, median of paired ratios, excluded pairs kept with a reason.

  ab.py --label L --pairs 16 --metric decode_tps --a-cmd 'CMD' --b-cmd 'CMD' [--ident KEY] [--cap-mib N] [--est-seconds S]

Each arm command must write a JSON object holding the metric (dotted path) to the file named by the `{out}` placeholder
(or to stdout when it has none); `{pair}` in a command is replaced by the pair index (0-based), so a pair can use its own
prompt. With `--ident KEY` the value at that path (a hash of the arm's output) is kept in every row as `a_ident`/`b_ident`.
B is the candidate: ratio = B / A. Creates $SLOTSTREAM_R/runs/YYYY/MM/<ts>-L/ with pairs.jsonl,
result.json and verdict.json, and writes $SLOTSTREAM_R/runs/LAST-L.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import eligibility, stats  # noqa: E402


def dotted(obj, path):
    for k in path.split("."):
        obj = obj[k]
    return obj


def run_arm_ident(cmd, metric, tmp, runner=subprocess.run, pair=None, ident=None):
    """One fresh process. -> (value or None, error or None, ident value or None)."""
    out = Path(tmp)
    out.unlink(missing_ok=True)
    c = cmd.replace("{out}", str(out))
    if pair is not None:
        c = c.replace("{pair}", str(pair))
    p = runner(shlex.split(c), capture_output=True, text=True)
    if p.returncode != 0:
        return None, f"rc={p.returncode}", None
    try:
        text = out.read_text() if "{out}" in cmd else p.stdout
        d = json.loads(text)
        return float(dotted(d, metric)), None, (None if ident is None else dotted(d, ident))
    except Exception as e:                                      # noqa: BLE001
        return None, f"unparsable:{type(e).__name__}", None


def run_arm(cmd, metric, tmp, runner=subprocess.run):
    """One fresh process. -> (value or None, error or None)."""
    v, e, _ = run_arm_ident(cmd, metric, tmp, runner)
    return v, e


def capture_dir(label, root=None):
    r = Path(root or os.environ["SLOTSTREAM_R"])
    d = r / "runs" / time.strftime("%Y/%m") / f"{time.strftime('%Y%m%dT%H%M%S')}-{label}"
    d.mkdir(parents=True, exist_ok=True)
    return d, r


def run_ab(a_cmd, b_cmd, *, pairs=16, metric="decode_tps", label="ab", snapshot=None, runner=subprocess.run, root=None,
           min_median=1.05, min_positive=0.80, a_name="A", b_name="B", sub=None, tmp=None, ident=None, keep_arms=False):
    """snapshot: callable -> eligibility snapshot dict (default: eligibility.collect)."""
    d, r = capture_dir(label, root)
    tmpf = tmp or str(d / "arm.json")
    snapshot = snapshot or (lambda: eligibility.collect(0, 0))
    rows, order = [], stats.balanced_order(pairs)
    stopped = None
    for i, o in enumerate(order):
        vals, idents = {}, {}
        errs = []
        for arm in o:
            v, e, idv = run_arm_ident(a_cmd if arm == "A" else b_cmd, metric, tmpf, runner, pair=i, ident=ident)
            vals[arm], idents[arm] = v, idv
            if e:
                errs.append(f"{arm}:{e}")
            if keep_arms and Path(tmpf).exists():              # the arm's own record (acceptance, scheduler counters)
                (d / "arms").mkdir(exist_ok=True)
                (d / "arms" / f"p{i:02d}-{arm}.json").write_text(Path(tmpf).read_text())
        snap = snapshot()
        ok, why = eligibility.evaluate(snap)
        why = list(why) + errs
        row = {"pair": i, "order": o, "a": vals.get("A"), "b": vals.get("B"), "eligible": ok and not errs, "reasons": why, "snapshot": snap}
        if ident is not None:
            row["a_ident"], row["b_ident"] = idents.get("A"), idents.get("B")
        row["ratio"] = (row["b"] / row["a"]) if row["a"] and row["b"] is not None else None
        rows.append(row)
        clean = [x["ratio"] for x in rows if x["eligible"] and x["ratio"] is not None]
        if stats.futility(clean, pairs, min_positive) and len(rows) < pairs:
            stopped = "futility"
            break
    clean = [x["ratio"] for x in rows if x["eligible"] and x["ratio"] is not None]
    res = {"label": label, "metric": metric, "arms": {"A": a_name, "B": b_name}, "pairs_planned": pairs, "pairs_run": len(rows),
           "pairs_clean": len(clean), "pairs_excluded": len(rows) - len(clean), "stopped": stopped,
           "median_ratio": stats.true_median(clean) if clean else None,
           "positive_fraction": stats.positive_fraction(clean, pairs) if clean else None,
           "pass": bool(clean) and stats.gate(clean, min_median, min_positive) and len(clean) == pairs, "pairs": rows}
    (d / "pairs.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    (d / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    (d / "verdict.json").write_text(json.dumps({"pass": res["pass"], "median_ratio": res["median_ratio"],
                                                  "pairs_clean": res["pairs_clean"], "stopped": stopped}) + "\n")
    (r / "runs" / f"LAST-{label}").write_text(str(d) + "\n")
    return res, d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--metric", default="decode_tps")
    ap.add_argument("--a-cmd", required=True)
    ap.add_argument("--b-cmd", required=True)
    ap.add_argument("--ident", help="dotted path of a hash of the arm's output, kept per row as a_ident/b_ident")
    ap.add_argument("--keep-arms", action="store_true", help="keep each arm's output file under arms/pNN-A|B.json")
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    ap.add_argument("--min-median", type=float, default=1.05)
    ap.add_argument("--min-positive", type=float, default=0.80)
    a = ap.parse_args(argv)
    res, d = run_ab(a.a_cmd, a.b_cmd, pairs=a.pairs, metric=a.metric, label=a.label,
                    min_median=a.min_median, min_positive=a.min_positive, ident=a.ident,
                    keep_arms=a.keep_arms, a_name=a.a_name, b_name=a.b_name)
    print(json.dumps({k: v for k, v in res.items() if k != "pairs"}))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
