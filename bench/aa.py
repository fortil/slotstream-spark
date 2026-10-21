"""A/A harness: the same command on both sides; the noise floor of the paired ratio.

  aa.py --label L --pairs 16 --metric decode_tps --cmd 'CMD'

Same capture layout as ab.py. Pass = the median ratio is within --tol (default 3 %) of 1.0 on the clean pairs.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench import ab  # noqa: E402


def run_aa(cmd, *, tol=0.03, **kw):
    res, d = ab.run_ab(cmd, cmd, a_name="A", b_name="A'", min_median=0.0, min_positive=0.0, **kw)
    m = res["median_ratio"]
    res["tol"] = tol
    res["pass"] = m is not None and abs(m - 1.0) <= tol and res["pairs_clean"] >= 1
    (d / "result.json").write_text(json.dumps(res, indent=1) + "\n")
    (d / "verdict.json").write_text(json.dumps({"pass": res["pass"], "median_ratio": m, "pairs_clean": res["pairs_clean"]}) + "\n")
    return res, d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True)
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--metric", default="decode_tps")
    ap.add_argument("--cmd", required=True)
    ap.add_argument("--tol", type=float, default=0.03)
    a = ap.parse_args(argv)
    res, _ = run_aa(a.cmd, tol=a.tol, pairs=a.pairs, metric=a.metric, label=a.label)
    print(json.dumps({k: v for k, v in res.items() if k != "pairs"}))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
