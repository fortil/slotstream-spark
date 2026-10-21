"""Paired-run statistics for the A/B and A/A harness. Medians of paired ratios only: no means, no best-of-N."""


def paired_ratios(base, cand):
    """cand_i / base_i for every pair where both values exist and base > 0 (input order kept)."""
    return [c / b for b, c in zip(base, cand) if b and c is not None and b > 0]


def true_median(values):
    """Median with the mean of the two middle values when n is even."""
    v = sorted(values)
    n = len(v)
    if n == 0:
        raise ValueError("median of an empty set")
    mid = n // 2
    return v[mid] if n % 2 else (v[mid - 1] + v[mid]) / 2


def positive_fraction(ratios, total=None):
    """Share of pairs with ratio > 1 over `total` (default: the pairs given)."""
    return sum(1 for r in ratios if r > 1.0) / (total or len(ratios))


def gate(ratios, min_median=1.05, min_positive=0.80):
    """Pass when the true median of the paired ratios is at least min_median and at least min_positive of the pairs are > 1."""
    if not ratios:
        return False
    return true_median(ratios) >= min_median and positive_fraction(ratios) >= min_positive


def futility(ratios, total_pairs=16, min_positive=0.80):
    """True when even if every remaining pair were positive the positive bar could not be reached (stop early)."""
    remaining = max(0, total_pairs - len(ratios))
    best = sum(1 for r in ratios if r > 1.0) + remaining
    return best / total_pairs < min_positive


def balanced_order(n):
    """Run order of each pair, ABBA...: pair 0 runs A then B, pair 1 B then A, and so on (n entries of 'AB' or 'BA')."""
    return ["AB" if i % 2 == 0 else "BA" for i in range(n)]


def adoption(ratios, margin, min_clean=5, min_positive=0.80):
    """The F6 adoption rule on the clean pairs of an A/B: -> (adopted, median or None, positive fraction or None, reason).
    Adopted iff there are at least `min_clean` clean pairs, the true median of the paired ratios is at least `margin`
    (1.02 lookahead, 1.05 MTP) and at least `min_positive` of the clean pairs have ratio > 1 (fraction over the clean pairs)."""
    n = len(ratios)
    if n < min_clean:
        return False, (true_median(ratios) if n else None), (positive_fraction(ratios) if n else None), \
            f"{n} clean pairs < {min_clean}"
    med, pos = true_median(ratios), positive_fraction(ratios)
    if med < margin:
        return False, med, pos, f"median {med:.4f} < {margin:g}"
    if pos < min_positive:
        return False, med, pos, f"positive fraction {pos:.3f} < {min_positive:g}"
    return True, med, pos, f"median {med:.4f} >= {margin:g}, positive fraction {pos:.3f} >= {min_positive:g}"


def main(argv=None):
    """stats.py adopt --result RESULT.json --margin 1.02 [--min-clean 5]: exit 0 adopted, 1 not (the F6 job's lookahead choice)."""
    import argparse
    import json
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("adopt")
    a.add_argument("--result", required=True)
    a.add_argument("--margin", type=float, required=True)
    a.add_argument("--min-clean", type=int, default=5)
    args = ap.parse_args(argv)
    res = json.loads(open(args.result).read())
    rows = [r for r in res["pairs"] if r["eligible"] and r.get("ratio") is not None]
    same = all(r.get("a_ident") is not None and r.get("a_ident") == r.get("b_ident") for r in rows)
    ok, med, pos, why = adoption([r["ratio"] for r in rows], args.margin, args.min_clean)
    ok = ok and same
    print(f"adopt: {'yes' if ok else 'no'} ({why}{'' if same else '; outputs differ across arms'})")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
