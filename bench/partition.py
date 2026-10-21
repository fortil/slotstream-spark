#!/usr/bin/env python3
"""Wall-time partition of the decode loop from a `cli bench decode --profile` result.json (step 57).

  partition.py RESULT_JSON

Pure arithmetic over the totals the engine recorded: the nine named phases (slotstream_spark.runtime.profile.PHASES),
`other = wall - sum(phases)` and the shares of the wall. `check()` is what tools/gates/records.py partition judges.
"""
import json
import sys

PHASES = ("sample", "ngram_hash", "embed_read", "ngram_wait", "launch", "gpu_wait", "ensure_hits",
          "expert_miss_read", "expert_copy_tail")
OTHER_MAX = 0.10


def table(entry):
    """Rows (phase, ns, share of wall) including `other` and the total (= wall)."""
    wall = entry["wall_ns"]
    rows = [(p, entry["phases_ns"][p], entry["phases_ns"][p] / wall) for p in PHASES]
    other = wall - sum(entry["phases_ns"].values())
    rows.append(("other", other, other / wall))
    rows.append(("wall", wall, sum(r[2] for r in rows)))
    return rows


def check(entry):
    """-> dict of the facts the gate judges for one prompt key."""
    wall = entry["wall_ns"]
    named = sum(entry["phases_ns"].values())
    other = wall - named
    return {"wall_ns": wall, "named_ns": named, "other_ns": other, "other_share": other / wall,
            "recorded_other_matches": other == entry["other_ns"],
            "overlap": other < 0,
            "read_split_exact": entry["phases_ns"]["expert_miss_read"] + entry["phases_ns"]["expert_copy_tail"]
            == entry["read_wall_ns"],
            "copy_separate": "expert_copy_tail" in entry["phases_ns"] and "expert_miss_read" in entry["phases_ns"],
            "tokens": entry["tokens_profiled"]}


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    res = json.loads(open(argv[0]).read())
    for k, e in res["prompts"].items():
        print(f"{k}: {e['prompt_tokens']} prompt tokens, {e['tokens_profiled']} profiled tokens, tok/s {e['tok_s']}, "
              f"h {e['hit_rate']}")
        for name, ns, share in table(e):
            print(f"  {name:18s} {ns / 1e6:12.3f} ms {100 * share:7.2f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
