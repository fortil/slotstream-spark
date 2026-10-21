#!/usr/bin/env python3
"""Perplexity of the engine on wikitext-2-raw-v1 (plan step 93, gate of step 95). Spark only for the scoring.

  ppl_wikitext.py [--mode D5] [--out DIR] [--windows 145] [--text-only]

The text is the test split joined with "\\n\\n" (rows pinned by tools/quality/fetch_wikitext.sh), tokenized with the
checkpoint tokenizer without special tokens. The first 145 non-overlapping windows of 2048 tokens are scored on the tokens
1..2047 of each window (token 0 has no context), so a full run asserts 145 windows and 145 x 2047 = 296,815 scored tokens
(README of the checkpoint). PPL = exp(sum NLL / 296,815). Each window starts from a fresh sequence state and goes through
the sweep prefill driver; the lm_head runs in 256-row tiles (slotstream_spark/engine/score.py).

Writes DIR/ppl.json (per-window NLL, PPL, the reference and its band) and DIR/ppl_windows.jsonl (one line per window as it
finishes). A full run exits 0 whatever the PPL: the band around the README reference 5.3914 is informative
(`reference_informative` in numeric_contract.toml, the README protocol is not published) and the gate is the ratio to the
transformers oracle judged by `decide.py f9-quality`. Exit 2 when a full run does not have the expected window or token counts. `--windows N` (N < 145) is a smoke run: it never
writes `full: true` and is never judged.
"""
import argparse
import hashlib
import json
import os
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FULL_WINDOWS = 145
WINDOW = 2048


def contract_ppl():
    c = tomllib.loads((ROOT / "slotstream_spark" / "config" / "numeric_contract.toml").read_text())["perplexity"]
    return float(c["wikitext2_ref"]), float(c["rel_tol"]), int(c["windows"]), int(c["window_tokens"])


def contract_reference_informative():
    c = tomllib.loads((ROOT / "slotstream_spark" / "config" / "numeric_contract.toml").read_text())["perplexity"]
    return bool(c["reference_informative"])


def rows_path():
    r = os.environ.get("SLOTSTREAM_R") or str(Path.home() / "slotstream-spark")
    return Path(r) / "state" / "wikitext"


def load_text(d=None):
    """-> (text, sha256 of the pinned rows file). Refuses a file whose sha256 differs from its pin."""
    d = Path(d) if d else rows_path()
    rows_file, pin = d / "test_rows.json", d / "test_rows.sha256"
    if not rows_file.exists() or not pin.exists():
        raise SystemExit(f"ppl_wikitext: {rows_file} or its pin is missing: run tools/quality/fetch_wikitext.sh")
    sha = hashlib.sha256(rows_file.read_bytes()).hexdigest()
    if sha != pin.read_text().strip():
        raise SystemExit(f"ppl_wikitext: {rows_file} sha256 {sha} differs from its pin")
    rows = json.loads(rows_file.read_text())
    return "\n\n".join(rows), sha


def tokenize(text, model_dir):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    return [int(t) for t in tok.encode(text, add_special_tokens=False).ids]


def ids_sha256(ids):
    import array
    return hashlib.sha256(array.array("i", ids).tobytes()).hexdigest()


def summarize(rows, ref, tol, windows):
    """-> dict with the PPL over `rows` (per-window {"nll", "tokens"}) and the band verdict."""
    from slotstream_spark.engine import score
    nll = sum(r["nll"] for r in rows)
    tokens = sum(r["tokens"] for r in rows)
    p = score.ppl(nll, tokens)
    lo, hi = ref * (1 - tol), ref * (1 + tol)
    full = len(rows) == windows
    return {"windows": len(rows), "tokens_scored": tokens, "nll_sum": nll, "ppl": p, "ref": ref, "rel_tol": tol,
            "band": [lo, hi], "rel_diff": p / ref - 1.0, "in_band": bool(lo <= p <= hi), "full": bool(full)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", default="D5")
    ap.add_argument("--out")
    ap.add_argument("--windows", type=int, default=FULL_WINDOWS)
    ap.add_argument("--text-only", action="store_true", help="tokenize and print the window math, no engine")
    a = ap.parse_args(argv)
    ref, tol, windows, window_tokens = contract_ppl()
    ref_informative = contract_reference_informative()
    assert (windows, window_tokens) == (FULL_WINDOWS, WINDOW), "numeric_contract.toml [perplexity] changed"
    text, rows_sha = load_text()
    model_dir = os.environ["SLOTSTREAM_MODEL"]
    ids = tokenize(text, model_dir)
    from slotstream_spark.engine import score
    sl = score.window_slices(len(ids), WINDOW, a.windows)
    full = a.windows >= FULL_WINDOWS
    info = {"text_chars": len(text), "rows_sha256": rows_sha, "tokens_total": len(ids), "ids_sha256": ids_sha256(ids),
            "windows_available": len(ids) // WINDOW, "windows_run": len(sl), "window": WINDOW}
    print(json.dumps(info), flush=True)
    if full and (len(sl) != FULL_WINDOWS or score.scored_tokens(len(sl), WINDOW) != 296_815):
        print(f"ppl_wikitext: expected {FULL_WINDOWS} windows and 296,815 scored tokens, the text gives "
              f"{len(sl)} windows ({score.scored_tokens(len(sl), WINDOW)} tokens)", file=sys.stderr)
        return 2
    if a.text_only:
        return 0
    out = Path(a.out or Path(os.environ.get("CAP", ".")) / "quality")
    out.mkdir(parents=True, exist_ok=True)
    part = out / "ppl_windows.jsonl"
    part.write_text("")
    from bench import ttft
    eng = ttft.bring_up(a.mode, WINDOW + 64, prefill=True)
    model, driver = eng["model"], eng["driver"]

    def on_window(row):
        with open(part, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"window {row['window']:3d}/{len(sl)}: nll {row['nll']:.2f} ({row['nll'] / row['tokens']:.4f}/token) "
              f"{row['seconds']:.1f} s", flush=True)

    rows = score.nll_windows(model, driver, ids, window=WINDOW, max_windows=a.windows, on_window=on_window)
    eng["meter"].stop()
    from slotstream_spark.model import layers
    res = summarize(rows, ref, tol, FULL_WINDOWS)
    res.update(info)
    res.update({"per_window": rows, "mode": a.mode, "reference_informative": ref_informative, "slots": eng["slots"], "gemm": dict(layers.GEMM_STATS),
                "lm_head_tile_rows": score.TILE, "versions": {"torch": eng["torch"].__version__}})
    res["full"] = bool(full and res["full"])
    (out / "ppl.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"ppl {res['ppl']:.4f} over {res['windows']} windows / {res['tokens_scored']:,} tokens "
          f"(ref {ref}, band {res['band'][0]:.4f}-{res['band'][1]:.4f}, rel diff {res['rel_diff'] * 100:+.2f}%)", flush=True)
    if not res["full"]:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
