#!/usr/bin/env python3
"""Perplexity of the transformers oracle on the wikitext-2 windows of the engine run (plan step 95b). Spark only.

  oracle_ppl.py [--windows 32] [--state FILE] [--out DIR] [--engine-ppl FILE] [--text-only]

The text, token ids, ids hash and window slices come from the same functions the engine run used (ppl_wikitext.load_text,
tokenize, ids_sha256 and slotstream_spark.engine.score.window_slices), so the windows are the engine's by construction.
Each window goes through tools/ref_layerwise.teacher_force in the checkpoint dtype (ORACLE_DTYPE unset) and is scored on
tokens 1..2047 (row i against ids[i+1], no BOS, no special tokens), the NLL summed in float64 in 256-row tiles.

The job is resumable: one JSON line per finished window is appended to the state file
(default $SLOTSTREAM_R/state/f9-oracle/windows.jsonl) and a restart scores only the windows that are missing. A state line
with another ids_sha256 or dtype is refused (exit 2), never mixed. At the end DIR/oracle_ppl.json holds the per-window rows,
the PPL and the shas; `full` is true when all `[perplexity] oracle_windows` windows are there. Exit 0 only with all
requested windows scored; 2 for a mismatch with the engine file or a foreign state. The oracle builds one layer at a time and
dequantizes the lm_head in 32,768-row blocks; no full fp32 copy of a large weight is made. `--text-only` prints the window math
and exits without the oracle.
"""
import argparse
import json
import math
import os
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "quality"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

WINDOW = 2048
TILE = 256
ORACLE = "tools/ref_layerwise.py teacher_force"


def contract_oracle():
    c = tomllib.loads((ROOT / "slotstream_spark" / "config" / "numeric_contract.toml").read_text())["perplexity"]
    return int(c["oracle_windows"]), int(c["window_tokens"])


def window_starts(n_tokens, n_windows, window=WINDOW):
    """Start offsets of the first `n_windows` non-overlapping windows (the engine's own slices)."""
    from slotstream_spark.engine import score
    return [a for a, _ in score.window_slices(n_tokens, window, n_windows)]


class StateError(SystemExit):
    """Exit code 2 (as documented above); the message goes to stderr."""
    def __init__(self, msg):
        print(msg, file=sys.stderr)
        super().__init__(2)


def plan_windows(lines, n_windows, ids_sha, dtype):
    """-> (done {window: row}, missing [k]). A line with another ids_sha256 or dtype stops the run (exit 2); a duplicate
    window line counts once."""
    done = {}
    for ln in lines:
        row = ln if isinstance(ln, dict) else json.loads(ln)
        if row["ids_sha256"] != ids_sha or row["dtype"] != dtype:
            raise StateError(f"oracle_ppl: state line for window {row.get('window')} has ids_sha256 {row['ids_sha256']} / dtype "
                             f"{row['dtype']}, this run has {ids_sha} / {dtype}; refusing to mix")
        k = int(row["window"])
        if 0 <= k < n_windows:
            done.setdefault(k, row)
    return done, [k for k in range(n_windows) if k not in done]


def read_state(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def build_result(rows, n_windows, oracle_windows, ids_sha, rows_sha, dtype, state_file, engine_capture, versions):
    rows = sorted(rows, key=lambda r: r["window"])
    tokens = sum(int(r["tokens"]) for r in rows)
    nll = math.fsum(float(r["nll"]) for r in rows)
    per = [{k: r[k] for k in ("window", "start", "nll", "tokens", "seconds")} for r in rows]
    return {"per_window": per, "windows": len(rows), "tokens_scored": tokens, "nll_sum": nll,
            "ppl": math.exp(nll / tokens) if tokens else None, "ids_sha256": ids_sha, "rows_sha256": rows_sha, "dtype": dtype,
            "oracle": ORACLE, "state_file": str(state_file), "engine_capture": engine_capture,
            "full": bool(len(rows) == oracle_windows and n_windows == oracle_windows), "versions": versions}


def nll_from_logits(logits, ids_window, tile=TILE):
    """-> (float64 NLL sum, scored): row i of `logits` against ids_window[i+1] for i in 0..len-2, in `tile`-row blocks."""
    import torch
    n = len(ids_window)
    tgt = torch.as_tensor(list(ids_window[1:]), dtype=torch.long, device=logits.device)
    total, scored = 0.0, 0
    for r0 in range(0, n - 1, tile):
        r1 = min(n - 1, r0 + tile)
        blk = logits[r0:r1].double()
        lse = torch.logsumexp(blk, dim=-1)
        pick = blk.gather(1, tgt[r0:r1, None])[:, 0]
        total += float((lse - pick).sum())
        scored += r1 - r0
    if scored != n - 1:
        raise RuntimeError(f"scored {scored} rows, expected {n - 1}")
    return total, scored


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--windows", type=int, default=None)
    ap.add_argument("--state")
    ap.add_argument("--out")
    ap.add_argument("--engine-ppl")
    ap.add_argument("--text-only", action="store_true")
    a = ap.parse_args(argv)
    oracle_windows, window_tokens = contract_oracle()
    n_windows = a.windows if a.windows is not None else oracle_windows
    assert window_tokens == WINDOW, "numeric_contract.toml [perplexity] window_tokens changed"
    import ppl_wikitext as pw
    text, rows_sha = pw.load_text()
    model_dir = os.environ["SLOTSTREAM_MODEL"]
    ids = pw.tokenize(text, model_dir)
    ids_sha = pw.ids_sha256(ids)
    starts = window_starts(len(ids), n_windows)
    print(json.dumps({"windows_available": len(ids) // WINDOW, "windows": n_windows, "windows_run": len(starts),
                      "ids_sha256": ids_sha, "rows_sha256": rows_sha}), flush=True)
    if len(starts) != n_windows:
        print(f"oracle_ppl: the text has {len(starts)} windows, {n_windows} requested", file=sys.stderr)
        return 2
    engine_capture = None
    if a.engine_ppl:
        eng = json.loads(Path(a.engine_ppl).read_text())
        if eng.get("ids_sha256") != ids_sha or eng.get("rows_sha256") != rows_sha:
            print(f"oracle_ppl: {a.engine_ppl} has ids_sha256 {eng.get('ids_sha256')} / rows_sha256 {eng.get('rows_sha256')}, "
                  f"this run {ids_sha} / {rows_sha}", file=sys.stderr)
            return 2
        engine_capture = str(a.engine_ppl)
    if a.text_only:
        return 0
    r = os.environ.get("SLOTSTREAM_R") or str(Path.home() / "slotstream-spark")
    state = Path(a.state or Path(r) / "state" / "f9-oracle" / "windows.jsonl")
    state.parent.mkdir(parents=True, exist_ok=True)
    out = Path(a.out or Path(os.environ.get("CAP", ".")) / "quality")
    out.mkdir(parents=True, exist_ok=True)
    import ref_layerwise as rl
    dtype = str(rl.DTYPE)
    done, missing = plan_windows(read_state(state), n_windows, ids_sha, dtype)
    print(f"oracle_ppl: {len(done)} windows in the state file, {len(missing)} to score ({dtype})", flush=True)
    if missing:
        import torch
        mdir = rl.model_dir()
        cfg = rl.load_cfg(mdir)
        ck = rl.Checkpoint(mdir)
        try:
            for k in missing:
                a0 = starts[k]
                win = ids[a0:a0 + WINDOW]
                t0 = time.perf_counter()
                logits = rl.teacher_force(win, ck=ck, cfg=cfg)
                nll, scored = nll_from_logits(logits, win)
                del logits
                torch.cuda.empty_cache()
                if scored != WINDOW - 1:
                    raise RuntimeError(f"window {k}: scored {scored} tokens")
                row = {"window": k, "start": a0, "nll": nll, "tokens": scored, "seconds": round(time.perf_counter() - t0, 3),
                       "ids_sha256": ids_sha, "dtype": dtype}
                with open(state, "a") as f:
                    f.write(json.dumps(row) + "\n")
                done[k] = row
                print(f"window {k}/{n_windows}: nll {nll:.2f} ({nll / scored:.4f}/token) {row['seconds']:.1f} s", flush=True)
        finally:
            ck.close()
    import transformers
    res = build_result([done[k] for k in range(n_windows)], n_windows, oracle_windows, ids_sha, rows_sha, dtype, state,
                       engine_capture, {"torch": rl.torch.__version__, "transformers": transformers.__version__})
    (out / "oracle_ppl.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"oracle ppl {res['ppl']:.4f} over {res['windows']} windows / {res['tokens_scored']:,} tokens ({dtype})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
