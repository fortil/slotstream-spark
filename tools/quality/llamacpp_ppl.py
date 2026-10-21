#!/usr/bin/env python3
"""llama.cpp perplexity on the engine's own wikitext windows (plan step 96, design decision 21). Spark only.

  llamacpp_ppl.py --engine-ppl FILE [--windows 32] [--weights q41exact] [--out DIR] [--unit slotstream-spark-f9-ref]
                  [--memfree-stop 14336] [--memfree-log FILE] [--threads 16]

Why not stock llama-perplexity: it tokenizes a text file itself and scores only the second half of each chunk, so it cannot
score the engine's token ids or its tokens 1..2047. The helper tools/quality/ppl_ids.cpp (built against the pinned llama.cpp
by build_ppl_ids.sh) takes the ids, parses the same flags through llama.cpp's own perplexity-example parser and scores
tokens 1..2047 of every window from an empty cache, no BOS. The ids are the ones ppl_wikitext.py tokenizes (same text pin,
same tokenizer, same window_slices); their sha256 is checked against the engine's ppl.json (exit 2 on a mismatch) and
recorded in result.json, which decide.py f9-diff checks again.

The run is wrapped like the LB server: MemoryMax is the MEMORYMAX_MIB that deploy/lb/llamacpp.sh --print-footprint prints for the
MemFree measured right before launch, inside `systemd-run --user --scope --unit=<unit>`, at oom_score_adj 1000, with
deploy/lb/memfree_watch.sh beside it. Fixed flags: -fit off -np 1 --no-repack -lm mmap -cmoe -b 2048
--no-op-offload (prefill on the CPU for the experts, the same as the LB yardstick). --cache-ram is a server-only
flag (the perplexity example rejects it) and there is no prompt cache to limit here.

Writes DIR/windows.jsonl (one line per window), DIR/ids.i32 and DIR/result.json (per_window, ppl, ids_sha256, op_offload,
prefill_device, memorymax_mib, memorymax_table_mib, memfree_at_start_mib, memfree_min_mib, memfree_stop, llama tok/s, memcg
peak). Exit 0 when all windows are scored, 2 for an ids mismatch or a bad argument, 5 for a MemFree soft stop (result.json has
memfree_stop=true), else the helper's code.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tools" / "quality"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

WINDOW = 2048
FIXED_FLAGS = ["-fit", "off", "-np", "1", "--no-repack", "-lm", "mmap", "-cmoe", "-b", "2048",
               "--no-op-offload"]
EXTRA_FLAGS = ["-ngl", "99", "-ub", "2048", "-fa", "on", "-c", str(WINDOW)]


def read_windows(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def check_windows(rows, n_windows, window=WINDOW):
    """-> rows in window order. Refuses a missing, duplicated or short window."""
    by = {}
    for r in rows:
        k = int(r["window"])
        if k in by:
            raise ValueError(f"window {k} appears twice")
        by[k] = r
    if sorted(by) != list(range(n_windows)):
        raise ValueError(f"windows {sorted(by)} are not 0..{n_windows - 1}")
    for k, r in sorted(by.items()):
        if int(r["start"]) != k * window or int(r["tokens"]) != window - 1:
            raise ValueError(f"window {k}: start/tokens are {r['start']}/{r['tokens']}")
    return [by[k] for k in range(n_windows)]


def summarize(rows):
    tokens = sum(int(r["tokens"]) for r in rows)
    nll = math.fsum(float(r["nll"]) for r in rows)
    return {"windows": len(rows), "tokens_scored": tokens, "nll_sum": nll, "ppl": math.exp(nll / tokens)}


def parse_footprint(text):
    """'FOOTPRINT_MIB=57344 MEMORYMAX_MIB=47616 MEMFREE_MIB=70000 TABLE_MIB=47616' -> dict of ints."""
    d = {k: int(v) for k, v in re.findall(r"([A-Z_]+)=(\d+)", text)}
    for k in ("MEMORYMAX_MIB", "TABLE_MIB", "MEMFREE_MIB"):
        if k not in d:
            raise ValueError(f"llamacpp.sh --print-footprint printed no {k}: {text!r}")
    return d


def memfree_min(path):
    """Smallest MemFree (MiB) in the 1 Hz CSV of memfree_watch.sh, or None."""
    p = Path(path)
    if not p.exists():
        return None
    vals = []
    for ln in p.read_text().splitlines()[1:]:
        f = ln.split(",")
        if len(f) > 1 and f[1].strip().lstrip("-").isdigit():
            vals.append(int(f[1]))
    return min(vals) if vals else None


def meminfo_mib(key):
    for ln in Path("/proc/meminfo").read_text().splitlines():
        if ln.startswith(key + ":"):
            return int(ln.split()[1]) // 1024
    raise RuntimeError(f"{key} not in /proc/meminfo")


def model_first(weights, r):
    d = Path(r) / "model" / "q4_1-exact" if weights == "q41exact" else Path(r) / "lb" / "gguf" / "Qwen3.8-Flash-Next-Q4_1"
    first = sorted(d.glob("*-00001-of-*.gguf")) or sorted(d.glob("model.gguf"))
    if not first:
        raise FileNotFoundError(f"no GGUF under {d}")
    return first[0]


def scope_cgroup(unit):
    out = subprocess.run(["systemctl", "--user", "show", "-p", "ControlGroup", "--value", unit + ".scope"],
                         capture_output=True, text=True).stdout.strip()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine-ppl", required=True)
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--weights", default="q41exact", choices=("q41exact", "q41"))
    ap.add_argument("--gguf", help="diagnostic: score this GGUF (first shard) instead of the one --weights names")
    ap.add_argument("--out")
    ap.add_argument("--unit", default="slotstream-spark-f9-ref")
    ap.add_argument("--memfree-stop", type=int, default=14336)
    ap.add_argument("--memfree-log")
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(argv)

    import ppl_wikitext as pw
    from slotstream_spark.engine import score
    text, rows_sha = pw.load_text()
    ids = pw.tokenize(text, os.environ["SLOTSTREAM_MODEL"])
    ids_sha = pw.ids_sha256(ids)
    sl = score.window_slices(len(ids), WINDOW, a.windows)
    if len(sl) != a.windows:
        print(f"llamacpp_ppl: the text has {len(sl)} windows, {a.windows} requested", file=sys.stderr)
        return 2
    eng = json.loads(Path(a.engine_ppl).read_text())
    if eng.get("ids_sha256") != ids_sha or eng.get("rows_sha256") != rows_sha:
        print(f"llamacpp_ppl: {a.engine_ppl} has ids_sha256 {eng.get('ids_sha256')} / rows_sha256 {eng.get('rows_sha256')}, "
              f"this run {ids_sha} / {rows_sha}", file=sys.stderr)
        return 2
    r = os.environ.get("SLOTSTREAM_R") or str(Path.home() / "slotstream-spark")
    out = Path(a.out or Path(os.environ.get("CAP", ".")) / "f9-ref")
    out.mkdir(parents=True, exist_ok=True)
    import array
    (out / "ids.i32").write_bytes(array.array("i", ids).tobytes())
    wfile = out / "windows.jsonl"
    wfile.write_text("")
    mflog = a.memfree_log or str(out / "ppl-memfree.csv")

    memfree = meminfo_mib("MemFree")
    hu = parse_footprint(subprocess.run(
        ["bash", str(ROOT / "deploy/lb/llamacpp.sh"), "--print-footprint", "--weights", "q41exact", "--arm", "A", "--mode", "D5",
         "--memfree-mib", str(memfree)], capture_output=True, text=True, check=True).stdout)
    maxm, table = hu["MEMORYMAX_MIB"], hu["TABLE_MIB"]
    first = Path(a.gguf) if a.gguf else model_first(a.weights, r)
    helper = Path(r) / "lb" / "bin" / "ppl-ids"
    cmd = ["systemd-run", "--user", "--scope", f"--unit={a.unit}", "-p", f"MemoryMax={maxm}M", "-p", f"MemoryHigh={maxm - 512}M",
           "-p", "MemorySwapMax=0", "nice", "-n", "10", str(helper), "-m", str(first), "-t", str(a.threads)]
    cmd += FIXED_FLAGS + EXTRA_FLAGS + ["--ids-file", str(out / "ids.i32"), "--windows", str(a.windows), "--out", str(wfile)]
    print("llamacpp_ppl: MemoryMax", maxm, "MiB (table", table, ", MemFree", memfree, ") ->", " ".join(cmd), flush=True)
    try:
        Path("/proc/self/oom_score_adj").write_text("1000")
    except OSError:
        pass
    env = dict(os.environ, XDG_RUNTIME_DIR=os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}")
    log = open(out / "ppl-ids.log", "w")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    Path(mflog + ".stop").unlink(missing_ok=True)
    watch = subprocess.Popen(["bash", str(ROOT / "deploy/lb/memfree_watch.sh"), str(proc.pid), str(a.memfree_stop), mflog])
    peak = 0
    while proc.poll() is None:
        cg = scope_cgroup(a.unit)
        if cg:
            try:
                peak = max(peak, int(Path(f"/sys/fs/cgroup{cg}/memory.peak").read_text()) // 1048576)
            except (OSError, ValueError):
                pass
        time.sleep(5)
    rc = proc.returncode
    try:
        watch.wait(timeout=30)
    except subprocess.TimeoutExpired:
        watch.kill()
    stopped = Path(mflog + ".stop").exists()
    tail = (out / "ppl-ids.log").read_text()
    m = re.search(r"done, (\d+) windows, ([\d.]+) s, ([\d.]+) tok/s", tail)
    res = {"op_offload": False, "prefill_device": "cpu", "memorymax_mib": maxm, "memorymax_table_mib": table,
           "memfree_at_start_mib": memfree, "memfree_min_mib": memfree_min(mflog), "memfree_stop": stopped,
           "memfree_stop_mib": int(Path(mflog + ".stop").read_text().split()[2]) if stopped else None,
           "memfree_floor_mib": a.memfree_stop, "memcg_peak_mib": peak, "unit": a.unit, "weights": a.weights,
           "model_first_shard": first.name, "flags": FIXED_FLAGS + EXTRA_FLAGS, "ids_sha256": ids_sha, "rows_sha256": rows_sha,
           "engine_ppl_file": str(a.engine_ppl), "rc": rc, "seconds": round(time.time() - t0, 1),
           "llama_tok_s": float(m.group(3)) if m else None}
    if stopped:
        (out / "result.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        print("llamacpp_ppl: MemFree soft stop", file=sys.stderr)
        return 5
    if rc != 0:
        (out / "result.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        print(f"llamacpp_ppl: the helper exited {rc}", file=sys.stderr)
        return rc
    try:
        rows = check_windows(read_windows(wfile), a.windows)
    except ValueError as e:
        (out / "result.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
        print(f"llamacpp_ppl: {e}", file=sys.stderr)
        return 1
    res.update(summarize(rows))
    res["per_window"] = [{k: x[k] for k in ("window", "start", "nll", "tokens", "seconds")} for x in rows]
    (out / "result.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"llama.cpp ppl {res['ppl']:.4f} over {res['windows']} windows / {res['tokens_scored']:,} tokens "
          f"(MemoryMax {maxm} MiB, peak {peak} MiB, {res['llama_tok_s']} tok/s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
