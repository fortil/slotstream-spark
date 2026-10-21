#!/usr/bin/env python3
"""The F9 measurement campaign of one memory mode (plan step 93; judged by `tools/decide.py f9-campaign`, step 95).

  run_campaign.py --mode D5 --samples 5 [--cap DIR] [--cells decode-8k,ttft-cold-2k,ttft-cold-8k,ttft-cold-32k]

Runs inside the f9-engine job (the guard holds the model lock; CAP is the job capture). Every cell writes
`$CAP/<mode>-<cell>/result.json`:

  decode-8k       `--samples` fresh engine processes (`--child decode`), one B1 prompt each (bench/cohorts), 128 warm-up +
                  512 measured greedy tokens, context <= 8K, the pool pre-filled from the frequency table at start, MTP per
                  policy.toml, lookahead off; metric (N-1)/sum(inter-token) as bench/f6_arm.py
  ttft-cold-2k    `--samples` fresh server processes (`cli serve`), each with SLOTSTREAM_PREFIX_DIR on a directory created
  ttft-cold-8k    for that sample and still empty, one HTTP request of N prompt tokens through /api/chat (streamed), time to
  ttft-cold-32k   the first streamed frame; the final frame gives prefill seconds and `cached_tokens` (= reused tokens, which
                  must be 0 for a cold sample). The 8K cell sends the follow-up turn on the same server right after.
                  The 32K cell sends the longest prompt of at most 32,000 tokens that the text can be cut to exactly (31,990 tokens;
                  the served window is 32,768 tokens and must hold the reply, so a 32,768-token prompt is not admissible).

Per cell: `snapshot` (eligibility.collect), the footprint `samples` and count-once `flags` that the peak is recomputed from
(`footprint.peak`), `chunks`, `memavailable_drop_bytes`, and the engine counters that exist (pool stats, reader bytes and jobs
per token, the page-cache change, `/metrics` before and after). A cell that fails keeps its error and the job goes on; the
exit code is 0 only when every cell is ok (a D1 failure is a result, the job script decides what to do with it).
"""
import argparse
import http.client
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

PORT = 18437
COLD_TOKENS = {"2k": 2048, "8k": 8192, "32k": 32000}
DECODE_WARMUP, DECODE_TOKENS = 128, 512
MAX_KEPT_SAMPLES = 160
SERVER_UP_S = 900


# ------------------------------------------------------------------------------------------------ pure helpers
def median(xs):
    from bench.stats import true_median
    return true_median(list(xs))


def prefill_tps(prompt_tokens, prefill_s):
    return None if not prefill_s else prompt_tokens / prefill_s


def thin_samples(samples, keep=MAX_KEPT_SAMPLES):
    """At most `keep` samples that still hold the peak: the first, the last and the argmax of VmHWM, NVML and VmRSS, then evenly spaced ones."""
    n = len(samples)
    if n <= keep:
        return list(samples)
    pick = {0, n - 1}
    for key in ("vmhwm", "nvml", "vmrss"):
        pick.add(max(range(n), key=lambda i: samples[i][key]))
    room = max(0, keep - len(pick))
    pick.update(int(i * (n - 1) / max(1, room)) for i in range(room))
    return [samples[i] for i in sorted(pick)]


def read_footprint(path):
    """(samples, flags) from a serve footprint JSONL (header line with the flags, then one sample per line)."""
    flags, samples = {}, []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d.get("header"):
            flags = {k: [bool(v[0]), bool(v[1])] for k, v in d["flags"].items()}
            continue
        samples.append({"tag": str(d.get("t_ns", len(samples))), "vmrss": int(d["vmrss"]), "vmhwm": int(d["vmhwm"]),
                        "nvml": int(d["nvml"]), "pinned": 0, "unified": {k: int(v) for k, v in d.get("unified", {}).items()}})
    return samples, flags


def peak_of(samples, flags):
    from slotstream_spark.memory import footprint as fp
    pk = fp.peak([fp.sample_from_dict(d) for d in samples], fp.flags_from_dict(flags))
    return fp.peak_to_dict(pk)


class AvailSampler:
    """1 s MemAvailable series (bytes) between start() and stop()."""

    def __init__(self, path="/proc/meminfo"):
        self.path, self.series, self._stop = path, [], threading.Event()
        self.start_bytes = self.now()
        self._th = threading.Thread(target=self._run, daemon=True)

    def now(self):
        with open(self.path) as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
        return 0

    def start(self):
        self._th.start()
        return self

    def _run(self):
        while not self._stop.wait(1.0):
            self.series.append(self.now())

    def stop(self):
        self._stop.set()
        self._th.join(3)
        self.series.append(self.now())
        low = min(self.series)
        return {"memavailable_start_bytes": self.start_bytes, "memavailable_min_bytes": low,
                "memavailable_drop_bytes": max(0, self.start_bytes - low)}


def cached_bytes():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("Cached:"):
                return int(line.split()[1]) * 1024
    return 0


# ------------------------------------------------------------------------------------------------ HTTP
def chat_ttft(port, model, messages, num_predict=8, timeout=1800, host="127.0.0.1"):
    """One streamed /api/chat request. -> dict with ttft_s (time to the first frame that carries content, or the final frame),
    total_s and the final frame's counters (prompt_eval_count, prompt_eval_duration, cached_tokens, eval_count, eval_duration)."""
    body = json.dumps({"model": model, "messages": messages, "stream": True, "options": {"temperature": 0, "num_predict": num_predict}})
    c = http.client.HTTPConnection(host, port, timeout=timeout)
    t0 = time.perf_counter()
    c.request("POST", "/api/chat", body, {"content-type": "application/json"})
    r = c.getresponse()
    if r.status != 200:
        raw = r.read()
        c.close()
        raise RuntimeError(f"/api/chat answered {r.status}: {raw[:300]!r}")
    first, final = None, None
    while True:
        line = r.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        now = time.perf_counter() - t0
        if first is None and (d.get("message", {}).get("content") or d.get("done")):
            first = now
        if d.get("done"):
            final = d
            final["_total_s"] = now
    c.close()
    if final is None or first is None:
        raise RuntimeError("the stream ended without a final frame")
    return {"ttft_s": first, "total_s": final["_total_s"], "prompt_tokens": int(final["prompt_eval_count"]),
            "prefill_s": final["prompt_eval_duration"] / 1e9, "reused_tokens": int(final.get("cached_tokens", 0)),
            "eval_count": int(final.get("eval_count", 0)), "eval_s": final.get("eval_duration", 0) / 1e9}


def get_json(port, path, timeout=30, host="127.0.0.1"):
    c = http.client.HTTPConnection(host, port, timeout=timeout)
    c.request("GET", path)
    r = c.getresponse()
    raw = r.read()
    c.close()
    return r.status, raw


def metrics(port):
    """/metrics as {name: value} for the plain slotstream_* counters."""
    try:
        st, raw = get_json(port, "/metrics")
    except OSError:
        return {}
    out = {}
    for line in raw.decode().splitlines():
        if line.startswith("slotstream_") and "{" not in line:
            k, _, v = line.partition(" ")
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


# ------------------------------------------------------------------------------------------------ server
class Server:
    def __init__(self, mode, port, cell_dir, prefix_dir):
        self.mode, self.port, self.cell_dir, self.prefix_dir = mode, port, Path(cell_dir), Path(prefix_dir)
        self.proc = None
        self.log = None

    def start(self):
        env = dict(os.environ, SLOTSTREAM_PREFIX_DIR=str(self.prefix_dir), CAP=str(self.cell_dir), SLOTSTREAM_MODE=self.mode)
        self.log = open(self.cell_dir / "serve.log", "ab")
        self.proc = subprocess.Popen([str(ROOT / "tools" / "spark" / "py.sh"), "-m", "slotstream_spark.cli", "serve",
                                      "--mode", self.mode, "--port", str(self.port)], cwd=str(ROOT), env=env,
                                     stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < SERVER_UP_S:
            if self.proc.poll() is not None:
                raise RuntimeError(f"the server exited with {self.proc.returncode} before it was ready (serve.log)")
            try:
                st, raw = get_json(self.port, "/health", timeout=3)
                if st == 200 and json.loads(raw).get("ok"):
                    return json.loads(raw)
            except (OSError, ValueError):
                pass
            time.sleep(1.0)
        raise RuntimeError(f"/health did not answer ok in {SERVER_UP_S} s")

    def oom_score_adj(self):
        try:
            return int(Path(f"/proc/{self.proc.pid}/oom_score_adj").read_text())
        except (OSError, ValueError):
            return None

    def model_id(self):
        st, raw = get_json(self.port, "/v1/models")
        return json.loads(raw)["data"][0]["id"]

    def stop(self):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(90)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(30)
        if self.log:
            self.log.close()


# ------------------------------------------------------------------------------------------------ cells
def snapshot(est_s, mode):
    from bench import eligibility
    snap = eligibility.collect(0, est_s, mode=mode)
    ok, why = eligibility.evaluate(snap)
    return snap, ok, why


def write_cell(cap, mode, cell, doc):
    d = Path(cap) / f"{mode.lower()}-{cell}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return d


def cell_dir(cap, mode, cell):
    d = Path(cap) / f"{mode.lower()}-{cell}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cut_at_most(tok, text_ids, n, floor=256):
    """(ids, body) of the longest prompt of at most `n` template tokens that can be cut exactly; n itself when it can.
    The text decodes and re-encodes to a few more tokens at some lengths (32,000 and 32,100 do not cut, 31,990 does)."""
    from bench import ttft
    last = None
    for m in range(n, n - floor, -1):
        try:
            return ttft.cut_prompt(tok, text_ids, m)
        except RuntimeError as e:
            last = e
    raise RuntimeError(f"no prompt of {n - floor + 1}..{n} tokens can be cut: {last}")


def ttft_sample(mode, cdir, tokens, with_followup, k):
    """One fresh server, one cold request (and the follow-up on the 8K cell). -> sample dict (raises on a server failure)."""
    from bench import ttft
    from slotstream_spark.engine.tokenizer import Tokenizer
    tok = Tokenizer.from_checkpoint(os.environ["SLOTSTREAM_MODEL"])
    prompts = ttft.load_prompts()
    ids, body = cut_at_most(tok, tok.encode(prompts["text"]), tokens)
    sdir = cdir / f"sample{k}"
    sdir.mkdir(parents=True, exist_ok=True)
    prefix_dir = Path(tempfile.mkdtemp(prefix="f9-prefix-", dir=os.environ.get("SLOTSTREAM_R") and str(Path(os.environ["SLOTSTREAM_R"]) / "tmp")))
    assert not any(prefix_dir.iterdir())                                 # a cold sample starts from an empty prefix directory
    avail = AvailSampler().start()
    srv = Server(mode, PORT, sdir, prefix_dir)
    out = {"sample": k, "prompt_tokens_expected": len(ids), "prompt_sha256": ttft.ids_sha256(ids)}
    try:
        health = srv.start()
        out["chunks"] = int(health["chunks"])
        out["oom_score_adj"] = srv.oom_score_adj()
        model = srv.model_id()
        m0 = metrics(PORT)
        out["prefix_dir_entries_after_start"] = sorted(x.name for x in prefix_dir.iterdir())   # the tier's own bookkeeping; the dir held nothing before
        cold = chat_ttft(PORT, model, [{"role": "user", "content": body}])
        cold["prefill_tps"] = prefill_tps(cold["prompt_tokens"], cold["prefill_s"])
        out["cold"] = cold
        if with_followup:
            msgs = [{"role": "user", "content": body}, {"role": "assistant", "content": prompts["reply"]},
                    {"role": "user", "content": prompts["followup"]}]
            f = chat_ttft(PORT, model, msgs)
            f["new_tokens"] = f["prompt_tokens"] - f["reused_tokens"]
            out["followup"] = f
        out["metrics_delta"] = {k2: v - m0.get(k2, 0.0) for k2, v in metrics(PORT).items()}
    finally:
        srv.stop()
        out.update(avail.stop())
        shutil_rmtree(prefix_dir)
    fp = sdir / "footprint-serve.jsonl"
    out["footprint_file"] = fp.name if fp.exists() else None
    return out


def shutil_rmtree(p):
    import shutil
    shutil.rmtree(p, ignore_errors=True)


def run_ttft_cell(cap, mode, name, samples, est_s):
    tokens = COLD_TOKENS[name.rsplit("-", 1)[1]]
    cdir = cell_dir(cap, mode, name)
    with_follow = name.endswith("8k")
    rows, failures, mem_samples, flags, chunks = [], [], [], {}, 0
    drop = 0
    snap, eligible, why = snapshot(est_s, mode)
    for k in range(samples):
        try:
            s = ttft_sample(mode, cdir, tokens, with_follow, k)
        except Exception as e:                                           # noqa: BLE001
            failures.append(f"sample {k}: {type(e).__name__}: {e}")
            print(f"{mode} {name} sample {k} FAILED: {e}", flush=True)
            continue
        if s["cold"]["reused_tokens"] != 0:
            failures.append(f"sample {k}: cold sample reused {s['cold']['reused_tokens']} tokens")
        if s["cold"]["prompt_tokens"] != s["prompt_tokens_expected"]:
            failures.append(f"sample {k}: the server counted {s['cold']['prompt_tokens']} prompt tokens, not {s['prompt_tokens_expected']}")
        if not name.endswith("32k") and s["prompt_tokens_expected"] != tokens:
            failures.append(f"sample {k}: the prompt has {s['prompt_tokens_expected']} tokens, not {tokens}")
        rows.append(s)
        chunks = max(chunks, s["chunks"])
        drop = max(drop, s["memavailable_drop_bytes"])
        fpf = cdir / f"sample{k}" / "footprint-serve.jsonl"
        if fpf.exists():
            sm, fl = read_footprint(fpf)
            flags = flags or fl
            for d in thin_samples(sm, MAX_KEPT_SAMPLES // max(1, samples)):
                d["tag"] = f"s{k}:{d['tag']}"
                mem_samples.append(d)
        print(f"{mode} {name} sample {k}: ttft {s['cold']['ttft_s']:.2f} s, prefill {s['cold']['prefill_s']:.2f} s "
              f"({s['cold']['prefill_tps']:.1f} tok/s), reused {s['cold']['reused_tokens']}"
              + (f", follow-up {s['followup']['ttft_s']:.2f} s (reused {s['followup']['reused_tokens']})" if with_follow and 'followup' in s else ""),
              flush=True)
    doc = {"cell": name, "mode": mode, "kind": "ttft-cold", "tokens": tokens,
           "prompt_tokens": rows[0]["prompt_tokens_expected"] if rows else None, "samples_requested": samples,
           "samples": rows, "failures": failures, "snapshot": snap, "eligible": eligible, "ineligible_reasons": why,
           "chunks": chunks, "memavailable_drop_bytes": drop, "footprint_samples": mem_samples, "flags": flags,
           "ok": bool(rows) and not failures and len(rows) == samples}
    if rows:
        doc["medians"] = {"ttft_s": median([r["cold"]["ttft_s"] for r in rows]),
                          "prefill_s": median([r["cold"]["prefill_s"] for r in rows]),
                          "prefill_tps": median([r["cold"]["prefill_tps"] for r in rows])}
        if with_follow and all("followup" in r for r in rows):
            doc["medians"]["followup_ttft_s"] = median([r["followup"]["ttft_s"] for r in rows])
    if mem_samples and flags:
        doc["peak"] = peak_of(mem_samples, flags)
    write_cell(cap, mode, name, doc)
    return doc["ok"]


def run_decode_cell(cap, mode, samples, est_s):
    cdir = cell_dir(cap, mode, "decode-8k")
    rows, failures, mem_samples, flags, chunks, drop = [], [], [], {}, 0, 0
    snap, eligible, why = snapshot(est_s, mode)
    me = str(Path(__file__).resolve())
    for k in range(samples):
        out = cdir / f"child{k}.json"
        out.unlink(missing_ok=True)
        t = time.time()
        rc = subprocess.run([str(ROOT / "tools" / "spark" / "py.sh"), me, "--child", "decode", "--mode", mode, "--pair", str(k),
                             "--out", str(out)], cwd=str(ROOT)).returncode
        d = json.loads(out.read_text()) if out.exists() else {}
        if rc != 0 or not d.get("ok"):
            failures.append(f"prompt {k}: rc {rc} ok {d.get('ok')}")
            print(f"{mode} decode-8k prompt {k} FAILED rc {rc}", flush=True)
            continue
        d["wall_s"] = round(time.time() - t, 1)
        rows.append({x: y for x, y in d.items() if x not in ("footprint_samples", "flags", "peak_child")})
        chunks = max(chunks, d["chunks"])
        drop = max(drop, d["memavailable_drop_bytes"])
        flags = flags or d["flags"]
        for s in thin_samples(d["footprint_samples"], MAX_KEPT_SAMPLES // max(1, samples)):
            s["tag"] = f"p{k}:{s['tag']}"
            mem_samples.append(s)
        print(f"{mode} decode-8k prompt {k}: {d['decode_tps']:.2f} tok/s (mtp {d['mtp_policy']}), prefill {d['prefill_s']} s", flush=True)
    doc = {"cell": "decode-8k", "mode": mode, "kind": "decode", "samples_requested": samples, "samples": rows,
           "failures": failures, "snapshot": snap, "eligible": eligible, "ineligible_reasons": why, "chunks": chunks,
           "memavailable_drop_bytes": drop, "footprint_samples": mem_samples, "flags": flags,
           "ok": len(rows) == samples and not failures}
    if rows:
        doc["decode_tps_median"] = median([r["decode_tps"] for r in rows])
    if mem_samples and flags:
        doc["peak"] = peak_of(mem_samples, flags)
    write_cell(cap, mode, "decode-8k", doc)
    return doc["ok"]


# ------------------------------------------------------------------------------------------------ decode child
def child_decode(a):
    """One fresh engine process, one B1 prompt: as bench/f6_arm.py plus footprint samples and the reader counters."""
    from bench import f6_arm, ttft
    from slotstream_spark.engine import lookahead as la
    from slotstream_spark.engine import mtp as mtp_mod
    from slotstream_spark.engine.generate import generate
    from slotstream_spark.engine.prefill import PrefillPlan
    from slotstream_spark.engine.tokenizer import Tokenizer
    from slotstream_spark.runtime.graphs import GraphSet
    from slotstream_spark.runtime.readback import Readback
    import torch

    req = f6_arm.load_request("B1", a.pair)
    tok = Tokenizer.from_checkpoint(os.environ["SLOTSTREAM_MODEL"])
    ids = tok.encode(tok.render_chat(req["messages"], enable_thinking=False))
    total = DECODE_WARMUP + DECODE_TOKENS
    lcfg = la.LookaheadConfig.load()
    if lcfg.on:
        raise SystemExit("run_campaign: lookahead is on; the F9 campaign runs with it off (db/records/decisions/f6-lookahead-d5.md)")
    mtp_mode = mtp_mod.policy(a.mode)
    max_ctx = -(-(max(len(ids) + total + 64, f6_arm.WARMUP_CTX + 64)) // 64) * 64
    eng = ttft.bring_up(a.mode, max_ctx, prefill=True)
    model, pool, cfg = eng["model"], eng["pool"], eng["cfg"]
    meter = eng["meter"]
    model.st.ensure_act(torch.bfloat16)
    rb = Readback(model.vocab, forecast=False)
    g1 = GraphSet(model, 1, readback=rb, forecast=None)
    head = g3 = None
    if mtp_mode == "resident":
        packed = Path(os.environ["SLOTSTREAM_PACKED"])
        head = mtp_mod.MTPHead(cfg, packed / "mtp_trunk.safetensors", packed / "mtp_experts.bin", embed=model.seg.embed,
                               lm_head=model.W("lm_head.weight"), device="cuda", max_ctx=max_ctx)
        g3 = GraphSet(model, 3, readback=rb, pool_handle=g1.pool_handle, forecast=None, record=True, multi=True)
    ttft.warm_up(eng, clear=False)
    model.st.reset_inplace()
    meter.snap("warm")
    plan = PrefillPlan.from_envelope(eng["env"], prefill_pass=4096, read_scope=8192)
    r0, cache0 = pool.reader.stats(), cached_bytes()
    stamps = []
    t0 = time.perf_counter()
    res = generate(model, ids, max_tokens=total, greedy=True, seed=42, eos=None, driver=eng["driver"], plan=plan,
                   on_token=lambda _t: stamps.append(time.perf_counter()), graphs=g1, mtp=head, graphs3=g3)
    torch.cuda.synchronize()
    meter.snap("decode")
    r1, cache1 = pool.reader.stats(), cached_bytes()
    meter.stop()
    tps = f6_arm.decode_tps(stamps, DECODE_WARMUP, DECODE_TOKENS)
    rec_bytes = int(pool.record_bytes)
    read_bytes = int(r1["bytes"]) - int(r0["bytes"])
    jobs = int(r1["jobs"]) - int(r0["jobs"])
    n_tok = max(1, len(res["tokens"]))
    mres = meter.result()
    out = {"prompt_index": a.pair, "prompt_id": req["id"], "prompt_sha256": req["prompt_sha256"], "prompt_tokens": len(ids),
           "decode_tps": tps, "tokens_sha256": f6_arm.ids_sha256(res["tokens"]), "n_tokens": len(res["tokens"]),
           "nan": bool(res["nan"]), "prefill_s": round(res["timings"]["prefill_s"], 3), "mtp_policy": mtp_mode,
           "lookahead_mode": lcfg.mode, "mtp": res["mtp"], "mode": a.mode, "slots": eng["slots"],
           "chunks": eng["slots"] // _slots_per_chunk(), "pool_prefill": eng["pool_prefill"],
           "pool": {k: v for k, v in pool.stats().items() if isinstance(v, (int, float))},
           "reader": {"bytes": read_bytes, "jobs": jobs, "records": read_bytes // rec_bytes if rec_bytes else None,
                      "records_per_token": (read_bytes / rec_bytes) / n_tok if rec_bytes else None,
                      "bytes_per_token": read_bytes / n_tok, "read_ns_per_job": ((int(r1["read_ns"]) - int(r0["read_ns"])) / jobs if jobs else None),
                      "memcpy_hist": r1.get("memcpy_hist")},
           "page_cache": {"cached_delta_bytes": cache1 - cache0, "read_bytes": read_bytes,
                          "ratio": ((cache1 - cache0) / read_bytes) if read_bytes else None},
           "prefetch": None, "prefetch_note": "lookahead off (KEEP_OFF): no prefetch precision or lateness exists",
           "nvme_latency": None, "nvme_latency_note": "the reader keeps mean read_ns per job and a memcpy histogram, no per-IO percentiles",
           "footprint_samples": mres["samples"], "flags": mres["flags"], "peak_child": mres["peak"],
           "memavailable_drop_bytes": mres["memavailable_drop_bytes"],
           "versions": {"torch": torch.__version__, "cuda": torch.version.cuda}}
    out["ok"] = bool((not res["nan"]) and len(res["tokens"]) == total and tps is not None
                     and not (res["mtp"] or {}).get("leaked_records") and eng["pool_prefill"]["exhausted"] == 0)
    Path(a.out).write_text(json.dumps(out, sort_keys=True) + "\n")
    print(json.dumps({k: out[k] for k in ("prompt_id", "decode_tps", "ok")}), flush=True)
    return 0 if out["ok"] else 1


def _slots_per_chunk():
    from slotstream_spark.config import modes
    return modes.SLOTS_PER_CHUNK


# ------------------------------------------------------------------------------------------------ main
ALL_CELLS = ("decode-8k", "ttft-cold-2k", "ttft-cold-8k", "ttft-cold-32k")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("D1", "D2", "D5"), required=True)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--cap", default=os.environ.get("CAP"))
    ap.add_argument("--cells", default=",".join(ALL_CELLS))
    ap.add_argument("--child", choices=("decode",))
    ap.add_argument("--pair", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.child:
        if not a.out:
            raise SystemExit("--child needs --out")
        return child_decode(a)
    if not a.cap:
        raise SystemExit("run_campaign: no CAP (run inside the guard) and no --cap")
    cells = [c for c in a.cells.split(",") if c]
    bad = [c for c in cells if c not in ALL_CELLS]
    if bad:
        raise SystemExit(f"unknown cells: {bad}")
    ok_all = True
    for name in cells:
        t = time.time()
        print(f"== {a.mode} {name} ({a.samples} samples)", flush=True)
        est = 1200 * a.samples if name != "ttft-cold-32k" else 1500 * a.samples
        ok = run_decode_cell(a.cap, a.mode, a.samples, est) if name == "decode-8k" else run_ttft_cell(a.cap, a.mode, name, a.samples, est)
        print(f"== {a.mode} {name}: {'ok' if ok else 'FAILED'} in {time.time() - t:.0f} s", flush=True)
        ok_all = ok_all and ok
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
