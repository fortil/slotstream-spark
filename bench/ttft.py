#!/usr/bin/env python3
"""Prefill and TTFT in D5 (plan step 72).

  python bench/ttft.py --build-prompts [--out bench/prompts/ttft.json]   Mac, once: freeze the text spans
  python bench/ttft.py --mode D5 --samples 5 [--out FILE]                Spark (through capture.sh)

The run process starts one fresh child process per measurement (`--child KIND`), each after the normal warm start
(engine up, kernels compiled by an unrelated warm-up prompt, then the pool emptied again, so the timed request meets
a cold pool and an empty prefix cache):
  cold8k   `--samples` times: cold 8,192-token TTFT split into queue, prefill, first sample; then, in the same
           process with the prefix cache in memory, the follow-up turn (at most 1,024 new tokens)
  once     2,048 tokens once, and 32,768 tokens once (published)
  scope16k 16,384 tokens with the read scope on and off (pool emptied between), at the D5 pass and at pass 1,024
Every child records `footprint.Sample` dicts (process start, after the pool, after the staging buffers, after the
warm-up, after each request, at the end, plus one per second from a sampler thread), their count-once flags,
`memavailable_drop_bytes` (MemAvailable at start minus the minimum of the 1 s series) and the peak recomputed by
`slotstream_spark.memory.footprint`. `tools/decide.py f5-prefill` judges the capture.

Prompts: frozen text spans of blobs the corpus train split cites (bench/prompts/ttft.json), cut at run time so the
rendered chat prompt has exactly N tokens; every prompt is a prefix of the same text and runs in its own process.
"""
import argparse
import array
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "bench" / "prompts" / "ttft.json"
CORPUS = ROOT / "third_party" / "slotstream" / "Tools" / "fixtures" / "expert-lookahead" / "corpus.json"
SOURCES = ROOT / "third_party" / "slotstream"
SIZES = (2048, 8192, 16384, 32768)
TEXT_CHARS = 160_000
WARMUP_CTX = 3072                        # the warm-up prompt (facts, ~2,735 template tokens) must fit every child
FOLLOWUP = ("Thanks. Now compare the two parts of the material you were given: which one is longer, which one is "
            "more technical, and what would you cut first if it had to fit on one page? Answer in a short list and "
            "finish with one sentence that names the single most important thing to keep.")
REPLY = ("The material is a sequence of source files and documents. I looked at its structure and the way each part "
         "is organized. The first parts define data types and their invariants, the later parts exercise them in "
         "scripts that print tables. Tell me what you want to compare and I will go through it part by part.")


# ------------------------------------------------------------------------------------------------ prompts
def build_prompts(out):
    """Mac: concatenate whole files of the blobs the train split cites (blob sha1 must match), sorted by path, until
    TEXT_CHARS; write the frozen json."""
    corpus = json.loads(CORPUS.read_text())
    blobs = {}
    for r in corpus["requests"]:
        if r["split"] != "train" or "source" not in r:
            continue
        p = SOURCES / r["source"]["path"]
        b = p.read_bytes()
        if hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() == r["source"]["blob"]:
            blobs[r["source"]["path"]] = r["source"]["blob"]
    files = [{"path": path, "blob": blobs[path]} for path in sorted(blobs)]
    base = "".join(f"### {f['path']}\n{(SOURCES / f['path']).read_bytes().decode('utf-8')}\n" for f in files)
    parts, size, k = [], 0, 0
    while size < TEXT_CHARS:                    # the train blobs hold ~100k chars: repeat them, each pass numbered
        parts.append(f"## Pass {k}\n{base}")
        size += len(parts[-1])
        k += 1
    text = "".join(parts)[:TEXT_CHARS]
    doc = {"version": 1, "source": "corpus.json train split (blobs whose git sha matches the corpus), whole files sorted "
           "by path, repeated with numbered pass headers up to text_chars", "files": files, "base_chars": len(base), "text_chars": len(text), "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
           "sizes": list(SIZES), "followup": FOLLOWUP, "reply": REPLY, "text": text}
    Path(out).write_text(json.dumps(doc, ensure_ascii=False, indent=0) + "\n")
    print(f"wrote {out}: {len(files)} files, {len(text)} chars, sha256 {doc['text_sha256']}")
    return 0


def load_prompts(path=PROMPTS):
    d = json.loads(Path(path).read_text())
    if hashlib.sha256(d["text"].encode()).hexdigest() != d["text_sha256"]:
        raise SystemExit(f"{path}: text does not match its sha256")
    return d


def ids_sha256(ids):
    return hashlib.sha256(array.array("i", [int(t) for t in ids]).tobytes()).hexdigest()


def cut_prompt(tok, text_ids, n, thinking=False):
    """(prompt ids, body) of exactly `n` template tokens: one user turn whose body is the first k tokens of the text."""
    def render(body):
        return tok.encode(tok.render_chat([{"role": "user", "content": body}], enable_thinking=thinking))
    k = n - len(render(""))
    for _ in range(40):
        body = tok.decode(text_ids[:k])
        ids = render(body)
        if len(ids) == n:
            return ids, body
        k += n - len(ids)
    for d in range(-12, 13):
        body = tok.decode(text_ids[:k + d])
        ids = render(body)
        if len(ids) == n:
            return ids, body
    raise RuntimeError(f"cannot cut a {n}-token prompt (last length {len(ids)})")


def build_ids(tok, prompts, n, thinking=False):
    return cut_prompt(tok, tok.encode(prompts["text"]), n, thinking)


def followup_ids(tok, body, prompts, thinking=False):
    msgs = [{"role": "user", "content": body}, {"role": "assistant", "content": prompts["reply"]},
            {"role": "user", "content": prompts["followup"]}]
    return tok.encode(tok.render_chat(msgs, enable_thinking=thinking))


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# ------------------------------------------------------------------------------------------------ meters
def vm_kb(field):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    return 0


def mem_available_bytes():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def nvml_used(pid):
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
            if p.pid == pid and p.usedGpuMemory is not None:
                return int(p.usedGpuMemory)
    except Exception:
        pass
    return 0


class Meter:
    """Named phase samples plus a 1 s sampler thread (each tick also a Sample, tagged t<n>)."""

    def __init__(self, flags_fn):
        self.samples, self.series, self.unified, self.flags = [], [], {}, {}
        self.flags_fn = flags_fn
        self.start_avail = mem_available_bytes()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._sampling = False                 # tick samples start once the unified allocations are registered
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def start_sampling(self):
        """From here the 1 s ticks also record Samples (before it only the MemAvailable series runs: a tick taken while the
        pool exists but is not yet registered would count its bytes as non-pool)."""
        self._sampling = True

    def _sample(self, tag):
        with self._lock:
            self.samples.append({"tag": tag, "vmrss": vm_kb("VmRSS"), "vmhwm": vm_kb("VmHWM"),
                                 "nvml": nvml_used(os.getpid()), "pinned": 0, "unified": dict(self.unified)})

    def _run(self):
        n = 0
        while not self._stop.wait(1.0):
            self.series.append(mem_available_bytes())
            if self._sampling:
                self._sample(f"t{n}")
            n += 1

    def snap(self, tag):
        self._sample(tag)

    def register(self, name, nbytes, mem_type):
        self.unified[name] = int(nbytes)
        self.flags[name] = list(self.flags_fn(mem_type))

    def stop(self):
        self._stop.set()
        self._th.join(3)
        self.series.append(mem_available_bytes())
        self._sample("end")

    def result(self):
        from slotstream_spark.memory import footprint as fp
        samples = [fp.sample_from_dict(d) for d in self.samples]
        flags = fp.flags_from_dict(self.flags)
        pk = fp.peak(samples, flags)
        low = min(self.series + [mem_available_bytes()]) if self.series else mem_available_bytes()
        return {"samples": self.samples, "flags": self.flags, "memavailable_start_bytes": self.start_avail,
                "memavailable_min_bytes": low, "memavailable_drop_bytes": max(0, self.start_avail - low),
                "peak": fp.peak_to_dict(pk)}


# ------------------------------------------------------------------------------------------------ child
class TimedCache:
    """A PrefixCache whose `take` is timed (the queue term of a request)."""

    def __new__(cls, *a, **kw):
        from slotstream_spark.engine.prefix_cache import PrefixCache

        class Timed(PrefixCache):
            take_s = 0.0

            def take(self, *aa, **kk):
                t = time.perf_counter()
                try:
                    return super().take(*aa, **kk)
                finally:
                    self.take_s = time.perf_counter() - t
        return Timed(*a, **kw)


def bring_up(mode, max_ctx, prefill=False):
    """The normal warm start: engine, sweep driver, then an unrelated warm-up prompt and an emptied pool.

    prefill=True fills the pool from the expert frequency table right after the engine is built (no GraphSet exists yet)."""
    import argparse as ap
    import torch
    from slotstream_spark import cli
    from slotstream_spark.engine import prefill_schedule as ps
    from slotstream_spark.engine.prefill import PrefillDriver
    from slotstream_spark.memory import signals
    from slotstream_spark.model.moe_sweep import SweepMoE
    signals.write_oom_score_adj()
    a = ap.Namespace(mode=mode, pool_slots=None, pool_epl=None, gdn="fused")
    slots = cli._slots(a)
    meter = Meter(lambda mt: cli.FootprintFeed.POOL_FLAGS[mt])
    meter.snap("start")
    model, pool, cfg, info = cli.build_engine(a, slots, max_ctx)
    meter.register("pool", pool.owned_bytes(), pool.mem_type)
    meter.snap("pool")
    pf = None
    if prefill:
        from slotstream_spark.engine import pool_prefill
        pf = pool_prefill.prefill_pool(model.seg, pool, pool_prefill.load_table())
        meter.snap("prefill")
    env = ps.envelope()
    moe = SweepMoE(model.seg, pool, max_rows_per_launch=int(env["max_rows_per_launch"]))
    meter.register("staging", moe.staging.bytes, pool.mem_type)
    meter.snap("staging")
    meter.start_sampling()
    driver = PrefillDriver(model, moe, env)
    return {"model": model, "pool": pool, "moe": moe, "driver": driver, "env": env, "cfg": cfg, "slots": slots,
            "meter": meter, "info": info, "torch": torch, "pool_prefill": pf}


def clear_pool(eng):
    eng["pool"].resize_slots(0)
    eng["pool"].resize_slots(eng["slots"])


def warm_up(eng, clear=True):
    from bench.lb.client import build_prompt
    from slotstream_spark.engine.prefill import PrefillPlan, d5_budget
    from slotstream_spark.engine.tokenizer import Tokenizer
    tok = Tokenizer.from_checkpoint(os.environ["SLOTSTREAM_MODEL"])
    ids = tok.encode(tok.render_user_turn(build_prompt({"kind": "facts", "seed": 7, "target_chars": 8000})))
    model = eng["model"]
    model.st.reset_inplace()
    plan = PrefillPlan.from_envelope(eng["env"], prefill_pass=4096, read_scope=8192)
    eng["driver"].run(ids, model.st, plan=plan, control=d5_budget(), should_continue=lambda: True)
    model.st.reset_inplace()
    if clear:
        clear_pool(eng)
    eng["torch"].cuda.synchronize()
    eng["meter"].snap("warm")


def request(eng, cache, key, ids, plan, target, tag):
    """One request up to its first sampled token; -> dict of the TTFT split."""
    import numpy as np
    from slotstream_spark.engine.generate import _prefill_cached
    from slotstream_spark.engine.sampler import SampleParams, Sampler
    model = eng["model"]
    info = {"reserve_tokens": len(ids) + 1}
    t0 = time.perf_counter()
    logits, pre = _prefill_cached(model, ids, eng["driver"], plan, None, None, cache, key, target, info)
    eng["torch"].cuda.synchronize()
    t1 = time.perf_counter()
    if pre is not None and pre.sweep.get("gemv_fallbacks"):
        raise RuntimeError("gemv fallback inside the sweep prefill")
    arr = logits.cpu().numpy()
    tok = Sampler(0).next(arr, SampleParams(temperature=0.0), set())
    t2 = time.perf_counter()
    queue = cache.take_s
    out = {"tag": tag, "prompt_tokens": len(ids), "reused_tokens": int(info["reused"]),
           "decision": info["decision"], "snapshot_end": info.get("snapshot_end"),
           "queue_s": queue, "prefill_s": pre.seconds if pre else 0.0, "first_sample_s": t2 - t1,
           "store_s": max(0.0, (t1 - t0) - queue - (pre.seconds if pre else 0.0)), "ttft_s": t2 - t0,
           "first_token": int(tok), "finite": bool(np.isfinite(arr).all())}
    if pre is not None:
        out.update({"passes": len(pre.passes), "bytes_requested": pre.bytes_requested,
                    "records_read": pre.records_read, "admitted": pre.admitted, "scopes": len(pre.scopes),
                    "gemm_calls": pre.sweep.get("gemm_calls"), "gemv_fallbacks": pre.sweep.get("gemv_fallbacks")})
    out["snapshot_admitted"] = info.get("snapshot_admitted")
    return out


def child_main(a):
    from slotstream_spark.engine import prefix_cache as pc
    from slotstream_spark.engine.prefill import PrefillPlan
    from slotstream_spark.engine.tokenizer import Tokenizer
    from slotstream_spark.memory.ledger import Ledger
    prompts = load_prompts()
    tok = Tokenizer.from_checkpoint(os.environ["SLOTSTREAM_MODEL"])
    im_start = tok.encode("<|im_start|>")[0]
    n = int(a.n)
    ids, body = build_ids(tok, prompts, n)
    ids2 = followup_ids(tok, body, prompts) if a.child == "cold8k" else None
    longest = max(len(ids), len(ids2 or ()))
    eng = bring_up(a.mode, -(-(max(longest, WARMUP_CTX) + 64) // 64) * 64)      # the warm-up prompt needs its own room
    res = {"child": a.child, "n": n, "prompt_sha256": ids_sha256(ids), "slots": eng["slots"],
           "chunks": eng["slots"] // _slots_per_chunk(), "info": eng["info"]}
    warm_up(eng)
    key = pc.CheckpointKey(Path(os.environ["SLOTSTREAM_MODEL"]).name, json.dumps({"gdn": "fused", "sweep": True}),
                           4096, False)
    max_tokens = Ledger(a.mode).prefix // pc.BYTES_PER_TOKEN
    plan = PrefillPlan.from_envelope(eng["env"], prefill_pass=4096, read_scope=8192)
    meter = eng["meter"]
    if a.child == "cold8k":
        cache = TimedCache(max_tokens=max_tokens)
        res["cold"] = request(eng, cache, key, ids, plan, pc.snapshot_target(ids, im_start), "cold")
        meter.snap("prefill")
        res["followup_sha256"] = ids_sha256(ids2)
        res["followup_new_tokens"] = len(ids2) - common_prefix(ids, ids2)
        res["followup"] = request(eng, cache, key, ids2, plan, pc.snapshot_target(ids2, im_start), "followup")
        meter.snap("followup")
        res["prefix_cache"] = cache.stats()
    elif a.child == "once":
        cache = TimedCache(max_tokens=max_tokens)
        res["cold"] = request(eng, cache, key, ids, plan, pc.snapshot_target(ids, im_start), "once")
        meter.snap("prefill")
    elif a.child == "scope16k":
        runs = []
        for pass_size in (4096, 1024):
            for scope in (8192, 0):
                clear_pool(eng)
                cache = TimedCache(max_tokens=max_tokens)
                p = PrefillPlan.from_envelope(eng["env"], prefill_pass=pass_size, read_scope=scope)
                k = pc.CheckpointKey(key.model_identity, key.optimizations_json, pass_size, False)
                r = request(eng, cache, k, ids, p, None, f"pass{pass_size}-scope{scope}")
                r.update({"prefill_pass": pass_size, "read_scope": scope})
                runs.append(r)
                meter.snap(f"prefill-{pass_size}-{scope}")
        res["runs"] = runs
    else:
        raise SystemExit(f"unknown child {a.child}")
    meter.stop()
    res.update(meter.result())
    res["ok"] = all(r.get("finite", True) for r in [res.get("cold", {}), res.get("followup", {})] + res.get("runs", []))
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: res[k] for k in ("child", "n", "ok") if k in res}), flush=True)
    return 0 if res["ok"] else 1


def _slots_per_chunk():
    from slotstream_spark.config import modes
    return modes.SLOTS_PER_CHUNK


# ------------------------------------------------------------------------------------------------ run
def run(a):
    from slotstream_spark.engine.tokenizer import Tokenizer
    cap = Path(os.environ.get("CAP", "."))
    prompts = load_prompts()
    tok = Tokenizer.from_checkpoint(os.environ["SLOTSTREAM_MODEL"])
    meta = {}
    for n in SIZES:
        ids, body = build_ids(tok, prompts, n)
        meta[str(n)] = {"tokens": len(ids), "sha256": ids_sha256(ids)}
        if n == 8192:
            f = followup_ids(tok, body, prompts)
            meta["followup"] = {"tokens": len(f), "sha256": ids_sha256(f), "new_tokens": len(f) - common_prefix(ids, f)}
    jobs = [("cold8k", 8192, f"cold8k-{i}") for i in range(a.samples)]
    jobs += [("once", 2048, "once2k"), ("scope16k", 16384, "scope16k"), ("once", 32768, "once32k")]
    res = {"mode": a.mode, "samples": a.samples, "prompts": meta, "text_sha256": prompts["text_sha256"],
           "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "children": {}, "failures": []}
    me = str(Path(__file__).resolve())
    for kind, n, name in jobs:
        out = cap / f"ttft-child-{name}.json"
        out.unlink(missing_ok=True)
        t = time.time()
        rc = subprocess.run([sys.executable, me, "--child", kind, "--n", str(n), "--mode", a.mode, "--out", str(out)],
                            cwd=str(ROOT)).returncode
        entry = json.loads(out.read_text()) if out.exists() else {}
        entry["returncode"] = rc
        entry["wall_s"] = round(time.time() - t, 1)
        if rc != 0 or not entry.get("ok"):
            res["failures"].append(name)
        elif entry.get("prompt_sha256") != meta[str(n)]["sha256"]:
            res["failures"].append(f"{name}: prompt sha256 differs from the run process")
        res["children"][name] = entry
        print(f"{name}: rc {rc} {entry['wall_s']} s", flush=True)
        time.sleep(5)
    res["ok"] = not res["failures"]
    dest = a.out or str(cap / "result.json")
    Path(dest).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(f"wrote {dest}; failures: {res['failures'] or 'none'}", flush=True)
    return 0 if res["ok"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-prompts", action="store_true")
    ap.add_argument("--mode", choices=("D5",), default="D5")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--out")
    ap.add_argument("--child", choices=("cold8k", "once", "scope16k"))
    ap.add_argument("--n", type=int, default=8192)
    a = ap.parse_args(argv)
    if a.build_prompts:
        return build_prompts(a.out or PROMPTS)
    sys.path.insert(0, str(ROOT))
    if a.child:
        if not a.out:
            raise SystemExit("--child needs --out")
        return child_main(a)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
