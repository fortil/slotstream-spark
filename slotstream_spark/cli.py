"""slotstream-spark command line: `run` (ids in, ids out), `doctor`; `bench decode` and `serve` (the HTTP server of step 84, 127.0.0.1 only).

  python -m slotstream_spark.cli run --mode D5 --pool-epl 48 --golden-prompt dev0 --max-tokens 16 --greedy
        --check-first-token
  python -m slotstream_spark.cli run --prompt "Why is the sky blue?" --max-tokens 64 --greedy
  python -m slotstream_spark.cli bench decode --mode D5 --prompts bench/lb/prompts.json --prompt-keys short,2k
        --tokens 256 --warmup 1 --profile --out FILE
  python -m slotstream_spark.cli doctor
  python -m slotstream_spark.cli serve --mode D5 --port 18435
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
import tomllib
import urllib.request
from pathlib import Path

from slotstream_spark.guard import paused

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = Path(__file__).resolve().parent / "config" / "numeric_contract.toml"
LAYERS = 48
RUNNING_URL = os.environ.get("SLOTSTREAM_RUNNING_URL", "")   # empty: no model router to watch


def _env_dir(var):
    v = os.environ.get(var)
    if not v:
        raise SystemExit(f"{var} is not set (run through tools/spark/py.sh)")
    return Path(v)


def fixture_kl_max():
    """[fixture] kl_max_nats of the numeric contract; None when it is not calibrated (-1.0)."""
    with open(CONTRACT, "rb") as f:
        v = float(tomllib.load(f)["fixture"]["kl_max_nats"])
    return None if v < 0 else v


def kl_nats(p_logits, q_logits):
    """KL(softmax(p) || softmax(q)) in nats, float64."""
    import numpy as np
    p = np.asarray(p_logits, dtype=np.float64)
    q = np.asarray(q_logits, dtype=np.float64)
    lp = p - p.max()
    lp -= math.log(np.exp(lp).sum())
    lq = q - q.max()
    lq -= math.log(np.exp(lq).sum())
    return float((np.exp(lp) * (lp - lq)).sum())


def _load_golden(pid):
    import importlib.util
    path = ROOT / "tools" / "make_goldens.py"
    if not path.is_file():
        raise SystemExit(f"--golden-prompt needs tools/make_goldens.py (a development checkout); not found at {path}")
    spec = importlib.util.spec_from_file_location("make_goldens", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_golden(pid)


def _ids_from_args(a):
    if a.golden_prompt:
        g = _load_golden(a.golden_prompt)
        return [int(t) for t in g["tokens"]], g
    if a.ids_json:
        d = json.loads(Path(a.ids_json).read_text())
        return [int(t) for t in (d["ids"] if isinstance(d, dict) else d)], None
    if getattr(a, "prompt", None):
        tok = _tokenizer()
        return tok.encode(tok.render_user_turn(a.prompt)), None
    raise SystemExit("run needs --golden-prompt ID, --ids-json FILE or --prompt TEXT")


def _tokenizer():
    from slotstream_spark.engine.tokenizer import Tokenizer
    return Tokenizer.from_checkpoint(_env_dir("SLOTSTREAM_MODEL"))


def _slots(a):
    from slotstream_spark.config import modes
    if a.pool_slots:
        return int(a.pool_slots)
    if a.pool_epl:
        return int(a.pool_epl) * LAYERS
    m = modes.get(a.mode)
    cap = modes.dynamic_cap_mib(m, modes.meminfo_mib("MemAvailable"))
    if os.environ.get("SLOTSTREAM_FOOTPRINT_MIB"):
        cap = min(cap, int(os.environ["SLOTSTREAM_FOOTPRINT_MIB"]))
    return planned_chunks(a.mode, cap) * modes.SLOTS_PER_CHUNK


def planned_chunks(mode, cap_mib):
    """Pool chunks from the ledger with the measured runtime (planner.plan), floored at the governor floor and
    capped by SLOTSTREAM_MAX_CHUNKS (drills only)."""
    from slotstream_spark.config import modes
    from slotstream_spark.memory import planner
    chunks = max(planner.plan(mode, int(cap_mib) << 20).chunks, modes.FLOOR_CHUNKS)
    if os.environ.get("SLOTSTREAM_MAX_CHUNKS"):
        chunks = min(chunks, int(os.environ["SLOTSTREAM_MAX_CHUNKS"]))
    return chunks


def served_ctx(plan):
    """Engine context for serve: the admission window rounded up to 64, so admit() and Model agree."""
    return -(-plan.window // 64) * 64


def build_engine(a, slots, max_ctx):
    """Load the trunk, open the rows and the expert pool, build the Model."""
    import slotio
    import torch
    from slotstream_spark.model.config import TextConfig
    from slotstream_spark.model.forward import Model
    from slotstream_spark.model.weights import Embedder, NgramSource, accessor, load_trunk
    from slotstream_spark.runtime.pool import ExpertPool

    model_dir, packed = _env_dir("SLOTSTREAM_MODEL"), _env_dir("SLOTSTREAM_PACKED")
    cfg = TextConfig.from_checkpoint(model_dir)
    trunk = packed / "trunk.safetensors"
    t0 = time.perf_counter()
    weights = load_trunk(trunk, device="cuda")
    t_trunk = time.perf_counter() - t0
    reader = slotio.Reader()
    embed = Embedder(reader, trunk, device="cuda")
    ngram = NgramSource(reader, packed, weights, cfg["eos_token_id"], device="cuda")
    pool = ExpertPool(packed, slots, reader=reader)
    model = Model(cfg, accessor(weights), embed, pool, ngram, device="cuda", gdn=a.gdn, max_ctx=max_ctx)
    torch.cuda.synchronize()
    return model, pool, cfg, {"trunk_load_s": round(t_trunk, 2), "trunk_tensors": len(weights)}


def cmd_run(a):
    kl_limit = None
    if a.check_first_token:
        kl_limit = fixture_kl_max()
        if kl_limit is None:
            print("contract not calibrated: [fixture] kl_max_nats is -1.0", file=sys.stderr)
            return 2
    ids, golden = _ids_from_args(a)
    if a.check_first_token and golden is None:
        raise SystemExit("--check-first-token needs --golden-prompt")
    from slotstream_spark.engine.generate import generate
    slots = _slots(a)
    max_ctx = max(256, -(-(len(ids) + a.max_tokens) // 64) * 64)
    model, pool, cfg, info = build_engine(a, slots, max_ctx)
    res = generate(model, ids, max_tokens=a.max_tokens, greedy=True if a.greedy else False,
                   eos=cfg["eos_token_id"])
    out = {"mode": a.mode, "slots": slots, "gdn": a.gdn, "prompt_tokens": len(ids), "tokens": res["tokens"],
           "n_tokens": len(res["tokens"]), "nan_logits": res["nan"],
           "prefill_s": round(res["timings"]["prefill_s"], 3),
           "decode_s": [round(x, 3) for x in res["timings"]["decode_s"]], **info,
           "pool": {k: v for k, v in pool.stats().items() if isinstance(v, (int, float))}}
    ok = not res["nan"] and (len(res["tokens"]) == a.max_tokens or res["tokens"][-1:] == [cfg["eos_token_id"]])
    if a.check_first_token:
        gl = golden["logits"][-1]
        kl = kl_nats(gl, res["first_logits"])
        match = int(gl.argmax()) == int(res["first_logits"].argmax())
        out["first_token"] = {"kl_nats": kl, "limit": kl_limit, "argmax_match": match,
                              "golden_argmax": int(gl.argmax()), "engine_argmax": int(res["first_logits"].argmax())}
        print(f"first position: KL {kl:.5f} nats (limit {kl_limit:.5f}), argmax match: {match}")
        ok = ok and kl <= kl_limit
    print(f"tokens ({len(res['tokens'])}): {res['tokens']}  nan={res['nan']}")
    if getattr(a, "prompt", None):
        out["text"] = _tokenizer().decode(res["tokens"])
        print("text:", out["text"])
    out["ok"] = bool(ok)
    dest = a.out or (os.path.join(os.environ["CAP"], "result.json") if os.environ.get("CAP") else None)
    if dest:
        Path(dest).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return 0 if ok else 1


def _decode_pass(model, g, sampler, params, ids, ntok, eos, clock):
    """Prefill `ids` (eager), then up to `ntok` graph decode steps; per-token phases go to `clock`."""
    import numpy as np
    g.detach()
    model.st.reset_inplace()
    t0 = time.perf_counter()
    logits = model.prefill(ids).cpu().numpy()
    prefill_s = time.perf_counter() - t0
    ph = clock.phase if clock is not None else None
    out, seen, stamps = [], set(), []
    for i in range(ntok):
        if clock is not None:
            clock.begin()
            with ph("sample"):
                tok = sampler.next(logits, params, seen)
        else:
            tok = sampler.next(logits, params, seen)
        out.append(tok)
        seen.add(tok)
        stamps.append(time.perf_counter())
        if tok == eos or i == ntok - 1:
            if clock is not None:
                clock.cancel()
            break
        logits = g.step([tok])[0].numpy()
        if not np.isfinite(logits).all():
            raise RuntimeError("non-finite logits")
        if clock is not None:
            clock.end()
    return out, prefill_s, stamps


def cmd_bench_decode(a):
    import numpy as np
    import torch
    from slotstream_spark.bench.prompts import build_prompt
    from slotstream_spark.engine.sampler import SampleParams, Sampler
    from slotstream_spark.runtime.graphs import GraphSet
    from slotstream_spark.runtime.profile import PHASES, PhaseClock

    tok = _tokenizer()
    specs = json.loads(Path(a.prompts).read_text())
    keys = [k for k in a.prompt_keys.split(",") if k]
    ids_by_key = {}
    for k in keys:
        if k not in specs:
            raise SystemExit(f"prompt key {k!r} is not in {a.prompts}")
        ids_by_key[k] = tok.encode(tok.render_user_turn(build_prompt(specs[k])))
    max_ctx = -(-(max(len(v) for v in ids_by_key.values()) + a.tokens + 16) // 64) * 64
    slots = _slots(a)
    model, pool, cfg, info = build_engine(a, slots, max_ctx)
    model.st.ensure_act(torch.bfloat16)
    eos = cfg["eos_token_id"]
    g = GraphSet(model, 1)
    model.nvtx = bool(a.profile)
    params = SampleParams(temperature=0.0)
    params.max_tokens = a.tokens
    res = {"mode": a.mode, "slots": slots, "gdn": a.gdn, "tokens_requested": a.tokens, "warmup": a.warmup,
           "profile": bool(a.profile), "prompts": {}, "info": info,
           "versions": {"torch": torch.__version__, "cuda": torch.version.cuda}}
    tau_ns = []
    for k in keys:
        ids = ids_by_key[k]
        for _ in range(a.warmup):                       # untimed passes: JIT, expert load, allocator
            _decode_pass(model, g, Sampler(0), params, ids, min(32, a.tokens), eos, None)
        clock = PhaseClock(nvtx=True) if a.profile else None
        g.clock = clock
        out, prefill_s, stamps = _decode_pass(model, g, Sampler(0), params, ids, a.tokens, eos, clock)
        g.clock = None
        n = len(out)
        entry = {"prompt_tokens": len(ids), "new_tokens": n, "prefill_s": round(prefill_s, 3),
                 "tok_s": None if n < 2 else round((n - 1) / (stamps[-1] - stamps[0]), 3), "tokens": out}
        if clock is not None:
            tot = {p: sum(t["phases"][p] for t in clock.tokens) for p in PHASES}
            wall = sum(t["wall_ns"] for t in clock.tokens)
            entry.update({"tokens_profiled": len(clock.tokens), "wall_ns": wall, "phases_ns": tot,
                          "other_ns": sum(t["other_ns"] for t in clock.tokens),
                          "read_wall_ns": clock.read_wall_ns, "hits": clock.hits, "misses": clock.misses,
                          "hit_rate": None if clock.hits + clock.misses == 0
                          else round(clock.hits / (clock.hits + clock.misses), 4),
                          "per_token_wall_ns": [t["wall_ns"] for t in clock.tokens],
                          "nvtx": clock.nvtx})
            tau_ns += clock.tau_ns
        res["prompts"][k] = entry
        print(f"{k}: {len(ids)} prompt tokens, prefill {prefill_s:.2f} s, {n} new tokens, tok/s {entry['tok_s']}"
              + (f", h {entry['hit_rate']}" if clock is not None else ""), flush=True)
    if tau_ns:
        s = sorted(tau_ns)
        res["tau"] = {"p50_us": round(s[len(s) // 2] / 1000, 2), "p99_us": round(s[int(len(s) * 0.99)] / 1000, 2),
                      "steps": len(s), "definition": "ensure + launch of all-hit layer steps (as M0' / step 54)"}
    res["pool"] = {kk: v for kk, v in pool.stats().items() if isinstance(v, (int, float))}
    res["ok"] = True
    dest = a.out or (os.path.join(os.environ["CAP"], "result.json") if os.environ.get("CAP") else None)
    if dest:
        Path(dest).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    return 0


def _running():
    if not RUNNING_URL:
        return "off"
    try:
        with urllib.request.urlopen(RUNNING_URL, timeout=3) as r:
            return json.loads(r.read().decode())
    except Exception as e:      # production service absent or slow: report, never fail the doctor on it
        return f"unreachable ({type(e).__name__})"


def cmd_doctor(a):
    from slotstream_spark import config
    from slotstream_spark.config import modes
    from slotstream_spark.memory import planner
    problems = []
    cfg = config.load()
    errs = config.validate(cfg)
    print(f"platform.toml: {'ok' if not errs else 'INVALID'}")
    for e in errs:
        print("  " + e)
        problems.append(e)
    print(f"  pool_memory={cfg['memory']['pool_memory']} ngram.row_bytes={cfg['ngram']['row_bytes']} "
          f"kernels.plan={cfg['kernels']['plan']}")
    avail, free = modes.meminfo_mib("MemAvailable"), modes.meminfo_mib("MemFree")
    print(f"MemAvailable {avail} MiB, MemFree {free} MiB")
    for name in ("D1", "D2", "D5"):
        m = modes.get(name)
        cap = modes.dynamic_cap_mib(m, avail)
        ch = modes.chunks_for(m, cap)
        try:
            pl = planner.plan(name, cap << 20).chunks
        except ValueError as e:
            pl = f"n/a ({e})"
        flag = "" if ch >= modes.FLOOR_CHUNKS else "  (below the floor)"
        print(f"  {name}: cap {cap} MiB, design chunks {max(ch, 0)}, planned chunks {pl} "
              f"(ceiling {m.ceiling_chunks}, floor {modes.FLOOR_CHUNKS}){flag}")
    print(f"/running: {_running()}")
    model_dir = os.environ.get("SLOTSTREAM_MODEL")
    packed = os.environ.get("SLOTSTREAM_PACKED")
    if not model_dir or not packed:
        print("SLOTSTREAM_MODEL / SLOTSTREAM_PACKED not set")
        return 1
    files = [Path(model_dir) / "config.json", Path(model_dir) / "model.safetensors.index.json"]
    files += [Path(packed) / n for n in ("trunk.safetensors", "experts.bin", "experts.manifest.json",
                                         "ngram.bin", "ngram.manifest.json")]
    seal = {}
    try:
        import slotio
        seal = {"direct": slotio.File.open_direct}
    except ImportError:
        print("slotio not importable: seals not checked")
    for f in files:
        if not f.exists():
            print(f"  MISSING {f.name}")
            problems.append(f"missing {f.name}")
            continue
        s = ""
        if seal and f.suffix in (".bin", ".safetensors"):
            s = f", seal {'ok' if seal['direct'](str(f)).check_seal() == 0 else 'CHANGED'}"
        print(f"  {f.name}: {f.stat().st_size} B{s}")
    try:
        from slotstream_spark.model.weights import check_ngram_row_bytes
        print(f"ngram row_bytes: {check_ngram_row_bytes(packed)} (manifest = platform.toml)")
    except Exception as e:
        print(f"ngram row_bytes: {e}")
        problems.append(str(e))
    return 1 if problems else 0


def take_model_lock():
    """SLOTSTREAM_TAKE_LOCK=1: flock $R/.model.lock without blocking; exit 75 when another model process holds it."""
    import fcntl
    path = _env_dir("SLOTSTREAM_R") / ".model.lock"
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"model lock {path} is held", file=sys.stderr)
        raise SystemExit(75)
    return fd


class ServeEngine:
    """What the minimal server and the signal thread see of the loaded engine."""

    def __init__(self, model, pool, cfg, plan, exit_event, generator=None):
        self.model, self.pool, self.cfg, self.plan, self._exit = model, pool, cfg, plan, exit_event
        self.generator = generator

    @property
    def chunks(self):
        return self.pool.pool.chunks

    def resize_chunks(self, n):
        from slotstream_spark.config import modes
        self.pool.resize_slots(int(n) * modes.SLOTS_PER_CHUNK)

    def drop_prefix(self):
        if self.generator is not None:
            self.generator.drop_prefix()

    def request_exit(self, reason):
        print(f"exit requested: {reason}", file=sys.stderr, flush=True)
        self._exit.set()

    def generate(self, ids, max_tokens, boundary):
        from slotstream_spark.engine.generate import generate
        return generate(self.model, ids, max_tokens=max_tokens, greedy=True, eos=self.cfg["eos_token_id"],
                        on_token=lambda _t: boundary.check())


class FootprintFeed:
    """The 20 ms slotio.Sampler of the engine process, drained to `<dest>` (JSONL: header with the count-once flags,
    then one sample per line) and exposed to the governor through `read()` (latest Sample, flags).

    Only the pool shows in two counters (VmRSS and NVML, unified memory: the H1 finding), so it is the one allocation
    registered; the reader scratch is anonymous memory already inside VmRSS and the embedding/n-gram landing buffers
    are plain RSS, so they have no byte accessor and need no subtraction."""

    POOL_FLAGS = {"host_alloc": (True, False), "vmm_host_numa": (True, True)}

    def __init__(self, pool, dest=None, period_ms=20, drain_ms=200):
        import threading
        import slotio
        from slotstream_spark.memory import footprint as fp
        self._fp = fp
        mem_type = getattr(pool, "mem_type", "vmm_host_numa")
        self.flags = {"pool": self.POOL_FLAGS[mem_type]}
        self.sampler = slotio.Sampler(period_ms=period_ms)
        self.sampler.register_pool("pool", pool.pool)
        self.latest, self.drain_s, self._stop = None, drain_ms / 1000, threading.Event()
        self._out = None
        if dest:
            self._out = open(dest, "a")
            self._out.write(json.dumps({"header": True, "flags": {k: list(v) for k, v in self.flags.items()},
                                        "period_ms": period_ms, "nvml_ok": self.sampler.nvml_ok,
                                        "mem_type": mem_type}, sort_keys=True) + "\n")
            self._out.flush()
        self.sampler.start()
        self._th = threading.Thread(target=self._run, daemon=True, name="footprint-drain")
        self._th.start()

    def _drain(self):
        rows = self.sampler.drain()
        if rows:
            d = rows[-1]
            self.latest = self._fp.Sample(str(d["t_ns"]), d["vmrss"], d["vmhwm"], d["nvml"], 0, dict(d["unified"]))
            if self._out:
                self._out.write("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
                self._out.flush()

    def _run(self):
        while not self._stop.wait(self.drain_s):
            self._drain()

    def read(self):
        return self.latest, self.flags

    def stop(self):
        self._stop.set()
        self._th.join(2)
        self.sampler.stop()
        self._drain()
        if self._out:
            self._out.close()


def nvme_throttle_c():
    """The Arbiter threshold: the device's temp1_max minus the margin, in C. Unreadable is an error, never a default."""
    from slotstream_spark.host import eligibility
    lim, src = eligibility.nvme_limit_mc()
    if lim is None:
        raise SystemExit(f"nvme hwmon limit unreadable ({eligibility.hwmon_input() or 'no nvme hwmon found'}): refusing to run without a throttle point")
    return lim / 1000.0


def start_arbiter(reader, throttle_c=None):
    """slotio.Arbiter on the engine's reader (it applies set_caps every 250 ms); -> (arbiter, throttle_c)."""
    import slotio
    from slotstream_spark.host import eligibility
    t = nvme_throttle_c() if throttle_c is None else throttle_c
    arb = slotio.Arbiter(reader, device=os.environ.get("SLOTSTREAM_NVME_DEVICE") or "nvme0n1", hwmon=eligibility.hwmon_input(), throttle_c=t)
    arb.start(reader)
    return arb, t


def pool_prefill_mode():
    from slotstream_spark import config
    v = os.environ.get("SLOTSTREAM_POOL_PREFILL")
    if v is None:
        return config.get("pool.prefill")
    if v not in ("off", "table"):
        raise ValueError(f"SLOTSTREAM_POOL_PREFILL must be off|table, got {v!r}")
    return v


class ServeFront:
    """ASGI wrapper of the server app: the two internal endpoints the drills and the front read (`/health`,
    `/internal/memory`) and the activity clock behind the idle exit. Everything else goes to the app untouched.
    The QUIET paths are read-only status calls: a poller that lists the models every second must not keep the engine
    alive, so they do not move the activity clock."""

    QUIET = ("/health", "/api/version", "/metrics", "/internal/memory", "/v1/models")

    def __init__(self, app, engine, ledger, cap_fn, idle_s, clock=time.monotonic):
        self.app, self.engine, self.ledger, self.cap_fn, self.idle_s, self.clock = app, engine, ledger, cap_fn, idle_s, clock
        self.ready, self.active, self.last_activity = False, 0, clock()

    def idle_expired(self):
        return self.active == 0 and self.clock() - self.last_activity >= self.idle_s

    async def _json(self, send, status, body):
        raw = json.dumps(body).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode())]})
        await send({"type": "http.response.body", "body": raw})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope["path"]
        if path == "/health":
            body = {"ok": self.ready, "busy": self.active > 0, "chunks": self.engine.chunks}
            return await self._json(send, 200 if self.ready else 503, body)
        if path == "/internal/memory":
            plan = self.engine.plan
            return await self._json(send, 200, {
                "chunks": self.engine.chunks, "plan_chunks": plan.chunks, "window": plan.window,
                "expected_peak": self.ledger.expected_peak_bytes(self.engine.chunks), "cap_bytes": self.cap_fn(),
                "terms": self.ledger.terms()})
        if path in self.QUIET:
            return await self.app(scope, receive, send)
        self.active += 1
        try:
            await self.app(scope, receive, send)
        finally:
            self.active -= 1
            self.last_activity = self.clock()


def serve_idle_exit_s():
    return float(os.environ.get("SLOTSTREAM_IDLE_EXIT_S", "1800"))


def cmd_serve(a):
    """Startup order (rule 8): seal check, measure, plan, reserve, trunk with DONTNEED, pool pre-fill, sweep driver,
    capture graphs, serve. Listens on 127.0.0.1 only."""
    import threading
    from slotstream_spark.config import modes
    from slotstream_spark.memory import governor as G
    from slotstream_spark.memory import planner, signals
    from slotstream_spark.memory.ledger import Ledger
    signals.write_oom_score_adj()
    if os.environ.get("SLOTSTREAM_TAKE_LOCK") == "1":
        take_model_lock()
    mode = a.mode or os.environ.get("SLOTSTREAM_MODE", "D5")
    packed = _env_dir("SLOTSTREAM_PACKED")
    import slotio
    for n in ("experts.bin", "ngram.bin", "trunk.safetensors"):                       # 1 seal check
        if slotio.File.open_direct(str(packed / n)).check_seal() != 0:
            print(f"seal changed: {n}", file=sys.stderr)
            return 1
    m = modes.get(mode)                                                               # 2 measure
    cap = modes.dynamic_cap_mib(m, modes.meminfo_mib("MemAvailable"))
    if os.environ.get("SLOTSTREAM_FOOTPRINT_MIB"):
        cap = min(cap, int(os.environ["SLOTSTREAM_FOOTPRINT_MIB"]))
    gcfg = G.load_config()
    run = signals.fetch_running(RUNNING_URL, 1.0) if RUNNING_URL else ()
    ok, why = G.may_start(time.localtime().tm_hour * 60 + time.localtime().tm_min, run, cap, G.floor_mib(
        G.GovInputs(mode, 0, (), 0, 0, 0, 0), gcfg), gcfg, no_windows=paused.no_windows())
    if not ok:
        print(f"may_start refused: {why}", file=sys.stderr)
        return 75
    chunks = planned_chunks(mode, cap)                                                # 3 plan
    plan = planner.plan(mode, cap << 20)
    ledger = Ledger(mode)                                                             # 4 reserve
    a.mode, a.gdn, a.pool_slots, a.pool_epl = mode, "fused", chunks * modes.SLOTS_PER_CHUNK, None
    model, pool, cfg, info = build_engine(a, a.pool_slots, served_ctx(plan))             # 5 trunk (DONTNEED inside)
    max_ctx = served_ctx(plan)
    from slotstream_spark.engine import prefill_schedule as ps
    from slotstream_spark.engine import prefix_cache as pc
    from slotstream_spark.engine.prefill import PrefillDriver, PrefillPlan
    from slotstream_spark.model.moe_sweep import SweepMoE
    from slotstream_spark.runtime.graphs import GraphSet
    import torch
    model.st.ensure_act(torch.bfloat16)
    if pool_prefill_mode() == "table":                                                # 5b pool pre-fill (no GraphSet yet)
        from slotstream_spark.engine import pool_prefill
        st_ = pool_prefill.prefill_pool(model.seg, pool, pool_prefill.load_table())
        print(f"pool prefill: {st_['records']} records in {st_['seconds']:.1f} s, per_layer {st_['per_layer']}, "
              f"exhausted {st_['exhausted']}", file=sys.stderr, flush=True)
    env = ps.envelope()                                                               # 5c sweep driver
    moe = SweepMoE(model.seg, pool, max_rows_per_launch=int(env["max_rows_per_launch"]))
    driver = PrefillDriver(model, moe, env)
    graphs = GraphSet(model, 1)                                                       # 6 capture graphs
    tokenizer = _tokenizer()
    model_name = Path(os.environ["SLOTSTREAM_MODEL"]).name
    key = pc.CheckpointKey(model_name, json.dumps({"gdn": "fused", "sweep": True}), 4096, False)
    cache = pc.PrefixCache(max_tokens=ledger.prefix // pc.BYTES_PER_TOKEN)
    from slotstream_spark.server import app as server_app
    from slotstream_spark.server import engine_adapter, http_limits, template
    generator = engine_adapter.ModelGenerator(
        model, driver, graphs, cache, key, PrefillPlan.from_envelope(env, prefill_pass=4096, read_scope=8192),
        cfg["eos_token_id"], max_ctx, tokenizer.encode("<|im_start|>")[0])
    exit_event = threading.Event()
    cap_dir = Path(os.environ.get("CAP", "/tmp"))
    feed = FootprintFeed(pool, cap_dir / "footprint-serve.jsonl")                    # 20 ms sampler -> JSONL + governor
    arbiter, throttle_c = start_arbiter(pool.reader)                                  # reader caps from NVMe/thermal state
    print(f"arbiter throttle_c={throttle_c}", file=sys.stderr, flush=True)
    engine = ServeEngine(model, pool, cfg, plan, exit_event, generator)
    ctl = signals.Controller(engine)
    cap_fn = lambda: cap << 20
    lock = ROOT / "model.lock.json"
    digest = "sha256:" + hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else ""
    adapter = engine_adapter.EngineAdapter(generator, tokenizer, template, ledger, ctl, model_name=model_name, plan=plan,
                                           window=plan.window, max_ctx=max_ctx, cap_fn=cap_fn,
                                           chunks_fn=lambda: engine.chunks, digest=digest, eos=cfg["eos_token_id"],
                                           pool_stats_fn=pool.stats)
    sig = signals.Signals(mode=mode, cfg=gcfg, ledger=ledger, engine=engine,
                          get_running=lambda: signals.fetch_running(RUNNING_URL, 1.0) if RUNNING_URL else (),
                          read_mem=lambda: (modes.meminfo_mib("MemAvailable"), modes.meminfo_mib("MemFree")),
                          read_footprint=feed.read, controller=ctl, log=open(cap_dir / "governor.jsonl", "a"))
    sig.start()
    import uvicorn
    server_app.ignore_sigpipe()
    front = ServeFront(server_app.create_app(adapter), engine, ledger, cap_fn, serve_idle_exit_s())
    options = http_limits.uvicorn_options()
    options["host"] = http_limits.bind_host(None)                                     # 127.0.0.1, never configurable
    server = uvicorn.Server(uvicorn.Config(front, port=a.port, log_level="warning", **options))
    front.ready = True                                                                # 7 health
    def watch():
        while not exit_event.is_set() and not front.idle_expired():
            exit_event.wait(1.0)
        server.should_exit = True
    threading.Thread(target=watch, daemon=True).start()
    server.run()
    graphs.close()                                                                    # stops the token graph's launch thread, if one was built
    sig.stop()
    arbiter.stop()
    feed.stop()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="slotstream_spark.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--mode", choices=("D1", "D2", "D5"), default="D5")
    g = r.add_mutually_exclusive_group()
    g.add_argument("--pool-epl", type=int, help="pool experts per layer (slots = N * 48)")
    g.add_argument("--pool-slots", type=int)
    p = r.add_mutually_exclusive_group()
    p.add_argument("--golden-prompt")
    p.add_argument("--ids-json")
    p.add_argument("--prompt", help="text; rendered as one user turn, thinking off")
    r.add_argument("--max-tokens", type=int, default=16)
    r.add_argument("--greedy", action="store_true")
    r.add_argument("--gdn", choices=("fused", "ref"), default="fused")
    r.add_argument("--check-first-token", action="store_true")
    r.add_argument("--out")
    sub.add_parser("doctor")
    b = sub.add_parser("bench")
    bsub = b.add_subparsers(dest="what")
    bd = bsub.add_parser("decode")
    bd.add_argument("--mode", choices=("D1", "D2", "D5"), default="D5")
    bg = bd.add_mutually_exclusive_group()
    bg.add_argument("--pool-epl", type=int)
    bg.add_argument("--pool-slots", type=int)
    bd.add_argument("--prompts", required=True)
    bd.add_argument("--prompt-keys", default="short")
    bd.add_argument("--tokens", type=int, default=256)
    bd.add_argument("--warmup", type=int, default=1)
    bd.add_argument("--profile", action="store_true")
    bd.add_argument("--gdn", choices=("fused", "ref"), default="fused")
    bd.add_argument("--out")
    sv = sub.add_parser("serve")
    sv.add_argument("--mode", choices=("D1", "D2", "D5"), default=None)
    sv.add_argument("--port", type=int, default=18435)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "doctor":
        return cmd_doctor(a)
    if a.cmd == "bench" and a.what == "decode":
        return cmd_bench_decode(a)
    if a.cmd == "serve":
        return cmd_serve(a)
    print(f"{a.cmd}: not implemented yet", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
