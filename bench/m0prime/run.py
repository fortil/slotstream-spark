#!/usr/bin/env python3
"""M0' : four real layers (0-3) of the model on the Spark, with the design triggers of the plan.

  bench/m0prime/run.py [--out DIR] [--positions N]

Loads the trunk subset for layers 0-3 (GDN, GDN+PLE, GDN, QSA), fills the expert pool with all 2,048
experts of those layers by the safe path, and measures:
  - fill: t_memcpy per record (log2 histogram of the reader), IO GB/s, lanes, chunk free_ns p99
  - GEMV GB/s on every shape of the loaded weights plus the 2560x2560 4-bit gate shape, timed with
    bench/m0/gemv_bench.py::cold_graph_us (cold-L2 CUDA graph, user decision 2026-10-11)
  - teacher-forced decode over >= 256 positions (the 8 dev prompts, then dev0 again), fused GDN and reference
    GDN: per-layer rel_l2 (layers 0-3) and KL / top-1 of the truncated logits (final mixer + lm_head)
  - per-layer step time (CUDA graphs of the attention+router segment and the MoE segment, L2 flushed
    before each replay), the projection inputs t_GDN, t_QSA, t_embed, t_lm_head
  - tau = wait_seq return -> ensure (all hits) -> cudaGraphLaunch return, p50/p99 over 1,024 layer steps
  - runtime footprint = (VmRSS + NVML) - pool - weights - state
Writes $CAP/result.json (or --out DIR/result.json). The judgement is tools/gates/records.py m0prime.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "bench" / "m0"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

LAYER_SET = (0, 1, 2, 3)
GDN_TIMED = (0, 2)          # layer 1 carries the PLE (host-side n-gram rows): not graph-capturable
QSA_LAYER = 3
TAU_STEPS = 1024
TAU_WARMUP = 16
FLUSH_BYTES = 128 * 1024 * 1024
MIB = 1 << 20


def pct(vals, q):
    s = sorted(vals)
    if not s:
        return None
    i = min(len(s) - 1, max(0, int(math.ceil(q / 100.0 * len(s))) - 1))
    return s[i]


def vm_kb(field):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    return 0


def nvml_used(pid):
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
            if p.pid == pid and p.usedGpuMemory is not None:
                return int(p.usedGpuMemory), "pynvml"
    except Exception:
        pass
    return 0, "unavailable"


def tensor_bytes(obj):
    import torch
    from slotstream_spark.kernels.gemv_affine import AffineWeight, DenseWeight
    if isinstance(obj, AffineWeight):
        return sum(t.numel() * t.element_size() for t in (obj.q, obj.s, obj.b))
    if isinstance(obj, DenseWeight):
        return obj.w.numel() * obj.w.element_size()
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    return 0


def state_bytes(st):
    import torch
    n = 0
    for t in (st.S, st.conv, st.ple_conv):
        if isinstance(t, torch.Tensor):
            n += t.numel() * t.element_size()
    for c in st.attn.values():
        for name in ("main_kv", "ring", "comp"):
            t = getattr(c, name, None)
            if isinstance(t, torch.Tensor):
                n += t.numel() * t.element_size()
    return n


class Collect:
    """Sink for Model._chunk: layer outputs (hyper width) and, when asked, the layer inputs of the last call."""

    def __init__(self, layers=LAYER_SET, keep_inputs=False):
        self.layers = layers
        self.out = {L: [] for L in layers}
        self.keep_inputs = keep_inputs
        self.inputs = {}
        self._prev = None

    def h_in(self, hyp):
        self._prev = hyp
        if self.keep_inputs:
            self.inputs = {0: hyp.clone()}

    def layer(self, L, hyp, ids, w):
        if L in self.out:
            self.out[L].append(hyp.float().cpu().numpy())
        if self.keep_inputs:
            self.inputs[L + 1] = hyp.clone()

    def ngram(self, gids):
        pass


def golden_tokens(pid):
    import make_goldens as mg
    d = mg.find_prompt_dir(mg.DEFAULT_DATA, pid)
    return [int(t) for t in json.load(open(d / "tokens.json"))["ids"]]


def stream_sequences(min_positions):
    seqs = [golden_tokens(f"dev{i}") for i in range(8)]
    total = sum(len(s) for s in seqs)
    while total < min_positions:
        seqs.append(golden_tokens("dev0"))
        total += len(seqs[-1])
    return seqs, total


def run_stream(model, seqs, mode):
    """Teacher-forced decode, one token per step, every sequence from a fresh state. -> (layer outs, logits)."""
    import numpy as np
    sink = Collect()
    logits = []
    for toks in seqs:
        model.reset()
        model.set_gdn(mode)
        for t in toks:
            final = model._chunk([t], sink)
            logits.append(model._logits(final)[0].cpu().numpy())
    model.set_gdn("fused")
    layers = {L: np.concatenate(v, 0) for L, v in sink.out.items()}
    return layers, np.stack(logits, 0)


def gemv_bytes(n, k, bits, gs):
    if bits == 16:
        return n * k * 2 + 2 * k + 4 * n
    return n * k * bits / 8 + 2 * (n * k // gs) * 2 + 2 * k + 4 * n


def gemv_shapes(torch, weights):
    from slotstream_spark.kernels.gemv_affine import AffineWeight, DenseWeight
    shapes = {}
    for name, w in weights.items():
        if isinstance(w, AffineWeight):
            shapes.setdefault((w.n, w.k, w.bits, w.gs), w)
        elif isinstance(w, torch.Tensor) and w.is_cuda and w.dtype == torch.bfloat16 and w.dim() == 2:
            shapes.setdefault((w.shape[0], w.shape[1], 16, 0), DenseWeight(w))
    return shapes


def gate_weight(torch, n=2560, k=2560):
    from slotstream_spark.kernels.gemv_affine import AffineWeight
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randint(-2**31, 2**31 - 1, (n, k // 8), device="cuda", dtype=torch.int32, generator=g)
    s = (torch.rand(n, k // 64, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    b = (torch.rand(n, k // 64, device="cuda", generator=g) * 0.02 - 0.01).to(torch.bfloat16)
    return AffineWeight(q, s, b, 4, 64, n, k)


def measure_gemv(torch, gemv_bench, weights):
    from slotstream_spark.kernels.gemv_affine import gemv
    shapes = gemv_shapes(torch, weights)
    shapes[(2560, 2560, 4, 64)] = gate_weight(torch)      # the gate shape of step 23, synthetic weights
    out = {}
    for (n, k, bits, gs), w in sorted(shapes.items()):
        x = torch.randn(1, k, device="cuda").to(torch.bfloat16)
        us = gemv_bench.cold_graph_us(torch, lambda: gemv(x, w))
        nb = gemv_bytes(n, k, bits, gs)
        gbs = nb / (us * 1e-6) / 1e9 if us > 0 else None
        out[f"{n}x{k}_b{bits}"] = {"n": n, "k": k, "bits": bits, "cold_graph_us": round(us, 2),
                                   "bytes": int(nb), "gbs_cold_graph": None if gbs is None else round(gbs, 2)}
        print(f"gemv {n}x{k} bits {bits}: {us:.2f} us, {gbs:.1f} GB/s (cold-L2 graph)", flush=True)
    return out


# ---------------------------------------------------------------- layer graphs
class LayerGraphs:
    """CUDA graphs of one layer's decode step: A = attention + router, B = routed + shared experts + residual.
    Static buffers carry the data between them; B needs the experts resident (they all are)."""

    def __init__(self, torch, model, L, hyp_in, patch_meta=None):
        import torch.nn.functional as F
        from slotstream_spark.kernels.moe_gemv_affine import moe_decode
        from slotstream_spark.model.layers import lin
        seg, st, cfg = model.seg, model.st, model.cfg
        self.L, self.torch = L, torch
        H, topk, inter = cfg["hidden_size"], cfg["num_experts_per_tok"], cfg["moe_intermediate_size"]
        self.hyp_in = hyp_in.clone()
        self.hyp = torch.empty_like(hyp_in)
        self.mixed = torch.empty(1, H, dtype=hyp_in.dtype, device="cuda")
        self.inj = torch.empty(1, cfg["hc_count"], dtype=hyp_in.dtype, device="cuda")
        self.rids = torch.empty(1, topk, dtype=torch.int32, device="cuda")
        self.rw = torch.empty(1, topk, dtype=torch.float32, device="cuda")
        self.out = torch.empty_like(hyp_in)
        h_buf = torch.empty(1, topk, inter, dtype=torch.float32, device="cuda")
        o_buf = torch.empty(1, topk, H, dtype=torch.float32, device="cuda")
        routed = torch.zeros(1, H, dtype=torch.float32, device="cuda")
        m_t, p_t = seg._device_tables()
        W = seg.W
        p = f"layers.{L}.mlp."

        def part_a():
            hyp = seg.seg_attn(L, st, self.hyp_in, [0])
            mixed, inj, rids, rw = seg.seg_router(L, hyp)
            self.hyp.copy_(hyp)
            self.mixed.copy_(mixed)
            self.inj.copy_(inj)
            self.rids.copy_(rids.to(torch.int32))
            self.rw.copy_(rw)

        def part_b():
            routed.zero_()
            moe_decode(self.mixed, self.rids, self.rw, L, m_t, p_t, h_buf, o_buf, routed)
            sh = lin(F.silu(lin(self.mixed, W(p + "shared_expert.gate_proj.weight"))) *
                     lin(self.mixed, W(p + "shared_expert.up_proj.weight")), W(p + "shared_expert.down_proj.weight"))
            sh = torch.sigmoid(lin(self.mixed, W(p + "shared_expert_gate.weight"))) * sh
            h = routed.to(self.mixed.dtype) + sh
            self.out.copy_(self.hyp + (h.unsqueeze(-2) * self.inj.unsqueeze(-1)).flatten(-2))

        self.part_a, self.part_b = part_a, part_b
        self.mode = "graph"
        for _ in range(3):                      # eager warm-up on a side stream: Triton JIT, allocations
            s = torch.cuda.Stream()
            with torch.cuda.stream(s):
                part_a()
                part_b()
            torch.cuda.synchronize()
        self.ga, self.gb = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        self.note = ""
        try:
            if patch_meta is not None:
                patch_meta(True)
            with torch.cuda.graph(self.ga):
                part_a()
            with torch.cuda.graph(self.gb):
                part_b()
        except Exception as e:                  # recorded, never silent
            self.mode = "eager"
            self.note = f"{type(e).__name__}: {str(e)[:200]}"
        finally:
            if patch_meta is not None:
                patch_meta(False)
        torch.cuda.synchronize()

    def run(self):
        if self.mode == "graph":
            self.ga.replay()
            self.gb.replay()
        else:
            self.part_a()
            self.part_b()


def cold_step_us(torch, lg, flush, reps=80):
    ts = []
    for _ in range(reps):
        flush.fill_(1)
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        lg.run()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000)
    return statistics.median(ts), pct(ts, 99)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--positions", type=int, default=256)
    a = ap.parse_args(argv)
    t_start = time.time()

    import numpy as np
    import slotio
    import torch
    import gemv_bench
    import calibrate_contract as cc
    from slotstream_spark.kernels import vendored
    from slotstream_spark.kernels.gemv_affine import gemv
    from slotstream_spark.model.config import TextConfig
    from slotstream_spark.model.forward import Model
    from slotstream_spark.model.layers import lin
    from slotstream_spark.model.weights import Embedder, NgramSource, accessor, load_trunk
    from slotstream_spark.runtime.pool import ExpertPool, wait_seq

    model_dir, packed = Path(os.environ["SLOTSTREAM_MODEL"]), Path(os.environ["SLOTSTREAM_PACKED"])
    cfg = TextConfig.from_checkpoint(model_dir)
    types = cfg.layer_types[:4]
    if types != ["linear_attention"] * 3 + ["full_attention"]:
        print(f"DECISION: layers 0-3 are {types}, M0' expects GDN, GDN, GDN, QSA")
        return 1
    res = {"versions": {"torch": torch.__version__, "cuda": torch.version.cuda}, "layer_types": types}

    # ---------------------------------------------------------------- load
    t0 = time.perf_counter()
    weights = load_trunk(packed / "trunk.safetensors", layers=LAYER_SET, device="cuda")
    res["trunk_load_s"] = round(time.perf_counter() - t0, 2)
    reader = slotio.Reader()
    embed = Embedder(reader, packed / "trunk.safetensors", device="cuda")
    ngram = NgramSource(reader, packed, weights, cfg["eos_token_id"], device="cuda")
    torch.cuda.synchronize()
    rss_before_pool = vm_kb("VmRSS")
    nvml_before_pool, _ = nvml_used(os.getpid())
    man = json.load(open(packed / "experts.manifest.json"))
    rec_bytes = int(man["record_bytes"])
    slots = 6 * 388
    pool = ExpertPool(packed, slots, reader=reader)
    model = Model(cfg, accessor(weights), embed, pool, ngram, device="cuda", gdn="fused", max_ctx=320)
    model.NL = 4
    seg = model.seg
    torch.cuda.synchronize()
    res["pool"] = {"slots": slots, "chunks": 6, "mem_type": pool.mem_type, "record_bytes": rec_bytes}

    # ---------------------------------------------------------------- fill 2,048 experts
    r0 = reader.stats()
    reads, walls, total_ns = 0, [], []
    t_fill = time.perf_counter()
    for L in LAYER_SET:
        for a0 in range(0, 512, 128):
            ids = np.arange(a0, a0 + 128, dtype=np.int32)
            _, gen = seg._ensure(L, ids)
            tm = pool.last_timing
            seg._done(gen)
            reads += int(tm["misses"])
            walls.append(int(tm["read_wall_ns"]))
            total_ns.append(int(tm["total_ns"]))
    fill_s = time.perf_counter() - t_fill
    r1 = reader.stats()
    fill_bytes = reads * rec_bytes
    d_copy, d_jobs = r1["copy_ns"] - r0["copy_ns"], r1["jobs"] - r0["jobs"]
    hist = [b - a_ for a_, b in zip(r0["memcpy_hist"], r1["memcpy_hist"])]
    n_copies = sum(hist)

    def hist_q(q):
        want, acc = q / 100.0 * n_copies, 0
        for b, c in enumerate(hist):
            acc += c
            if acc >= want and c:
                return b
        return None
    b50, b99 = hist_q(50), hist_q(99)
    res["fill"] = {
        "records": reads, "bytes": fill_bytes, "expected_bytes": 2048 * rec_bytes, "seconds": round(fill_s, 2),
        "io_gbs": round(fill_bytes / (sum(walls) * 1e-9) / 1e9, 3) if walls else None,
        "wall_gbs": round(fill_bytes / fill_s / 1e9, 3), "reader_jobs": d_jobs,
        "t_memcpy_mean_ms": round(d_copy / max(d_jobs, 1) / 1e6, 4),
        "t_memcpy_p50_ms": None if b50 is None else round(2 ** (b50 + 0.5) / 1e6, 4),
        "t_memcpy_p99_ms": None if b99 is None else round(2 ** (b99 + 1) / 1e6, 4),
        "t_memcpy_hist_note": "log2 ns buckets: p50 = bucket geometric midpoint, p99 = bucket upper bound",
        "lanes": len(reader.lane_cpus()), "lane_cpus": list(reader.lane_cpus())}
    print("fill:", json.dumps(res["fill"]), flush=True)
    if reads != 2048:
        print(f"DECISION: the fill read {reads} records, expected 2,048")
        return 1

    # ---------------------------------------------------------------- GEMV on every shape
    res["gemv"] = measure_gemv(torch, gemv_bench, weights)

    # ---------------------------------------------------------------- A/B teacher-forced decode
    seqs, positions = stream_sequences(a.positions)
    res["positions"] = positions
    fused, lg_f = run_stream(model, seqs, "fused")
    ref, lg_r = run_stream(model, seqs, "ref")
    per = {}
    for L in LAYER_SET:
        per[str(L)] = cc.rel_l2(fused[L], ref[L])
    kls, agree = [], []
    for a0 in range(0, positions, 16):
        kl, ag = cc.logits_metrics(lg_r[a0:a0 + 16], lg_f[a0:a0 + 16])
        kls.append(kl)
        agree.append(ag)
    kl = np.concatenate(kls)
    res["gdn_ab"] = {"rel_l2_by_layer": per, "kl_mean": float(kl.mean()), "kl_p99": float(np.percentile(kl, 99)),
                     "kl_max": float(kl.max()), "top1": float(np.concatenate(agree).mean()), "positions": positions,
                     "finite": bool(np.isfinite(lg_f).all() and np.isfinite(lg_r).all()),
                     "direction": "reference = engine with set_gdn('ref'), other = fused"}
    print("gdn_ab:", json.dumps(res["gdn_ab"]), flush=True)
    del lg_f, lg_r, fused, ref

    # ---------------------------------------------------------------- layer step graphs
    model.reset()
    model.set_gdn("fused")
    toks = golden_tokens("dev0")
    sink = Collect(keep_inputs=True)
    for t in toks:
        model._chunk([t], sink)
    nxt = toks[-1]
    # inputs of the next decode step: run it eagerly once to capture each layer's input hyper stream
    sink.out = {L: [] for L in LAYER_SET}
    model._chunk([nxt], sink)
    st = model.st
    st.pos -= 1                                  # the timing replays rewrite position pos, not pos+1
    st.tokens.pop()
    inputs = {L: sink.inputs[L] for L in LAYER_SET}
    flush = torch.empty(FLUSH_BYTES // 4, device="cuda", dtype=torch.int32)

    saved_meta = {}
    orig_meta = vendored._metadata

    def patch_meta(on):
        if on:
            vendored._metadata = lambda cache, t, start: saved_meta["m"]
        else:
            vendored._metadata = orig_meta

    graphs, step = {}, {}
    for L in (*GDN_TIMED, QSA_LAYER):
        if L == QSA_LAYER:
            # one real metadata build at this position, reused while capturing (its host tensors cannot be captured)
            def spy(cache, t, start, _o=orig_meta):
                m = _o(cache, t, start)
                saved_meta["m"] = m
                return m
            vendored._metadata = spy
            lg = LayerGraphs(torch, model, L, inputs[L], patch_meta=patch_meta)
            vendored._metadata = orig_meta
        else:
            lg = LayerGraphs(torch, model, L, inputs[L])
        graphs[L] = lg
        us, p99 = cold_step_us(torch, lg, flush)
        step[str(L)] = {"mode": lg.mode, "note": lg.note, "cold_us_median": round(us, 2), "cold_us_p99": round(p99, 2)}
        print(f"layer {L} step ({lg.mode}): {us:.1f} us (p99 {p99:.1f})", flush=True)
    t_gdn = statistics.mean(step[str(L)]["cold_us_median"] for L in GDN_TIMED)
    t_qsa = step[str(QSA_LAYER)]["cold_us_median"]

    # t_embed: the engine's eager embed of one token (host read + dequant), t_lm_head: cold-L2 graph of the gemv
    ts = []
    for _ in range(200):
        t1 = time.perf_counter()
        embed([nxt])
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t1) * 1e6)
    t_embed = statistics.median(ts[20:])
    xh = torch.randn(1, cfg["hidden_size"], device="cuda").to(torch.bfloat16)
    w_head = seg.W("lm_head.weight")
    t_head = gemv_bench.cold_graph_us(torch, lambda: lin(xh, w_head))
    t_comp_ms = (36 * t_gdn + 12 * t_qsa + t_embed + t_head) / 1000.0
    res["timings"] = {"layer_steps": step, "t_gdn_us": round(t_gdn, 2), "t_qsa_us": round(t_qsa, 2),
                      "t_embed_us": round(t_embed, 2), "t_lm_head_us": round(t_head, 2),
                      "t_comp_48_ms": round(t_comp_ms, 3),
                      "basis": "full layer step (attention+router+routed+shared experts, experts resident), "
                               "L2 flushed before each replay; t_gdn = mean of layers 0 and 2 (layer 1 has the host-side PLE)"}
    print("timings:", json.dumps({k: v for k, v in res["timings"].items() if k != "layer_steps"}), flush=True)

    # ---------------------------------------------------------------- tau
    flag = np.zeros(1, dtype=np.uint64)
    flag_addr = flag.ctypes.data
    samples, k = [], 0
    ids_host = {L: graphs[L].rids.cpu().numpy().reshape(-1).astype(np.int32) for L in GDN_TIMED}
    for i in range(TAU_WARMUP + TAU_STEPS):
        L = GDN_TIMED[i % len(GDN_TIMED)]
        lg = graphs[L]
        lg.ga.replay()                    # attention + router of this layer (not part of tau)
        torch.cuda.synchronize()
        k += 1
        flag[0] = k                       # the "ids ready" word
        t0n = time.perf_counter_ns()
        wait_seq(flag_addr, k, 1_000_000_000)
        t1n = time.perf_counter_ns()
        seg.gen += 1
        slots_ = pool.ensure(L, ids_host[L], seg.gen)
        t2n = time.perf_counter_ns()
        lg.gb.replay()
        t3n = time.perf_counter_ns()
        torch.cuda.synchronize()
        seg._done(seg.gen)
        if i >= TAU_WARMUP:
            samples.append((t3n - t0n, t1n - t0n, t2n - t1n, t3n - t2n))
    tau = [s[0] / 1000 for s in samples]
    res["tau"] = {"p50_us": round(pct(tau, 50), 3), "p99_us": round(pct(tau, 99), 3),
                  "wait_p50_us": round(pct([s[1] / 1000 for s in samples], 50), 3),
                  "ensure_p50_us": round(pct([s[2] / 1000 for s in samples], 50), 3),
                  "launch_p50_us": round(pct([s[3] / 1000 for s in samples], 50), 3),
                  "steps": len(samples), "warmup": TAU_WARMUP,
                  "segments": "wait_seq + ensure (all hits, 10 experts) + gb.replay (routed+shared experts graph)"}
    print("tau:", json.dumps(res["tau"]), flush=True)

    # ---------------------------------------------------------------- footprint
    nv, src = nvml_used(os.getpid())
    rss = vm_kb("VmRSS")
    pool_b = pool.owned_bytes()
    pool_in_rss = (rss - rss_before_pool) >= 0.9 * pool_b
    pool_in_nvml = (nv - nvml_before_pool) >= 0.9 * pool_b
    weights_b = sum(tensor_bytes(w) for w in weights.values())
    state_b = state_bytes(model.st)
    foot = rss + nv - pool_b * (int(pool_in_rss) + int(pool_in_nvml)) - weights_b - state_b
    res["runtime"] = {"vmrss": rss, "vmhwm": vm_kb("VmHWM"), "nvml": nv, "nvml_source": src, "pool_bytes": pool_b,
                      "pool_in_rss": bool(pool_in_rss), "pool_in_nvml": bool(pool_in_nvml),
                      "rss_before_pool": rss_before_pool, "nvml_before_pool": nvml_before_pool,
                      "weights_bytes": weights_b, "state_bytes": state_b, "pinned_term": 0,
                      "footprint_bytes": int(foot),
                      "formula": "(VmRSS + NVML + pinned) - pool - weights - state; pinned buffers are inside VmRSS; the "
                                 "pool is subtracted once for each counter (VmRSS, NVML) whose growth when the pool was "
                                 "created is >= 90% of its owned bytes (on this machine the pool shows in both)"}
    print("runtime:", json.dumps(res["runtime"]), flush=True)

    # ---------------------------------------------------------------- chunk free p99 (grow / shrink cycles)
    before = len(pool.pool.stats()["free_ns"])
    for _ in range(20):
        pool.resize_slots(388)
        pool.resize_slots(slots)
    free = list(pool.pool.stats()["free_ns"])[before:]
    res["chunk_free"] = {"samples": len(free), "p50_ns": pct(free, 50), "p99_ns": pct(free, 99),
                         "max_ns": max(free) if free else None}
    print("chunk_free:", json.dumps(res["chunk_free"]), flush=True)

    res["elapsed_s"] = round(time.time() - t_start, 1)
    out = a.out or os.environ.get("CAP")
    if out:
        os.makedirs(out, exist_ok=True)
        Path(out, "result.json").write_text(json.dumps(res, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
