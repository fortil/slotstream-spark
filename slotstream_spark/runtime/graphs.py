# SPDX-License-Identifier: MIT AND Apache-2.0
# Portions ported from helgard-orlm/qwen-flash-next-8gb@958d22e7ba0317750ccc38862b9667b1034a4dd1 qwen_delta_graph.py:17-19,54-55 (Apache-2.0, section 4); modified.
"""CUDA graphs per segment for the decode step, in 1-row and 3-row variants (plan section 3.3).

GraphSet(model, rows) captures G(0) ... G(48):
  G(0)   control words -> embedded rows, PLE, HC, mixer(0), router(0), signal(0)
  G(l)   MoE(l-1) on the ids of the previous graph, HC write, mixer(l), router(l), signal(l)        l = 1 .. 47
  G(48)  MoE(47), final mixer, lm_head -> fp32 logits in pinned memory, signal(48)
Between two graphs the host waits on the sequence word (Readback.wait: bounded spin, no stream synchronize), reads
the layer's ids from the pinned slot and runs ExpertPool.ensure; the next graph then reads the experts through the
device tables. Tensors that cross a graph boundary are static buffers (hyp, mixed, inj, ids, weights), everything
else lives in one shared graph pool. The position of the first row is a device scalar loaded from pinned memory at
the start of G(0); the QSA metadata is rebuilt from it on the device, so a graph is valid for every position.

Both variants use the decode kernels (recurrent GDN, select-and-attend QSA, our batch-invariant GEMV and the
rank-order MoE sum), so a 3-row step equals three 1-row steps bit for bit and a graph step equals the eager decode.

Generation of the pins of ensure(l) at step t is t * 49 + l; the sequence word is the Ensurer's progress pointer
while a GraphSet is attached (`attach`), and `detach` hands the eager path back a consistent generation counter.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from slotstream_spark.kernels import vendored
from slotstream_spark.kernels.moe_gemv_affine import moe_decode
from slotstream_spark.kernels.signal import signal
from slotstream_spark.model.forward import ngram_finish, ngram_start
from slotstream_spark.model.layers import lin
from slotstream_spark.runtime.readback import LAYERS, Readback


class _Ctx:
    """What Segments.seg_attn needs on the graph path: the device position, the n-gram embedding and `store_kv`."""

    def __init__(self, st_device, pos_dev, ple_emb):
        self.device, self.pos, self.ple_emb = st_device, pos_dev, ple_emb
        self._static = {}

    def _meta_inputs(self, cache, t):
        key = (id(cache), t)
        if key not in self._static:
            qsl_cpu = torch.tensor([0, t], dtype=torch.int32)
            self._static[key] = (qsl_cpu, qsl_cpu.to(self.device), torch.arange(t, dtype=torch.int64, device=self.device))
        return self._static[key]

    def metadata(self, cache, t):
        """vendored._metadata with the position taken from the device scalar (everything else is shape-only)."""
        m = vendored._ops()
        shims, qc = m["shims"], m["cache"]
        s = cache.s
        qsl_cpu, qsl, ar = self._meta_inputs(cache, t)
        seq = (self.pos + t).to(torch.int32)
        slots = self.pos + ar

        def cam(table):
            return shims.CommonAttentionMetadata(
                query_start_loc=qsl, query_start_loc_cpu=qsl_cpu, seq_lens=seq, num_reqs=1, num_actual_tokens=t,
                max_query_len=t, max_seq_len=cache.max_ctx, block_table_tensor=table, slot_mapping=slots)

        b = cache.buf
        ring = qc.build_qsa_metadata(cam(cache.ring_table), b["t2r"][0], b["lp"][0], b["vb"][0], b["sm"][0],
                                     storage_block_size=s.ring, compress_ratio=1, circular_buffer_size=s.ring)
        n_work = (t + (s.ratio - 1)) // s.ratio
        kw = cache.k_work[:n_work]
        comp = qc.build_qsa_metadata(cam(cache.comp_table), b["t2r"][1], b["lp"][1], b["vb"][1], b["sm"][1],
                                     storage_block_size=s.comp_page, compress_ratio=s.ratio, circular_buffer_size=0,
                                     k_work_metadata_buffer=kw, request_capacity=1)
        return dict(qsl=qsl, seq=seq, slots=slots, ring=ring, comp=comp, k_work=kw, t=t, start=None)

    def store_kv(self, cache, qkv, iq, ik, _pos, q_norm, k_norm, iq_norm, ik_norm):
        """vendored.qsa_store_kv on the device position."""
        m = vendored._ops()
        s = cache.s
        t = qkv.shape[0]
        meta = self.metadata(cache, t)
        positions = self.pos + self._meta_inputs(cache, t)[2]
        q_idx = torch.empty(t, s.index_heads, s.index_dim, dtype=torch.bfloat16, device=qkv.device)
        q, gate = m["prepare"].qsa_prepare(
            iq, ik, positions, cache.cos_sin, vendored.gemma_weight(iq_norm), vendored.gemma_weight(ik_norm), s.eps,
            q_idx, cache.ring, meta["ring"][3], cache.ring_table, meta["qsl"], meta["comp"][1], cache.comp,
            meta["comp"][3], meta["k_work"], compress_ratio=s.ratio, mrope_section=None, rope_pos_offset=None,
            main_qkv=qkv, main_q_norm_weight=vendored.gemma_weight(q_norm),
            main_k_norm_weight=vendored.gemma_weight(k_norm), main_eps=s.eps, main_kv_cache=cache.main_kv,
            main_slot_mapping=meta["slots"], main_k_scale=1.0, main_v_scale=1.0)
        cache.last = meta
        return q_idx, q, gate


class GraphSet:
    """Captured decode graphs of `rows` (1 or 3) rows bound to model.st (use model.st.reset_inplace() to start
    over without invalidating them) and to the model's expert pool."""

    def __init__(self, model, rows, readback=None, pool_handle=None, forecast=None, record=False, multi=False):
        if rows not in (1, 3):
            raise ValueError("rows must be 1 or 3")
        self.model, self.rows = model, rows
        self.seg, self.cfg = model.seg, model.cfg
        self.device = model.device
        # forecast: None (no tap, the graphs are the ones without it) or a ForecastTap (engine.lookahead); the tap of
        # layer l for layer l + 1 is captured into G(l) before the layer's signal and needs a Readback with buffers
        self.forecast = forecast
        self.lookahead = None              # engine.lookahead.Scheduler while speculative prefetch is on (step 74); needs the tap
        self.tap_probe = None              # debugging hook (eager runs): called with (layer, mixed, raw, used)
        # record: the 3-row graphs write the GDN/conv/PLE state after each row into a StepRecord (step 76, rollback of a
        # rejected draft); multi: G(48) also copies the pre-mixer stream [rows, hc*H] to `multi` (the MTP head's input)
        if record and rows != 3:
            raise ValueError("a step record needs the 3-row graph set")
        self.record_on, self.multi_on = bool(record), bool(multi)
        self.record = None
        self.last_multi = None
        self.rb = readback or Readback(model.vocab, forecast=forecast is not None)
        if forecast is not None and self.rb.f_ids is None:
            raise ValueError("a forecast tap needs a Readback built with forecast=True")
        self.probe = None                  # debugging hook called after each eagerly run segment
        self.clock = None                  # runtime.profile.PhaseClock while profiling (step 57)
        self.timings = None
        H, hc = self.cfg["hidden_size"], self.cfg["hc_count"]
        topk, R = self.cfg["num_experts_per_tok"], rows
        self.NL = model.NL
        dt = self._act_dtype()
        dev = self.device
        mk = lambda *s, dtype=dt: torch.zeros(*s, dtype=dtype, device=dev)
        self.hyp_in = mk(R, hc * H)
        self.hyp = mk(R, hc * H)
        self.mixed, self.inj = mk(R, H), mk(R, hc)
        self.rids = mk(R, topk, dtype=torch.int32)
        self.rw = mk(R, topk, dtype=torch.float32)
        w = self._ple_width()
        self.ple_emb = mk(R, w) if w else None
        # the prefetch overlaps G(0): only valid when no PLE layer is in segment 0 and the source can split read()
        self.ple_late = (self.ple_emb is not None and min(self.seg.ple_layers) >= 1
                         and hasattr(getattr(self.seg.ngram, "rows", None), "prefetch"))
        self.ctrl_dev = mk(2, dtype=torch.int64)
        self.h_buf = mk(R, topk, self.cfg["moe_intermediate_size"], dtype=torch.float32)
        self.o_buf = mk(R, topk, H, dtype=torch.float32)
        self.routed = mk(R, H, dtype=torch.float32)
        self.ctx = _Ctx(dev, self.ctrl_dev[:1], self.ple_emb)
        model.st.ensure_act(dt)
        self.multi = mk(R, hc * H) if self.multi_on else None
        if self.record_on:
            from slotstream_spark.kernels.gdn_step_record import StepRecord
            self.record = StepRecord(model.st, R)
            self.seg.record = self.record
        self.graphs = None
        self.pool_handle = pool_handle or torch.cuda.graph_pool_handle()
        self._build()

    # ------------------------------------------------------------------ construction
    def _act_dtype(self):
        return self.seg.embed([0]).dtype

    def _ple_width(self):
        if not self.seg.ple_layers or self.seg.ngram is None:
            return 0
        return int(self.seg.ngram(self.model.st.tokens, [0]).shape[-1])

    def _moe(self, L):
        seg, W = self.seg, self.seg.W
        m_t, p_t = seg._device_tables()
        moe_decode(self.mixed, self.rids, self.rw, L, m_t, p_t, self.h_buf, self.o_buf, self.routed)
        p = f"layers.{L}.mlp."
        sh = lin(F.silu(lin(self.mixed, W(p + "shared_expert.gate_proj.weight"))) *
                 lin(self.mixed, W(p + "shared_expert.up_proj.weight")), W(p + "shared_expert.down_proj.weight"))
        sh = torch.sigmoid(lin(self.mixed, W(p + "shared_expert_gate.weight"))) * sh
        h = self.routed.to(self.mixed.dtype) + sh
        return self.hyp + (h.unsqueeze(-2) * self.inj.unsqueeze(-1)).flatten(-2)

    def _attn_router(self, L, hyp):
        seg, st = self.seg, self.model.st
        hyp = seg.seg_attn(L, st, hyp, None, self.ctx)
        mixed, inj, rids, rw = seg.seg_router(L, hyp)
        self.hyp.copy_(hyp)
        self.mixed.copy_(mixed)
        self.inj.copy_(inj)
        self.rids.copy_(rids.to(torch.int32))
        self.rw.copy_(rw)
        if self.forecast is not None and L + 1 < self.NL:
            self.forecast.write(self, L, self.hyp)
        signal(self.rids, self.rw, self.rb.ids, self.rb.w, self.rb.seq, self.ctrl_dev, L)

    def _body(self, k):
        """Segment k of the step (a function with no arguments; every input is a static buffer)."""
        NL = self.NL
        if k == 0:
            def g0():
                self.ctrl_dev.copy_(self.rb.ctrl, non_blocking=True)
                self._attn_router(0, self.hyp_in)
            return g0
        if k < NL:
            def gl():
                self._attn_router(k, self._moe(k - 1))
            return gl

        def gf():
            hyp = self._moe(NL - 1)
            if self.multi is not None:
                self.multi.copy_(hyp)
            final = self.seg.seg_final(hyp)
            lg = lin(final, self.seg.W("lm_head.weight")).float()
            self.rb.logits[:self.rows].copy_(lg, non_blocking=True)
            signal(None, None, self.rb.ids, self.rb.w, self.rb.seq, self.ctrl_dev, NL)
        return gf

    def _rowwise(self, body):
        """Run a segment with the torch reductions applied one row at a time (Segments._rw)."""
        def run():
            self.seg.rowwise = True
            try:
                body()
            finally:
                self.seg.rowwise = False
        return run

    def _build(self):
        bodies = [self._rowwise(self._body(k)) for k in range(LAYERS)]
        self.bodies = bodies
        snap = self.snapshot()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(2):                              # Triton JIT, allocator, metadata inputs
                self._run(self._probe_tokens(), eager=True)
                self.restore(snap)
        torch.cuda.synchronize()
        self.graphs = []
        for k, body in enumerate(bodies):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool_handle, capture_error_mode="thread_local"):
                body()
            self.graphs.append(g)
        torch.cuda.synchronize()

    def _probe_tokens(self):
        return [1] * self.rows

    # ------------------------------------------------------------------ state snapshot (warm-up, tests)
    def snapshot(self):
        return snapshot_state(self.model.st)

    def restore(self, snap):
        restore_state(self.model.st, snap)

    # ------------------------------------------------------------------ attach / detach
    def attach(self):
        if self.rb.attached:
            return
        seg = self.seg
        self.rb.base = max(self.rb.base, ((int(seg.gen) + 1 + LAYERS - 1) // LAYERS) * LAYERS)
        # seq must stay below the first target of the step (base): when the eager passes left progress == base (gen + 1 a
        # multiple of 49), max(progress, base - 1) made wait(base + 0) return before G(0) had written layer 0's ids
        self.rb.set_seq(self.rb.base - 1)
        seg.pool.set_progress_ptr(self.rb.seq_addr)
        self.rb.attached = True

    def close(self):
        """Teardown: stops the token graph's launch thread (step 91) when one was built; idempotent."""
        tg = getattr(self, "token_graph", None)
        if tg is not None:
            tg.close()
            self.token_graph = None

    def detach(self):
        """Give the eager path back a progress word and a generation counter above every graph pin."""
        if not self.rb.attached:
            return
        torch.cuda.synchronize()
        seg = self.seg
        seg.gen = int(self.rb.base) + LAYERS
        seg.progress[0] = seg.gen
        seg.pool.set_progress_ptr(seg.progress.ctypes.data)
        self.rb.attached = False

    # ------------------------------------------------------------------ one decode step
    def _run(self, toks, eager=False, timings=None):
        seg, st, rb, R = self.seg, self.model.st, self.rb, self.rows
        ids = [int(t) for t in toks]
        if len(ids) != R:
            raise ValueError(f"this graph set takes {R} tokens per step")
        if st.pos + R > self.model.max_ctx:
            raise ValueError(f"context {st.pos + R} exceeds max_ctx {self.model.max_ctx}")
        if not self.rb.attached:
            self.attach()
        if self.record is not None and not eager:
            self.record.begin()
        clk = self.clock
        ph = clk.phase if clk is not None else _no_phase
        started = None
        with torch.no_grad():
            if self.ple_emb is not None:
                if self.ple_late:                                  # (2)-(3) hash and prefetch; (4) embed read below
                    with ph("ngram_hash"):
                        started = ngram_start(seg.ngram, st.tokens, ids)
                else:
                    with ph("ngram_hash"):
                        self.ple_emb.copy_(seg.ngram(st.tokens, ids).to(self.ple_emb.dtype))
            with ph("embed_read"):
                self.hyp_in.copy_(seg.embed(ids).repeat(1, seg.hc))
        base = self.rb.base
        self.rb.base += LAYERS
        rb.begin(st.pos, base)
        run = (lambda k: self.bodies[k]()) if eager else (lambda k: self.graphs[k].replay())
        pool = seg.pool
        with ph("launch"):
            run(0)                                                  # (5) launch G(0)
        if started is not None:                                     # (6) rows land while G(0) runs; G(1) reads them
            with ph("ngram_wait"), torch.no_grad():
                self.ple_emb.copy_(ngram_finish(seg.ngram, started).to(self.ple_emb.dtype))
        for L in range(self.NL):
            t0 = time.perf_counter_ns()
            rb.wait(base + L)
            t1 = time.perf_counter_ns()
            uniq = np.unique(rb.layer_ids(L, R * seg.topk)).astype(np.int32)
            pool.ensure(L, uniq, base + L)
            if self.lookahead is not None:
                f_ids, f_logits = rb.forecast(L, R) if L + 1 < self.NL else (None, None)
                self.lookahead.on_layer(L, f_ids, f_logits, self.NL)
            t2 = time.perf_counter_ns()
            run(L + 1)
            if eager and self.probe is not None:
                torch.cuda.synchronize()
                self.probe(L + 1, self)
            t3 = time.perf_counter_ns()
            if timings is not None:
                timings.append((t3 - t0, t1 - t0, t2 - t1, t3 - t2, int(pool.last_timing["misses"])))
            if clk is not None:
                clk.layer(t1 - t0, t2 - t1, t3 - t2, pool.last_timing)
        with ph("gpu_wait"):
            rb.wait(base + self.NL)
        st.pos += R
        st.tokens += ids
        for c in st.attn.values():
            c.pos = st.pos
        if self.multi is not None:
            self.last_multi = self.multi.clone()
        return rb.logits[:R].clone()

    def step(self, toks, timings=None):
        """Decode `rows` tokens: logits fp32 [rows, V] (CPU)."""
        return self._run(toks, eager=False, timings=timings)


class _NoPhase:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NO_PHASE = _NoPhase()


def _no_phase(name):
    return _NO_PHASE


def snapshot_state(st):
    """Copies of everything a decode step mutates (device tensors cloned, python scalars kept)."""
    snap = {"pos": st.pos, "tokens": list(st.tokens), "S": st.S.clone(),
            "conv": None if st.conv is None else st.conv.clone(),
            "ple_conv": None if st.ple_conv is None else st.ple_conv.clone(), "attn": {}}
    for L, c in st.attn.items():
        snap["attn"][L] = (c.main_kv.clone(), c.ring.clone(), c.comp.clone(), c.pos)
    return snap


def restore_state(st, snap):
    st.pos, st.tokens = snap["pos"], list(snap["tokens"])
    st.S.copy_(snap["S"])
    if snap["conv"] is not None:
        st.conv.copy_(snap["conv"])
        st.ple_conv.copy_(snap["ple_conv"])
    for L, (kv, ring, comp, pos) in snap["attn"].items():
        c = st.attn[L]
        c.main_kv.copy_(kv)
        c.ring.copy_(ring)
        c.comp.copy_(comp)
        c.pos = pos
        c.last = None
