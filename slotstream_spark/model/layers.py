"""Per-layer segments of the eager forward (plan section 3.3) and the per-sequence state.

Segments, in call order for layer L:
  seg_begin(ids)               embed rows -> hyper-width stream [T, hc*H]
  seg_attn(L, st, hyp, ...)    PLE (layer 1), attention HC mix, GDN or QSA, HC residual -> hyp
  seg_router(L, hyp)           MLP HC mix, router logits, softmax, stable top-10 -> (mixed, inj, ids, weights)
  seg_moe(L, mixed, ids, w)    ids to host, ExpertPool.ensure per row group, routed + shared expert -> [T, H]
  seg_final(hyp)               final HC mixer -> [T, H] (then lm_head in forward.py)

Two backends share the block code. `cuda`: weights are AffineWeight (gemv) or bf16 tensors (gemv dense),
attention is the vendored QSA with a paged cache, GDN is the vendored fla kernel (`fused`) or the
reference_torch scan (`ref`), the experts are read by moe_decode from pool slots. `ref` (CPU tests, tiny
models): weights are plain tensors, attention is RefModel.attention, the experts come from the pool object's
`slot_weights(slot)`. Every op of the `ref` path is the same reference_torch function RefModel calls, in the
same order, so a model on this backend equals RefModel bit for bit (both are fp32 CPU paths).
"""
import math
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from slotstream_spark.kernels import gemm_affine
from slotstream_spark.kernels.gemv_affine import AffineWeight, DenseWeight, gemv
from slotstream_spark.model import reference_torch as R

DN_CHUNK = R.RefModel.DN_CHUNK
MAX_GROUP_EXPERTS = 32


_DENSE_GEMM = False   # set only by PrefillDriver._scope through prefill_gemm(); decode, eager prefill and graphs never set it
GEMM_STATS = {"gemm_calls": 0, "gemv_fallbacks": 0}


@contextmanager
def prefill_gemm():
    """Inside the context, `lin` runs the dense calls on the tensor-core GEMM (a row's bits depend on that row and the
    weight only). Restored on exit, also on exceptions."""
    global _DENSE_GEMM
    prev = _DENSE_GEMM
    _DENSE_GEMM = True
    try:
        yield
    finally:
        _DENSE_GEMM = prev


def _gemm_weight(x, w):
    """The GEMM's weight object for this call, or None when the call stays on the GEMV/F.linear dispatch."""
    if x.dtype != torch.bfloat16 or not x.is_cuda or x.dim() != 2:
        return None
    if isinstance(w, AffineWeight):
        return w if w.bits == 4 else None
    if isinstance(w, DenseWeight):
        return w
    if isinstance(w, torch.Tensor) and w.is_cuda and w.dtype == torch.bfloat16 and w.dim() == 2:
        return DenseWeight(w)
    return None


def lin(x, w):
    """x [T, K] @ w^T. AffineWeight and bf16 CUDA tensors go through the batch-invariant gemv (fp32 accumulate,
    result in the dtype of x); other tensors through F.linear. Inside `prefill_gemm()` the 4-bit and bf16 cases
    take gemm_affine.gemm instead."""
    if _DENSE_GEMM:
        gw = _gemm_weight(x, w)
        if gw is not None:
            GEMM_STATS["gemm_calls"] += 1
            return gemm_affine.gemm(x.contiguous(), gw).to(x.dtype)
        GEMM_STATS["gemv_fallbacks"] += 1
    if isinstance(w, (AffineWeight, DenseWeight)):
        return gemv(x.contiguous(), w).to(x.dtype)
    if w.is_cuda and w.dtype == torch.bfloat16 and w.dim() == 2 and x.dtype == torch.bfloat16:
        return gemv(x.contiguous(), DenseWeight(w)).to(x.dtype)
    return F.linear(x, w)


def conv_weight(w):
    """[C, 4, 1] (MLX), [C, 1, 4] or [C, 4] -> [C, 4]."""
    return w.reshape(w.shape[0], -1)


class SeqState:
    """Everything one sequence carries between steps: position, tokens, GDN state fp32 [n_lin, nv, dk, dv],
    GDN conv state [n_lin, ck, C], PLE conv state [hist, hc*H], and the attention caches (QSACache per full
    layer on cuda: paged bf16 KV 256+256 per kv head, 128-wide indexer keys; dict of growing tensors on ref).
    The n-gram context is the last two entries of `tokens`."""

    def __init__(self, cfg, device, max_ctx=4096, cuda=False):
        self.cfg, self.device, self.max_ctx, self.cuda = cfg, device, max_ctx, cuda
        types = cfg.layer_types
        self.lin_idx = {L: j for j, L in enumerate(i for i, t in enumerate(types) if t == "linear_attention")}
        self.full_layers = [i for i, t in enumerate(types) if t == "full_attention"]
        nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
        dk, dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
        self.conv_ch = 2 * nk * dk + nv * dv
        self.ck = cfg["linear_conv_kernel_dim"] - 1
        self.ple_hist = 3 * (cfg["ple_conv_kernel_size"] - 1) if "ple_conv_kernel_size" in cfg.raw else 0
        self.ple_ch = cfg["hc_count"] * cfg["hidden_size"]
        self.S = torch.zeros(len(self.lin_idx), nv, dk, dv, dtype=torch.float32, device=device)
        self.conv = None
        self.ple_conv = None
        self.pos = 0
        self.tokens = []
        if cuda:
            from slotstream_spark.kernels.vendored import QSACache
            self.attn = {L: QSACache(max_ctx, device) for L in self.full_layers}
        else:
            self.attn = {L: dict(k=None, v=None, ik=None, bk=None) for L in self.full_layers}

    def reset_inplace(self):
        """Back to an empty sequence keeping every tensor address (CUDA graphs are bound to them)."""
        self.S.zero_()
        if self.conv is not None:
            self.conv.zero_()
            self.ple_conv.zero_()
        for c in self.attn.values():
            if isinstance(c, dict):
                for k in c:
                    c[k] = None
            else:
                c.main_kv.zero_()
                c.ring.zero_()
                c.comp.zero_()
                c.pos = 0
                c.last = None
        self.pos = 0
        self.tokens = []

    def ensure_act(self, dtype):
        """Conv states live in the activation dtype; created zero on first use (zero == no history)."""
        if self.conv is None:
            self.conv = torch.zeros(len(self.lin_idx), self.ck, self.conv_ch, dtype=dtype, device=self.device)
            self.ple_conv = torch.zeros(self.ple_hist, self.ple_ch, dtype=dtype, device=self.device)


def row_groups(ids_host, max_rows, max_experts=MAX_GROUP_EXPERTS):
    """Split rows [0, T) into consecutive groups of <= max_rows rows whose union of experts is <= max_experts."""
    out, start, cur = [], 0, set()
    for t in range(len(ids_host)):
        new = cur | set(int(e) for e in ids_host[t])
        if t > start and (len(new) > max_experts or t - start >= max_rows):
            out.append((start, t))
            start, new = t, set(int(e) for e in ids_host[t])
        cur = new
    out.append((start, len(ids_host)))
    return out


class Segments:
    def __init__(self, cfg, W, embed, pool, ngram=None, device="cuda", cuda=True, gdn="fused", max_rows=32):
        self.cfg, self.W, self.embed, self.pool, self.ngram = cfg, W, embed, pool, ngram
        self.device, self.cuda, self.max_rows = device, cuda, max_rows
        self.gdn = gdn
        self.H, self.hc, self.eps = cfg["hidden_size"], cfg["hc_count"], cfg["rms_norm_eps"]
        self.topk = cfg["num_experts_per_tok"]
        self.types = cfg.layer_types
        self.ple_layers = set(cfg.ple_layers)
        self.rm = None if cuda else R.RefModel(cfg, W, None, None, None, device)
        if cuda:
            # vendored._ops() loads its modules only while the cache dict is empty, and _fla() (GDN) fills it
            # with another key first; loading the QSA/HC modules here, before any GDN call, avoids that.
            from slotstream_spark.kernels import vendored
            vendored._ops()
        self.gen = 0
        self.progress = np.zeros(1, dtype=np.uint64)
        self._tables = None
        if hasattr(pool, "set_progress_ptr"):
            pool.set_progress_ptr(self.progress.ctypes.data)
        self.rowwise = False        # graph path: torch reductions run one row at a time (see _rw)
        self.record = None          # kernels.gdn_step_record.StepRecord: the graph rows of that width write per-position state
        self.last_routing = {}      # layer -> (ids int32 [T,10], w fp32 [T,10]) of the last call
        self.last_ngram_ids = None

    def _rw(self, fn, *ts):
        """fn(*ts) as is; on the graph path with several rows, fn per row and the rows concatenated. A torch reduction
        can pick another summation order for [3, n] than for [1, n], so a 3-row step would not equal three 1-row
        steps bit for bit; per row it is the same call on the same shapes as the 1-row step."""
        n = ts[0].shape[0]
        if not self.rowwise or n == 1:
            return fn(*ts)
        return torch.cat([fn(*[x[i:i + 1] for x in ts]) for i in range(n)], 0)

    # ------------------------------------------------------------ begin / final
    def seg_begin(self, ids):
        x = self.embed(ids)
        return x.repeat(1, self.hc)

    def hc_mix(self, prefix, hyp, inject=True):
        W, H, hc = self.W, self.H, self.hc
        hn = self._rw(lambda h: R.rmsnorm(h, W(prefix + ".hc_norm.weight"), self.eps, group=H), hyp)
        m = F.silu(lin(hn, W(prefix + ".input_mix_weight_down.weight")) / hc)
        m = torch.sigmoid(lin(m, W(prefix + ".input_mix_weight_up.weight"))).unflatten(-1, (hc, H))
        mixed = self._rw(lambda a, b: (a * b.unflatten(-1, (hc, H))).mean(dim=-2), m, hn)
        if not inject:
            return mixed, None
        return mixed, 2 * torch.sigmoid(lin(hn, W(prefix + ".block_inject_weight.weight")) / hc)

    def seg_final(self, hyp):
        return self.hc_mix("hyper_connection_mixer", hyp, inject=False)[0]

    # ------------------------------------------------------------ attention segment
    def seg_attn(self, L, st, hyp, ids, g=None):
        """`g` is the CUDA-graph context of runtime/graphs.py (device position, precomputed n-gram rows, device
        metadata); without it this is the eager path. The graph path always uses the decode kernels (recurrent GDN,
        select-and-attend QSA), also for the 3 rows of a verify step."""
        T = hyp.shape[0]
        if L in self.ple_layers and self.ngram is not None:
            hyp = hyp + (self.ple(L, st, hyp, ids) if g is None else self.ple_apply(L, st, hyp, g.ple_emb))
        h, inj = self.hc_mix(f"layers.{L}.attn_hyper_connection", hyp)
        if self.types[L] == "linear_attention":
            h = self.deltanet(L, st, h, decode_rows=g is not None)
        elif self.cuda:
            h = self.qsa(L, st, h, g)
        else:
            cos, sin = self.rm.rope(torch.arange(st.pos, st.pos + T))
            h = self.rm.attention(L, st.attn[L], h, cos, sin)
        return hyp + (h.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)

    def ple(self, L, st, hyp, ids):
        emb = self.ngram(st.tokens, ids).to(hyp.dtype)
        self.last_ngram_ids = getattr(self.ngram, "last_ids", None)
        return self.ple_apply(L, st, hyp, emb)

    def ple_apply(self, L, st, hyp, emb):
        W, H, hc = self.W, self.H, self.hc
        p = f"layers.{L}.ple."
        kn = self._rw(lambda e: R.rmsnorm(lin(e, W(p + "key_proj.weight")), W(p + "norm_key.weight"), self.eps,
                                          group=H), emb).unflatten(-1, (hc, H))
        val = lin(emb, W(p + "value_proj.weight"))
        qn = self._rw(lambda h: R.rmsnorm(h, W(p + "norm_query.weight"), self.eps, group=H), hyp).unflatten(-1, (hc, H))
        gate = self._rw(lambda a, b: (a * b).sum(-1, keepdim=True), kn, qn) / math.sqrt(H)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gv = torch.sigmoid(gate) * val.unsqueeze(-2)
        gvn = self._rw(lambda g_: R.rmsnorm(g_.flatten(-2), W(p + "norm_conv.weight"), self.eps, group=H), gv)
        gv = gv.flatten(-2)
        hist = st.ple_hist
        st.ensure_act(gvn.dtype)
        full = torch.cat([st.ple_conv, gvn], 0)
        st.ple_conv.copy_(full[-hist:])
        rec = self.record
        if rec is not None and gvn.shape[0] == rec.rows and self.rowwise:
            for i in range(rec.rows):                  # state after row i is the window full[i + 1 : i + 1 + hist]
                rec.ple[i].copy_(full[i + 1:i + 1 + hist])
        cw = conv_weight(W(p + "conv1d.weight"))
        if self.rowwise and gvn.shape[0] > 1:       # output i only needs the window full[i : i + hist + 1]
            conv = torch.cat([F.conv1d(full[i:i + hist + 1].T.unsqueeze(0), cw.unsqueeze(1), None,
                                       groups=full.shape[1], dilation=3) for i in range(gvn.shape[0])], 2)
        else:
            conv = F.conv1d(full.T.unsqueeze(0), cw.unsqueeze(1), None, groups=full.shape[1], dilation=3)
        return gv + F.silu(conv)[0].T

    def deltanet(self, L, st, x, decode_rows=False):
        T = x.shape[0]
        if T > DN_CHUNK and self.gdn == "ref":
            return torch.cat([self.deltanet(L, st, x[a:a + DN_CHUNK]) for a in range(0, T, DN_CHUNK)], 0)
        c, W = self.cfg, self.W
        nk, nv = c["linear_num_key_heads"], c["linear_num_value_heads"]
        dk, dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
        kd, vd = nk * dk, nv * dv
        p = f"layers.{L}.linear_attn."
        j = st.lin_idx[L]
        qkv = lin(x, W(p + "in_proj_qkv.weight"))
        z = lin(x, W(p + "in_proj_z.weight")).reshape(T, -1, dv)
        b = lin(x, W(p + "in_proj_b.weight"))
        a = lin(x, W(p + "in_proj_a.weight"))
        st.ensure_act(qkv.dtype)
        ck = st.ck
        full = torch.cat([st.conv[j], qkv], 0)
        st.conv[j].copy_(full[-ck:])
        rec = self.record if (decode_rows and self.record is not None and T == self.record.rows) else None
        if rec is not None:
            for i in range(T):                          # state after row i is the window full[i + 1 : i + 1 + ck]
                rec.conv[i, j].copy_(full[i + 1:i + 1 + ck])
        cw = conv_weight(W(p + "conv1d.weight"))
        if self.rowwise and T > 1:                  # output i only needs the window full[i : i + ck + 1]
            conv = torch.cat([F.conv1d(full[i:i + ck + 1].T.unsqueeze(0), cw.unsqueeze(1), None, padding=0,
                                       groups=full.shape[1]) for i in range(T)], 2)
        else:
            conv = F.conv1d(full.T.unsqueeze(0), cw.unsqueeze(1), None, padding=0, groups=full.shape[1])
        qkv = F.silu(conv)[0].T.to(x.dtype)
        q, k, v = torch.split(qkv, [kd, kd, vd], dim=-1)
        q = q.reshape(1, T, nk, dk)
        k = k.reshape(1, T, nk, dk)
        v = v.reshape(1, T, nv, dv)
        beta = b.sigmoid().unsqueeze(0)
        g = (-W(p + "A_log").float().exp() * F.softplus(a.float() + W(p + "dt_bias"))).unsqueeze(0)
        started = st.pos > 0 or decode_rows      # a captured graph must not depend on the position at capture
        S_in = st.S[j].unsqueeze(0) if started else None
        if self.gdn == "ref":
            r = nv // nk
            q, k = q.repeat_interleave(r, dim=2), k.repeat_interleave(r, dim=2)
            if started and T == 1:
                o, S = R.recurrent_gated_delta_rule(q, k, v, g, beta, S_in)
            else:
                o, S = R.chunk_gated_delta_rule(q, k, v, g, beta, initial_state=S_in)
        else:
            from slotstream_spark.kernels import vendored
            if rec is not None:
                from slotstream_spark.kernels.gdn_step_record import gdn_step_record
                o, S = gdn_step_record(q, k, v, g, beta.float(), S_in, rec.S[:, j])
            elif (started and T == 1) or decode_rows:
                o, S = vendored.gdn_decode(q, k, v, g, beta.float(), S_in)
            else:
                o, S = vendored.gdn_prefill(q, k, v, g, beta.float(), S_in)
        st.S[j].copy_(S[0])
        o = o.reshape(-1, dv).to(x.dtype)
        z = z.reshape(-1, dv)
        of = o.float()
        nw = W(p + "norm.weight")
        o = self._rw(lambda f: nw * (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype),
                     of.reshape(T, -1, dv)).reshape(-1, dv)
        o = (o * torch.sigmoid(z.float())).to(x.dtype).reshape(T, -1)
        return lin(o, W(p + "out_proj.weight"))

    def qsa(self, L, st, x, gctx=None):
        from slotstream_spark.kernels import vendored
        W = self.W
        p = f"layers.{L}.self_attn."
        cache = st.attn[L]
        qkv = torch.cat([lin(x, W(p + "q_proj.weight")), lin(x, W(p + "k_proj.weight")),
                         lin(x, W(p + "v_proj.weight"))], dim=-1).contiguous()
        qk = lin(x, W(p + "indexer.index_qk_proj.weight"))
        ni = self.cfg["indexer_n_heads"] * self.cfg["indexer_head_dim"]
        iq, ik = qk[:, :ni].contiguous(), qk[:, ni:].contiguous()
        store = vendored.qsa_store_kv if gctx is None else gctx.store_kv
        q_idx, q, gate = store(cache, qkv, iq, ik, st.pos if gctx is None else gctx.pos, W(p + "q_norm.weight"),
                               W(p + "k_norm.weight"), W(p + "indexer.q_layernorm.weight"),
                               W(p + "indexer.k_layernorm.weight"))
        out = vendored.qsa_decode(cache, q_idx, q, gate) if (x.shape[0] == 1 or gctx is not None) else \
            vendored.qsa_prefill(cache, q_idx, q, gate)
        return lin(out.reshape(x.shape[0], -1), W(p + "o_proj.weight"))

    # ------------------------------------------------------------ router segment
    def seg_router(self, L, hyp):
        W = self.W
        mixed, inj = self.hc_mix(f"layers.{L}.mlp_hyper_connection", hyp)
        logits = lin(mixed, W(f"layers.{L}.mlp.gate.weight"))
        probs = self._rw(lambda l_: F.softmax(l_, dtype=torch.float, dim=-1), logits)
        if self.cuda:
            from slotstream_spark.kernels.topk_stable import topk_stable
            ids, tv = topk_stable(probs.contiguous(), self.topk, out_val=torch.empty(
                probs.shape[0], self.topk, dtype=torch.float32, device=probs.device))
        else:
            tv, ids = torch.topk(probs, self.topk, dim=-1)
        tv = tv / self._rw(lambda v: v.sum(-1, keepdim=True), tv)
        return mixed, inj, ids, tv

    # ------------------------------------------------------------ MoE segment
    def _device_tables(self):
        if self._tables is None:
            m, s = self.pool.map_tensor, self.pool.slot_ptr_tensor
            if s.dtype != torch.int64:
                s = s.view(torch.int64)
            if m.dtype != torch.int32:
                m = m.view(torch.int32)
            self._tables = (m, s)
        return self._tables

    def _ensure(self, L, ids_group):
        """Pin the unique experts of a row group; returns {expert: slot}."""
        uniq = np.unique(ids_group).astype(np.int32)
        self.gen += 1
        slots = self.pool.ensure(L, uniq, self.gen)
        return dict(zip(uniq.tolist(), [int(s) for s in slots])), self.gen

    def _done(self, gen):
        """The group's kernels finished: pins of generations <= gen may be recycled."""
        self.progress[0] = gen + 1

    def seg_moe(self, L, mixed, ids, w):
        W = self.W
        T = mixed.shape[0]
        ids_host = ids.detach().cpu().numpy().astype(np.int32)         # router ids -> host (sync)
        self.last_routing[L] = (ids_host, w.detach().float().cpu().numpy())
        routed = torch.zeros(T, self.H, dtype=torch.float32, device=mixed.device)
        for a, b in row_groups(ids_host, self.max_rows):
            slot_of, gen = self._ensure(L, ids_host[a:b])
            if self.cuda:
                from slotstream_spark.kernels.moe_gemv_affine import moe_decode
                m_t, p_t = self._device_tables()
                r = b - a
                h_buf = torch.empty(r, self.topk, self.cfg["moe_intermediate_size"], dtype=torch.float32,
                                    device=mixed.device)
                o_buf = torch.empty(r, self.topk, self.H, dtype=torch.float32, device=mixed.device)
                moe_decode(mixed[a:b].contiguous(), ids[a:b].to(torch.int32).contiguous(),
                           w[a:b].contiguous(), L, m_t, p_t, h_buf, o_buf, routed[a:b])
                torch.cuda.synchronize()
            else:
                for t in range(a, b):
                    xt = mixed[t].float()
                    for rk in range(self.topk):
                        gw, uw, dw = self.pool.slot_weights(slot_of[int(ids_host[t, rk])])
                        hh = F.silu(gw.float() @ xt) * (uw.float() @ xt)
                        routed[t] += w[t, rk] * (dw.float() @ hh)
            self._done(gen)
        p = f"layers.{L}.mlp."
        sh = lin(F.silu(lin(mixed, W(p + "shared_expert.gate_proj.weight"))) *
                 lin(mixed, W(p + "shared_expert.up_proj.weight")), W(p + "shared_expert.down_proj.weight"))
        sh = torch.sigmoid(lin(mixed, W(p + "shared_expert_gate.weight"))) * sh
        return routed.to(mixed.dtype) + sh
