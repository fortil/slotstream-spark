# SPDX-License-Identifier: MIT AND Apache-2.0
# Portions ported from helgard-orlm/qwen-flash-next-8gb@958d22e7ba0317750ccc38862b9667b1034a4dd1 qwen_engine.py:91-174 and qwen_engine.py:684-902 (Apache-2.0, section 4); modified.
"""Plain-torch reference forward of Qwen3.8-Flash-Next (no quantization, no streaming).

It is the fp32 yardstick for the kernels and for the vendored ops. Differences from the helgard
original: norm weights are the *folded* ones (the MLX checkpoint stores `1 + w` already, qwen4_exp.py
CENTERED_NORMS), so `rmsnorm` multiplies by `w` and not by `1 + w`; every weight and every expert
comes through injected accessors, so the model runs on CPU with a tiny configuration; the dtype of
the activations follows the input instead of being fixed to bf16.

Weight accessor names (relative to the text model, dense tensors in the `nn.Linear` layout):
  embed_tokens (via `embed(ids)`), lm_head.weight, hyper_connection_mixer.{hc_norm,
  input_mix_weight_down,input_mix_weight_up}.weight, layers.{i}.{attn,mlp}_hyper_connection.{hc_norm,
  input_mix_weight_down,input_mix_weight_up,block_inject_weight}.weight, layers.{i}.mlp.gate.weight,
  layers.{i}.mlp.shared_expert.{gate,up,down}_proj.weight, layers.{i}.mlp.shared_expert_gate.weight,
  layers.{i}.linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,conv1d,norm,out_proj}.weight,
  layers.{i}.linear_attn.{A_log,dt_bias}, layers.{i}.self_attn.{q,k,v,o}_proj.weight,
  layers.{i}.self_attn.{q,k}_norm.weight, layers.{i}.self_attn.indexer.{index_qk_proj,q_layernorm,
  k_layernorm}.weight, layers.{i}.ple.{key_proj,value_proj,norm_key,norm_query,norm_conv,conv1d}.weight.
Experts come from `expert(layer, e) -> (gate[I,H], up[I,H], down[H,I])` and the n-gram rows from
`ngram_rows(history_ids, ids) -> [T, ple_embed_dim]`.
"""
import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- small pieces
def rmsnorm(x, w, eps, group=None):
    """Qwen4ExpTextRMSNorm with folded weights: (x / rms) * w, in fp32, result in the dtype of x."""
    xf = x.float()
    if group is not None:
        xf = xf.reshape(*xf.shape[:-1], -1, group)
    o = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    if group is not None:
        o = o.flatten(-2)
    return (o * w.float()).type_as(x)


def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x, cos, sin):
    """x [..., D]; cos/sin broadcast against x[..., :rd] (partial rotary: the rest passes through)."""
    rd = cos.shape[-1]
    xr, xp = x[..., :rd], x[..., rd:]
    return torch.cat([xr * cos + rotate_half(xr) * sin, xp], dim=-1)


def chunk_gated_delta_rule(query, key, value, g, beta, chunk_size=64, initial_state=None):
    """torch_chunk_gated_delta_rule of the transformers reference (l2 norm of q/k in the kernel, final state returned)."""
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    query, key, value, beta, decay = [x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
                                      for x in (query, key, value, beta, g)]
    query = l2norm(query)
    key = l2norm(key)
    query = query * query.shape[-1] ** -0.5
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))
    num_chunks = (sequence_length + pad_size) // chunk_size
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
                                  for x in (query, key, k_beta, v_beta)]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
    upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    cum = decay.cumsum(dim=3)
    pw = (cum.unsqueeze(4) - cum.unsqueeze(3)).masked_fill(upper, float("-inf")).exp()
    ut = (k_beta @ key.transpose(-1, -2)) * pw
    intra = (query @ key.transpose(-1, -2)) * pw
    dkb = k_beta * cum.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(ut, v_beta, upper=False, unitriangular=True)
    k_cum = torch.linalg.solve_triangular(ut, dkb, upper=False, unitriangular=True)
    S = (torch.zeros(batch_size, num_v_heads, k_head_dim, v_head_dim, dtype=new_values.dtype, device=new_values.device)
         if initial_state is None else initial_state.to(new_values))
    out = torch.zeros_like(new_values)
    query = query * cum.exp().unsqueeze(-1)
    key = key * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    cd = cum[..., -1].exp()[..., None, None]
    for i in range(num_chunks):
        v_new = new_values[:, :, i] - k_cum[:, :, i] @ S
        out[:, :, i] = query[:, :, i] @ S + intra[:, :, i] @ v_new
        S = S * cd[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new
    out = out.reshape(batch_size, num_v_heads, -1, v_head_dim)[:, :, :sequence_length]
    return out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format), S


def recurrent_gated_delta_rule(query, key, value, g, beta, initial_state):
    """torch_recurrent_gated_delta_rule of the transformers reference; works for any T, the state is returned."""
    initial_dtype = query.dtype
    query, key, value, beta, decay = [x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
                                      for x in (query, key, value, beta, g)]
    query = l2norm(query)
    key = l2norm(key)
    query = query / (query.shape[-1] ** 0.5)
    S = initial_state.float()
    out = torch.zeros_like(value)
    for i in range(query.shape[2]):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        S = S * decay[:, :, i].exp()[..., None, None]
        kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta[:, :, i].unsqueeze(-1)
        S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (S * q_t.unsqueeze(-1)).sum(dim=-2)
    return out.transpose(1, 2).contiguous().to(initial_dtype), S


def moe_ref(x, router_w, topk, expert, layer, sh_gate, sh_up, sh_down, sh_g):
    """Routed + shared expert block per transformers modeling:963-967: softmax in fp32, top-k, renormalize,
    outputs summed in rank order, in fp32; the shared expert is gated by sigmoid(sh_g x)."""
    logits = F.linear(x, router_w)
    probs = F.softmax(logits, dtype=torch.float, dim=-1)
    tv, ti = torch.topk(probs, topk, dim=-1)
    tv = tv / tv.sum(-1, keepdim=True)
    out = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    for t in range(x.shape[0]):
        xt = x[t].float()
        for r in range(topk):
            gw, uw, dw = expert(layer, int(ti[t, r]))
            h = F.silu(gw.float() @ xt) * (uw.float() @ xt)
            out[t] += tv[t, r] * (dw.float() @ h)
    sh = F.linear(F.silu(F.linear(x, sh_gate)) * F.linear(x, sh_up), sh_down)
    sh = torch.sigmoid(F.linear(x, sh_g)) * sh
    return out.to(x.dtype) + sh, ti


# ---------------------------------------------------------------- model
class RefModel:
    DN_CHUNK = 512      # multiple of the 64-token algorithm block: same arithmetic as one call

    def __init__(self, cfg, weight, embed, expert, ngram_rows=None, device="cpu"):
        self.cfg = cfg
        self.w = weight
        self.embed = embed
        self.expert = expert
        self.ngram_rows = ngram_rows
        self.dev = device
        self.H = cfg["hidden_size"]
        self.hc = cfg["hc_count"]
        self.eps = cfg["rms_norm_eps"]
        self.NL = cfg["num_hidden_layers"]
        self.types = cfg.layer_types
        self.topk = cfg["num_experts_per_tok"]
        rd = cfg.rotary_dim
        self.inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, rd, 2, dtype=torch.float) / rd))
        self.reset()

    def reset(self):
        self.pos = 0
        self.tokens = []
        self.st = []
        for t in self.types:
            self.st.append(dict(conv=None, S=None) if t == "linear_attention"
                           else dict(k=None, v=None, ik=None, bk=None))
        self.ple_conv = None
        self.last_selected = {}     # layer -> bool mask [T, S] of the indexer (None = dense), for tests

    def rope(self, positions):
        f = positions.to(self.dev).float()[:, None] * self.inv_freq.to(self.dev)[None, :]
        emb = torch.cat([f, f], -1)
        return emb.cos(), emb.sin()

    # ------------------------------------------------------------ blocks
    def gated_residual(self, prefix, hyp, inject=True):
        """hyp [T, 4H] -> (mixed [T, H], inj [T, 4] | None)."""
        w = self.w
        hn = rmsnorm(hyp, w(prefix + ".hc_norm.weight"), self.eps, group=self.H)
        m = F.silu(F.linear(hn, w(prefix + ".input_mix_weight_down.weight")) / self.hc)
        m = torch.sigmoid(F.linear(m, w(prefix + ".input_mix_weight_up.weight"))).unflatten(-1, (self.hc, self.H))
        mixed = (m * hn.unflatten(-1, (self.hc, self.H))).mean(dim=-2)
        if not inject:
            return mixed, None
        return mixed, 2 * torch.sigmoid(F.linear(hn, w(prefix + ".block_inject_weight.weight")) / self.hc)

    def deltanet(self, i, st, x):
        T = x.shape[0]
        if T > self.DN_CHUNK:
            return torch.cat([self.deltanet(i, st, x[a:a + self.DN_CHUNK]) for a in range(0, T, self.DN_CHUNK)], 0)
        c = self.cfg
        nk, nv = c["linear_num_key_heads"], c["linear_num_value_heads"]
        dk, dv = c["linear_key_head_dim"], c["linear_value_head_dim"]
        kd, vd = nk * dk, nv * dv
        p = f"layers.{i}.linear_attn."
        w = self.w
        qkv = F.linear(x, w(p + "in_proj_qkv.weight"))
        z = F.linear(x, w(p + "in_proj_z.weight")).reshape(T, -1, dv)
        b = F.linear(x, w(p + "in_proj_b.weight"))
        a = F.linear(x, w(p + "in_proj_a.weight"))
        ck = c["linear_conv_kernel_dim"] - 1
        prev = st["conv"] if st["conv"] is not None else qkv.new_zeros(ck, qkv.shape[1])
        full = torch.cat([prev, qkv], 0)
        st["conv"] = full[-ck:].clone()
        cw = w(p + "conv1d.weight")
        cw = cw if cw.dim() == 2 else cw.squeeze(1)
        conv = F.conv1d(full.T.unsqueeze(0), cw.unsqueeze(1), None, padding=0, groups=full.shape[1])
        qkv = F.silu(conv)[0].T.to(x.dtype)
        q, k, v = torch.split(qkv, [kd, kd, vd], dim=-1)
        q = q.reshape(1, T, nk, dk)
        k = k.reshape(1, T, nk, dk)
        v = v.reshape(1, T, nv, dv)
        beta = b.sigmoid().unsqueeze(0)
        g = (-w(p + "A_log").float().exp() * F.softplus(a.float() + w(p + "dt_bias"))).unsqueeze(0)
        r = nv // nk
        q = q.repeat_interleave(r, dim=2)
        k = k.repeat_interleave(r, dim=2)
        if st["S"] is not None and T == 1:
            o, S = recurrent_gated_delta_rule(q, k, v, g, beta, st["S"])
        else:
            o, S = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=st["S"])
        st["S"] = S
        o = o.reshape(-1, dv)
        z = z.reshape(-1, dv)
        of = o.float()
        o = w(p + "norm.weight") * (of * torch.rsqrt(of.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype)
        o = (o * torch.sigmoid(z.float())).to(x.dtype).reshape(T, -1)
        return F.linear(o, w(p + "out_proj.weight"))

    def indexer_mask(self, i, st, x, cs, sn):
        """Qwen4ExpTextQSAIndexer for one sequence without padding -> bool [T, S], or None when every query
        sees all its causal blocks. A block is `ratio` consecutive tokens; its key is the mean of the raw
        keys (fp32 -> input dtype), k_layernorm, RoPE at the position of the block start. Query p sees
        (p+1)//ratio full blocks: if they are <= budget/ratio all are taken (dense causal attention, None),
        otherwise the top budget/ratio by sum over heads of relu(q.k) / sqrt(d), plus the tail of the
        incomplete block."""
        c = self.cfg
        T = x.shape[0]
        ihd, inh = c["indexer_head_dim"], c["indexer_n_heads"]
        cr = c["indexer_compress_ratio"]
        btop = c["indexer_budget"] // cr
        p = f"layers.{i}.self_attn.indexer."
        qk = F.linear(x, self.w(p + "index_qk_proj.weight"))
        q, rk = torch.split(qk, [inh * ihd, ihd], dim=-1)
        st["ik"] = rk if st.get("ik") is None else torch.cat([st["ik"], rk], 0)
        S = st["ik"].shape[0]
        nb = S // cr
        if nb <= btop:
            return None
        q = apply_rope(rmsnorm(q.view(T, inh, ihd), self.w(p + "q_layernorm.weight"), self.eps), cs, sn)
        have = 0 if st.get("bk") is None else st["bk"].shape[0]
        if nb > have:
            raw = st["ik"][have * cr: nb * cr].view(nb - have, cr, ihd)
            pooled = rmsnorm(raw.float().mean(dim=1).to(raw.dtype), self.w(p + "k_layernorm.weight"), self.eps)
            cc, ss = self.rope(torch.arange(have, nb) * cr)
            bk = apply_rope(pooled.unsqueeze(1), cc[:, None, :].to(x.dtype), ss[:, None, :].to(x.dtype)).squeeze(1)
            st["bk"] = bk if st.get("bk") is None else torch.cat([st["bk"], bk], 0)
        bk = st["bk"]
        sc = torch.relu(torch.einsum("thd,bd->tbh", q.float(), bk.float())).sum(-1) / math.sqrt(ihd)
        pos = torch.arange(S - T, S, device=x.device)
        ncb = (pos + 1) // cr
        bidx = torch.arange(nb, device=x.device)
        sc = sc.masked_fill(bidx[None, :] >= ncb[:, None], float("-inf"))
        sel_b = torch.zeros(T, nb, dtype=torch.bool, device=x.device)
        many = ncb > btop
        if many.any():
            # ties (relu leaves many exact zeros) go to the lower block index, like kernels.topk_stable
            top = torch.sort(-sc[many], dim=-1, stable=True).indices[:, :btop]
            sel_b[many] = torch.zeros_like(sel_b[many]).scatter(1, top, True)
        sel_b[~many] = bidx[None, :] < ncb[~many, None]
        j = torch.arange(S, device=x.device)
        jb = (j // cr).clamp_max(nb - 1)
        in_block = sel_b.gather(1, jb[None, :].expand(T, -1)) & (j[None, :] < (ncb * cr)[:, None])
        tail = (j[None, :] >= (ncb * cr)[:, None]) & (j[None, :] <= pos[:, None])
        return in_block | tail

    def attention(self, i, st, x, cos, sin):
        c = self.cfg
        T = x.shape[0]
        hd, nh, nkv = c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"]
        cs, sn = cos[:, None, :].to(x.dtype), sin[:, None, :].to(x.dtype)
        p = f"layers.{i}.self_attn."
        w = self.w
        sel = self.indexer_mask(i, st, x, cs, sn)
        self.last_selected[i] = sel
        qg = F.linear(x, w(p + "q_proj.weight")).view(T, -1, hd * 2)
        q, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(T, -1)
        q = rmsnorm(q, w(p + "q_norm.weight"), self.eps)
        k = rmsnorm(F.linear(x, w(p + "k_proj.weight")).view(T, -1, hd), w(p + "k_norm.weight"), self.eps)
        v = F.linear(x, w(p + "v_proj.weight")).view(T, -1, hd)
        q = apply_rope(q, cs, sn)
        k = apply_rope(k, cs, sn)
        st["k"] = k if st["k"] is None else torch.cat([st["k"], k], 0)
        st["v"] = v if st["v"] is None else torch.cat([st["v"], v], 0)
        K, V = st["k"], st["v"]
        Sn = K.shape[0]
        qh = q.transpose(0, 1).unsqueeze(0)
        kh = K.transpose(0, 1).unsqueeze(0)
        vh = V.transpose(0, 1).unsqueeze(0)
        mask = sel if sel is not None else torch.ones(T, Sn, dtype=torch.bool, device=x.device).tril(Sn - T)
        QB = 256 if Sn <= 2048 else 128 if Sn <= 4096 else 64
        o = torch.cat([F.scaled_dot_product_attention(qh[:, :, q0:q0 + QB], kh, vh, attn_mask=mask[q0:q0 + QB],
                                                      scale=hd ** -0.5, enable_gqa=True)
                       for q0 in range(0, T, QB)], dim=2)
        o = o[0].transpose(0, 1).reshape(T, -1)
        o = o * torch.sigmoid(gate)
        return F.linear(o, w(p + "o_proj.weight"))

    def moe(self, i, x):
        p = f"layers.{i}.mlp."
        w = self.w
        out, ti = moe_ref(x, w(p + "gate.weight"), self.topk, self.expert, i,
                          w(p + "shared_expert.gate_proj.weight"), w(p + "shared_expert.up_proj.weight"),
                          w(p + "shared_expert.down_proj.weight"), w(p + "shared_expert_gate.weight"))
        self.last_routing = ti
        return out

    def ple(self, i, hyp, ids):
        c = self.cfg
        T = len(ids)
        p = f"layers.{i}.ple."
        w = self.w
        emb = self.ngram_rows(self.tokens, ids).to(hyp.dtype)
        kn = rmsnorm(F.linear(emb, w(p + "key_proj.weight")), w(p + "norm_key.weight"), self.eps,
                     group=self.H).unflatten(-1, (self.hc, self.H))
        val = F.linear(emb, w(p + "value_proj.weight"))
        qn = rmsnorm(hyp, w(p + "norm_query.weight"), self.eps, group=self.H).unflatten(-1, (self.hc, self.H))
        gate = (kn * qn).sum(-1, keepdim=True) / math.sqrt(self.H)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gv = torch.sigmoid(gate) * val.unsqueeze(-2)
        gvn = rmsnorm(gv.flatten(-2), w(p + "norm_conv.weight"), self.eps, group=self.H)
        gv = gv.flatten(-2)
        k = c["ple_conv_kernel_size"]
        hist = 3 * (k - 1)      # dilation 3
        prevc = self.ple_conv if self.ple_conv is not None else gvn.new_zeros(hist, gvn.shape[1])
        full = torch.cat([prevc, gvn], 0)
        self.ple_conv = full[-hist:].clone()
        cw = w(p + "conv1d.weight")
        cw = cw if cw.dim() == 2 else cw.squeeze(1)
        conv = F.conv1d(full.T.unsqueeze(0), cw.unsqueeze(1), None, groups=full.shape[1], dilation=3)
        return gv + F.silu(conv)[0].T

    # ------------------------------------------------------------ pass
    @torch.no_grad()
    def forward(self, ids, all_logits=False, logits_from=0, dtype=None):
        """ids: list[int] of new tokens -> logits of the last one [V] fp32 (all_logits: [T - logits_from, V])."""
        T = len(ids)
        x = self.embed(torch.tensor(ids))
        if dtype is not None:
            x = x.to(dtype)
        cos, sin = self.rope(torch.arange(self.pos, self.pos + T))
        hyp = x.repeat(1, self.hc)
        ple = set(self.cfg.ple_layers)
        for L in range(self.NL):
            if L in ple and self.ngram_rows is not None:
                hyp = hyp + self.ple(L, hyp, ids)
            h, inj = self.gated_residual(f"layers.{L}.attn_hyper_connection", hyp)
            if self.types[L] == "linear_attention":
                h = self.deltanet(L, self.st[L], h)
            else:
                h = self.attention(L, self.st[L], h, cos, sin)
            hyp = hyp + (h.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
            h, inj = self.gated_residual(f"layers.{L}.mlp_hyper_connection", hyp)
            h = self.moe(L, h)
            hyp = hyp + (h.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
        mixed, _ = self.gated_residual("hyper_connection_mixer", hyp, inject=False)
        sel = mixed[logits_from:] if all_logits else mixed[-1:]
        logits = F.linear(sel, self.w("lm_head.weight"))
        self.tokens += list(ids)
        self.pos += T
        return logits.float() if all_logits else logits[0].float()
