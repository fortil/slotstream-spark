"""Prefill mixers: the QSA and GDN blocks run over a pass in chronological tiles, plus the raw attention kernel.

The mixers reuse the Segments blocks (`seg.qsa`, `seg.deltanet`) so the weights, norms, RoPE, the key compression and the
state updates are the ones decode expects: they leave the KV, indexer, GDN and conv state exactly as `Segments` does
after a chunk. What the prefill adds is the position: the state's `pos` is set to `base + offset` before each tile, so the
driver can run a layer over several passes (a read scope) and the tiles stay chronological per layer.

QSA uses the vendored select-and-attend ops (indexer, `topk_stable` blocks, sparse attention, store and compress);
GDN uses the vendored FLA 0.5.2 `chunk_gated_delta_rule` (FlashInfer is out of scope).

`prefill_attention` is the raw masked attention (no norm, RoPE or gate): bf16 operands, fp32 scores, P = exp(s - max) carried as a
bf16 hi term plus a bf16 lo term (16 mantissa bits), fp32 accumulation and normalization, one bf16 store. It is the quantity the float64 oracle gate (tests/gates/test_prefill_attention_oracle.py)
judges; its rounding floor is P -> bf16 plus the output store.
"""
import torch

QSA_TILE = 1024          # rows per QSA tile (store + select + attend); fixed, never a function of the scope
ATTN_BLOCK = 256         # query rows per block inside prefill_attention


def qsa_prefill(seg, L, x, st, *, base):
    """QSA block of layer L over x [T, H] bf16 whose first row sits at position `base`; -> [T, H]."""
    outs = []
    for a in range(0, x.shape[0], QSA_TILE):
        st.pos = base + a
        outs.append(seg.qsa(L, st, x[a:a + QSA_TILE]))
    return outs[0] if len(outs) == 1 else torch.cat(outs, 0)


def gdn_prefill(seg, L, x, st, *, base):
    """GDN block of layer L over x [T, H]; the state carried in `st` is the one before row `base`."""
    st.pos = base
    return seg.deltanet(L, st, x)


def attn_block(seg, L, st, hyp, ids, *, base, history):
    """Attention segment of layer L for a tile `ids` at position `base` (Segments.seg_attn with the prefill mixers).
    `history` is the token list before the tile (PLE n-gram context). -> hyp after the attention residual."""
    st.pos = base
    st.tokens = history
    if L in seg.ple_layers and seg.ngram is not None:
        hyp = hyp + seg.ple(L, st, hyp, ids)
    h, inj = seg.hc_mix(f"layers.{L}.attn_hyper_connection", hyp)
    if seg.types[L] == "linear_attention":
        h = gdn_prefill(seg, L, h, st, base=base)
    else:
        h = qsa_prefill(seg, L, h, st, base=base)
    return hyp + (h.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)


def _pv(p, v):
    """P.V with bf16 operands, fp32 accumulate and an fp32 result (no intermediate bf16 store)."""
    return torch.bmm(p, v, out_dtype=torch.float32)


def prefill_attention(q, k, v, *, base, visible=None, scale=1 / 16):
    """q [B,H,S,D], k and v [B,Hk,N,D] (GQA: H a multiple of Hk) bf16 -> [B,H,S,D] bf16.

    Row i attends keys j <= base + i (causal, aligned to the end of the keys) and, when `visible` ([S,N] bool) is given,
    only keys it marks. Scores are fp32 (exact bf16 products); P = exp(s - max) goes through the P.V product as a bf16 hi
    and a bf16 lo term (fp32 accumulate), is divided by the fp32 sum of P, and the output is stored bf16. Query blocks of ATTN_BLOCK rows bound the score memory;
    q may be strided."""
    b, h, s, d = q.shape
    hk, n = k.shape[1], k.shape[2]
    if h % hk:
        raise ValueError("query heads must be a multiple of key heads")
    g = h // hk
    kf = k.float()
    out = torch.empty(b, h, s, v.shape[3], dtype=torch.bfloat16, device=q.device)
    keys = torch.arange(n, device=q.device)
    for a in range(0, s, ATTN_BLOCK):
        e = min(s, a + ATTN_BLOCK)
        qb = q[:, :, a:e].float().reshape(b, hk, g * (e - a), d)
        sc = torch.matmul(qb, kf.transpose(-1, -2)).mul_(scale).view(b, hk, g, e - a, n)
        rows = torch.arange(a, e, device=q.device)
        ok = keys[None, :] <= (base + rows)[:, None]
        if visible is not None:
            ok = ok & visible[a:e]
        sc.masked_fill_(~ok, float("-inf"))
        p = torch.exp(sc - sc.amax(dim=-1, keepdim=True))                       # unnormalized, max element exactly 1
        denom = p.sum(dim=-1, keepdim=True)
        hi = p.to(torch.bfloat16)                                               # P as two bf16 terms: 16 mantissa bits
        lo = (p - hi.float()).to(torch.bfloat16)
        rows_ = g * (e - a)
        o = _pv(hi.view(b * hk, rows_, n), v.view(b * hk, n, -1))
        o = o + _pv(lo.view(b * hk, rows_, n), v.view(b * hk, n, -1))
        o = o.view(b, hk, g, e - a, -1) / denom
        out[:, :, a:e] = o.to(torch.bfloat16).reshape(b, h, e - a, -1)
    return out
