"""Adapters over the vendored ops: vLLM Qwen4Exp QSA and hyper-connection kernels (187a0eb9) and the
flash-linear-attention 0.5.2 gated delta rule.

The vendored packages live under third_party/vendor and are put on sys.path here. Everything works
on one sequence (request 0): the paged metadata is block_table = arange, `seq_lens` a device tensor.
Norm weights in this project are folded (the checkpoint stores 1 + w); the vendored kernels compute
x * (1 + w), so the adapters pass `w_folded.float() - 1` (exact in fp32 for bf16 w).

QSA per full-attention layer, in call order:
  cache = QSACache(max_ctx, device)
  q_idx, q, gate = qsa_store_kv(cache, qkv, iq, ik, positions, weights)   # norm, RoPE, KV store, key compression
  out = qsa_decode(cache, q_idx, q, gate)  |  qsa_prefill(cache, q_idx, q, gate)    # select + sparse attention
`out` is the attention output already multiplied by sigmoid(gate), before o_proj.
"""
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from .gemv_affine import AffineWeight, DenseWeight, gemv

_VENDOR = Path(__file__).resolve().parents[2] / "third_party" / "vendor"
for _p in (_VENDOR / "vllm-187a0eb9", _VENDOR / "fla-0.5.2"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_MOD = {}


def _ops():
    """Import the vendored modules on first use (they need torch and triton, and CUDA only when called)."""
    if not _MOD:
        from qwen4_exp_ops import _shims, hc, qsa, qsa_cache, qsa_indexer, qsa_prepare
        _MOD.update(shims=_shims, hc=hc, qsa=qsa, cache=qsa_cache, indexer=qsa_indexer, prepare=qsa_prepare)
    return _MOD


def _fla():
    if "fla" not in _MOD:
        from fla_v052.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
        _MOD["fla"] = (chunk_gated_delta_rule, fused_recurrent_gated_delta_rule)
    return _MOD["fla"]


def gemma_weight(w_folded: torch.Tensor) -> torch.Tensor:
    """Folded norm weight (1 + w) -> the `w` the vendored kernels expect (they add 1 themselves)."""
    return (w_folded.float() - 1.0).contiguous()


def _lin(w):
    if isinstance(w, (AffineWeight, DenseWeight)):
        return w
    return DenseWeight(w.to(torch.bfloat16).contiguous())


# ---------------------------------------------------------------- hyper-connections
def hc_mix(hyp, norm_w, down, up, inject=None, hc_count=4, eps=1e-6):
    """Gated residual mix of one HC block: hyp [R, 4H] bf16 -> (mixed [R, H] bf16, inj_logits [R, 4] | None, hn).

    hn = group RMSNorm of hyp (folded `norm_w` [4H]); m = silu(down(hn) / hc); gate = up(m);
    mixed = mean over streams of sigmoid(gate) * hn; inj_logits = inject(hn) (None for the final mixer).
    The low-rank linears go through `gemv` (bf16 dense or affine weights)."""
    hc = _ops()["hc"]
    hn = hc.grouped_gemma_rmsnorm(hyp, gemma_weight(norm_w), eps, hc_count)
    d = gemv(hn, _lin(down)).to(hn.dtype)
    m = hc.hc_silu(d, hc_count)
    g = gemv(m, _lin(up)).to(hn.dtype)
    mixed = hc.hc_gate_mix(hn, g, hc_count)
    inj = None if inject is None else gemv(hn, _lin(inject)).to(hn.dtype)
    return mixed, inj, hn


def hc_residual(hyp, block_out, inj_logits, hc_count=4):
    """hyp + block_out * 2 sigmoid(inj / hc) per stream (no injection when inj_logits is None)."""
    return _ops()["hc"].hc_combine(hyp, block_out, inj_logits, hc_count)


def hc_residual_norm(hyp, block_out, inj_logits, next_norm_w, eps=1e-6, hc_count=4):
    """hc_residual fused with the next block's group RMSNorm: returns (new hyp, hn)."""
    return _ops()["hc"].hc_combine_norm(hyp, block_out, inj_logits, gemma_weight(next_norm_w), eps, hc_count)


# ---------------------------------------------------------------- gated delta rule
def gdn_decode(q, k, v, g, beta, state):
    """T recurrent steps. q, k [1, T, 16, 128] (repeated x3 inside), v [1, T, 48, 128], g, beta [1, T, 48],
    state fp32 [1, 48, 128, 128] -> (o [1, T, 48, 128], new state). q/k are l2-normalised in the kernel."""
    _, fused = _fla()
    r = v.shape[2] // q.shape[2]
    return fused(q.repeat_interleave(r, dim=2), k.repeat_interleave(r, dim=2), v, g=g, beta=beta,
                 initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)


def gdn_prefill(q, k, v, g, beta, state=None):
    """Chunked form of gdn_decode for T > 1 (same arguments, state may be None)."""
    chunk, _ = _fla()
    r = v.shape[2] // q.shape[2]
    return chunk(q.repeat_interleave(r, dim=2), k.repeat_interleave(r, dim=2), v, g=g, beta=beta,
                 initial_state=state, output_final_state=True, use_qk_l2norm_in_kernel=True)


# ---------------------------------------------------------------- QSA
@dataclass
class QSAShapes:
    num_q_heads: int = 24
    num_kv_heads: int = 2
    head_dim: int = 256
    index_heads: int = 4
    index_dim: int = 128
    ratio: int = 4
    budget: int = 2048
    rotary_dim: int = 64
    rope_theta: float = 1e7
    eps: float = 1e-6
    page: int = 64          # rows per main KV page
    comp_page: int = 64     # compressed rows per page
    ring: int = 8           # raw-key ring: a group (4) plus up to 3 speculative rows, in whole groups


class QSACache:
    """Caches, block tables and metadata scratch of one QSA layer for a single sequence."""

    MIN_COLUMNS = 512

    def __init__(self, max_ctx, device, shapes=None):
        self.s = s = shapes or QSAShapes()
        self.device = device
        self.pos = 0
        self.max_ctx = max_ctx
        n_pages = -(-max_ctx // s.page)
        comp_rows = max(-(-max_ctx // s.ratio), self.MIN_COLUMNS)
        n_comp = -(-comp_rows // s.comp_page)
        bf = torch.bfloat16
        self.main_kv = torch.zeros(n_pages, s.page, s.num_kv_heads, 2 * s.head_dim, dtype=bf, device=device)
        self.ring = torch.zeros(2, s.ring, 1, s.index_dim, dtype=bf, device=device)    # block 0 is the null block
        self.comp = torch.zeros(n_comp, s.comp_page, 1, s.index_dim, dtype=bf, device=device)
        self.main_table = torch.arange(n_pages, dtype=torch.int32, device=device)[None]
        self.ring_table = torch.ones(1, 1, dtype=torch.int32, device=device)
        self.comp_table = torch.arange(n_comp, dtype=torch.int32, device=device)[None]
        rd = s.rotary_dim
        inv = 1.0 / (s.rope_theta ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
        f = torch.arange(max_ctx, dtype=torch.float32)[:, None] * inv[None, :]
        self.cos_sin = torch.cat([f.cos(), f.sin()], dim=-1).to(bf).to(device).contiguous()
        self.cap = 4096   # token capacity of the metadata scratch
        self._meta_buffers()
        self.last = None

    def _meta_buffers(self):
        d = self.device
        mk = lambda dt: torch.empty(self.cap, dtype=dt, device=d)
        self.buf = dict(
            t2r=[mk(torch.int32), mk(torch.int32)], lp=[mk(torch.int64), mk(torch.int64)],
            vb=[mk(torch.int32), mk(torch.int32)], sm=[mk(torch.int64), mk(torch.int64)])
        self.k_work = torch.empty(-(-self.cap // self.s.ratio) + self.s.ratio, 2, dtype=torch.int32, device=d)

    def clone(self):
        c = object.__new__(QSACache)
        c.__dict__.update(self.__dict__)
        for name in ("main_kv", "ring", "comp"):
            setattr(c, name, getattr(self, name).clone())
        c._meta_buffers()
        return c


def _metadata(cache, t, start):
    """Run the vendored metadata builder for the ring and the compressed cache."""
    m = _ops()
    shims, qc = m["shims"], m["cache"]
    dev = cache.device
    s = cache.s
    if t > cache.cap:
        raise ValueError(f"at most {cache.cap} tokens per call")
    qsl_cpu = torch.tensor([0, t], dtype=torch.int32)
    qsl = qsl_cpu.to(dev)
    seq = torch.tensor([start + t], dtype=torch.int32, device=dev)
    slots = torch.arange(start, start + t, dtype=torch.int64, device=dev)

    def cam(table):
        return shims.CommonAttentionMetadata(
            query_start_loc=qsl, query_start_loc_cpu=qsl_cpu, seq_lens=seq, num_reqs=1, num_actual_tokens=t,
            max_query_len=t, max_seq_len=start + t, block_table_tensor=table, slot_mapping=slots)

    b = cache.buf
    ring = qc.build_qsa_metadata(cam(cache.ring_table), b["t2r"][0], b["lp"][0], b["vb"][0], b["sm"][0],
                                 storage_block_size=s.ring, compress_ratio=1, circular_buffer_size=s.ring)
    n_work = (t + (s.ratio - 1)) // s.ratio
    kw = cache.k_work[:n_work]
    comp = qc.build_qsa_metadata(cam(cache.comp_table), b["t2r"][1], b["lp"][1], b["vb"][1], b["sm"][1],
                                 storage_block_size=s.comp_page, compress_ratio=s.ratio, circular_buffer_size=0,
                                 k_work_metadata_buffer=kw, request_capacity=1)
    return dict(qsl=qsl, seq=seq, slots=slots, ring=ring, comp=comp, k_work=kw, t=t, start=start)


def qsa_store_kv(cache, qkv, iq, ik, positions_start, q_norm, k_norm, iq_norm, ik_norm):
    """Norm + RoPE of the main and indexer queries, K/V store, raw-key ring update and compression of the
    groups that complete. qkv [T, (2*24+2*2)*256] bf16 = per head [q, gate] then k then v; iq [T, 4*128],
    ik [T, 128] (raw indexer projections); norm weights are the folded ones. Returns (q_idx [T, 4, 128],
    q [T, 24, 256], gate [T, 24, 256]) and advances cache.pos."""
    m = _ops()
    s = cache.s
    t = qkv.shape[0]
    start = int(positions_start)
    meta = _metadata(cache, t, start)
    positions = torch.arange(start, start + t, dtype=torch.int64, device=qkv.device)
    ring_slots = meta["ring"][3]
    comp_slots = meta["comp"][3]
    q_idx = torch.empty(t, s.index_heads, s.index_dim, dtype=torch.bfloat16, device=qkv.device)
    q, gate = m["prepare"].qsa_prepare(
        iq, ik, positions, cache.cos_sin, gemma_weight(iq_norm), gemma_weight(ik_norm), s.eps, q_idx,
        cache.ring, ring_slots, cache.ring_table, meta["qsl"], meta["comp"][1], cache.comp, comp_slots,
        meta["k_work"], compress_ratio=s.ratio, mrope_section=None, rope_pos_offset=None,
        main_qkv=qkv, main_q_norm_weight=gemma_weight(q_norm), main_k_norm_weight=gemma_weight(k_norm),
        main_eps=s.eps, main_kv_cache=cache.main_kv, main_slot_mapping=meta["slots"],
        main_k_scale=1.0, main_v_scale=1.0)
    cache.last = meta
    cache.pos = start + t
    return q_idx, q, gate


def _attend(cache, q_idx, q, gate, decode):
    m = _ops()
    s = cache.s
    meta = cache.last
    t = q.shape[0]
    comp_vb = meta["comp"][2]
    comp_lp = meta["comp"][1]
    block_topk = s.budget // s.ratio
    block_indices = torch.empty(t, block_topk, dtype=torch.int32, device=q.device)
    if decode:
        m["indexer"].qsa_select_paged_decode(q_idx, cache.comp, cache.comp_table, comp_vb, s.budget, s.ratio,
                                             t, block_indices)
    else:
        m["indexer"].qsa_select_paged_prefill(q_idx, cache.comp, cache.comp_table, meta["qsl"], comp_vb, s.budget,
                                              s.ratio, t, block_indices, meta["start"] + t)
    logical = torch.empty(t, s.budget + s.ratio - 1 + 1, dtype=torch.int32, device=q.device)
    m["indexer"].expand_qsa_block_indices(block_indices, comp_lp, comp_vb, s.ratio, s.budget, logical)
    token_to_req = torch.zeros(t, dtype=torch.int32, device=q.device)
    hd = s.head_dim
    out = m["qsa"].qsa_sparse_paged_attention(
        q, cache.main_kv[..., :hd], cache.main_kv[..., hd:], logical, cache.main_table, token_to_req,
        not decode, output_gate=gate)
    cache.last = dict(meta, block_indices=block_indices, logical=logical)
    return out


def qsa_decode(cache, q_idx, q, gate):
    """Decode (or verify) the T rows just stored by qsa_store_kv: select blocks, sparse attention.
    Returns [T, 24, 256] bf16, already gated by sigmoid(gate)."""
    return _attend(cache, q_idx, q, gate, decode=True)


def qsa_prefill(cache, q_idx, q, gate):
    """Same for a prompt chunk (T queries of one request)."""
    return _attend(cache, q_idx, q, gate, decode=False)


def selected_tokens(cache):
    """Token-level boolean mask [T, pos] of what the last qsa_decode/qsa_prefill attended to (for tests)."""
    lg = cache.last["logical"][:, :-1]      # the last column is the count, not a token
    t = lg.shape[0]
    mask = torch.zeros(t, cache.pos + 1, dtype=torch.bool, device=lg.device)    # extra column swallows -1 padding
    idx = torch.where(lg >= 0, lg, torch.full_like(lg, cache.pos)).long()
    mask.scatter_(1, idx, torch.ones_like(idx, dtype=torch.bool))
    return mask[:, :-1]
