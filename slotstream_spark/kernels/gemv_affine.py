"""Dense GEMV kernels (Triton): MLX affine 4/8-bit (gs 32/64) and bf16 dense.

y[r, n] = sum_k x[r, k] * (q[n, k] * s[n, k//gs] + b[n, k//gs]), fp32 accumulate.
Grid (cdiv(N, BLOCK_N), SPLIT_K, R). Each program reduces its K slice in a fixed order and
writes a partial to a [R, SPLIT_K, N] workspace; `_splitk_reduce` adds k = 0..SPLIT_K-1 in
order. No atomics. The split and tiles come from tuning.config_for(n, k, bits, gs), never
from R. The only thing R selects is the program: a 3-row call runs one program per tile with three accumulators;
each row is the same _v3_chunk helper call as the 1-row kernel, so it equals three 1-row calls bit for bit.
"""
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from . import tuning

SHARED_BYTES: dict[str, int] = {}


def _note(name, handle):
    meta = getattr(handle, "metadata", None)
    sh = getattr(meta, "shared", None)
    if sh is not None:
        SHARED_BYTES[name] = int(sh)


@dataclass
class AffineWeight:
    q: torch.Tensor   # int32 [n, k*bits/32]
    s: torch.Tensor   # bf16 [n, k/gs]
    b: torch.Tensor   # bf16 [n, k/gs]
    bits: int
    gs: int
    n: int
    k: int

    def __post_init__(self):
        if self.bits not in (4, 8) or self.gs not in (32, 64):
            raise ValueError(f"unsupported bits={self.bits} gs={self.gs}")
        if self.k % self.gs:
            raise ValueError(f"k={self.k} not a multiple of gs={self.gs}")
        if tuple(self.q.shape) != (self.n, self.k * self.bits // 32) or self.q.dtype != torch.int32:
            raise ValueError(f"q must be int32 {(self.n, self.k * self.bits // 32)}, got {tuple(self.q.shape)} {self.q.dtype}")
        for t in (self.s, self.b):
            if tuple(t.shape) != (self.n, self.k // self.gs) or t.dtype != torch.bfloat16:
                raise ValueError("scales and biases must be bf16 [n, k/gs]")


@dataclass
class DenseWeight:
    w: torch.Tensor  # bf16 [n, k]

    def __post_init__(self):
        if self.w.dtype != torch.bfloat16 or self.w.dim() != 2:
            raise ValueError("DenseWeight needs a 2-D bf16 tensor")

    @property
    def n(self):
        return self.w.shape[0]

    @property
    def k(self):
        return self.w.shape[1]


@triton.jit
def _v3_chunk(words, sc, bi, x_row_ptr, ws, kmask, NG: tl.constexpr, WPG: tl.constexpr):
    """One K chunk of one x row against an already loaded weight tile: words [BN, NW] uint32 (8 nibbles each),
    sc/bi [BN, NG] fp32. Returns sum over the chunk's groups of sc * sum(q*x) + bi * sum(x), [BN] fp32.
    The 1-row and the 3-row kernels both call this per row, so a row's op sequence is the same in both."""
    gq = tl.zeros_like(sc)
    gx = tl.zeros([NG], dtype=tl.float32)
    for j in tl.static_range(8):
        xj = tl.load(x_row_ptr + ws * 8 + j, mask=kmask, other=0.0).to(tl.float32)
        qj = ((words >> (4 * j)) & 0xF).to(tl.float32)
        gq += tl.sum(tl.reshape(qj * xj[None, :], [sc.shape[0], NG, WPG]), axis=2)
        gx += tl.sum(tl.reshape(xj, [NG, WPG]), axis=1)
    return tl.sum(sc * gq + bi * gx[None, :], axis=1)


@triton.jit
def _gemv_affine4(x_ptr, q_ptr, s_ptr, b_ptr, ws_ptr, N, K, KCHUNK,
                  GS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
                  KITER: tl.constexpr):
    pn = tl.program_id(0)
    pk = tl.program_id(1)
    r = tl.program_id(2)
    r64 = r.to(tl.int64)
    rows = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    rows64 = rows.to(tl.int64)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    k_lo = pk * KCHUNK
    k_hi = tl.minimum(k_lo + KCHUNK, K)
    NW: tl.constexpr = BLOCK_K // 8
    NG: tl.constexpr = BLOCK_K // GS
    WPG: tl.constexpr = GS // 8
    for it in tl.static_range(KITER):
        k0 = k_lo + it * BLOCK_K
        ws = k0 // 8 + tl.arange(0, NW)
        kmask = ws * 8 < k_hi
        words = tl.load(q_ptr + rows64[:, None] * (K // 8) + ws[None, :],
                        mask=rmask[:, None] & kmask[None, :], other=0).to(tl.uint32)
        gidx = k0 // GS + tl.arange(0, NG)
        gm = rmask[:, None] & (gidx * GS < k_hi)[None, :]
        sc = tl.load(s_ptr + rows64[:, None] * (K // GS) + gidx[None, :], mask=gm, other=0.0).to(tl.float32)
        bi = tl.load(b_ptr + rows64[:, None] * (K // GS) + gidx[None, :], mask=gm, other=0.0).to(tl.float32)
        acc += _v3_chunk(words, sc, bi, x_ptr + r64 * K, ws, kmask, NG, WPG)
    tl.store(ws_ptr + (r64 * SPLIT_K + pk) * N + rows, acc, mask=rmask)


@triton.jit
def _gemv_affine4_r3(x_ptr, q_ptr, s_ptr, b_ptr, ws_ptr, N, K, KCHUNK,
                     GS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
                     KITER: tl.constexpr):
    """Three rows in one program: the weight tile is loaded once, each row runs the same _v3_chunk as the 1-row kernel."""
    pn = tl.program_id(0)
    pk = tl.program_id(1)
    rows = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    rows64 = rows.to(tl.int64)
    acc0 = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_N], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_N], dtype=tl.float32)
    k_lo = pk * KCHUNK
    k_hi = tl.minimum(k_lo + KCHUNK, K)
    NW: tl.constexpr = BLOCK_K // 8
    NG: tl.constexpr = BLOCK_K // GS
    WPG: tl.constexpr = GS // 8
    for it in tl.static_range(KITER):
        k0 = k_lo + it * BLOCK_K
        ws = k0 // 8 + tl.arange(0, NW)
        kmask = ws * 8 < k_hi
        words = tl.load(q_ptr + rows64[:, None] * (K // 8) + ws[None, :],
                        mask=rmask[:, None] & kmask[None, :], other=0).to(tl.uint32)
        gidx = k0 // GS + tl.arange(0, NG)
        gm = rmask[:, None] & (gidx * GS < k_hi)[None, :]
        sc = tl.load(s_ptr + rows64[:, None] * (K // GS) + gidx[None, :], mask=gm, other=0.0).to(tl.float32)
        bi = tl.load(b_ptr + rows64[:, None] * (K // GS) + gidx[None, :], mask=gm, other=0.0).to(tl.float32)
        acc0 += _v3_chunk(words, sc, bi, x_ptr, ws, kmask, NG, WPG)
        acc1 += _v3_chunk(words, sc, bi, x_ptr + K, ws, kmask, NG, WPG)
        acc2 += _v3_chunk(words, sc, bi, x_ptr + 2 * K, ws, kmask, NG, WPG)
    tl.store(ws_ptr + (0 * SPLIT_K + pk) * N + rows, acc0, mask=rmask)
    tl.store(ws_ptr + (1 * SPLIT_K + pk) * N + rows, acc1, mask=rmask)
    tl.store(ws_ptr + (2 * SPLIT_K + pk) * N + rows, acc2, mask=rmask)


@triton.jit
def _gemv_affine8(x_ptr, q_ptr, s_ptr, b_ptr, ws_ptr, N, K, KCHUNK,
                  GS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    pn = tl.program_id(0)
    pk = tl.program_id(1)
    r = tl.program_id(2)
    rows = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    rows64 = rows.to(tl.int64)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    k_lo = pk * KCHUNK
    k_hi = tl.minimum(k_lo + KCHUNK, K)
    kk = tl.arange(0, BLOCK_K)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        e = k0 + kk
        km = e < k_hi
        mask2 = rmask[:, None] & km[None, :]
        xv = tl.load(x_ptr + r.to(tl.int64) * K + e, mask=km, other=0.0).to(tl.float32)
        qv = tl.load(q_ptr + rows64[:, None] * K + e[None, :], mask=mask2, other=0).to(tl.float32)
        g = e // GS
        sc = tl.load(s_ptr + rows64[:, None] * (K // GS) + g[None, :], mask=mask2, other=0.0).to(tl.float32)
        bi = tl.load(b_ptr + rows64[:, None] * (K // GS) + g[None, :], mask=mask2, other=0.0).to(tl.float32)
        p = (qv * sc + bi) * xv[None, :]
        p = tl.where(mask2, p, 0.0)
        acc += tl.sum(p, axis=1)
    tl.store(ws_ptr + (r.to(tl.int64) * SPLIT_K + pk) * N + rows, acc, mask=rmask)


@triton.jit
def _gemv_dense(x_ptr, w_ptr, ws_ptr, N, K, KCHUNK,
                BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    pn = tl.program_id(0)
    pk = tl.program_id(1)
    r = tl.program_id(2)
    rows = pn * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask = rows < N
    rows64 = rows.to(tl.int64)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    k_lo = pk * KCHUNK
    k_hi = tl.minimum(k_lo + KCHUNK, K)
    kk = tl.arange(0, BLOCK_K)
    for k0 in range(k_lo, k_hi, BLOCK_K):
        e = k0 + kk
        km = e < k_hi
        mask2 = rmask[:, None] & km[None, :]
        xv = tl.load(x_ptr + r.to(tl.int64) * K + e, mask=km, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + rows64[:, None] * K + e[None, :], mask=mask2, other=0.0).to(tl.float32)
        p = tl.where(mask2, wv * xv[None, :], 0.0)
        acc += tl.sum(p, axis=1)
    tl.store(ws_ptr + (r.to(tl.int64) * SPLIT_K + pk) * N + rows, acc, mask=rmask)


@triton.jit
def _splitk_reduce(ws_ptr, out_ptr, N, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    pn = tl.program_id(0)
    r = tl.program_id(1)
    cols = pn * BLOCK + tl.arange(0, BLOCK)
    m = cols < N
    base = r.to(tl.int64) * SPLIT_K * N
    acc = tl.load(ws_ptr + base + cols, mask=m, other=0.0)
    for j in range(1, SPLIT_K):  # k = 0..SPLIT_K-1 in order
        acc += tl.load(ws_ptr + base + j * N + cols, mask=m, other=0.0)
    tl.store(out_ptr + r.to(tl.int64) * N + cols, acc, mask=m)


def gemv(x: torch.Tensor, w, out: torch.Tensor | None = None, ws: torch.Tensor | None = None,
         cfg: dict | None = None) -> torch.Tensor:
    """x [R, K] bf16|fp32 -> [R, N] fp32. `ws` optionally supplies the split-K workspace
    (>= R*SPLIT_K*N fp32 elements) so the call allocates nothing under CUDA-graph capture.
    `cfg` is for the tuner only: it replaces tuning.config_for for this call (never a function of R)."""
    if x.dim() != 2 or x.dtype not in (torch.bfloat16, torch.float32) or not x.is_contiguous():
        raise ValueError("x must be a contiguous 2-D bf16 or fp32 tensor")
    r, k = x.shape
    if isinstance(w, AffineWeight):
        n, bits, gs = w.n, w.bits, w.gs
    elif isinstance(w, DenseWeight):
        n, bits, gs = w.n, 16, 0
    else:
        raise TypeError("w must be AffineWeight or DenseWeight")
    if k != w.k:
        raise ValueError(f"x has K={k}, weight has K={w.k}")
    cfg = dict(cfg) if cfg is not None else tuning.config_for(n, k, bits, gs)
    bn, bk, split = cfg["block_n"], cfg["block_k"], cfg["split_k"]
    if bits == 4 and bk % gs:
        raise ValueError("block_k must be a multiple of gs")
    kchunk = -(-(-(-k // bk)) // split) * bk
    kiter = -(-kchunk // bk)
    if out is None:
        out = torch.empty((r, n), dtype=torch.float32, device=x.device)
    need = r * split * n
    if ws is None:
        ws = torch.empty(need, dtype=torch.float32, device=x.device)
    elif ws.numel() < need or ws.dtype != torch.float32:
        raise ValueError("workspace too small")
    grid = (triton.cdiv(n, bn), split, r)
    kw = dict(num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    if isinstance(w, DenseWeight):
        h = _gemv_dense[grid](x, w.w, ws, n, k, kchunk, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split, **kw)
        _note("gemv_dense", h)
    elif bits == 4 and r == 3:
        h = _gemv_affine4_r3[(grid[0], split)](x, w.q, w.s, w.b, ws, n, k, kchunk, GS=gs, BLOCK_N=bn, BLOCK_K=bk,
                                               SPLIT_K=split, KITER=kiter, **kw)
        _note("gemv_affine4_r3", h)
    elif bits == 4:
        h = _gemv_affine4[grid](x, w.q, w.s, w.b, ws, n, k, kchunk,
                                GS=gs, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split, KITER=kiter, **kw)
        _note("gemv_affine4", h)
    else:
        h = _gemv_affine8[grid](x, w.q.view(torch.uint8), w.s, w.b, ws, n, k, kchunk,
                                GS=gs, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split, **kw)
        _note("gemv_affine8", h)
    h = _splitk_reduce[(triton.cdiv(n, 256), r)](ws, out, n, SPLIT_K=split, BLOCK=256)
    _note("splitk_reduce", h)
    return out
