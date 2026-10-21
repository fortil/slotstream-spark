"""Tensor-core GEMM (Triton) for the dense prefill rows: MLX affine 4-bit (gs 32/64) and bf16 dense.

y[m, n] = sum_k x[m, k] * w[n, k], fp32 accumulate. The weight tile is dequantized as q*s + b in fp32 and rounded
to bf16 once, exactly like fused_moe_affine._wtile, then goes through tl.dot. Grid (cdiv(N, BN), cdiv(M, BM)).
The tile configuration comes from tuning.gemm_config_for(n, k, bits), never from M, and the K loop runs in one fixed
order, so each output row depends only on its own x row and the weight: rows i..j of an M-row call equal a
(j - i)-row call on those rows bit for bit. M is only a mask.
"""
import torch
import triton
import triton.language as tl

from . import tuning
from .gemv_affine import AffineWeight, DenseWeight, _note

STATS = {"calls": 0, "rows": 0}


@triton.jit
def _gemm_affine4(x_ptr, q_ptr, s_ptr, b_ptr, out_ptr, M, N, K,
                  GS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mm = offs_m < M
    nm = offs_n < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        a = tl.load(x_ptr + offs_m[:, None].to(tl.int64) * K + kk[None, :], mask=mm[:, None], other=0.0)
        byte = tl.load(q_ptr + offs_n[None, :].to(tl.int64) * (K // 2) + (kk // 2)[:, None],
                       mask=nm[None, :], other=0).to(tl.int32)
        qv = ((byte >> ((kk % 2) * 4)[:, None]) & 15).to(tl.float32)
        g = kk // GS
        sc = tl.load(s_ptr + offs_n[None, :].to(tl.int64) * (K // GS) + g[:, None],
                     mask=nm[None, :], other=0.0).to(tl.float32)
        bi = tl.load(b_ptr + offs_n[None, :].to(tl.int64) * (K // GS) + g[:, None],
                     mask=nm[None, :], other=0.0).to(tl.float32)
        w = (qv * sc + bi).to(tl.bfloat16)
        acc = tl.dot(a, w, acc=acc)
    tl.store(out_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :], acc, mask=mm[:, None] & nm[None, :])


@triton.jit
def _gemm_dense(x_ptr, w_ptr, out_ptr, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    mm = offs_m < M
    nm = offs_n < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + tl.arange(0, BK)
        a = tl.load(x_ptr + offs_m[:, None].to(tl.int64) * K + kk[None, :], mask=mm[:, None], other=0.0)
        w = tl.load(w_ptr + offs_n[None, :].to(tl.int64) * K + kk[:, None], mask=nm[None, :], other=0.0)
        acc = tl.dot(a, w, acc=acc)
    tl.store(out_ptr + offs_m[:, None].to(tl.int64) * N + offs_n[None, :], acc, mask=mm[:, None] & nm[None, :])


def gemm(x: torch.Tensor, w, out: torch.Tensor | None = None, cfg: dict | None = None) -> torch.Tensor:
    """x [M, K] bf16 contiguous -> [M, N] fp32. `cfg` is for the tuner only (never a function of M)."""
    if x.dim() != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
        raise ValueError("x must be a contiguous 2-D bf16 tensor")
    m, k = x.shape
    if isinstance(w, AffineWeight):
        if w.bits != 4:
            raise ValueError(f"gemm supports 4-bit affine weights only, got bits={w.bits}")
        n, bits = w.n, 4
    elif isinstance(w, DenseWeight):
        n, bits = w.n, 16
    else:
        raise TypeError("w must be AffineWeight or DenseWeight")
    if k != w.k:
        raise ValueError(f"x has K={k}, weight has K={w.k}")
    cfg = dict(cfg) if cfg is not None else tuning.gemm_config_for(n, k, bits)
    bm, bn, bk = cfg["block_m"], cfg["block_n"], cfg["block_k"]
    if k % bk:
        raise ValueError(f"K={k} is not a multiple of block_k={bk}")
    if out is None:
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    elif (tuple(out.shape) != (m, n) or out.dtype != torch.float32 or out.device != x.device
          or not out.is_contiguous()):
        raise ValueError(f"out must be a contiguous fp32 [{m}, {n}] tensor on {x.device}, got "
                         f"{tuple(out.shape)} {out.dtype} {out.device}")
    grid = (triton.cdiv(n, bn), triton.cdiv(m, bm))
    kw = dict(num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    if bits == 16:
        h = _gemm_dense[grid](x, w.w, out, m, n, k, BM=bm, BN=bn, BK=bk, **kw)
        _note("gemm_dense", h)
    else:
        h = _gemm_affine4[grid](x, w.q.view(torch.uint8), w.s, w.b, out, m, n, k,
                                GS=w.gs, BM=bm, BN=bn, BK=bk, **kw)
        _note("gemm_affine4", h)
    STATS["calls"] += 1
    STATS["rows"] += m
    return out
