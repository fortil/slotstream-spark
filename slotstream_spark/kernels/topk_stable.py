# SPDX-License-Identifier: MIT
# Port of BlockSelection.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""Stable top-k for the router (k = 10) and the QSA indexer (k = 512), ties broken by lower index.

Same selection as BlockSelection.swift:9-75 (the Metal kernel): scores map to order-preserving
uint32 keys (positive: flip the sign bit; negative: bitwise not; NaN: key 0, lowest), a binary
search over the 32 key bits finds the k-th largest key, and ties at that key are taken in
original-index order. The subnormal flush of the Metal comparator is not copied: subnormals keep
their own keys, and -0.0 sorts just below +0.0 like any other pair of distinct bit patterns.

Output order: k <= 16 sorted by (value desc, index asc); larger k sorted by index asc. A row with
fewer than k valid entries (n_valid <= k, the QSA no-sparsity regime) selects every valid index
and pads the rest with index -1 and value -inf.
"""
import torch
import triton
import triton.language as tl

from .gemv_affine import _note

# Rows up to 65,536 columns: the indexer compresses 4 positions into one block, so this covers the model's 262,144
# positions. One program per row holds the whole row; at 32,768 columns and above it runs with 32 warps. Measured on
# the GB10 with 256 rows and k = 512 (tools/probes/topk_wide.py, 2026-10-21): 0.13 ms at 16,384 columns, 1.3 ms at
# 32,768, 5.6 ms at 65,536, 65,536 B of shared memory, every row equal to the CPU reference.
MAX_N = 65536


@triton.jit
def _topk_stable(scores_ptr, nvalid_ptr, idx_ptr, val_ptr, N, K,
                 HAS_NV: tl.constexpr, HAS_VAL: tl.constexpr, SORT_BY_VALUE: tl.constexpr,
                 NP: tl.constexpr, KP: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, NP)
    nv = N
    if HAS_NV:
        nv = tl.minimum(tl.load(nvalid_ptr + row), N)
    valid = offs < nv
    x = tl.load(scores_ptr + row * N + offs, mask=valid, other=0.0)
    bits = x.to(tl.int32, bitcast=True)
    ubits = bits.to(tl.uint32, bitcast=True)
    is_nan = (bits & 0x7FFFFFFF) > 0x7F800000
    key = tl.where(bits < 0, ~ubits, ubits ^ 0x80000000)
    key = tl.where(is_nan, 0, key).to(tl.uint32)

    thr = tl.zeros([], dtype=tl.uint32)
    for i in tl.static_range(32):
        trial = thr | (tl.full([], 1, tl.uint32) << (31 - i))
        cnt = tl.sum((valid & (key >= trial)).to(tl.int32), axis=0)
        thr = tl.where(cnt >= K, trial, thr)

    greater = valid & (key > thr)
    eq = valid & (key == thr)
    n_greater = tl.sum(greater.to(tl.int32), axis=0)
    budget = K - n_greater
    tie_rank = tl.cumsum(eq.to(tl.int32), axis=0)
    sel = greater | (eq & (tie_rank <= budget))
    count = tl.sum(sel.to(tl.int32), axis=0)

    kk = tl.arange(0, KP)
    obase = row * K
    if SORT_BY_VALUE:
        rem = sel
        for j in range(0, K):
            mk = tl.max(tl.where(rem, key, 0).to(tl.uint32), axis=0)
            cand = rem & (key == mk)
            idx = tl.min(tl.where(cand, offs, NP), axis=0)
            ok = idx < NP
            pos = tl.where(ok, idx, 0)
            tl.store(idx_ptr + obase + j, tl.where(ok, idx, -1))
            if HAS_VAL:
                v = tl.load(scores_ptr + row * N + pos)
                tl.store(val_ptr + obase + j, tl.where(ok, v, float("-inf")))
            rem = rem & (offs != idx)
    else:
        pos = tl.cumsum(sel.to(tl.int32), axis=0) - 1
        tl.store(idx_ptr + obase + pos, offs, mask=sel)
        if HAS_VAL:
            tl.store(val_ptr + obase + pos, x, mask=sel)
        pad = (kk >= count) & (kk < K)
        tl.store(idx_ptr + obase + kk, -1, mask=pad)
        if HAS_VAL:
            tl.store(val_ptr + obase + kk, float("-inf"), mask=pad)


def topk_stable(scores: torch.Tensor, k: int, n_valid: torch.Tensor | None = None,
                out_idx: torch.Tensor | None = None, out_val: torch.Tensor | None = None):
    """scores f32 [R, N] (N <= MAX_N) -> (out_idx int32 [R, k], out_val f32 [R, k] or None).

    `out_val` is filled only when a tensor is passed. `n_valid` (int32 [R], device) is read inside
    the kernel, so the call is CUDA-graph capturable."""
    if scores.dim() != 2 or scores.dtype != torch.float32 or not scores.is_contiguous():
        raise ValueError("scores must be a contiguous fp32 [R, N] tensor")
    r, n = scores.shape
    if not 1 <= k <= n or n > MAX_N:
        raise ValueError(f"need 1 <= k <= N <= {MAX_N}, got k={k} N={n}")
    if n_valid is not None and (n_valid.dtype != torch.int32 or tuple(n_valid.shape) != (r,)):
        raise ValueError("n_valid must be int32 [R]")
    if out_idx is None:
        out_idx = torch.empty((r, k), dtype=torch.int32, device=scores.device)
    if out_idx.dtype != torch.int32 or tuple(out_idx.shape) != (r, k) or not out_idx.is_contiguous():
        raise ValueError("out_idx must be contiguous int32 [R, k]")
    if out_val is not None and (out_val.dtype != torch.float32 or tuple(out_val.shape) != (r, k)
                                or not out_val.is_contiguous()):
        raise ValueError("out_val must be contiguous fp32 [R, k]")
    np_ = max(triton.next_power_of_2(n), 16)
    h = _topk_stable[(r,)](
        scores, n_valid if n_valid is not None else scores, out_idx,
        out_val if out_val is not None else scores, n, k,
        HAS_NV=n_valid is not None, HAS_VAL=out_val is not None, SORT_BY_VALUE=k <= 16,
        NP=np_, KP=max(triton.next_power_of_2(k), 16),
        num_warps=32 if np_ >= 32768 else (16 if np_ >= 8192 else 8))   # 32 elements per thread from 16,384 up
    _note("topk_stable", h)
    return out_idx, out_val
