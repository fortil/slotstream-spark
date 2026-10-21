# SPDX-License-Identifier: Apache-2.0
# Portions derived from vllm/model_executor/layers/fused_moe/fused_moe.py, function `fused_moe_kernel_gptq_awq`
# (lines 64-296), vLLM v0.28.0 image file (sha256 in third_party/vendor/vllm-187a0eb9/fused_moe/ORIGIN.md);
# Copyright contributors to the vLLM project, Apache-2.0; modified by slotstream-spark: MLX affine 4-bit
# gs64 dequant (q*s + b, element i in bits 4i of each u32), expert records addressed through a pointer table,
# gate/up fused with the SiLU product, no atomics, fixed tiles, routed weight applied in the down launch.
"""Grouped affine 4-bit MoE for prefill: tensor-core `tl.dot` over dequantized bf16 tiles, fp32 accumulate.

One group = at most 32 experts whose 2,764,800-B records sit at `group_ptr[g]` (a pool slot or a staging
slot; the kernel cannot tell). `expert_map[e]` is the local index g or -1 (rows of such experts are skipped,
vLLM expert_map semantics). Two launches: gate_up writes h = silu(x Wg^T) * (x Wu^T) as bf16 [R,640];
down writes y = (h Wd^T) * w as fp32 [R,2560] into `out_rows`.

Rounding path (the floor behind `[ops] fused_moe_rel_l2_max`): the dequantized gate, up and down tiles are cast
to bf16 for `tl.dot`, and h is stored bf16: four bf16 roundings. Tiles, warps and stages are constants of the
shape class, never functions of R or of residency, so a row's bits depend on that row and that record only.

Record layout (bytes): gate.w 0, gate.s 819,200, gate.b 870,400, up.w 921,600, up.s 1,740,800,
up.b 1,792,000, down.w 1,843,200, down.s 2,662,400, down.b 2,713,600.
"""
import torch
import triton
import triton.language as tl

from .gemv_affine import _note
from .moe_align import moe_align

HID, INTER, TOPK = 2560, 640, 10
RECORD_BYTES = 2_764_800
O_GATE_W, O_GATE_S, O_GATE_B = 0, 819_200, 870_400
O_UP_W, O_UP_S, O_UP_B = 921_600, 1_740_800, 1_792_000
O_DOWN_W, O_DOWN_S, O_DOWN_B = 1_843_200, 2_662_400, 2_713_600
GROUP_MAX = 32
# Fixed shape class (prefill): independent of R and of residency.
BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS, NUM_STAGES = 64, 64, 64, 4, 2


@triton.jit
def _wtile(base, w_off, s_off, b_off, offs_n, k0, K: tl.constexpr, BK: tl.constexpr):
    """Dequantized bf16 tile [BK, BN] of rows offs_n, columns k0..k0+BK: q*s + b in fp32, rounded to bf16."""
    kk = k0 + tl.arange(0, BK)
    wp = (base + w_off).to(tl.pointer_type(tl.uint8))
    sp = (base + s_off).to(tl.pointer_type(tl.bfloat16))
    bp = (base + b_off).to(tl.pointer_type(tl.bfloat16))
    byte = tl.load(wp + offs_n[None, :] * (K // 2) + (kk // 2)[:, None]).to(tl.int32)
    q = ((byte >> ((kk % 2) * 4)[:, None]) & 15).to(tl.float32)
    g = kk // 64
    sc = tl.load(sp + offs_n[None, :] * (K // 64) + g[:, None]).to(tl.float32)
    bi = tl.load(bp + offs_n[None, :] * (K // 64) + g[:, None]).to(tl.float32)
    return (q * sc + bi).to(tl.bfloat16)


@triton.jit
def _gate_up_kernel(x_ptr, row_token_ptr, sorted_ptr, block_expert_ptr, num_padded_ptr, group_ptr, h_ptr, R,
                    O_GW: tl.constexpr, O_GS: tl.constexpr, O_GB: tl.constexpr,
                    O_UW: tl.constexpr, O_US: tl.constexpr, O_UB: tl.constexpr,
                    HIDDEN: tl.constexpr, INT: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    if pid_m * BM >= tl.load(num_padded_ptr):
        return
    e = tl.load(block_expert_ptr + pid_m)
    if e < 0:
        return
    offs_m = pid_m * BM + tl.arange(0, BM)
    row = tl.load(sorted_ptr + offs_m).to(tl.int64)
    rmask = row < R
    tok = tl.load(row_token_ptr + row, mask=rmask, other=0).to(tl.int64)
    base = tl.load(group_ptr + e.to(tl.int64))
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = x_ptr + tok[:, None] * HIDDEN + offs_k[None, :]
    accg = tl.zeros((BM, BN), dtype=tl.float32)
    accu = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, HIDDEN, BK):
        a = tl.load(a_ptrs + k0, mask=rmask[:, None], other=0.0)
        accg = tl.dot(a, _wtile(base, O_GW, O_GS, O_GB, offs_n, k0, HIDDEN, BK), acc=accg)
        accu = tl.dot(a, _wtile(base, O_UW, O_US, O_UB, offs_n, k0, HIDDEN, BK), acc=accu)
    h = accg / (1.0 + tl.exp(-accg)) * accu
    tl.store(h_ptr + row[:, None] * INT + offs_n[None, :], h.to(tl.bfloat16), mask=rmask[:, None])


@triton.jit
def _down_kernel(h_ptr, row_weight_ptr, sorted_ptr, block_expert_ptr, num_padded_ptr, group_ptr, out_ptr, R,
                 O_DW: tl.constexpr, O_DS: tl.constexpr, O_DB: tl.constexpr,
                 HIDDEN: tl.constexpr, INT: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    if pid_m * BM >= tl.load(num_padded_ptr):
        return
    e = tl.load(block_expert_ptr + pid_m)
    if e < 0:
        return
    offs_m = pid_m * BM + tl.arange(0, BM)
    row = tl.load(sorted_ptr + offs_m).to(tl.int64)
    rmask = row < R
    base = tl.load(group_ptr + e.to(tl.int64))
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = h_ptr + row[:, None] * INT + offs_k[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, INT, BK):
        a = tl.load(a_ptrs + k0, mask=rmask[:, None], other=0.0)
        acc = tl.dot(a, _wtile(base, O_DW, O_DS, O_DB, offs_n, k0, INT, BK), acc=acc)
    wt = tl.load(row_weight_ptr + row, mask=rmask, other=0.0)
    tl.store(out_ptr + row[:, None] * HIDDEN + offs_n[None, :], acc * wt[:, None], mask=rmask[:, None])


def launch_gate_up(x, row_token, aligned, group_ptr, h_buf, r):
    sorted_rows, block_expert, num_padded = aligned
    grid = (INTER // BLOCK_N, block_expert.numel())
    hd = _gate_up_kernel[grid](x, row_token, sorted_rows, block_expert, num_padded, group_ptr, h_buf, r,
                               O_GW=O_GATE_W, O_GS=O_GATE_S, O_GB=O_GATE_B, O_UW=O_UP_W, O_US=O_UP_S, O_UB=O_UP_B,
                               HIDDEN=HID, INT=INTER, BM=BLOCK_M, BN=BLOCK_N, BK=BLOCK_K,
                               num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    _note("fused_moe_gate_up", hd)


def launch_down(h_buf, row_weight, aligned, group_ptr, out_rows, r):
    sorted_rows, block_expert, num_padded = aligned
    grid = (HID // BLOCK_N, block_expert.numel())
    hd = _down_kernel[grid](h_buf, row_weight, sorted_rows, block_expert, num_padded, group_ptr, out_rows, r,
                            O_DW=O_DOWN_W, O_DS=O_DOWN_S, O_DB=O_DOWN_B, HIDDEN=HID, INT=INTER,
                            BM=BLOCK_M, BN=BLOCK_N, BK=BLOCK_K, num_warps=NUM_WARPS, num_stages=NUM_STAGES)
    _note("fused_moe_down", hd)


def align_rows(row_expert, expert_map, num_local):
    """Local expert of every row through `expert_map` (-1 outside the group), aligned to BLOCK_M."""
    local = expert_map[row_expert.long()].to(torch.int32)
    return moe_align(local, num_local, BLOCK_M)


def fused_moe_affine(x, row_token, row_weight, expert_map, group_ptr, out_rows, *, row_expert, h_buf=None, aligned=None):
    """x bf16 [T,2560] · row_token int32 [R] · row_weight fp32 [R] · expert_map int32 [512] (local index or -1) ·
    group_ptr int64 [G<=32] record base addresses · out_rows fp32 [R,2560] (rows outside the group are left alone) ->
    None. `row_expert` int32 [R] (keyword) is the global expert of each row, needed to align rows by expert;
    `aligned` may carry a precomputed `align_rows` result and `h_buf` a bf16 [R,640] buffer."""
    r = row_token.numel()
    if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != HID or not x.is_contiguous():
        raise ValueError("x must be a contiguous bf16 [T, 2560] tensor")
    if row_token.dtype != torch.int32 or row_weight.dtype != torch.float32 or row_weight.numel() != r:
        raise ValueError("row_token must be int32 [R] and row_weight fp32 [R]")
    if row_expert.dtype != torch.int32 or row_expert.numel() != r:
        raise ValueError("row_expert must be int32 [R]")
    if expert_map.dtype != torch.int32 or group_ptr.dtype != torch.int64:
        raise ValueError("expert_map must be int32 and group_ptr int64")
    if not 1 <= group_ptr.numel() <= GROUP_MAX:
        raise ValueError(f"a group holds 1..{GROUP_MAX} experts")
    if tuple(out_rows.shape) != (r, HID) or out_rows.dtype != torch.float32 or not out_rows.is_contiguous():
        raise ValueError("out_rows must be contiguous fp32 [R, 2560]")
    if h_buf is None:
        h_buf = torch.empty((r, INTER), dtype=torch.bfloat16, device=x.device)
    elif tuple(h_buf.shape) != (r, INTER) or h_buf.dtype != torch.bfloat16:
        raise ValueError("h_buf must be bf16 [R, 640]")
    if aligned is None:
        aligned = align_rows(row_expert, expert_map, group_ptr.numel())
    launch_gate_up(x, row_token, aligned, group_ptr, h_buf, r)
    launch_down(h_buf, row_weight, aligned, group_ptr, out_rows, r)


def combine_rank_order(out_rows, T, K=TOPK):
    """out_rows fp32 [T*K, 2560], row = token*K + rank -> fp32 [T, 2560], summed in router rank order (sequential fp32)."""
    if out_rows.shape[0] != T * K:
        raise ValueError("out_rows must have T*K rows")
    v = out_rows.view(T, K, -1)
    acc = v[:, 0].clone()
    for k in range(1, K):
        acc = acc + v[:, k]
    return acc
