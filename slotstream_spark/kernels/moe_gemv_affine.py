# SPDX-License-Identifier: MIT AND Apache-2.0
# Portions ported from helgard-orlm/qwen-flash-next-8gb@958d22e7ba0317750ccc38862b9667b1034a4dd1 qmoe.py:24-83 (Apache-2.0, section 4); modified.
"""MoE expert GEMVs for one token step, MLX affine 4-bit gs64 (q*s + b, fp32 accumulate).

Each expert lives in a 2,764,800 B record inside a pool slot. The kernels never see a slot
index from the host: they read `slot = map[layer*512 + e]` and `base = slot_ptr[slot]` from
device tables, so a CUDA graph depends only on routing. The per-row sum over the 10 routed
experts is its own kernel and adds in rank order (no torch.sum, no atomics), so a 3-row verify
equals three 1-row calls bit for bit and a record can move between slots without changing bits.

Record layout (bytes): gate.w 0, gate.s 819,200, gate.b 870,400, up.w 921,600, up.s 1,740,800,
up.b 1,792,000, down.w 1,843,200, down.s 2,662,400, down.b 2,713,600.
"""
import torch
import triton
import triton.language as tl

from . import tuning
from .gemv_affine import _note

HID, INTER, TOPK = 2560, 640, 10
RECORD_BYTES = 2_764_800
O_GATE_W, O_GATE_S, O_GATE_B = 0, 819_200, 870_400
O_UP_W, O_UP_S, O_UP_B = 921_600, 1_740_800, 1_792_000
O_DOWN_W, O_DOWN_S, O_DOWN_B = 1_843_200, 2_662_400, 2_713_600
GS = 64
EXPERTS_PER_LAYER = 512


@triton.jit
def _tile4(base, w_off, s_off, b_off, rows, rmask, k0, K, x_ptr, BK: tl.constexpr, GSZ: tl.constexpr):
    """Partial dot of BN rows x one chunk of BK elements: word-wise nibbles, one scale and bias per group.
    The K tail is masked in whole groups (K % GSZ == 0), so a masked word adds exact zeros."""
    wp = (base + w_off).to(tl.pointer_type(tl.uint32))
    sp = (base + s_off).to(tl.pointer_type(tl.bfloat16))
    bp = (base + b_off).to(tl.pointer_type(tl.bfloat16))
    NG: tl.constexpr = BK // GSZ
    ws = k0 // 8 + tl.arange(0, BK // 8)
    kmask = ws * 8 < K
    words = tl.load(wp + rows[:, None] * (K // 8) + ws[None, :], mask=rmask[:, None] & kmask[None, :], other=0)
    gidx = k0 // GSZ + tl.arange(0, NG)
    gm = rmask[:, None] & (gidx * GSZ < K)[None, :]
    sc = tl.load(sp + rows[:, None] * (K // GSZ) + gidx[None, :], mask=gm, other=0.0).to(tl.float32)
    bi = tl.load(bp + rows[:, None] * (K // GSZ) + gidx[None, :], mask=gm, other=0.0).to(tl.float32)
    gq = tl.zeros_like(sc)
    gx = tl.zeros([NG], dtype=tl.float32)
    for j in tl.static_range(8):
        xj = tl.load(x_ptr + ws * 8 + j, mask=kmask, other=0.0).to(tl.float32)
        qj = ((words >> (4 * j)) & 0xF).to(tl.float32)
        gq += tl.sum(tl.reshape(qj * xj[None, :], [sc.shape[0], NG, GSZ // 8]), axis=2)
        gx += tl.sum(tl.reshape(xj, [NG, GSZ // 8]), axis=1)
    return tl.sum(sc * gq + bi * gx[None, :], axis=1)


@triton.jit
def _gate_up(x_ptr, ids_ptr, map_ptr, slot_ptr, h_ptr, layer,
             O_GW: tl.constexpr, O_GS: tl.constexpr, O_GB: tl.constexpr,
             O_UW: tl.constexpr, O_US: tl.constexpr, O_UB: tl.constexpr,
             HIDDEN: tl.constexpr, INT: tl.constexpr, TK: tl.constexpr,
             BN: tl.constexpr, BK: tl.constexpr, KITER: tl.constexpr):
    pn = tl.program_id(0)
    j = tl.program_id(1)           # row * TOPK + rank
    r = j // TK
    e = tl.load(ids_ptr + j)
    slot = tl.load(map_ptr + layer * 512 + e).to(tl.int64)
    base = tl.load(slot_ptr + slot)
    rows = pn * BN + tl.arange(0, BN)
    rmask = rows < INT
    xr = x_ptr + r.to(tl.int64) * HIDDEN
    accg = tl.zeros([BN], dtype=tl.float32)
    accu = tl.zeros([BN], dtype=tl.float32)
    for it in tl.static_range(KITER):
        k0 = it * BK
        accg += _tile4(base, O_GW, O_GS, O_GB, rows, rmask, k0, HIDDEN, xr, BK, 64)
        accu += _tile4(base, O_UW, O_US, O_UB, rows, rmask, k0, HIDDEN, xr, BK, 64)
    h = accg / (1.0 + tl.exp(-accg)) * accu
    tl.store(h_ptr + j.to(tl.int64) * INT + rows, h, mask=rmask)


@triton.jit
def _down(h_ptr, ids_ptr, w_ptr, map_ptr, slot_ptr, o_ptr, layer,
          O_DW: tl.constexpr, O_DS: tl.constexpr, O_DB: tl.constexpr,
          HIDDEN: tl.constexpr, INT: tl.constexpr,
          BR: tl.constexpr, BKD: tl.constexpr, KITER_D: tl.constexpr):
    pn = tl.program_id(0)
    j = tl.program_id(1)
    e = tl.load(ids_ptr + j)
    slot = tl.load(map_ptr + layer * 512 + e).to(tl.int64)
    base = tl.load(slot_ptr + slot)
    rows = pn * BR + tl.arange(0, BR)
    rmask = rows < HIDDEN
    hr = h_ptr + j.to(tl.int64) * INT
    acc = tl.zeros([BR], dtype=tl.float32)
    for it in tl.static_range(KITER_D):
        acc += _tile4(base, O_DW, O_DS, O_DB, rows, rmask, it * BKD, INT, hr, BKD, 64)
    wt = tl.load(w_ptr + j)
    tl.store(o_ptr + j.to(tl.int64) * HIDDEN + rows, acc * wt, mask=rmask)


@triton.jit
def _rank_sum(o_ptr, out_ptr, HIDDEN: tl.constexpr, TK: tl.constexpr, BLOCK: tl.constexpr):
    pn = tl.program_id(0)
    r = tl.program_id(1)
    cols = pn * BLOCK + tl.arange(0, BLOCK)
    m = cols < HIDDEN
    base = o_ptr + r.to(tl.int64) * TK * HIDDEN
    acc = tl.load(base + cols, mask=m, other=0.0)
    for k in tl.static_range(1, TK):
        acc = acc + tl.load(base + k * HIDDEN + cols, mask=m, other=0.0)   # rank order, sequential fp32
    tl.store(out_ptr + r.to(tl.int64) * HIDDEN + cols, acc, mask=m)


def moe_decode(x, expert_ids, weights, layer: int, map_t, slot_ptr_t, h_buf, o_buf, out, cfg=None):
    """x [R,2560] bf16|fp32 · expert_ids int32 [R,10] (device) · weights fp32 [R,10] ·
    map_t int32 [49*512] · slot_ptr_t int64 [<=25088] · h_buf [R,10,640] f32 ·
    o_buf [R,10,2560] f32 · out [R,2560] f32 -> out."""
    r = x.shape[0]
    if x.dim() != 2 or x.shape[1] != HID or not x.is_contiguous():
        raise ValueError("x must be a contiguous [R, 2560] tensor")
    if tuple(expert_ids.shape) != (r, TOPK) or expert_ids.dtype != torch.int32 or not expert_ids.is_contiguous():
        raise ValueError("expert_ids must be contiguous int32 [R, 10]")
    if tuple(weights.shape) != (r, TOPK) or weights.dtype != torch.float32 or not weights.is_contiguous():
        raise ValueError("weights must be contiguous fp32 [R, 10]")
    if map_t.dtype != torch.int32 or slot_ptr_t.dtype != torch.int64:
        raise ValueError("map_t must be int32 and slot_ptr_t int64")
    for t, shape in ((h_buf, (r, TOPK, INTER)), (o_buf, (r, TOPK, HID)), (out, (r, HID))):
        if tuple(t.shape) != shape or t.dtype != torch.float32 or not t.is_contiguous():
            raise ValueError(f"buffer must be contiguous fp32 {shape}")
    c = dict(cfg) if cfg is not None else tuning.moe_config()
    c.setdefault("bkd", tuning.MOE_DEFAULT["bkd"])
    if c["bk"] % 64 or c["bkd"] % 64:
        raise ValueError("bk and bkd must be multiples of 64")
    n = r * TOPK
    hd = _gate_up[(INTER // c["bn"], n)](x, expert_ids, map_t, slot_ptr_t, h_buf, layer,
                                    O_GW=O_GATE_W, O_GS=O_GATE_S, O_GB=O_GATE_B,
                                    O_UW=O_UP_W, O_US=O_UP_S, O_UB=O_UP_B,
                                    HIDDEN=HID, INT=INTER, TK=TOPK, BN=c["bn"], BK=c["bk"], KITER=-(-HID // c["bk"]),
                                    num_warps=c["num_warps"])
    _note("moe_gate_up", hd)
    hd = _down[(HID // c["br"], n)](h_buf, expert_ids, weights, map_t, slot_ptr_t, o_buf, layer,
                               O_DW=O_DOWN_W, O_DS=O_DOWN_S, O_DB=O_DOWN_B,
                               HIDDEN=HID, INT=INTER, BR=c["br"], BKD=c["bkd"], KITER_D=-(-INTER // c["bkd"]),
                               num_warps=c["num_warps"])
    _note("moe_down", hd)
    hd = _rank_sum[(triton.cdiv(HID, 256), r)](o_buf, out, HIDDEN=HID, TK=TOPK, BLOCK=256)
    _note("moe_rank_sum", hd)
    return out
