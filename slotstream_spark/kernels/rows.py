"""Row dequant kernels: PLE n-gram rows (gs32) and embedding rows (gs64).

Both read raw bytes that slotio fetched: nibbles first (low nibble = even element), then the
bf16 scales, then the bf16 biases. Output is q*s + b in fp32 rounded to bf16 (RNE).
"""
import torch
import triton
import triton.language as tl

from .gemv_affine import _note

PLE_VALUES = 160     # 16 rows * 160 = 2560
PLE_NIB_BYTES = 80
PLE_GROUPS = 5
EMBED_ROW_BYTES = 1440  # 1280 B U32 + 80 B scales + 80 B biases
EMBED_VALUES = 2560
EMBED_NIB_BYTES = 1280
EMBED_GROUPS = 40


@triton.jit
def _rows_dequant(rows_ptr, out_ptr, ROW_BYTES: tl.constexpr, NIB: tl.constexpr, GROUPS: tl.constexpr,
                  GS: tl.constexpr, NVALS: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    pb = tl.program_id(1)
    i = pb * BLOCK + tl.arange(0, BLOCK)
    m = i < NVALS
    base = rows_ptr + r.to(tl.int64) * ROW_BYTES
    byte = tl.load(base + i // 2, mask=m, other=0).to(tl.int32)
    q = ((byte >> ((i % 2) * 4)) & 15).to(tl.float32)
    g = i // GS
    s_lo = tl.load(base + NIB + 2 * g, mask=m, other=0).to(tl.int32)
    s_hi = tl.load(base + NIB + 2 * g + 1, mask=m, other=0).to(tl.int32)
    b_lo = tl.load(base + NIB + 2 * GROUPS + 2 * g, mask=m, other=0).to(tl.int32)
    b_hi = tl.load(base + NIB + 2 * GROUPS + 2 * g + 1, mask=m, other=0).to(tl.int32)
    s = ((s_lo | (s_hi << 8)).to(tl.uint16)).to(tl.bfloat16, bitcast=True).to(tl.float32)
    b = ((b_lo | (b_hi << 8)).to(tl.uint16)).to(tl.bfloat16, bitcast=True).to(tl.float32)
    v = q * s + b
    tl.store(out_ptr + r.to(tl.int64) * NVALS + i, v.to(tl.bfloat16), mask=m)


def _check(rows_u8, row_bytes):
    if rows_u8.dtype != torch.uint8 or rows_u8.dim() != 2 or rows_u8.shape[1] != row_bytes:
        raise ValueError(f"rows must be uint8 [R, {row_bytes}], got {tuple(rows_u8.shape)} {rows_u8.dtype}")
    if not rows_u8.is_contiguous():
        raise ValueError("rows must be contiguous")


def ple_gather_gs32(rows_u8: torch.Tensor, row_bytes: int = 100) -> torch.Tensor:
    """rows_u8 [16, row_bytes] (row_bytes in {100, 128}) -> bf16 [2560]; 128-B rows carry 28 B padding."""
    if row_bytes not in (100, 128):
        raise ValueError("row_bytes must be 100 or 128")
    _check(rows_u8, row_bytes)
    if rows_u8.shape[0] != 16:
        raise ValueError("ple_gather_gs32 takes exactly 16 rows")
    out = torch.empty(16 * PLE_VALUES, dtype=torch.bfloat16, device=rows_u8.device)
    h = _rows_dequant[(16, 1)](rows_u8, out, ROW_BYTES=row_bytes, NIB=PLE_NIB_BYTES, GROUPS=PLE_GROUPS,
                               GS=32, NVALS=PLE_VALUES, BLOCK=256)
    _note("ple_gather", h)
    return out


def embed_rows_affine(rows_u8: torch.Tensor) -> torch.Tensor:
    """rows_u8 [R, 1440] -> bf16 [R, 2560] (gs64: 1280 B nibbles, 80 B scales, 80 B biases)."""
    _check(rows_u8, EMBED_ROW_BYTES)
    r = rows_u8.shape[0]
    out = torch.empty((r, EMBED_VALUES), dtype=torch.bfloat16, device=rows_u8.device)
    h = _rows_dequant[(r, triton.cdiv(EMBED_VALUES, 1024))](
        rows_u8, out, ROW_BYTES=EMBED_ROW_BYTES, NIB=EMBED_NIB_BYTES, GROUPS=EMBED_GROUPS,
        GS=64, NVALS=EMBED_VALUES, BLOCK=1024)
    _note("embed_rows", h)
    return out
