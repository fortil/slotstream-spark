"""Reference math for MLX affine quantization (q*s + b), in torch.

Layout (MLX): a row of K values with `bits` bits each is a little-endian bit stream packed in
int32 words; element i sits at bits [bits*i, bits*(i+1)), so the low nibble holds the even
element for 4 bits. Scales and biases are bf16, one pair per group of `gs` elements.
"""
import torch


def unpack(q_u32: torch.Tensor, bits: int) -> torch.Tensor:
    """int32 [N, K*bits/32] -> uint8 [N, K], element i at bits [bits*i, bits*(i+1))."""
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    b = q_u32.contiguous().view(torch.uint8)  # little-endian bytes
    if bits == 8:
        return b.clone()
    lo = b & 15
    hi = b >> 4
    return torch.stack([lo, hi], dim=-1).reshape(b.shape[0], -1)


def dequant(q, s_bf16, b_bf16, bits: int, gs: int) -> torch.Tensor:
    """q int32 [N, K*bits/32], s/b bf16 [N, K/gs] -> fp32 [N, K] = q*s + b."""
    u = unpack(q, bits).to(torch.float32)
    n, k = u.shape
    if k % gs or s_bf16.shape != (n, k // gs) or b_bf16.shape != (n, k // gs):
        raise ValueError(f"shape mismatch: q {tuple(u.shape)} s {tuple(s_bf16.shape)} gs {gs}")
    s = s_bf16.to(torch.float32).repeat_interleave(gs, dim=1)
    b = b_bf16.to(torch.float32).repeat_interleave(gs, dim=1)
    return u * s + b


def gemv_ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x [R, K], w [N, K] (already dequantized or bf16) -> fp32 [R, N], accumulated in fp64."""
    return (x.to(torch.float64) @ w.to(torch.float64).T).to(torch.float32)


def gemv_ref64(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Same as gemv_ref but keeps the fp64 result (for error measurement)."""
    return x.to(torch.float64) @ w.to(torch.float64).T
