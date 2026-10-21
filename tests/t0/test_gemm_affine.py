import tomllib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slotstream_spark.kernels import affine_ref, gemm_affine, gemv_affine, tuning  # noqa: E402
from slotstream_spark.kernels.gemm_affine import gemm  # noqa: E402
from slotstream_spark.kernels.gemv_affine import AffineWeight, DenseWeight  # noqa: E402

from test_affine_kernels import AFFINE_SHAPES, _affine, _x  # noqa: E402

pytestmark = pytest.mark.cuda

OPS = tomllib.loads((Path(gemm_affine.__file__).parents[1] / "config" / "numeric_contract.toml").read_text())["ops"]
LIMIT = OPS["fused_moe_rel_l2_max"]
CONTROL = OPS["control_min_ratio"] * LIMIT
SLICE = 2048


def _rel_sliced(y, x, w, bf16_tile=False):
    """rel_l2 of y [M, N] against the fp64 reference of x @ w^T, one 2,048-column slice of the weight at a time."""
    num = den = 0.0
    for c0 in range(0, w.n, SLICE):
        c1 = min(c0 + SLICE, w.n)
        wd = affine_ref.dequant(w.q[c0:c1], w.s[c0:c1], w.b[c0:c1], w.bits, w.gs)
        if bf16_tile:
            wd = wd.to(torch.bfloat16)
        ref = affine_ref.gemv_ref64(x, wd)
        num += ((y[:, c0:c1].double() - ref) ** 2).sum().item()
        den += (ref ** 2).sum().item()
        del wd, ref
    return (num / den) ** 0.5


@pytest.mark.parametrize("n,k", AFFINE_SHAPES)
def test_affine_accuracy(n, k):
    w = _affine(n, k, 4, 64, seed=n + k + 4)
    x = _x(64, k, seed=7)
    y = gemm(x, w)
    rel = _rel_sliced(y, x, w)
    floor = _rel_sliced(y, x, w, bf16_tile=True)
    print(f"gemm {n}x{k} rel_l2 {rel:.3e} distance to the bf16-dequant reference {floor:.3e}")
    assert rel <= LIMIT


def test_affine_gs32_accuracy():
    w = _affine(2560, 2560, 4, 32, seed=11)
    x = _x(64, 2560, seed=7)
    assert _rel_sliced(gemm(x, w), x, w) <= LIMIT


def test_nibble_swap_control():
    w = _affine(2560, 2560, 4, 64, seed=5)
    x = _x(64, 2560, seed=7)
    y = gemm(x, w)
    u = affine_ref.unpack(w.q, 4)
    sw = u.view(2560, -1, 2).flip(-1).reshape(2560, -1).to(torch.float32)
    wd = sw * w.s.float().repeat_interleave(64, 1) + w.b.float().repeat_interleave(64, 1)
    ref = affine_ref.gemv_ref64(x, wd)
    assert _relerr_(y, ref) > CONTROL


def _relerr_(y, ref):
    return ((y.double() - ref).norm() / ref.norm()).item()


ROW_SHAPES = [(10240, 2560, 4), (2560, 6144, 4), (10240, 320, 4), (512, 2560, 16)]


def _weight(n, k, bits):
    if bits == 16:
        g = torch.Generator(device="cuda").manual_seed(n)
        return DenseWeight(torch.randn(n, k, device="cuda", generator=g).to(torch.bfloat16))
    return _affine(n, k, bits, 64, seed=n + k)


@pytest.mark.parametrize("n,k,bits", ROW_SHAPES)
def test_row_invariance(n, k, bits):
    w = _weight(n, k, bits)
    x = _x(4096, k, seed=3)
    big = gemm(x, w)
    assert torch.equal(big, gemm(x, w))
    for m in (1, 17, 64, 4096):
        assert torch.equal(big[:m], gemm(x[:m].contiguous(), w))


def test_dense_accuracy():
    w = _weight(512, 2560, 16)
    x = _x(64, 2560, seed=9)
    assert _relerr_(gemm(x, w), affine_ref.gemv_ref64(x, w.w)) <= LIMIT


def test_errors():
    w = _affine(2560, 2560, 4, 64, seed=1)
    with pytest.raises(ValueError):
        gemm(_x(4, 2560, seed=1), w, cfg={"block_m": 64, "block_n": 64, "block_k": 96, "num_warps": 4, "num_stages": 2})
    with pytest.raises(ValueError):
        gemm(_x(4, 2560, seed=1, dtype=torch.float32), w)
    with pytest.raises(ValueError):
        gemm(_x(4, 2048, seed=1), w)
    w8 = _affine(256, 256, 8, 64, seed=2)
    with pytest.raises(ValueError):
        gemm(_x(4, 256, seed=1), w8)


def test_out_is_validated_and_written():
    w = _affine(2560, 2560, 4, 64, seed=1)
    x = _x(8, 2560, seed=3)
    out = torch.empty((8, 2560), dtype=torch.float32, device=x.device)
    assert gemm(x, w, out=out) is out
    assert torch.equal(out, gemm(x, w))
    for bad in (torch.empty((7, 2560), dtype=torch.float32, device=x.device),
                torch.empty((8, 2560), dtype=torch.bfloat16, device=x.device),
                torch.empty((2560, 8), dtype=torch.float32, device=x.device).t(),
                torch.empty((8, 2560), dtype=torch.float32, device="cpu")):
        with pytest.raises(ValueError):
            gemm(x, w, out=bad)


def test_every_override_launches_and_fits_shared_memory():
    seen = {}
    for (n, k, bits) in tuning.GEMM_OVERRIDES:
        w = _weight(n, k, bits)
        x = _x(256, k, seed=2)
        y = gemm(x, w)
        assert y.shape == (256, n)
        seen[(n, k, bits)] = gemv_affine.SHARED_BYTES["gemm_affine4" if bits == 4 else "gemm_dense"]
        del w, y
    print("gemm shared bytes per override", seen)
    assert seen
    for shape, b in seen.items():
        assert b <= tuning.SMEM_LIMIT, (shape, b)
