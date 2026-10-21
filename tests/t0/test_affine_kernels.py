import json

import pytest

torch = pytest.importorskip("torch")

from slotstream_spark.kernels import affine_ref, gemv_affine, rows, tuning  # noqa: E402
from slotstream_spark.kernels.gemv_affine import AffineWeight, DenseWeight, gemv  # noqa: E402

pytestmark = pytest.mark.cuda

AFFINE_SHAPES = [(2560, 2560), (10240, 2560), (6144, 2560), (2560, 6144), (12288, 2560), (512, 2560),
                 (320, 10240), (10240, 320), (640, 2560), (2560, 640), (248320, 2560)]
DENSE_SHAPES = [(512, 2560), (640, 2560), (48, 2560), (1, 2560), (4, 10240)]
SMEM_LIMIT = 101_376


def _affine(n, k, bits, gs, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randint(-2**31, 2**31 - 1, (n, k * bits // 32), dtype=torch.int32, device="cuda", generator=g)
    s = (torch.randn(n, k // gs, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    b = (torch.randn(n, k // gs, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    return AffineWeight(q, s, b, bits, gs, n, k)


def _x(r, k, seed, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(r, k, device="cuda", generator=g).to(dtype)


def _relerr(y, ref64):
    return ((y.double() - ref64).norm() / ref64.norm()).item()


def test_unpack_nibble_order():
    q = torch.tensor([[0x76543210]], dtype=torch.int32)
    assert affine_ref.unpack(q, 4).tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert affine_ref.unpack(q, 8).tolist() == [[0x10, 0x32, 0x54, 0x76]]


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("n,k", AFFINE_SHAPES)
def test_affine_vs_fp64_and_row_invariance(n, k, bits):
    w = _affine(n, k, bits, 64, seed=n + k + bits)
    x = _x(3, k, seed=7)
    y = gemv(x, w)
    ref = affine_ref.gemv_ref64(x, affine_ref.dequant(w.q, w.s, w.b, bits, 64))
    assert _relerr(y, ref) <= 1e-4
    for r in range(3):
        assert torch.equal(gemv(x[r:r + 1], w)[0], y[r])


@pytest.mark.parametrize("gs", [32, 64])
def test_affine_gs_and_fp32_x(gs):
    w = _affine(2560, 2560, 4, gs, seed=3)
    x = _x(3, 2560, seed=4, dtype=torch.float32)
    y = gemv(x, w)
    ref = affine_ref.gemv_ref64(x, affine_ref.dequant(w.q, w.s, w.b, 4, gs))
    assert _relerr(y, ref) <= 1e-4
    assert torch.equal(gemv(x[1:2], w)[0], y[1])


@pytest.mark.parametrize("n,k", DENSE_SHAPES)
def test_dense_bf16(n, k):
    g = torch.Generator(device="cuda").manual_seed(n)
    wt = torch.randn(n, k, device="cuda", generator=g).to(torch.bfloat16)
    x = _x(3, k, seed=9)
    y = gemv(x, DenseWeight(wt))
    assert _relerr(y, affine_ref.gemv_ref64(x, wt)) <= 1e-4
    for r in range(3):
        assert torch.equal(gemv(x[r:r + 1], DenseWeight(wt))[0], y[r])


def _table_shapes():
    import json
    return [tuple(int(v) for v in key.split("x")) for key in json.loads(tuning.TABLE_PATH.read_text())["shapes"]]


@pytest.mark.parametrize("n,k,bits,gs", _table_shapes())
def test_frozen_table_shapes(n, k, bits, gs):
    """Every shape of tuning_gb10.json: the frozen config is valid, exact, row-invariant and within smem."""
    cfg = tuning.config_for(n, k, bits, gs)
    assert cfg == json.loads(tuning.TABLE_PATH.read_text())["shapes"][tuning.shape_key(n, k, bits, gs)]["config"]
    m = min(n, 4096)
    if bits == 16:
        g = torch.Generator(device="cuda").manual_seed(n)
        wt = torch.randn(n, k, device="cuda", generator=g).to(torch.bfloat16)
        w, deq = DenseWeight(wt), wt[:m]
        name = "gemv_dense"
    else:
        w = _affine(n, k, bits, gs, seed=n + k)
        deq = affine_ref.dequant(w.q[:m], w.s[:m], w.b[:m], bits, gs)
        name = "gemv_affine4"
    x = _x(3, k, seed=5)
    y = gemv(x, w)
    ref = affine_ref.gemv_ref64(x, deq)
    assert _relerr(y[:, :m], ref) <= 1e-4
    for r in range(3):
        assert torch.equal(gemv(x[r:r + 1], w)[0], y[r])
    assert gemv_affine.SHARED_BYTES[name] <= SMEM_LIMIT


@pytest.mark.parametrize("n,k", [(10240, 2560), (10240, 320)])
def test_r3_program_is_used_and_bit_equal(n, k):
    w = _affine(n, k, 4, 64, seed=n + k + 1)
    x = _x(5, k, seed=21)
    gemv_affine.SHARED_BYTES.pop("gemv_affine4_r3", None)
    y3 = gemv(x[:3].contiguous(), w)
    assert gemv_affine.SHARED_BYTES["gemv_affine4_r3"] <= SMEM_LIMIT
    for r in range(3):
        assert torch.equal(gemv(x[r:r + 1].contiguous(), w)[0], y3[r])
    for rr in (2, 5):  # grid-z path
        y = gemv(x[:rr].contiguous(), w)
        for r in range(rr):
            assert torch.equal(gemv(x[r:r + 1].contiguous(), w)[0], y[r])


def test_block_k_multiple_of_gs():
    w = _affine(256, 2560, 4, 64, seed=2)
    with pytest.raises(ValueError):
        gemv(_x(1, 2560, 1), w, cfg={"block_n": 16, "block_k": 32, "split_k": 1, "num_warps": 4, "num_stages": 2})


def test_config_independent_of_rows():
    # config_for takes no row count; same shape gives the same dict
    assert tuning.config_for(2560, 2560, 4, 64) == tuning.config_for(2560, 2560, 4, 64)


def _ple_rows(row_bytes, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    nib = torch.randint(0, 256, (16, 80), dtype=torch.uint8, device="cuda", generator=g)
    s = (torch.randn(16, 5, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    b = (torch.randn(16, 5, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    pad = torch.randint(0, 256, (16, row_bytes - 100), dtype=torch.uint8, device="cuda", generator=g)
    raw = torch.cat([nib, s.view(torch.uint8).reshape(16, 10), b.view(torch.uint8).reshape(16, 10), pad], dim=1)
    ref = (affine_ref.unpack(nib.view(torch.int32), 4).float().reshape(16, 160)
           * s.float().repeat_interleave(32, 1) + b.float().repeat_interleave(32, 1))
    return raw.contiguous(), ref.to(torch.bfloat16).reshape(-1)


@pytest.mark.parametrize("row_bytes", [100, 128])
def test_ple_gather(row_bytes):
    raw, ref = _ple_rows(row_bytes, 11)
    assert torch.equal(rows.ple_gather_gs32(raw, row_bytes), ref)


def test_embed_rows():
    g = torch.Generator(device="cuda").manual_seed(5)
    r = 6
    nib = torch.randint(-2**31, 2**31 - 1, (r, 320), dtype=torch.int32, device="cuda", generator=g)
    s = (torch.randn(r, 40, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    b = (torch.randn(r, 40, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    raw = torch.cat([nib.view(torch.uint8).reshape(r, 1280), s.view(torch.uint8).reshape(r, 80),
                     b.view(torch.uint8).reshape(r, 80)], dim=1).contiguous()
    ref = affine_ref.dequant(nib, s, b, 4, 64).to(torch.bfloat16)
    assert torch.equal(rows.embed_rows_affine(raw), ref)


def test_shared_memory_limit():
    for n, k in [(2560, 2560), (320, 10240)]:
        gemv(_x(1, k, 1), _affine(n, k, 4, 64, 1))
        gemv(_x(1, k, 1), _affine(n, k, 8, 64, 1))
    gemv(_x(1, 2560, 1), DenseWeight(torch.randn(48, 2560, device="cuda").to(torch.bfloat16)))
    rows.embed_rows_affine(torch.zeros(1, 1440, dtype=torch.uint8, device="cuda"))
    rows.ple_gather_gs32(torch.zeros(16, 100, dtype=torch.uint8, device="cuda"))
    assert gemv_affine.SHARED_BYTES, "no kernel reported metadata.shared"
    for name, sh in gemv_affine.SHARED_BYTES.items():
        assert sh <= SMEM_LIMIT, f"{name} uses {sh} B of shared memory"
