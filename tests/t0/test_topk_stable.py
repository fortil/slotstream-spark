import pytest

torch = pytest.importorskip("torch")

from slotstream_spark.kernels import gemv_affine  # noqa: E402
from slotstream_spark.kernels.topk_stable import topk_stable  # noqa: E402

pytestmark = pytest.mark.cuda

NS = [512, 513, 2048, 8192, 16384]
KS = [10, 512]


def _keys(x):
    """Order-preserving key in int64 (NaN -> 0), computed on the CPU from the raw bits."""
    b = x.cpu().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    key = torch.where(b >= 0x80000000, (~b) & 0xFFFFFFFF, b ^ 0x80000000)
    nan = (b & 0x7FFFFFFF) > 0x7F800000
    return torch.where(nan, torch.zeros_like(key), key)


def _ref_row(x, k, nv):
    key = _keys(x[:nv])
    order = torch.sort(-key, stable=True).indices[:k]   # (-key, index): ties keep the lower index
    if k <= 16:
        sel = order
    else:
        sel = torch.sort(order).values
    idx = torch.full((k,), -1, dtype=torch.int64)
    idx[:sel.numel()] = sel
    return idx


def _check(x, k, n_valid=None):
    x = x.contiguous().cuda()
    nvt = None if n_valid is None else torch.tensor(n_valid, dtype=torch.int32, device="cuda")
    out_val = torch.empty((x.shape[0], k), dtype=torch.float32, device="cuda")
    idx, val = topk_stable(x, k, nvt, out_val=out_val)
    torch.cuda.synchronize()
    for r in range(x.shape[0]):
        nv = x.shape[1] if n_valid is None else min(n_valid[r], x.shape[1])
        ref = _ref_row(x[r].cpu(), k, nv)
        assert idx[r].cpu().tolist() == ref.tolist(), f"row {r}"
        ok = ref >= 0
        got = val[r].cpu()
        want = x[r].cpu()[ref[ok]]
        assert torch.equal(got[ok].view(torch.int32), want.view(torch.int32))
        assert bool((got[~ok] == float("-inf")).all())
    return idx


def _rand(r, n, seed, levels=None):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(r, n, generator=g)
    if levels:
        x = torch.round(x * levels) / levels   # many exact ties
    return x


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("k", KS)
def test_random_and_ties(n, k):
    _check(_rand(3, n, n + k), k)
    _check(_rand(3, n, n * 3 + k, levels=2), k)


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("k", KS)
def test_all_equal_selects_lowest_indices(n, k):
    x = torch.full((2, n), 1.5)
    idx = _check(x, k)
    assert idx[0].cpu().tolist() == list(range(k))


def test_special_values_placement():
    n = 512
    x = _rand(4, n, 1)
    x[0, :20] = float("nan")
    x[1, 5:15] = float("-inf")
    x[1, 100:104] = float("inf")
    x[2, ::2] = 0.0
    x[2, 1::2] = -0.0
    x[3, :] = float("nan")
    for k in (10, 512):
        _check(x, k)
    idx = topk_stable(x.cuda(), 10)[0].cpu()
    assert not set(idx[0].tolist()) & set(range(20))       # NaN is lowest, never beats a number
    assert idx[1, :4].tolist() == [100, 101, 102, 103]     # +inf first, by index
    assert idx[3].tolist() == list(range(10))              # all NaN: lowest indices
    # +0.0 (even positions) outranks -0.0 (odd positions)
    assert idx[2].tolist() == list(range(0, 20, 2))


@pytest.mark.parametrize("k", KS)
def test_n_valid_masking(k):
    n = 2048
    x = _rand(5, n, 11, levels=4)
    x[:, 700:] = 1e9    # masked-out entries must never be selected
    nv = [2048, 700, 513, k, 3]
    idx = _check(x, k, n_valid=nv)
    for r, v in enumerate(nv):
        got = [i for i in idx[r].cpu().tolist() if i >= 0]
        assert all(i < v for i in got)
        if v <= k:
            assert sorted(got) == list(range(v))     # no-sparsity regime: every valid index


@pytest.mark.parametrize("k", KS)
def test_row_invariance(k):
    x = _rand(3, 8192, 21, levels=3).cuda()
    both = topk_stable(x, k)[0]
    for r in range(3):
        assert torch.equal(topk_stable(x[r:r + 1].contiguous(), k)[0][0], both[r])


def test_shared_memory_limit():
    topk_stable(_rand(1, 16384, 5).cuda(), 512)
    assert gemv_affine.SHARED_BYTES["topk_stable"] <= 101_376
