import pytest

torch = pytest.importorskip("torch")

from slotstream_spark.kernels import affine_ref, gemv_affine, tuning  # noqa: E402
from slotstream_spark.kernels import moe_gemv_affine as moe  # noqa: E402

pytestmark = pytest.mark.cuda

REC = moe.RECORD_BYTES
LAYER = 3
MAP_SIZE = 49 * 512


def _make_record(seed):
    """One synthetic expert record as bytes (uint8 CPU tensor) plus its fp32 dequantized weights."""
    g = torch.Generator().manual_seed(seed)

    def piece(n, k):
        q = torch.randint(-2**31, 2**31 - 1, (n, k // 8), dtype=torch.int32, generator=g)
        s = (torch.randn(n, k // 64, generator=g) * 0.02).to(torch.bfloat16)
        b = (torch.randn(n, k // 64, generator=g) * 0.02).to(torch.bfloat16)
        return q, s, b

    gate, up, down = piece(640, 2560), piece(640, 2560), piece(2560, 640)
    rec = torch.zeros(REC, dtype=torch.uint8)
    for (q, s, b), (ow, os_, ob) in ((gate, (moe.O_GATE_W, moe.O_GATE_S, moe.O_GATE_B)),
                                     (up, (moe.O_UP_W, moe.O_UP_S, moe.O_UP_B)),
                                     (down, (moe.O_DOWN_W, moe.O_DOWN_S, moe.O_DOWN_B))):
        for t, off in ((q, ow), (s, os_), (b, ob)):
            raw = t.contiguous().view(torch.uint8).reshape(-1)
            rec[off:off + raw.numel()] = raw
    deq = [affine_ref.dequant(q, s, b, 4, 64) for (q, s, b) in (gate, up, down)]
    return rec, deq


def _pool(records, slots, n_slots):
    """Pinned host pool of n_slots records; records[i] sits in slot slots[i]."""
    pool = torch.zeros((n_slots, REC), dtype=torch.uint8).pin_memory()
    for rec, slot in zip(records, slots):
        pool[slot] = rec
    return pool


def _tables(pool, experts, slots):
    n_slots = pool.shape[0]
    mp = torch.full((MAP_SIZE,), -1, dtype=torch.int32)
    for e, s in zip(experts, slots):
        mp[LAYER * 512 + e] = s
    ptrs = torch.tensor([pool.data_ptr() + i * REC for i in range(n_slots)], dtype=torch.int64)
    return mp.cuda(), ptrs.cuda()


def _run(pool, experts, slots, x, ids, w):
    mp, ptrs = _tables(pool, experts, slots)
    r = x.shape[0]
    h = torch.empty(r, 10, 640, device="cuda")
    o = torch.empty(r, 10, 2560, device="cuda")
    out = torch.empty(r, 2560, device="cuda")
    moe.moe_decode(x, ids, w, LAYER, mp, ptrs, h, o, out)
    torch.cuda.synchronize()
    return out


def _inputs(r, seed, n_experts):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(r, 2560, generator=g).to(torch.bfloat16).cuda()
    ids = torch.stack([torch.randperm(n_experts, generator=g)[:10] for _ in range(r)]).to(torch.int32).cuda()
    w = torch.softmax(torch.randn(r, 10, generator=g), dim=-1).cuda()
    return x, ids, w


def _reference(x, ids, w, deqs):
    out = torch.zeros(x.shape[0], 2560, dtype=torch.float64)
    for r in range(x.shape[0]):
        xr = x[r].double().cpu()
        for k in range(10):
            dg, du, dd = (d.double() for d in deqs[int(ids[r, k])])
            g, u = dg @ xr, du @ xr
            h = g / (1 + torch.exp(-g)) * u
            out[r] += float(w[r, k]) * (dd @ h)
    return out


@pytest.fixture(scope="module")
def experts():
    recs, deqs = zip(*[_make_record(100 + i) for i in range(16)])
    return list(recs), list(deqs)


def test_matches_fp64_reference(experts):
    recs, deqs = experts
    ids_all = list(range(16))
    slots = [(i * 5 + 2) % 16 for i in ids_all]
    pool = _pool(recs, slots, 16)
    x, ids, w = _inputs(2, 1, 16)
    out = _run(pool, ids_all, slots, x, ids, w)
    ref = _reference(x, ids, w, deqs)
    err = ((out.double().cpu() - ref).norm() / ref.norm()).item()
    assert err <= 1e-4, err


@pytest.mark.parametrize("n_slots", [13, 64])
def test_placement_bitwise(experts, n_slots):
    recs, _ = experts
    ids_all = list(range(13))
    base_slots = list(range(13))
    base_pool = _pool(recs[:13], base_slots, 13)
    x, ids, w = _inputs(2, 2, 13)
    ids = ids % 13
    ref = _run(base_pool, ids_all, base_slots, x, ids, w)
    moved = [(i * 7 + 3) % n_slots for i in ids_all]
    assert len(set(moved)) == 13
    pool = _pool(recs[:13], moved, n_slots)
    out = _run(pool, ids_all, moved, x, ids, w)
    assert torch.equal(out, ref)


def test_row_invariance_and_duplicates(experts):
    recs, _ = experts
    slots = list(range(16))
    pool = _pool(recs, slots, 16)
    x, ids, w = _inputs(3, 3, 16)
    ids[1] = ids[0]          # same experts in two rows
    ids[2, :5] = ids[0, :5]  # partial overlap
    out = _run(pool, list(range(16)), slots, x, ids, w)
    for r in range(3):
        one = _run(pool, list(range(16)), slots, x[r:r + 1].contiguous(), ids[r:r + 1].contiguous(),
                   w[r:r + 1].contiguous())
        assert torch.equal(one[0], out[r])


def test_down_tail_mask_and_config_keys(experts):
    recs, deqs = experts
    ids_all = list(range(16))
    pool = _pool(recs, ids_all, 16)
    mp, ptrs = _tables(pool, ids_all, ids_all)
    cfg = dict(tuning.MOE_DEFAULT, bkd=256)   # INT 640 leaves a masked 128-element tail
    x, ids, w = _inputs(3, 5, 16)

    def run(xx, ii, ww, c):
        r = xx.shape[0]
        h = torch.empty(r, 10, 640, device="cuda")
        o = torch.empty(r, 10, 2560, device="cuda")
        out = torch.empty(r, 2560, device="cuda")
        moe.moe_decode(xx, ii, ww, LAYER, mp, ptrs, h, o, out, cfg=c)
        torch.cuda.synchronize()
        return out

    out = run(x, ids, w, cfg)
    ref = _reference(x, ids, w, deqs)
    assert ((out.double().cpu() - ref).norm() / ref.norm()).item() <= 1e-4
    for r in range(3):
        one = run(x[r:r + 1].contiguous(), ids[r:r + 1].contiguous(), w[r:r + 1].contiguous(), cfg)
        assert torch.equal(one[0], out[r])
    with pytest.raises(ValueError):
        run(x, ids, w, dict(tuning.MOE_DEFAULT, bk=96))


def test_shared_memory_limit(experts):
    recs, _ = experts
    pool = _pool(recs, list(range(16)), 16)
    x, ids, w = _inputs(1, 4, 16)
    _run(pool, list(range(16)), list(range(16)), x, ids, w)
    for name in ("moe_gate_up", "moe_down", "moe_rank_sum"):
        assert gemv_affine.SHARED_BYTES[name] <= 101_376
