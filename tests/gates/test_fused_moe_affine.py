"""Step 67 gate: the grouped affine 4-bit MoE kernel against an fp64 exact-dequant reference (synthetic records).

Records live in the platform.toml pool memory type (slotio.Pool) so the kernel reads them the way the engine's
does. The limit comes from numeric_contract.toml [ops]; the floor witness (distance to a bf16-dequant fp32
reference) and the negative control (nibble order swapped) are printed and asserted.
"""
import inspect
import tomllib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slotstream_spark.config import PLATFORM_PATH  # noqa: E402
from slotstream_spark.kernels import fused_moe_affine as fm  # noqa: E402

pytestmark = [pytest.mark.gate, pytest.mark.cuda]

ROOT = Path(__file__).resolve().parents[2]
OPS = tomllib.loads((ROOT / "slotstream_spark" / "config" / "numeric_contract.toml").read_text())["ops"]
LIMIT, CONTROL = float(OPS["fused_moe_rel_l2_max"]), float(OPS["control_min_ratio"])
POOL_MEMORY = tomllib.loads(Path(PLATFORM_PATH).read_text())["memory"]["pool_memory"]
SEED, NEXP, TOK, TOPK = 1729, 32, 4096, 10
RMAX = TOK * TOPK
REC = fm.RECORD_BYTES
OFFS = ((fm.O_GATE_W, fm.O_GATE_S, fm.O_GATE_B), (fm.O_UP_W, fm.O_UP_S, fm.O_UP_B), (fm.O_DOWN_W, fm.O_DOWN_S, fm.O_DOWN_B))


def _record(seed):
    g = torch.Generator().manual_seed(seed)
    pieces = []
    rec = torch.zeros(REC, dtype=torch.uint8)
    for (n, k), (ow, os_, ob) in zip(((640, 2560), (640, 2560), (2560, 640)), OFFS):
        q = torch.randint(-2**31, 2**31 - 1, (n, k // 8), dtype=torch.int32, generator=g)
        s = (torch.randn(n, k // 64, generator=g) * 0.02).to(torch.bfloat16)
        b = (torch.randn(n, k // 64, generator=g) * 0.02).to(torch.bfloat16)
        for t, off in ((q, ow), (s, os_), (b, ob)):
            raw = t.contiguous().view(torch.uint8).reshape(-1)
            rec[off:off + raw.numel()] = raw
        pieces.append((q, s, b))
    return rec, pieces


def _deq(piece, dtype, swap=False):
    q, s, b = piece
    by = q.contiguous().view(torch.uint8)
    lo, hi = (by >> 4, by & 15) if swap else (by & 15, by >> 4)
    u = torch.stack([lo, hi], dim=-1).reshape(by.shape[0], -1).to(dtype)
    gs = lambda t: t.to(dtype).repeat_interleave(64, dim=1)  # noqa: E731
    return u * gs(s) + gs(b)


@pytest.fixture(scope="module")
def world():
    import slotio
    recs, pieces = zip(*[_record(SEED + i) for i in range(NEXP)])
    pool = slotio.Pool(POOL_MEMORY, slots_per_chunk=8, chunk_bytes=24 << 20, max_slots=64)
    pool.resize_slots(2 * NEXP)
    for i, r in enumerate(recs):
        pool.slot_view(i)[:] = r.numpy()
        pool.slot_view(NEXP + i)[:] = r.numpy()          # the same bytes at other addresses (another chunk range)
    g = torch.Generator().manual_seed(SEED)
    x = torch.randn(TOK, 2560, generator=g).to(torch.bfloat16).cuda()
    expert = torch.randint(0, NEXP, (RMAX,), generator=g, dtype=torch.int32)
    weight = torch.rand(RMAX, generator=g) + 0.1
    return dict(pool=pool, recs=recs, pieces=pieces, x=x, expert=expert.cuda(), weight=weight.cuda(),
                token=(torch.arange(RMAX, dtype=torch.int32) // TOPK).cuda())


def _ptrs(pool, order, offset=0):
    return torch.tensor([pool.slot_dev(offset + int(i)) for i in order], dtype=torch.int64).cuda()


def _map(order):
    m = torch.full((512,), -1, dtype=torch.int32)
    for local, e in enumerate(order):
        m[int(e)] = local
    return m.cuda()


def _run(w, r, order=None, offset=0, experts=None):
    """Run the first r rows; `order` = experts of the group in local order (default all 32), rows of others skipped."""
    order = list(range(NEXP)) if order is None else list(order)
    out = torch.zeros(r, 2560, dtype=torch.float32, device="cuda")
    fm.fused_moe_affine(w["x"], w["token"][:r], w["weight"][:r], _map(order), _ptrs(w["pool"], order, offset), out,
                        row_expert=w["expert"][:r])
    torch.cuda.synchronize()
    return out


def _reference(w, rows, dtype_w, swap=False, bf16_path=False):
    out = torch.zeros(len(rows), 2560, dtype=torch.float64)
    for n, r in enumerate(rows):
        e = int(w["expert"][r])
        xr = w["x"][int(w["token"][r])].cpu()
        dg, du, dd = (_deq(p, dtype_w, swap) for p in w["pieces"][e])
        if bf16_path:
            dg, du, dd = (d.to(torch.bfloat16).float() for d in (dg, du, dd))
            xf = xr.float()
            g, u = dg @ xf, du @ xf
            h = (g / (1 + torch.exp(-g)) * u).to(torch.bfloat16).float()
            y = (dd @ h).double()
        else:
            xd = xr.double()
            g, u = dg @ xd, du @ xd
            y = dd @ (g / (1 + torch.exp(-g)) * u)
        out[n] = y * float(w["weight"][r])
    return out


def _rel(a, b):
    return ((a.double().cpu() - b).norm() / b.norm()).item()


def test_matches_fp64_reference_floor_and_control(world, capsys):
    w, r = world, 2560
    out = _run(w, r)
    rows = list(range(0, r, 10))[:200]
    sel = out[rows]
    ref = _reference(w, rows, torch.float64)
    err = _rel(sel, ref)
    floor = _rel(sel, _reference(w, rows, torch.float32, bf16_path=True))
    ctrl = _rel(sel, _reference(w, rows, torch.float64, swap=True))
    print(f"fused_moe rel_l2 vs fp64 exact {err:.3e} (limit {LIMIT:g}); floor witness vs bf16-dequant fp32 {floor:.3e}; "
          f"nibble-swap control {ctrl:.3f} (needs > {CONTROL * LIMIT:g})")
    assert err <= LIMIT
    assert ctrl > CONTROL * LIMIT


def test_bit_identical_across_r(world):
    ref = _run(world, 17)
    for r in (2560, RMAX):
        assert torch.equal(_run(world, r)[:17], ref), r


def test_grouping_and_order_do_not_change_bits(world):
    r = 2560
    base = _run(world, r)
    rev = _run(world, r, order=list(reversed(range(NEXP))))
    assert torch.equal(rev, base)
    res_first = [3, 17, 5, 29] + [e for e in range(NEXP) if e not in (3, 17, 5, 29)]
    assert torch.equal(_run(world, r, order=res_first), base)
    a, b = _run(world, r, order=range(0, 16)), _run(world, r, order=range(16, 32))
    assert torch.equal(a + b, base)           # each half writes only its rows; the other rows stay zero


def test_pool_address_equals_staging_address(world):
    assert torch.equal(_run(world, 2560, offset=NEXP), _run(world, 2560))


def test_two_runs_identical(world):
    assert torch.equal(_run(world, 2560), _run(world, 2560))


def test_kernel_source_has_no_atomics():
    for k in (fm._gate_up_kernel, fm._down_kernel, fm._wtile):
        assert "atomic" not in inspect.getsource(getattr(k, "fn", k))


def test_launch_time_at_40960_rows(world, capsys):
    w, r = world, RMAX
    order = list(range(NEXP))
    out = torch.zeros(r, 2560, dtype=torch.float32, device="cuda")
    h = torch.empty(r, 640, dtype=torch.bfloat16, device="cuda")
    mp, gp = _map(order), _ptrs(w["pool"], order)
    aligned = fm.align_rows(w["expert"], mp, NEXP)
    fm.fused_moe_affine(w["x"], w["token"], w["weight"], mp, gp, out, row_expert=w["expert"], h_buf=h, aligned=aligned)
    torch.cuda.synchronize()                     # warm-up: the first call includes the Triton compile
    times = {"gate_up": [], "down": []}
    for _ in range(3):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        ev[0].record()
        fm.launch_gate_up(w["x"], w["token"], aligned, gp, h, r)
        ev[1].record()
        fm.launch_down(h, w["weight"], aligned, gp, out, r)
        ev[2].record()
        torch.cuda.synchronize()
        times["gate_up"].append(ev[0].elapsed_time(ev[1]))
        times["down"].append(ev[1].elapsed_time(ev[2]))
    worst = max(max(v) for v in times.values())
    print(f"fused_moe launch times at R={r} (ms, warm): {times}; max {worst:.1f} ms (limit 200)")
    assert worst <= 200.0


def test_combine_rank_order_equals_sequential_fp32_bitwise():
    """Step 70 carries this helper: its sum is the router-rank-order fp32 sum, checked against an independent
    sequential numpy float32 reference, bit for bit, and it is not invariant to the rank order (so the order matters)."""
    import numpy as np
    t, k = 257, fm.TOPK
    g = torch.Generator(device="cuda").manual_seed(SEED)
    rows = (torch.randn(t * k, fm.HID, generator=g, device="cuda") * torch.logspace(-3, 3, k, device="cuda").repeat(t)[:, None])
    got = fm.combine_rank_order(rows, t)
    assert got.dtype == torch.float32 and tuple(got.shape) == (t, fm.HID)
    host = rows.cpu().numpy().reshape(t, k, fm.HID)
    ref = host[:, 0].copy()
    for r in range(1, k):
        ref = (ref + host[:, r]).astype(np.float32)
    assert np.array_equal(got.cpu().numpy().view(np.uint32), ref.view(np.uint32))
    again = fm.combine_rank_order(rows, t)
    assert torch.equal(got, again)
    shuffled = fm.combine_rank_order(rows.view(t, k, -1)[:, torch.arange(k - 1, -1, -1, device="cuda")].reshape(t * k, -1), t)
    assert not torch.equal(got, shuffled), "the sum should depend on the rank order for this data"
    with pytest.raises(ValueError):
        fm.combine_rank_order(rows[:-1], t)
