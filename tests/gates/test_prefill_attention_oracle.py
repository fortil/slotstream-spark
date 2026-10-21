"""Float64 oracle for the raw prefill attention kernel (port of Diagnostics+FusedPrefill.swift:8-92).

Needs CUDA, no weights. Inputs ((i*m + a) % mod - sub) / 64 with Q (31,7,257,128), K (43,13,263,131), V (59,17,269,134);
(B,H,Hk,S,N) in three shapes; Q strided (transpose, contiguous, transpose); dense causal and sparse masks at QSA block
granularity; the oracle is float64 on rows {0, S/2, S-1} and heads {0, H-1}.

Pass iff rel_l2 <= max(1.1 x rel_l2(torch SDPA math backend in bf16, same inputs, vs the oracle), [ops] attn_oracle_rel_l2_max).
Negative control: the oracle with V rolled by one key row differs from the kernel by more than control_min_ratio x the
limit on every checked row. Two runs are bit-identical. The limit and the ratio come from numeric_contract.toml.
"""
import tomllib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slotstream_spark.model.prefill_mixers import prefill_attention  # noqa: E402

pytestmark = pytest.mark.gate

OPS = tomllib.loads((Path(__file__).resolve().parents[2] / "slotstream_spark" / "config" / "numeric_contract.toml")
                    .read_text())["ops"]
LIMIT = float(OPS["attn_oracle_rel_l2_max"])
RATIO = float(OPS["control_min_ratio"])
WITNESS_FACTOR = 1.1
D = 256
SHAPES = ((2, 4, 2, 17, 257), (1, 24, 2, 256, 8192), (1, 24, 2, 259, 8211))


def values(count, mult, add, mod, sub, device):
    i = torch.arange(count, dtype=torch.int64, device=device)
    return ((i * mult + add) % mod - sub).float() / 64


def visible_fn(rows, keys, base, sparse):
    """[S,N] bool: key <= base + row and, when sparse, key/4 % 4 == row % 4 or key >= (base + row)/4 * 4."""
    r, k = rows[:, None], keys[None, :]
    ok = k <= base + r
    if sparse:
        ok = ok & (((k // 4) % 4 == r % 4) | (k >= ((base + r) // 4) * 4))
    return ok


def oracle_rows(qa, ka, va, vis, b, h, hk, s, n, rolled=False):
    """float64 attention of the checked (batch, head, row) triples; -> dict[(batch, head, row)] = [D] float64."""
    out = {}
    for batch in range(b):
        for head in (0, h - 1):
            kv = batch * hk + head // (h // hk)
            kk = ka[kv]                                     # [N, D] float64
            vv = torch.roll(va[kv], 1, 0) if rolled else va[kv]
            for row in (0, s // 2, s - 1):
                sel = vis[row]
                sc = (kk[sel] @ qa[batch * h + head, row]) / 16
                w = torch.exp(sc - sc.max())
                out[(batch, head, row)] = (w[:, None] * vv[sel]).sum(0) / w.sum()
    return out


def rel(got, want):
    num = sum(float(((g - w) ** 2).sum()) for g, w in zip(got, want))
    den = sum(float((w ** 2).sum()) for w in want)
    return (num / max(den, 1e-30)) ** 0.5


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda t: "B%d.H%d.Hk%d.S%d.N%d" % t)
def test_prefill_attention_vs_float64_oracle(shape, sparse):
    b, h, hk, s, n = shape
    dev = "cuda"
    base = n - s
    q0 = values(b * h * s * D, 31, 7, 257, 128, dev).view(b, h, s, D).to(torch.bfloat16)
    q = q0.transpose(2, 3).contiguous().transpose(2, 3)                    # strided Q
    assert not q.is_contiguous()
    k = values(b * hk * n * D, 43, 13, 263, 131, dev).view(b, hk, n, D).to(torch.bfloat16)
    v = values(b * hk * n * D, 59, 17, 269, 134, dev).view(b, hk, n, D).to(torch.bfloat16)
    rows = torch.arange(s, device=dev)
    keys = torch.arange(n, device=dev)
    vis = visible_fn(rows, keys, base, sparse)

    got = prefill_attention(q, k, v, base=base, visible=vis if sparse else None)
    again = prefill_attention(q, k, v, base=base, visible=vis if sparse else None)
    assert got.shape == q.shape and torch.isfinite(got.float()).all()
    assert torch.equal(got, again), "two runs differ in bits"

    # same-dtype floor witness: torch SDPA math backend in bf16 on the same inputs
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH):
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=vis, scale=1 / 16, enable_gqa=True)

    qa = q.double().reshape(b * h, s, D)
    ka = k.double().reshape(b * hk, n, D)
    va = v.double().reshape(b * hk, n, D)
    want = oracle_rows(qa, ka, va, vis, b, h, hk, s, n)
    keys_ = list(want)
    g = [got[bt, hd, r].double() for bt, hd, r in keys_]
    w = [want[x] for x in keys_]
    r_ref = rel([ref[bt, hd, r].double() for bt, hd, r in keys_], w)
    r_got = rel(g, w)
    bound = max(WITNESS_FACTOR * r_ref, LIMIT)
    label = "B%d.H%d.S%d.N%d.%s" % (b, h, s, n, "sparse" if sparse else "dense")
    print(f"{label}: kernel rel_l2 {r_got:.3e} sdpa-bf16 floor witness {r_ref:.3e} limit {LIMIT:.1e} bound {bound:.3e}",
          flush=True)
    assert r_got <= bound

    rolled = oracle_rows(qa, ka, va, vis, b, h, hk, s, n, rolled=True)
    worst = min(rel([got[bt, hd, r].double()], [rolled[(bt, hd, r)]]) for bt, hd, r in keys_)
    print(f"{label}: V-rolled control min per-row rel {worst:.3e} (must exceed {RATIO * LIMIT:.3e})", flush=True)
    assert worst > RATIO * LIMIT


def test_negative_control_catches_a_wrong_kernel():
    """A kernel that attends the wrong keys (shifted by one) must fail the bound on the dense small shape."""
    b, h, hk, s, n = SHAPES[0]
    dev = "cuda"
    base = n - s
    q = values(b * h * s * D, 31, 7, 257, 128, dev).view(b, h, s, D).to(torch.bfloat16)
    k = values(b * hk * n * D, 43, 13, 263, 131, dev).view(b, hk, n, D).to(torch.bfloat16)
    v = values(b * hk * n * D, 59, 17, 269, 134, dev).view(b, hk, n, D).to(torch.bfloat16)
    vis = visible_fn(torch.arange(s, device=dev), torch.arange(n, device=dev), base, False)
    wrong = prefill_attention(q, k, torch.roll(v, 1, 2), base=base)
    qa, ka, va = q.double().reshape(b * h, s, D), k.double().reshape(b * hk, n, D), v.double().reshape(b * hk, n, D)
    want = oracle_rows(qa, ka, va, vis, b, h, hk, s, n)
    ks = list(want)
    assert rel([wrong[bt, hd, r].double() for bt, hd, r in ks], [want[x] for x in ks]) > LIMIT
