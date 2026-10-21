"""Vendored QSA / hyper-connection / GDN ops against the fp32 torch reference, on the GPU.

Each case feeds the vendored bf16 op and `reference_torch` (fp32 math) the same bf16 values. The
tolerance of one bf16 op against its fp32 reference is `[ops] rel_l2_max` of numeric_contract.toml
(read, not copied); the hyper-connection limit and the control ratio are read from the same file; the bitwise checks are exact. A table of the
largest relative error per op is printed when the run ends."""
import atexit
import sys
import tomllib
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from slotstream_spark.kernels import vendored  # noqa: E402
from slotstream_spark.model import reference_torch as R  # noqa: E402
from slotstream_spark.model.config import EXPECTED, TextConfig  # noqa: E402

pytestmark = pytest.mark.cuda

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = tomllib.loads((ROOT / "slotstream_spark" / "config" / "numeric_contract.toml").read_text())
OPS_REL = float(CONTRACT["ops"]["rel_l2_max"])
# Hyper-connection limit and the negative-control ratio also come from the contract (floor and control
# are documented next to the keys in the toml).
HC_REL = float(CONTRACT["ops"]["hc_rel_l2_max"])
CONTROL = float(CONTRACT["ops"]["control_min_ratio"])

H, HC = 2560, 4
NQ, NKV, HD = 24, 2, 256
IH, ID = 4, 128
TABLE: dict[str, float] = {}


def _print_table():
    if TABLE:
        w = max(len(k) for k in TABLE)
        out = ["max relative error per op (rel_l2 vs fp32 reference_torch):"]
        out += [f"  {k:<{w}}  {v:.3e}" for k, v in sorted(TABLE.items())]
        print("\n".join(out), file=sys.__stdout__)


atexit.register(_print_table)


def note(name, value):
    TABLE[name] = max(TABLE.get(name, 0.0), float(value))


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@pytest.fixture(scope="module")
def dev():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return torch.device("cuda")


def ref_config():
    tc = dict(EXPECTED)
    tc["layer_types"] = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(48)]
    tc["rope_parameters"] = {"rope_theta": 1e7}
    return TextConfig.from_dict(tc)


# ---------------------------------------------------------------- QSA
class QSAWeights:
    def __init__(self, dev, seed=0):
        self.dev = dev
        g = torch.Generator(device="cpu").manual_seed(seed)
        r = lambda *s, sc=0.02: (torch.randn(*s, generator=g) * sc).to(torch.bfloat16).to(dev)
        n = lambda d: (1 + 0.1 * torch.randn(d, generator=g)).to(torch.bfloat16).to(dev)
        self.qkv = r(2 * NQ * HD + 2 * NKV * HD, H)
        self.iqk = r((IH + 1) * ID, H)
        self.o = r(H, NQ * HD)
        self.qn, self.kn, self.iqn, self.ikn = n(HD), n(HD), n(ID), n(ID)
        p = "layers.3.self_attn."
        qd = 2 * NQ * HD
        f = lambda t: t.float()
        self.ref = {
            p + "q_proj.weight": f(self.qkv[:qd]), p + "k_proj.weight": f(self.qkv[qd:qd + NKV * HD]),
            p + "v_proj.weight": f(self.qkv[qd + NKV * HD:]), p + "o_proj.weight": f(self.o),
            p + "q_norm.weight": f(self.qn), p + "k_norm.weight": f(self.kn),
            p + "indexer.index_qk_proj.weight": f(self.iqk),
            p + "indexer.q_layernorm.weight": f(self.iqn), p + "indexer.k_layernorm.weight": f(self.ikn),
        }

    def project(self, x):
        qkv = (x @ self.qkv.T).contiguous()
        iqk = (x @ self.iqk.T).contiguous()
        return qkv, iqk[:, :IH * ID], iqk[:, IH * ID:]

    def post(self, out):
        return F.linear(out.reshape(out.shape[0], -1).float(), self.o.float())

    def ref_model(self):
        return R.RefModel(ref_config(), self.ref.__getitem__, None, None, device=self.dev)


def inputs(dev, n, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n, H, generator=g).to(torch.bfloat16).to(dev)


class ForcedRef(R.RefModel):
    """Reference whose indexer selection for the last chunk can be forced to the vendored one.

    Near the 512-block cutoff the bf16 and fp32 indexer scores differ by rounding noise, so the two
    selections may differ in a few tied blocks; with diffuse attention a handful of swapped blocks moves
    the output more than the op's own error. The attention op is therefore measured under the same
    selection, and the selection is checked separately (check_block_set)."""
    forced = None
    natural = None

    def indexer_mask(self, i, st, x, cs, sn):
        sel = super().indexer_mask(i, st, x, cs, sn)
        if self.forced is not None and x.shape[0] == self.forced.shape[0]:
            self.natural = sel
            return self.forced
        self.natural = sel
        return sel


def ref_attention(w, x, last, forced=None):
    """Reference over x: the context in chunks, then the last `last` tokens in one call (selection forced
    to `forced` [last, n] when given). Returns the outputs of the last tokens, the model and its state."""
    m = ForcedRef(ref_config(), w.ref.__getitem__, None, None, device=w.dev)
    xf = x.float()
    st = m.st[3]
    n = xf.shape[0]
    a = 0
    while a < n - last:
        b = min(a + 2048, n - last)
        cos, sin = m.rope(torch.arange(a, b))
        m.attention(3, st, xf[a:b], cos, sin)
        a = b
    m.forced = forced
    cos, sin = m.rope(torch.arange(n - last, n))
    o = m.attention(3, st, xf[n - last:], cos, sin)
    return o, m, st


def store(w, cache, x, a, b):
    qkv, iq, ik = w.project(x[a:b])
    return vendored.qsa_store_kv(cache, qkv, iq, ik, a, w.qn, w.kn, w.iqn, w.ikn)


def build_context(w, x, upto, dev, max_ctx):
    cache = vendored.QSACache(max_ctx, dev)
    a = 0
    while a < upto:
        b = min(a + 2048, upto)
        store(w, cache, x, a, b)
        a = b
    return cache


def block_scores(w, m, st, x_row, pos):
    """fp32 indexer scores of the last query over the blocks it sees (reference side, for tie checks)."""
    cr = 4
    p = "layers.3.self_attn.indexer."
    qk = F.linear(x_row, m.w(p + "index_qk_proj.weight"))
    q = qk[:, :IH * ID].view(1, IH, ID)
    cos, sin = m.rope(torch.tensor([pos]))
    q = R.apply_rope(R.rmsnorm(q, m.w(p + "q_layernorm.weight"), m.eps), cos[:, None, :], sin[:, None, :])
    bk = st["bk"]
    sc = torch.relu(torch.einsum("thd,bd->tbh", q.float(), bk.float())).sum(-1) / ID ** 0.5
    return sc[0, : (pos + 1) // cr]


def check_block_set(w, m, st, x_row, pos, ref_mask, ours_mask):
    """Selected block sets equal apart from ties: every block on which they differ scores within a
    bf16 margin of the cutoff."""
    cr = 4
    ncb = (pos + 1) // cr
    ref_b = ref_mask[:ncb * cr].view(ncb, cr)[:, 0]
    our_b = ours_mask[:ncb * cr].view(ncb, cr)[:, 0]
    diff = (ref_b != our_b).nonzero().flatten()
    if diff.numel() == 0:
        return 0
    sc = block_scores(w, m, st, x_row, pos)
    cut = sc.topk(512).values[-1]
    margin = 0.03 * float(sc.abs().max()) + 1e-3
    far = (sc[diff] - cut).abs() > margin
    assert not far.any(), f"{int(far.sum())} blocks differ beyond the tie margin {margin:.3g} (cut {float(cut):.4g})"
    return int(diff.numel())


@pytest.mark.parametrize("kv_len", [64, 2048, 2049, 4096, 8192, 16384])
def test_qsa_decode_vs_reference(dev, kv_len):
    w = QSAWeights(dev)
    x = inputs(dev, kv_len, seed=kv_len)
    cache = build_context(w, x, kv_len - 1, dev, kv_len)
    q_idx, q, gate = store(w, cache, x, kv_len - 1, kv_len)
    out = vendored.qsa_decode(cache, q_idx, q, gate)
    torch.cuda.synchronize()
    ours = vendored.selected_tokens(cache)
    ref_out, m, st = ref_attention(w, x, 1, forced=ours)
    e = rel(w.post(out), ref_out)
    note(f"qsa_decode kv_len={kv_len}", e)
    assert e <= OPS_REL, f"qsa decode kv_len={kv_len}: rel {e:.3e} > {OPS_REL}"
    ref_mask = m.natural
    ref_mask = torch.ones(1, kv_len, dtype=torch.bool, device=dev) if ref_mask is None else ref_mask
    if kv_len <= 2051:
        assert torch.equal(ours.cpu(), ref_mask.cpu()), "dense regime must select every causal token"
    else:
        ties = check_block_set(w, m, st, x[-1:].float(), kv_len - 1, ref_mask[0], ours[0])
        assert int(ours.sum()) == int(ref_mask.sum()), "same number of selected tokens"
        note("qsa_decode selection ties (blocks)", ties)


def test_qsa_prefill_t300_vs_reference(dev):
    w = QSAWeights(dev)
    x = inputs(dev, 300, seed=300)
    cache = vendored.QSACache(300, dev)
    qkv, iq, ik = w.project(x)
    q_idx, q, gate = vendored.qsa_store_kv(cache, qkv, iq, ik, 0, w.qn, w.kn, w.iqn, w.ikn)
    out = vendored.qsa_prefill(cache, q_idx, q, gate)
    torch.cuda.synchronize()
    ours = vendored.selected_tokens(cache)
    ref_out, m, st = ref_attention(w, x, 300, forced=ours)
    assert torch.equal(ours.cpu(), torch.ones(300, 300).tril().bool()), "dense prefill sees every causal token"
    e = rel(w.post(out), ref_out)
    note("qsa_prefill T=300", e)
    assert e <= OPS_REL, f"qsa prefill T=300: rel {e:.3e} > {OPS_REL}"


def test_qsa_prefill_sparse_regime(dev):
    """T = 3000 puts more than 512 compressed blocks in view, so the selection is active."""
    w = QSAWeights(dev)
    t = 3000
    x = inputs(dev, t, seed=3000)
    cache = vendored.QSACache(t, dev)
    qkv, iq, ik = w.project(x)
    q_idx, q, gate = vendored.qsa_store_kv(cache, qkv, iq, ik, 0, w.qn, w.kn, w.iqn, w.ikn)
    out = vendored.qsa_prefill(cache, q_idx, q, gate)
    torch.cuda.synchronize()
    ours = vendored.selected_tokens(cache)
    ref_out, m, st = ref_attention(w, x, t, forced=ours)
    assert m.natural is not None, "the selection must be active at T = 3000"
    for row in (2100, 2500, 2999):
        check_block_set(w, m, st, x[row:row + 1].float(), row, m.natural[row], ours[row])
    e = rel(w.post(out), ref_out)
    note("qsa_prefill T=3000 (sparse)", e)
    assert e <= OPS_REL, f"qsa prefill T=3000: rel {e:.3e} > {OPS_REL}"


@pytest.mark.parametrize("kv_len", [64, 4096])
def test_qsa_decode_r3_equals_three_r1(dev, kv_len):
    w = QSAWeights(dev)
    x = inputs(dev, kv_len, seed=7 + kv_len)
    base = build_context(w, x, kv_len - 3, dev, kv_len)
    a = base.clone()
    qkv, iq, ik = w.project(x[kv_len - 3:])
    q_idx, q, gate = vendored.qsa_store_kv(a, qkv, iq, ik, kv_len - 3, w.qn, w.kn, w.iqn, w.ikn)
    out3 = vendored.qsa_decode(a, q_idx, q, gate)
    blocks3 = a.last["block_indices"].clone()
    outs1, blocks1 = [], []
    b = base.clone()
    for i in range(3):
        # the projection is done once for the three rows: a GEMM with a different M may round differently,
        # which is not what is under test
        q_idx, q, gate = vendored.qsa_store_kv(b, qkv[i:i + 1], iq[i:i + 1], ik[i:i + 1], kv_len - 3 + i,
                                               w.qn, w.kn, w.iqn, w.ikn)
        outs1.append(vendored.qsa_decode(b, q_idx, q, gate).clone())
        blocks1.append(b.last["block_indices"].clone())
    torch.cuda.synchronize()
    assert torch.equal(out3, torch.cat(outs1, 0)), "R=3 must equal three R=1 bit for bit"
    assert torch.equal(blocks3, torch.cat(blocks1, 0))
    assert torch.equal(a.comp, b.comp) and torch.equal(a.main_kv, b.main_kv)


# ---------------------------------------------------------------- hyper-connections
class HCWeights:
    def __init__(self, dev, seed=1):
        g = torch.Generator(device="cpu").manual_seed(seed)
        r = lambda *s, sc=0.03: (torch.randn(*s, generator=g) * sc).to(torch.bfloat16).to(dev)
        self.norm = (1 + 0.1 * torch.randn(HC * H, generator=g)).to(torch.bfloat16).to(dev)
        self.down, self.up = r(320, HC * H), r(HC * H, 320)
        self.inj = r(HC, HC * H)

    def ref(self, dtype):
        c = lambda t: t.to(dtype)
        return {"b.hc_norm.weight": c(self.norm), "b.input_mix_weight_down.weight": c(self.down),
                "b.input_mix_weight_up.weight": c(self.up), "b.block_inject_weight.weight": c(self.inj)}


@pytest.mark.parametrize("name,inject", [("attention", True), ("mlp", True), ("final mixer", False)])
def test_hc_vs_gated_residual(dev, name, inject):
    w = HCWeights(dev)
    g = torch.Generator(device="cpu").manual_seed(11)
    hyp = torch.randn(5, HC * H, generator=g).to(torch.bfloat16).to(dev)
    mixed, inj, hn = vendored.hc_mix(hyp, w.norm, w.down, w.up, w.inj if inject else None)
    errs = {}
    for label, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        wt = w.ref(dt)
        m = R.RefModel(ref_config(), wt.__getitem__, None, None, device=dev)
        rm, ri = m.gated_residual("b", hyp.to(dt), inject=inject)
        errs[label] = rel(mixed, rm)
        if inject:
            errs[label + " inj"] = rel(2 * torch.sigmoid(inj.float() / HC), ri)
    note(f"hc_mix {name} (vs bf16 ref)", errs["bf16"])
    note(f"hc_mix {name} (vs fp32 ref)", errs["fp32"])
    assert errs["fp32"] <= HC_REL, f"hc {name}: rel {errs['fp32']:.3e} > {HC_REL} against the fp32 reference ({errs})"
    if inject:
        assert errs["fp32 inj"] <= HC_REL, errs
    # combine step against the reference residual update
    block = torch.randn(5, H, generator=g).to(torch.bfloat16).to(dev)
    out = vendored.hc_residual(hyp, block, inj)
    if inject:
        gate = 2 * torch.sigmoid(inj.float() / HC)
        want = hyp.float() + (block.float().unsqueeze(-2) * gate.unsqueeze(-1)).flatten(-2)
    else:
        want = hyp.float() + block.float().repeat(1, HC)
    e = rel(out, want)
    note(f"hc_residual {name}", e)
    assert e <= HC_REL, f"hc_residual {name}: rel {e:.3e}"
    nh, hn2 = vendored.hc_residual_norm(hyp, block, inj, w.norm)
    e2 = rel(hn2, R.rmsnorm(nh.float(), w.norm.float(), 1e-6, group=H))
    note(f"hc_residual_norm {name}", e2)
    assert e2 <= HC_REL, f"hc_residual_norm {name}: rel {e2:.3e}"


# ---------------------------------------------------------------- gated delta rule
def gdn_inputs(dev, t, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(1, t, 16, 128, generator=g).to(torch.bfloat16).to(dev)
    k = torch.randn(1, t, 16, 128, generator=g).to(torch.bfloat16).to(dev)
    v = torch.randn(1, t, 48, 128, generator=g).to(torch.bfloat16).to(dev)
    gate = (-(0.02 + 0.3 * torch.rand(1, t, 48, generator=g))).to(dev)
    beta = torch.sigmoid(torch.randn(1, t, 48, generator=g)).to(dev)
    return q, k, v, gate, beta


def ref_inputs(q, k, v):
    return q.float().repeat_interleave(3, dim=2), k.float().repeat_interleave(3, dim=2), v.float()


def test_gdn_fused_256_steps(dev):
    t = 256
    q, k, v, gate, beta = gdn_inputs(dev, t, 5)
    qr, kr, vr = ref_inputs(q, k, v)
    s_v = torch.zeros(1, 48, 128, 128, dtype=torch.float32, device=dev)
    s_r = s_v.clone()
    worst, drift = 0.0, 0.0
    for i in range(t):
        sl = slice(i, i + 1)
        o_v, s_v = vendored.gdn_decode(q[:, sl], k[:, sl], v[:, sl], gate[:, sl], beta[:, sl], s_v)
        o_r, s_r = R.recurrent_gated_delta_rule(qr[:, sl], kr[:, sl], vr[:, sl], gate[:, sl], beta[:, sl], s_r)
        e = rel(o_v, o_r)
        worst = max(worst, e)
        drift = max(drift, rel(s_v, s_r))
        assert e <= OPS_REL, f"gdn step {i}: rel {e:.3e} > {OPS_REL}"
    note("gdn fused, 256 steps (output)", worst)
    note("gdn fused, 256 steps (state drift)", drift)
    print(f"gdn state drift over 256 steps: max rel_l2 {drift:.3e}", file=sys.__stdout__)


@pytest.mark.parametrize("t", [64, 300, 600])
def test_gdn_chunk(dev, t):
    q, k, v, gate, beta = gdn_inputs(dev, t, 100 + t)
    qr, kr, vr = ref_inputs(q, k, v)
    o_v, s_v = vendored.gdn_prefill(q, k, v, gate, beta, None)
    o_r, s_r = R.chunk_gated_delta_rule(qr, kr, vr, gate, beta)
    e, es = rel(o_v, o_r), rel(s_v, s_r)
    note(f"gdn chunk T={t}", e)
    note(f"gdn chunk T={t} (state)", es)
    assert e <= OPS_REL, f"gdn chunk T={t}: rel {e:.3e} > {OPS_REL}"
    assert es <= OPS_REL, f"gdn chunk T={t} state: rel {es:.3e}"


def test_gdn_r3_equals_three_r1(dev):
    q, k, v, gate, beta = gdn_inputs(dev, 3, 9)
    g = torch.Generator(device="cpu").manual_seed(2)
    s0 = (torch.randn(1, 48, 128, 128, generator=g) * 0.1).to(dev)
    o3, s3 = vendored.gdn_decode(q, k, v, gate, beta, s0.clone())
    s = s0.clone()
    outs = []
    for i in range(3):
        sl = slice(i, i + 1)
        o, s = vendored.gdn_decode(q[:, sl], k[:, sl], v[:, sl], gate[:, sl], beta[:, sl], s)
        outs.append(o)
    assert torch.equal(o3, torch.cat(outs, 1)), "fused GDN R=3 must equal three R=1 bit for bit"
    assert torch.equal(s3, s)


def test_double_plus_one_is_caught(dev):
    """Falsifier: handing the folded weight (already 1 + w) to the kernel without `gemma_weight` applies the +1
    twice, and the error is O(1), far above every limit here."""
    w = HCWeights(dev)
    g = torch.Generator(device="cpu").manual_seed(13)
    hyp = torch.randn(5, HC * H, generator=g).to(torch.bfloat16).to(dev)
    hc = vendored._ops()["hc"]
    wrong = hc.grouped_gemma_rmsnorm(hyp, w.norm.float().contiguous(), 1e-6, HC)
    want = R.rmsnorm(hyp.float(), w.norm.float(), 1e-6, group=H)
    right = hc.grouped_gemma_rmsnorm(hyp, vendored.gemma_weight(w.norm), 1e-6, HC)
    assert rel(wrong, want) > CONTROL * HC_REL
    assert rel(right, want) <= HC_REL
