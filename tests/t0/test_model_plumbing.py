"""Plumbing of the eager forward on CPU (needs torch: runs on the Spark; skipped on the Mac)."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from slotstream_spark.kernels import affine_ref  # noqa: E402
from slotstream_spark.model import reference_torch as R  # noqa: E402
from slotstream_spark.model import weights as WT  # noqa: E402
from slotstream_spark.model.forward import Model  # noqa: E402
from slotstream_spark.model.layers import row_groups  # noqa: E402
from synth_ckpt import bf16, write_safetensors  # noqa: E402
from test_reference_torch import make_weights, tiny_cfg  # noqa: E402

IDS = [3, 1, 4, 1, 5, 9, 2, 6]


class FakePool:
    """ExpertPool stand-in: a slot per (layer, expert) in first-use order; weights by slot."""

    def __init__(self, experts):
        self.experts = experts
        self.slot = {}
        self.rev = []
        self.groups = []
        self.progress = None

    def set_progress_ptr(self, addr):
        self.progress = addr

    def ensure(self, layer, ids, gen=None):
        out = []
        for e in ids:
            key = (layer, int(e))
            if key not in self.slot:
                self.slot[key] = len(self.rev)
                self.rev.append(key)
            out.append(self.slot[key])
        self.groups.append((layer, len(set(int(e) for e in ids)), gen))
        return np.array(out, dtype=np.int32)

    def slot_weights(self, slot):
        return self.experts[self.rev[slot]]


def build(max_rows=32, seed=0):
    cfg = tiny_cfg()
    W, experts, table, emb = make_weights(cfg, seed)
    ref = R.RefModel(cfg, W.__getitem__, lambda ids: emb[ids], lambda L, e: experts[(L, e)],
                     lambda hist, ids: table[torch.tensor(ids)])
    pool = FakePool(experts)
    model = Model(cfg, W.__getitem__, lambda ids: emb[torch.tensor(list(ids))], pool,
                  lambda hist, ids: table[torch.tensor(ids)], device="cpu", gdn="ref", max_ctx=64,
                  max_rows=max_rows, cuda=False)
    return ref, model, pool


def run_ref(ref):
    out = [ref.forward(IDS[:5])]
    for t in IDS[5:]:
        out.append(ref.forward([t]))
    return torch.stack(out)


def run_model(model):
    out = [model.prefill(IDS[:5])]
    for t in IDS[5:]:
        out.append(model.decode(t))
    return torch.stack(out)


def test_model_equals_refmodel_bitwise():
    """Precondition for a bitwise claim: both sides are fp32 CPU paths running the same reference_torch
    functions in the same order (set_gdn("ref")); a different op order would be an fp32 reorder of about 1e-6,
    which would be a numeric difference, not a plumbing bug."""
    ref, model, pool = build()
    assert model.gdn == "ref"
    a, b = run_ref(ref), run_model(model)
    assert torch.equal(a, b)
    assert len(pool.groups) >= 4 * 4        # at least one group per layer per pass
    assert pool.progress is not None


@pytest.mark.parametrize("rows", [1, 3, 7])
def test_row_group_size_does_not_change_the_result(rows):
    ref, _, _ = build()
    want = run_ref(ref)
    _, model, pool = build(max_rows=rows)
    assert torch.equal(run_model(model), want)
    assert max(n for _, n, _ in pool.groups) <= 32
    # groups really were split: the 5-token prefill needs ceil(5 / rows) groups per layer
    per_layer = [g for g in pool.groups if g[0] == 0][0:-3]
    assert len(per_layer) >= -(-5 // rows)


def test_teacher_force_matches_prefill_and_dumps_the_golden_layout(tmp_path):
    _, model, _ = build()
    lg = model.teacher_force(IDS, dump_dir=str(tmp_path))
    assert lg.shape == (len(IDS), 32)
    hid = 4 * 16
    for L in range(4):
        assert (tmp_path / f"layer_{L:02d}_out.f32").stat().st_size == len(IDS) * hid * 4
        assert (tmp_path / f"router_{L:02d}_ids.i32").stat().st_size == len(IDS) * 2 * 4
        assert (tmp_path / f"router_{L:02d}_w.f32").stat().st_size == len(IDS) * 2 * 4
    assert (tmp_path / "logits.f32").stat().st_size == len(IDS) * 32 * 4
    assert json.loads((tmp_path / "meta.json").read_text())["gdn_impl"] == "ref"
    _, m2, _ = build()
    assert torch.equal(m2.prefill(IDS), lg[-1])


def test_set_gdn_validates():
    _, model, _ = build()
    model.set_gdn("fused")
    assert model.gdn == "fused"
    with pytest.raises(ValueError):
        model.set_gdn("nope")


def test_row_groups_limits():
    ids = np.array([[0, 1], [2, 3], [4, 5], [6, 7]])
    assert row_groups(ids, 32, 32) == [(0, 4)]
    assert row_groups(ids, 2, 32) == [(0, 2), (2, 4)]
    assert row_groups(ids, 32, 4) == [(0, 2), (2, 4)]
    assert row_groups(ids, 32, 2) == [(0, 1), (1, 2), (2, 3), (3, 4)]      # one row always fits


def test_ngram_row_bytes_mismatch_fails_loudly(tmp_path):
    (tmp_path / "ngram.manifest.json").write_text(json.dumps({"row_bytes": 128}))
    with pytest.raises(RuntimeError, match="row_bytes"):
        WT.check_ngram_row_bytes(tmp_path, platform_row_bytes=100)
    assert WT.check_ngram_row_bytes(tmp_path, platform_row_bytes=128) == 128


def test_load_trunk_builds_affine_and_skips_embed(tmp_path):
    rng = np.random.default_rng(1)
    p = "language_model.model."

    def q(n, k):
        return (rng.integers(0, 2**32, size=(n, k // 8), dtype=np.uint32),
                bf16(rng.uniform(0.01, 0.1, size=(n, k // 64))), bf16(rng.uniform(-0.5, 0.5, size=(n, k // 64))))

    t = {}
    for name, (n, k) in {p + "embed_tokens": (8, 128), p + "layers.0.linear_attn.out_proj": (16, 128),
                         p + "layers.1.linear_attn.out_proj": (16, 128), "language_model.lm_head": (32, 128)}.items():
        t[name + ".weight"], t[name + ".scales"], t[name + ".biases"] = q(n, k)
    t[p + "layers.0.mlp.gate.weight"] = bf16(rng.standard_normal((4, 16)))
    t[p + "layers.1.ple.ple_embedding.layer_multipliers"] = np.array([3, 5, 7], dtype=np.int64)
    path = tmp_path / "trunk.safetensors"
    write_safetensors(path, t)
    w = WT.load_trunk(path, device="cpu")
    assert not any(k.startswith("embed_tokens") for k in w)
    a = w["layers.0.linear_attn.out_proj.weight"]
    assert (a.n, a.k, a.bits, a.gs) == (16, 128, 4, 64)
    want = affine_ref.dequant(a.q, a.s, a.b, 4, 64)
    from slotstream_spark.model.mlx_ckpt import dequant_affine
    src = t[p + "layers.0.linear_attn.out_proj.weight"]
    ref = dequant_affine(src, t[p + "layers.0.linear_attn.out_proj.scales"],
                         t[p + "layers.0.linear_attn.out_proj.biases"], 4)
    assert np.allclose(want.numpy(), ref, atol=1e-6)
    assert w["layers.0.mlp.gate.weight"].dtype == torch.bfloat16
    assert w["layers.1.ple.ple_embedding.layer_multipliers"].tolist() == [3, 5, 7]
    assert w["lm_head.weight"].n == 32
    only = WT.load_trunk(path, layers=[0], device="cpu")
    assert "layers.1.linear_attn.out_proj.weight" not in only and "lm_head.weight" in only
    with pytest.raises(KeyError):
        WT.accessor(only)("layers.1.linear_attn.out_proj.weight")
