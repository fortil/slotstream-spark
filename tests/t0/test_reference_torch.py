"""CPU tests of the torch reference forward on tiny configurations (needs torch: run on the Spark)."""
import pytest

torch = pytest.importorskip("torch")

from slotstream_spark.model import reference_torch as R  # noqa: E402
from slotstream_spark.model.config import TextConfig  # noqa: E402

H, HC, I, E, TOPK, V = 16, 4, 8, 6, 2, 32


def tiny_cfg(budget=8, ratio=2):
    tc = dict(hidden_size=H, num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
              num_attention_heads=2, num_key_value_heads=1, head_dim=8, partial_rotary_factor=0.25,
              rope_parameters={"rope_theta": 10000.0}, rms_norm_eps=1e-6,
              linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=4,
              linear_value_head_dim=4, linear_conv_kernel_dim=4, num_experts=E, num_experts_per_tok=TOPK,
              moe_intermediate_size=I, hc_count=HC, hc_lowrank=6, indexer_n_heads=2, indexer_head_dim=4,
              indexer_budget=budget, indexer_compress_ratio=ratio, ple_layer_ids=[2], ple_embed_dim=H,
              ple_conv_kernel_size=4, vocab_size=V)
    return TextConfig.from_dict(tc, validate=False)


def make_weights(cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = {}

    def r(name, *shape, scale=0.3):
        W[name] = torch.randn(*shape, generator=g) * scale

    def ones(name, n):
        W[name] = 1 + 0.1 * torch.randn(n, generator=g)

    def hc(p, inject=True):
        ones(p + ".hc_norm.weight", HC * H)
        r(p + ".input_mix_weight_down.weight", 6, HC * H)
        r(p + ".input_mix_weight_up.weight", HC * H, 6)
        if inject:
            r(p + ".block_inject_weight.weight", HC, HC * H)

    hc("hyper_connection_mixer", inject=False)
    r("lm_head.weight", V, H)
    nk, nv, dk, dv = 2, 4, 4, 4
    for i, t in enumerate(cfg.layer_types):
        p = f"layers.{i}."
        hc(p + "attn_hyper_connection")
        hc(p + "mlp_hyper_connection")
        r(p + "mlp.gate.weight", E, H)
        for n in ("gate", "up"):
            r(p + f"mlp.shared_expert.{n}_proj.weight", I, H)
        r(p + "mlp.shared_expert.down_proj.weight", H, I)
        r(p + "mlp.shared_expert_gate.weight", 1, H)
        if t == "linear_attention":
            a = p + "linear_attn."
            r(a + "in_proj_qkv.weight", 2 * nk * dk + nv * dv, H)
            r(a + "in_proj_z.weight", nv * dv, H)
            r(a + "in_proj_b.weight", nv, H)
            r(a + "in_proj_a.weight", nv, H)
            r(a + "conv1d.weight", 2 * nk * dk + nv * dv, 4)
            r(a + "A_log", nv, scale=0.3)
            r(a + "dt_bias", nv, scale=0.3)
            ones(a + "norm.weight", dv)
            r(a + "out_proj.weight", H, nv * dv)
        else:
            a = p + "self_attn."
            r(a + "q_proj.weight", 2 * 8 * 2, H)
            r(a + "k_proj.weight", 8, H)
            r(a + "v_proj.weight", 8, H)
            r(a + "o_proj.weight", H, 16)
            ones(a + "q_norm.weight", 8)
            ones(a + "k_norm.weight", 8)
            r(a + "indexer.index_qk_proj.weight", 3 * 4, H)
            ones(a + "indexer.q_layernorm.weight", 4)
            ones(a + "indexer.k_layernorm.weight", 4)
        if i == 1:
            e = p + "ple."
            r(e + "key_proj.weight", HC * H, H)
            r(e + "value_proj.weight", H, H)
            for n in ("norm_key", "norm_query", "norm_conv"):
                ones(e + n + ".weight", HC * H)
            r(e + "conv1d.weight", HC * H, 4)
    experts = {(i, e): tuple(torch.randn(*s, generator=g) * 0.3 for s in ((I, H), (I, H), (H, I)))
               for i in range(4) for e in range(E)}
    table = torch.randn(V, H, generator=g)
    emb = torch.randn(V, H, generator=g)
    return W, experts, table, emb


def make_model(cfg, seed=0):
    W, experts, table, emb = make_weights(cfg, seed)
    return R.RefModel(cfg, W.__getitem__, lambda ids: emb[ids], lambda L, e: experts[(L, e)],
                      lambda hist, ids: table[torch.tensor(ids)]), W


def test_folded_norm_equals_unfolded():
    x = torch.randn(5, 12)
    raw = torch.randn(12)
    folded = 1 + raw
    unfolded = R.rmsnorm(x, torch.ones(12), 1e-6) * (1.0 + raw)
    assert torch.allclose(R.rmsnorm(x, folded, 1e-6), unfolded, atol=1e-6)
    # applying 1 + w to an already folded weight is wrong by O(1)
    wrong = R.rmsnorm(x, 1 + folded, 1e-6)
    assert (wrong - unfolded).abs().max() > 0.1


def test_recurrent_matches_chunk_over_70_positions():
    g = torch.Generator().manual_seed(1)
    T, nh, dk, dv = 70, 3, 8, 8
    q, k, v = (torch.randn(1, T, nh, d, generator=g) for d in (dk, dk, dv))
    gate = -torch.rand(1, T, nh, generator=g) * 0.5
    beta = torch.rand(1, T, nh, generator=g)
    oc, Sc = R.chunk_gated_delta_rule(q, k, v, gate, beta)
    orr, Sr = R.recurrent_gated_delta_rule(q, k, v, gate, beta, torch.zeros(1, nh, dk, dv))
    assert (oc - orr).abs().max() < 1e-5
    assert (Sc - Sr).abs().max() < 1e-5


def test_indexer_none_below_budget_and_selects_blocks_above():
    cfg = tiny_cfg(budget=8, ratio=2)      # keeps top 4 blocks of 2 tokens
    m, _ = make_model(cfg)
    st = dict(k=None, v=None, ik=None, bk=None)
    cos, sin = m.rope(torch.arange(0, 8))
    x = torch.randn(8, H)
    assert m.indexer_mask(3, st, x, cos[:, None, :], sin[:, None, :]) is None     # 4 blocks <= 4
    cos, sin = m.rope(torch.arange(8, 24))
    x2 = torch.randn(16, H)
    mask = m.indexer_mask(3, st, x2, cos[:, None, :], sin[:, None, :])
    assert mask.shape == (16, 24)
    pos = torch.arange(8, 24)
    for t in range(16):
        p = int(pos[t])
        ncb = (p + 1) // 2
        if p % 2 == 0:
            assert mask[t, p], "own token is in the incomplete block tail"
        assert not mask[t, p + 1:].any(), "causal"
        tail = list(range(ncb * 2, p + 1))
        assert mask[t, tail].all()
        full_blocks = mask[t, :ncb * 2].view(ncb, 2)
        assert (full_blocks[:, 0] == full_blocks[:, 1]).all()
        assert int(full_blocks[:, 0].sum()) == min(4, ncb)


def test_ple_five_tokens_equal_five_single_calls():
    cfg = tiny_cfg()
    m, _ = make_model(cfg)
    g = torch.Generator().manual_seed(3)
    ids = [3, 7, 1, 9, 4]
    hyp = torch.randn(5, HC * H, generator=g)
    a = m.ple(1, hyp, ids)
    m.reset()
    b = torch.cat([m.ple(1, hyp[t:t + 1], ids[t:t + 1]) for t in range(5)], 0)
    assert torch.allclose(a, b, atol=1e-5)
    # the conv state carries 9 rows and is what makes the calls continuous
    assert m.ple_conv.shape[0] == 9
    m.reset()
    broken = torch.cat([(m.reset(), m.ple(1, hyp[t:t + 1], ids[t:t + 1]))[1] for t in range(5)], 0)
    assert (a - broken).abs().max() > 1e-3


def test_full_forward_chunked_equals_token_by_token():
    cfg = tiny_cfg(budget=8, ratio=2)
    m, _ = make_model(cfg)
    ids = [5, 2, 9, 1, 14, 3, 8, 8, 0, 21, 6, 30, 11, 2]
    full = m.forward(ids, all_logits=True)
    sel_full = m.last_selected[3]
    assert sel_full is not None, "the tiny budget must activate the indexer"
    m.reset()
    one = torch.cat([m.forward([t], all_logits=True) for t in ids], 0)
    assert full.shape == (len(ids), V)
    assert (full - one).abs().max() < 1e-3
