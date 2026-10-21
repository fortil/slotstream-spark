"""tools/repack_q4_1.py on synthetic tensors (plan step 94): the repack is exact, nothing is requantized."""
import csv
import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from slotstream_spark.model import mlx_ckpt
import synth_ckpt

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("repack_q4_1_mod", ROOT / "tools/repack_q4_1.py")
rq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rq)
bf16 = synth_ckpt.bf16


def f16_round(bf):
    """bf16 bits -> the fp32 value of the same number rounded through fp16 (what Q4_1 stores)."""
    return rq.bf16_to_f32(bf).astype(np.float16).astype(np.float32)


def synth(rng, rows, K, gs, neg=True):
    q = rng.integers(0, 16, size=(rows, K), dtype=np.uint8)
    w = np.zeros((rows, K // 8), dtype=np.uint32)
    for i in range(8):
        w |= q[:, i::8].astype(np.uint32) << (4 * i)
    sc = rng.uniform(0.002, 0.08, size=(rows, K // gs)).astype(np.float32)
    if neg:
        sc *= rng.choice([-1.0, 1.0], size=sc.shape)
    bi = rng.uniform(-0.4, 0.4, size=(rows, K // gs)).astype(np.float32)
    return q, w, bf16(sc), bf16(bi)


@pytest.mark.parametrize("gs", [64, 32])
def test_ggml_dequant_equals_mlx_bit_for_bit(gs):
    rng = np.random.default_rng(1)
    q, w, s, b = synth(rng, 7, 256, gs)
    blocks = rq.pack_q4_1(rq.unpack_mlx(w), s, b, gs, "t")
    mlx = mlx_ckpt.dequant_affine(w, bf16(f16_round(s)), bf16(f16_round(b)))        # q*s + b with s, b rounded through fp16
    assert np.array_equal(rq.dequant_q4_1(blocks), mlx)
    assert np.array_equal(rq.dequant_q4_1(blocks).view(np.uint32), mlx.view(np.uint32))


def test_negative_scales_are_kept_and_half_the_groups_are_negative():
    rng = np.random.default_rng(2)
    q, w, s, b = synth(rng, 16, 512, 64)
    blocks = rq.pack_q4_1(rq.unpack_mlx(w), s, b, 64, "t")
    _, d, m = rq.unpack_q4_1(blocks)
    assert (d < 0).any() and (d > 0).any()
    assert np.array_equal(d[:, ::2], d[:, 1::2]) and np.array_equal(m[:, ::2], m[:, 1::2])       # two blocks share (d, m)
    assert np.array_equal(d[:, ::2], f16_round(s))


def test_nibble_round_trip_and_order():
    rng = np.random.default_rng(3)
    q, w, s, b = synth(rng, 5, 128, 64)
    assert np.array_equal(rq.unpack_mlx(w), q)
    blocks = rq.pack_q4_1(q, s, b, 64, "t")
    back, _, _ = rq.unpack_q4_1(blocks)
    assert np.array_equal(back, q)
    qs = blocks[:, :, 4:]
    assert np.array_equal(qs & 0xF, q.reshape(5, 4, 2, 16)[:, :, 0, :])          # low nibble of qs[j] is element j
    assert np.array_equal(qs >> 4, q.reshape(5, 4, 2, 16)[:, :, 1, :])           # high nibble is element j + 16
    word0 = int(w[0, 0])
    assert [(word0 >> (4 * i)) & 0xF for i in range(8)] == list(q[0, :8])        # MLX order: element i at bits 4i


def test_a_group_that_uses_few_levels_is_not_rescaled():
    """A requantizer would recompute (d, m) from min and max; the repack keeps the q codes and (d, m) as they are."""
    q = np.full((1, 64), 5, dtype=np.uint8)
    q[0, ::3] = 6
    w = np.zeros((1, 8), dtype=np.uint32)
    for i in range(8):
        w |= q[:, i::8].astype(np.uint32) << (4 * i)
    s, b = bf16(np.array([[-0.0625]], dtype=np.float32)), bf16(np.array([[0.75]], dtype=np.float32))
    blocks = rq.pack_q4_1(rq.unpack_mlx(w), s, b, 64, "t")
    back, d, m = rq.unpack_q4_1(blocks)
    assert np.array_equal(back, q) and (d == -0.0625).all() and (m == 0.75).all()


def test_the_tool_has_no_requantize_path():
    src = (ROOT / "tools/repack_q4_1.py").read_text()
    import re
    for banned in (r"make_qkx", r"(?<!de)quantize_row", r"llama-quantize", r"imatrix_weights", r"def requantize"):
        assert not re.search(banned, src.replace("quantize.imatrix.", "")), banned


@pytest.mark.parametrize("value,ok", [(2.0 ** -17, True), (65280.0, True), (0.0, True), (-0.0625, True), (2.0 ** -20, True),
                                      (1.9921875 * 2.0 ** -20, False), (3e-7, False), (1e-6, False), (65536.0, False),
                                      (70000.0, False), (-1e9, False)])
def test_fp16_exactness_boundaries(value, ok):
    a = bf16(np.array([[value]], dtype=np.float32))
    if ok:
        assert rq.bf16_to_f16_checked(a, "x").shape == (1, 1)
    else:
        with pytest.raises(rq.ExactnessError, match="layers.3.mlp.scales"):
            rq.bf16_to_f16_checked(a, "layers.3.mlp.scales")


def test_non_finite_is_refused_and_pack_names_the_tensor():
    bad = np.array([[0x7F80]], dtype=np.uint16)                                   # +inf in bf16
    with pytest.raises(rq.ExactnessError, match="non-finite"):
        rq.bf16_to_f16_checked(bad, "t")
    rng = np.random.default_rng(4)
    q, w, s, b = synth(rng, 2, 64, 64)
    s[1, 0] = bf16(np.array([3e-7], dtype=np.float32))[0]
    with pytest.raises(rq.ExactnessError, match="blk.0.ffn_up_exps"):
        rq.pack_q4_1(q, s, b, 64, "blk.0.ffn_up_exps")


def test_row_and_column_permutations_equal_permuting_the_dequantized_matrix():
    rng = np.random.default_rng(5)
    q, w, s, b = synth(rng, 12, 256, 64)

    class Src:
        def __init__(self):
            self.t = {"x.weight": w, "x.scales": s, "x.biases": b}

        def get(self, n):
            return self.t[n]

    ref = mlx_ckpt.dequant_affine(w, bf16(f16_round(s)), bf16(f16_round(b)))
    ridx = rng.permutation(12)
    sp = rq.AffineSpec("t", (256, 12), Src(), "x", row_idx=ridx)
    assert np.array_equal(rq.dequant_q4_1(sp.rows(0, 12).reshape(12, -1, 20)), ref[ridx])
    cp = rq.v_reorder_index(256, 2, 2, 64)                                          # moves whole 64-column runs
    sp = rq.AffineSpec("t", (256, 12), Src(), "x", col_perm=cp)
    assert np.array_equal(rq.dequant_q4_1(sp.rows(0, 12).reshape(12, -1, 20)), ref[:, cp])
    with pytest.raises(rq.MappingError, match="whole quantization groups"):
        rq.group_perm_from_cols(rng.permutation(256), 64)


def test_v_reorder_is_the_converter_permutation():
    # grouped [G0v0 G0v1 G0v2 G1v0 G1v1 G1v2] -> tiled [G0v0 G1v0 G0v1 G1v1 G0v2 G1v2]
    assert list(rq.v_reorder_index(6, 2, 3, 1)) == [0, 3, 1, 4, 2, 5]
    assert sorted(rq.v_reorder_index(48 * 128, 16, 3, 128)) == list(range(48 * 128))


def test_q8_0_and_q6_k_dequant_follow_ggml():
    rng = np.random.default_rng(6)
    d = np.float16(0.0123)
    qs = rng.integers(-128, 127, size=(2, 32), dtype=np.int8)
    raw = b"".join(d.tobytes() + qs[r].tobytes() for r in range(2))
    assert np.allclose(rq.dequant_q8_0(raw, 2, 32), qs.astype(np.float32) * np.float32(d))
    # Q6_K: encode known 6-bit values (q in 0..63, value (q - 32) * sc * d) with ggml's bit layout
    vals = rng.integers(0, 64, size=256)
    sc = rng.integers(-20, 20, size=16).astype(np.int8)
    dd = np.float16(0.01)
    ql, qh = np.zeros(128, dtype=np.uint8), np.zeros(64, dtype=np.uint8)
    for half in range(2):
        for l in range(32):
            v = vals[128 * half + l:128 * half + 128:32][:4] if False else [vals[128 * half + l + 32 * k] for k in range(4)]
            ql[64 * half + l] = (v[0] & 0xF) | ((v[2] & 0xF) << 4)
            ql[64 * half + l + 32] = (v[1] & 0xF) | ((v[3] & 0xF) << 4)
            qh[32 * half + l] = (v[0] >> 4) | ((v[1] >> 4) << 2) | ((v[2] >> 4) << 4) | ((v[3] >> 4) << 6)
    got = rq.dequant_q6_k(ql.tobytes() + qh.tobytes() + sc.tobytes() + dd.tobytes(), 1, 256)[0]
    want = np.empty(256, dtype=np.float32)
    for half in range(2):
        for k in range(4):
            for l in range(32):
                want[128 * half + 32 * k + l] = float(dd) * float(sc[8 * half + 2 * k + l // 16]) * (int(vals[128 * half + 32 * k + l]) - 32)
    assert np.allclose(got, want, rtol=1e-6, atol=1e-7)


# ------------------------------------------------------------------------------------------------ gguf + build
ALIGN = 32


def kv_str(key, value):
    return key, rq._gstr(key) + struct.pack("<I", 8) + rq._gstr(value)


def kv_u32(key, value):
    return rq.kv_record_u32(key, value)


def kv_i64_arr(key, vals):
    return key, rq._gstr(key) + struct.pack("<IIQ", 9, 11, len(vals)) + b"".join(struct.pack("<q", v) for v in vals)


def write_template(path, tensors, kv):
    """tensors: [(name, ne, gtype, raw bytes)]; a single-file GGUF."""
    infos = [(n, ne, t, len(raw)) for n, ne, t, raw in tensors]
    head = rq.gguf_header(kv, infos, ALIGN)
    with open(path, "wb") as f:
        f.write(head)
        for _n, _ne, _t, raw in tensors:
            f.write(raw + b"\0" * (-len(raw) % ALIGN))


def make_world(tmp_path, seed=7, noise=0.01):
    rng = np.random.default_rng(seed)
    q, w, s, b = synth(rng, 16, 128, 64)
    norm = bf16(rng.uniform(0.5, 1.5, size=64).astype(np.float32))
    consts = [23703573157769, 20109073645365]
    ck = tmp_path / "mlx"
    ck.mkdir()
    synth_ckpt.write_safetensors(ck / "model.safetensors", {
        "m.lin.weight": w, "m.lin.scales": s, "m.lin.biases": b, "m.norm.weight": norm,
        "language_model.model.layers.1.ple.ple_embedding.layer_multipliers": np.array(consts, dtype=np.int64)})
    (ck / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
        k: "model.safetensors" for k in ("m.lin.weight", "m.lin.scales", "m.lin.biases", "m.norm.weight",
                                                      "language_model.model.layers.1.ple.ple_embedding.layer_multipliers")}}))
    # the template: a noisy Q4_1 of the same matrix, an F32 norm, split and imatrix KV that must not reach the output
    dq = mlx_ckpt.dequant_affine(w, s, b)
    tb = rq.pack_q4_1(rq.unpack_mlx(w), s, b, 64, "tmpl")
    tn = rq.bf16_to_f32(norm).astype("<f4")
    kv = [kv_str("general.architecture", "qwen4exp"), kv_u32("qwen4exp.block_count", 2), kv_i64_arr("qwen4exp.ple.layer_multipliers", consts),
          kv_u32("split.no", 0), kv_u32("split.count", 1), kv_str("quantize.imatrix.file", "x.gguf")]
    tpl = tmp_path / "tmpl.gguf"
    write_template(tpl, [("lin.weight", (128, 16), rq.GGML_Q4_1, tb.tobytes()), ("norm.weight", (64,), rq.GGML_F32, tn.tobytes())], kv)
    return {"ck": ck, "tpl": tpl, "q": q, "w": w, "s": s, "b": b, "norm": norm, "dq": dq, "consts": consts}


def rules_ok(src, cfg):
    return [(r"lin\.weight$", lambda m, t: rq.AffineSpec(t.name, t.ne, src, "m.lin")),
            (r"norm\.weight$", lambda m, t: rq.DenseSpec(t.name, t.ne, t.gtype, src, ["m.norm.weight"]))]


def build(world, out, rules=rules_ok):
    return rq.run_build(world["ck"], world["tpl"], out, max_mbps=0, allow_buffered=True, rules_fn=rules, o_direct=False,
                        log=lambda *a: None, cfg={})


def test_gguf_header_round_trip(tmp_path):
    w = make_world(tmp_path)
    t = rq.read_gguf([w["tpl"]])
    assert [x.name for x in t.tensors] == ["lin.weight", "norm.weight"]
    assert t.tensors[0].ne == (128, 16) and t.tensors[0].gtype == rq.GGML_Q4_1 and t.tensors[0].offset % ALIGN == 0
    assert t.values["qwen4exp.block_count"] == 2 and t.values["qwen4exp.ple.layer_multipliers"] == w["consts"]


def test_end_to_end_build_is_exact_and_clean(tmp_path):
    w = make_world(tmp_path)
    out = tmp_path / "out"
    man = build(w, out)
    assert sorted(p.name for p in out.iterdir()) == ["manifest.json", "model.gguf"]                # atomic: no .tmp left
    g = rq.read_gguf([out / "model.gguf"])
    assert [t.name for t in g.tensors] == ["lin.weight", "norm.weight"]
    assert all(t.offset % ALIGN == 0 for t in g.tensors)
    keys = [k for k, _ in g.kv]
    assert keys == ["general.architecture", "qwen4exp.block_count", "qwen4exp.ple.layer_multipliers"]     # split.* and imatrix dropped
    tmpl_kv = dict(rq.read_gguf([w["tpl"]]).kv)
    assert all(dict(g.kv)[k] == tmpl_kv[k] for k in keys)                                           # the others byte for byte
    data = (out / "model.gguf").read_bytes()
    lin = g.tensors[0]
    want = rq.pack_q4_1(rq.unpack_mlx(w["w"]), w["s"], w["b"], 64, "x").tobytes()
    assert data[lin.offset:lin.offset + len(want)] == want
    got = rq.dequant_q4_1(np.frombuffer(data[lin.offset:lin.offset + len(want)], dtype=np.uint8).reshape(16, 4, 20))
    assert np.array_equal(got, mlx_ckpt.dequant_affine(w["w"], bf16(f16_round(w["s"])), bf16(f16_round(w["b"]))))
    nrm = g.tensors[1]
    assert np.array_equal(np.frombuffer(data[nrm.offset:nrm.offset + 256], dtype="<f4"), rq.bf16_to_f32(w["norm"]))
    import hashlib
    m = {t["name"]: t for t in man["tensors"]}
    assert m["lin.weight"]["sha256"] == hashlib.sha256(want).hexdigest() and m["lin.weight"]["corr"] > 0.999
    assert json.loads((out / "manifest.json").read_text())["checkpoint_revision"] == "aa7c790e"


def test_unmapped_template_tensor_is_refused(tmp_path):
    w = make_world(tmp_path)
    with pytest.raises(rq.MappingError, match="norm.weight"):
        build(w, tmp_path / "out", lambda src, cfg: rules_ok(src, cfg)[:1])
    assert not (tmp_path / "out" / "model.gguf").exists()


def test_a_wrong_mapping_is_caught_by_the_template_correlation(tmp_path):
    w = make_world(tmp_path)
    rng = np.random.default_rng(99)
    synth_ckpt.write_safetensors(w["ck"] / "other.safetensors", {
        "o.weight": synth(rng, 16, 128, 64)[1], "o.scales": bf16(rng.uniform(0.002, 0.08, size=(16, 2)).astype(np.float32)),
        "o.biases": bf16(rng.uniform(-0.4, 0.4, size=(16, 2)).astype(np.float32))})
    idx = json.loads((w["ck"] / "model.safetensors.index.json").read_text())
    idx["weight_map"].update({k: "other.safetensors" for k in ("o.weight", "o.scales", "o.biases")})
    (w["ck"] / "model.safetensors.index.json").write_text(json.dumps(idx))

    def wrong(src, cfg):
        return [(r"lin\.weight$", lambda m, t: rq.AffineSpec(t.name, t.ne, src, "o")), rules_ok(src, cfg)[1]]
    with pytest.raises(rq.MappingError, match="correlation"):
        build(w, tmp_path / "out", wrong)


def test_a_missing_or_doubled_plus_one_is_caught(tmp_path):
    w = make_world(tmp_path)

    def doubled(src, cfg):
        return [rules_ok(src, cfg)[0], (r"norm\.weight$", lambda m, t: rq.DenseSpec(t.name, t.ne, t.gtype, src, ["m.norm.weight"],
                                                                                    lambda a: a + 1.0))]
    with pytest.raises(rq.MappingError, match=r"mean offset"):
        build(w, tmp_path / "out", doubled)


def test_checkpoint_hash_constants_must_equal_the_templates(tmp_path):
    w = make_world(tmp_path)
    ck = mlx_ckpt.Checkpoint(w["ck"], allow_buffered=True)
    t = rq.read_gguf([w["tpl"]])

    class Ck:
        def read_array(self, name):
            return np.array([1, 2], dtype=np.int64)
    with pytest.raises(rq.MappingError, match="int64 constants differ"):
        rq.check_hash_constants(None, t, Ck())
    ck.close()


def test_sweep_finds_the_inexact_scale_and_names_it(tmp_path):
    rng = np.random.default_rng(8)
    q, w, s, b = synth(rng, 4, 128, 64)
    ok = {"m.a.weight": w, "m.a.scales": s, "m.a.biases": b, "m.n.weight": bf16(np.ones(8, dtype=np.float32))}
    d = tmp_path / "ck"
    d.mkdir()
    synth_ckpt.write_safetensors(d / "model.safetensors", ok)
    with mlx_ckpt.Checkpoint(d, allow_buffered=True) as ck:
        res = rq.sweep(ck, log=lambda *a: None)
    assert res["exact"] and res["tensors"] == 4 and res["scales_biases"] == 2
    s2 = s.copy()
    s2[2, 1] = bf16(np.array([3e-7], dtype=np.float32))[0]
    synth_ckpt.write_safetensors(d / "model.safetensors", dict(ok, **{"m.a.scales": s2}))
    with mlx_ckpt.Checkpoint(d, allow_buffered=True) as ck:
        res = rq.sweep(ck, log=lambda *a: None)
    assert not res["exact"] and "m.a.scales" in res["bad"][0]


# ------------------------------------------------------------------------------------------------ the real template
def template_rows():
    out = []
    for line in (ROOT / "tests/fixtures/q41_template_tensors.txt").read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, ne, gt = line.split()
        out.append(rq.GgufTensor(name, tuple(int(x) for x in ne.split(",")), int(gt), 0, "fixture"))
    return out


class FakeSrc:
    def get(self, name):
        if name.endswith("ple.conv1d.weight"):
            return np.zeros((10240, 4, 1), dtype=np.uint16)
        return np.zeros((640, 2560), dtype=np.uint16)


def real_specs():
    cfg = {"linear_num_key_heads": 16, "linear_num_value_heads": 48, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
           "indexer_n_heads": 4, "indexer_head_dim": 128}
    tmpl = rq.GgufFile(tensors=template_rows())
    return rq.plan_tensors(tmpl, rq.make_rules(FakeSrc(), cfg), FakeSrc()), tmpl


def test_every_tensor_of_the_real_template_has_one_rule_and_a_source_in_the_checkpoint():
    specs, tmpl = real_specs()
    assert len(specs) == len(tmpl.tensors) == 1224
    csv_path = ROOT / "third_party/qwen38-flash-next-metadata/tensors.csv"
    if not csv_path.exists():
        pytest.skip("third_party/qwen38-flash-next-metadata is not synced to the Spark")
    rows = {r["name"]: r for r in csv.DictReader(open(csv_path))}
    for s in specs:
        for n in s.sources:
            assert n in rows, f"{s.name}: source {n} is not in the checkpoint"
        if isinstance(s, rq.AffineSpec):
            assert tuple(int(x) for x in rows[s.sources[0]]["logical_shape"].split("x")) == tuple(reversed(s.ne)), s.name
            assert s.gtype == (rq.GGML_BF16 if rq.BF16_AFFINE_RE.match(s.name) else rq.GGML_Q4_1)
        elif isinstance(s, rq.DenseSpec) and s.names[0] in rows and "index_qk" not in s.names[0]:
            numel = int(np.prod([int(x) for x in rows[s.names[0]]["shape"].split("x")]))
            assert numel == int(np.prod(s.ne)), s.name


def test_real_template_coverage_counts():
    specs, _ = real_specs()
    aff = [s for s in specs if isinstance(s, rq.AffineSpec)]
    assert sum(1 for s in specs if isinstance(s, rq.NgramSpec)) == 1
    assert len(aff) + 1 + sum(1 for s in specs if isinstance(s, rq.DenseSpec)) == 1224
    for s in specs:
        if ("_exps" in s.name or ("_shexp" in s.name and "gate_inp" not in s.name)) or s.name.endswith(("attn_qkv.weight", "attn_gate.weight", "ssm_out.weight")):
            assert isinstance(s, rq.AffineSpec), s.name
    names = {s.name for s in specs}
    assert "per_layer_token_embd.weight" in names and "blk.47.ffn_down_exps.weight" in names and "blk.3.indexer.k_proj.weight" in names


def test_v_reordered_tensors_use_the_converter_permutations():
    specs, _ = real_specs()
    by = {s.name: s for s in specs}
    q = by["blk.0.attn_qkv.weight"]
    assert list(q.row_idx[:4096]) == list(range(4096)) and sorted(q.row_idx) == list(range(10240))
    assert q.row_idx[4096] == 4096 and q.row_idx[4096 + 128] == 4096 + 3 * 128 and q.row_idx[4096 + 127] == 4096 + 127
    assert by["blk.0.attn_gate.weight"].row_idx[1] == 1 and by["blk.0.attn_gate.weight"].row_idx[128] == 3 * 128
    assert by["blk.0.ssm_out.weight"].col_perm[128] == 3 * 128 and by["blk.0.ssm_out.weight"].row_idx is None


def test_a_template_f16_tensor_that_is_not_exact_in_fp16_is_written_as_f32():
    class S:
        def __init__(self, a):
            self.a = a

        def get(self, n):
            return self.a
    exact = rq.DenseSpec("c", (4, 2), rq.GGML_F16, S(bf16(np.array([[0.5, 1.0, -2.0, 0.25], [0.125, 3.0, 1.5, -1.0]], dtype=np.float32))), ["x"])
    assert exact.gtype == rq.GGML_F16 and exact.override is None and exact.rbytes == 8
    odd = bf16(np.array([[1.9921875 * 2.0 ** -20, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], dtype=np.float32))
    over = rq.DenseSpec("c", (4, 2), rq.GGML_F16, S(odd), ["x"])
    assert over.gtype == rq.GGML_F32 and over.rbytes == 16 and "not all exact in fp16" in over.override
    assert np.array_equal(np.frombuffer(over.rows(0, 2).tobytes(), dtype="<f4").reshape(2, 4), rq.bf16_to_f32(odd))


def test_q5_1_dequant_follows_ggml():
    rng = np.random.default_rng(11)
    x = rng.integers(0, 32, size=(2, 32))
    d, m = np.float16(0.02), np.float16(-0.3)
    raw = b""
    for r in range(2):
        qh = 0
        qs = bytearray(16)
        for j in range(16):
            qh |= (int(x[r, j]) >> 4 & 1) << j
            qh |= (int(x[r, j + 16]) >> 4 & 1) << (j + 16)
            qs[j] = (int(x[r, j]) & 0xF) | ((int(x[r, j + 16]) & 0xF) << 4)
        raw += d.tobytes() + m.tobytes() + struct.pack("<I", qh) + bytes(qs)
    want = x.astype(np.float32) * np.float32(d) + np.float32(m)
    assert np.array_equal(rq.dequant_q5_1(raw, 2, 32), want)


# ------------------------------------------------------------------------------------------------ named exceptions (step 94a)
def scale_with(value, rows=2, cols=2):
    rng = np.random.default_rng(11)
    q, w, s, b = synth(rng, rows, 64 * cols, 64)
    s = s.copy()
    s[1, 0] = bf16(np.array([value], dtype=np.float32))[0]
    return q, s, b


def test_a_named_tensor_with_a_tiny_scale_packs_and_is_recorded():
    q, s, b = scale_with(3e-7)
    exc = {"blk.t.scales": 2.0 ** -24}
    rq.TOLERATED.clear()
    rq.pack_q4_1(q, s, b, 64, "blk.t", exc)
    assert rq.TOLERATED["blk.t.scales"]["count"] == 1 and rq.TOLERATED["blk.t.scales"]["max_abs_err"] <= 2.0 ** -24
    assert "blk.t.biases" not in rq.TOLERATED
    with pytest.raises(rq.ExactnessError, match="blk.u.scales"):                    # same scale, tensor not named
        rq.pack_q4_1(q, s, b, 64, "blk.u", exc)
    with pytest.raises(rq.ExactnessError, match="blk.t.scales"):
        rq.pack_q4_1(q, s, b, 64, "blk.t")                                         # no exceptions at all


def test_a_named_tensor_that_overflows_fp16_still_raises():
    q, s, b = scale_with(70000.0)
    with pytest.raises(rq.ExactnessError, match="blk.t.scales"):
        rq.pack_q4_1(q, s, b, 64, "blk.t", {"blk.t.scales": 2.0 ** -24})


def test_a_named_tensor_beyond_its_tolerance_raises():
    v = bf16(np.array([[3e-7]], dtype=np.float32))                                  # fp16 rounding error about 2^-26 here
    err = float(abs(rq.bf16_to_f32(v).astype(np.float16).astype(np.float64) - rq.bf16_to_f32(v).astype(np.float64))[0, 0])
    assert 0 < err <= 2.0 ** -24
    with pytest.raises(rq.ExactnessError, match=r"x.scales: 1 values exceed the tolerated 2\^-24"):
        rq.bf16_to_f16_checked(v, "x.scales", {"x.scales": err / 2})
    assert rq.bf16_to_f16_checked(v, "x.scales", {"x.scales": 2.0 ** -24}).shape == (1, 1)


def test_sweep_reports_a_named_inexact_tensor_as_tolerated_and_an_unnamed_one_as_bad(tmp_path):
    rng = np.random.default_rng(8)
    q, w, s, b = synth(rng, 4, 128, 64)
    s2 = s.copy()
    s2[2, 1] = bf16(np.array([3e-7], dtype=np.float32))[0]
    d = tmp_path / "ck"
    d.mkdir()
    synth_ckpt.write_safetensors(d / "model.safetensors", {"m.a.weight": w, "m.a.scales": s2, "m.a.biases": b})
    with mlx_ckpt.Checkpoint(d, allow_buffered=True) as ck:
        res = rq.sweep(ck, log=lambda *a: None, exceptions={})
        assert not res["exact"] and res["tolerated"] == [] and "m.a.scales" in res["bad"][0]
        res = rq.sweep(ck, log=lambda *a: None, exceptions={"m.a.scales": 2.0 ** -24})
    assert res["exact"] and res["bad"] == []
    assert res["tolerated"] == [{"name": "m.a.scales", "count": 1, "max_abs_err": res["tolerated"][0]["max_abs_err"]}]
    assert res["tolerated"][0]["max_abs_err"] <= 2.0 ** -24 and res["abs_tol"] == 2.0 ** -24 and res["exceptions_file"].endswith("q4_1_exceptions.json")


def test_family_correlation_floors():
    assert rq.corr_floor("blk.3.hc_attn_up.weight") == (0.95, "hyper_connection")
    assert rq.corr_floor("output_hc_down.weight") == (0.95, "hyper_connection")
    assert rq.corr_floor("blk.12.hc_ffn_down.weight") == (0.95, "hyper_connection")
    assert rq.corr_floor("blk.3.ffn_gate_exps.weight") == (0.95, "other")
    assert rq.corr_floor("blk.3.hc_attn_inject.weight") == (0.95, "other")


class FakeRowsSpec:
    kind, gtype, ne, sources, sample_k = "dense", None, (64,), ("src",), 4

    def __init__(self, name, ours):
        self.name, self.ours = name, ours
        self.gtype, self.nrows = rq.GGML_F32, ours.shape[0]

    def rows(self, i0, i1):
        return self.ours[i0:i1].astype("<f4").view(np.uint8)


def test_check_against_template_uses_one_floor(monkeypatch):
    rng = np.random.default_rng(5)
    theirs = rng.standard_normal((4, 64)).astype(np.float32)
    near = theirs + 0.25 * rng.standard_normal((4, 64)).astype(np.float32)           # correlation about 0.97
    far = theirs + 1.0 * rng.standard_normal((4, 64)).astype(np.float32)             # correlation about 0.7
    monkeypatch.setattr(rq, "read_template_rows", lambda t, rows: theirs[list(rows)])
    for name, fam in (("blk.3.hc_attn_up.weight", "hyper_connection"), ("blk.3.ffn_gate_exps.weight", "other")):
        r = rq.check_against_template(FakeRowsSpec(name, near), None, rows=[0, 1, 2, 3])
        assert 0.95 < r["corr"] < 0.99 and r["corr_floor"] == 0.95 and r["family"] == fam and r["ok"] and r["reason"] is None
        with pytest.raises(rq.MappingError, match=r"0\.95"):
            rq.check_against_template(FakeRowsSpec(name, far), None, rows=[0, 1, 2, 3])
        r = rq.check_against_template(FakeRowsSpec(name, far), None, rows=[0, 1, 2, 3], strict=False)
        assert r["ok"] is False and "0.95" in r["reason"] and name in r["reason"]


def test_min_corr_applies_to_every_family(monkeypatch):
    monkeypatch.setattr(rq, "MIN_CORRELATION", 0.5)
    assert rq.corr_floor("blk.3.hc_attn_up.weight") == (0.5, "hyper_connection")
    assert rq.corr_floor("blk.3.ffn_up.weight") == (0.5, "other")


def test_the_tracked_exceptions_file():
    d = json.loads((ROOT / "slotstream_spark/config/q4_1_exceptions.json").read_text())
    assert len(d["tensors"]) == 7 and len(set(d["tensors"])) == 7 and d["abs_tol"] == 2.0 ** -24 and d["abs_tol_pow2"] == -24
    csv_path = ROOT / "third_party/qwen38-flash-next-metadata/tensors.csv"
    if not csv_path.exists():
        pytest.skip("third_party/qwen38-flash-next-metadata is not synced to the Spark")
    rows = {r["name"]: r for r in csv.DictReader(open(csv_path))}
    assert rows, "tensors.csv has no rows"
    key = next(k for k in next(iter(rows.values())) if k.lower() in ("dtype", "tipo"))
    for n in d["tensors"]:
        assert n.endswith((".scales", ".biases")) and n in rows and rows[n][key] == "BF16", n
    assert set(rq.load_exceptions()) == set(d["tensors"])


def test_the_manifest_carries_exceptions_overrides_and_floors(tmp_path):
    w = make_world(tmp_path)
    man = build(w, tmp_path / "out")
    assert man["exceptions"] == [] and man["type_overrides"] == []
    for t in man["tensors"]:
        assert t["corr_floor"] == 0.95 and t["family"] in ("hyper_connection", "other")
    assert man["corr_floor"] == 0.95 and man["failures"] == [] and man["below_099"] == []


def _wrong_and_doubled(w):
    rng = np.random.default_rng(99)
    synth_ckpt.write_safetensors(w["ck"] / "other.safetensors", {
        "o.weight": synth(rng, 16, 128, 64)[1], "o.scales": bf16(rng.uniform(0.002, 0.08, size=(16, 2)).astype(np.float32)),
        "o.biases": bf16(rng.uniform(-0.4, 0.4, size=(16, 2)).astype(np.float32))})
    idx = json.loads((w["ck"] / "model.safetensors.index.json").read_text())
    idx["weight_map"].update({k: "other.safetensors" for k in ("o.weight", "o.scales", "o.biases")})
    (w["ck"] / "model.safetensors.index.json").write_text(json.dumps(idx))

    def rules(src, cfg):
        return [(r"lin\.weight$", lambda m, t: rq.AffineSpec(t.name, t.ne, src, "o")),
                (r"norm\.weight$", lambda m, t: rq.DenseSpec(t.name, t.ne, t.gtype, src, ["m.norm.weight"], lambda a: a + 1.0))]
    return rules


def test_check_only_collects_every_failure_and_writes_check_json(tmp_path):
    w = make_world(tmp_path)
    rules = _wrong_and_doubled(w)
    out = tmp_path / "out"
    with pytest.raises(rq.MappingError) as e:
        rq.run_build(w["ck"], w["tpl"], out, max_mbps=0, allow_buffered=True, rules_fn=rules, o_direct=False,
                     log=lambda *a: None, cfg={}, write=False)
    assert "lin.weight" in str(e.value) and "norm.weight" in str(e.value)
    chk = json.loads((out / "check.json").read_text())
    assert len(chk["failures"]) == 2 and all({"corr", "reason", "sources"} <= set(f) for f in chk["failures"])
    assert not (out / "model.gguf").exists()
    out2 = tmp_path / "out2"
    with pytest.raises(rq.MappingError, match="correlation"):
        rq.run_build(w["ck"], w["tpl"], out2, max_mbps=0, allow_buffered=True, rules_fn=rules, o_direct=False,
                     log=lambda *a: None, cfg={}, write=True)
    assert not (out2 / "model.gguf").exists()


def test_check_only_lists_the_band_between_the_floor_and_099(tmp_path, monkeypatch):
    w = make_world(tmp_path)
    real = rq.read_template_rows
    noise = np.random.default_rng(3)

    def noisy(t, rows):
        r = real(t, rows)
        if t.name == "lin.weight":
            r = r + 0.25 * float(r.std()) * noise.standard_normal(r.shape).astype(r.dtype)
        return r
    monkeypatch.setattr(rq, "read_template_rows", noisy)
    out = tmp_path / "out"
    man = rq.run_build(w["ck"], w["tpl"], out, max_mbps=0, allow_buffered=True, rules_fn=rules_ok, o_direct=False,
                       log=lambda *a: None, cfg={}, write=False)
    chk = json.loads((out / "check.json").read_text())
    assert chk["failures"] == [] and man["failures"] == []
    assert [x["name"] for x in chk["below_099"]] == ["lin.weight"] and chk["below_099"][0]["family"] == "other"
    assert 0.95 <= chk["below_099"][0]["corr"] < 0.99


# ---------------------------------------------------------------- F9c: hyper-connection matrices as BF16, row-level layout check
def test_f32_to_bf16_rounds_to_nearest_even():
    a = np.array([1.0, 1.00390625, 1.005859375, -2.5, 0.0], dtype=np.float32)       # 1 + 2^-8 is a tie: goes to the even 1.0
    bits = rq.f32_to_bf16_bits(a)
    assert list(bits) == [0x3F80, 0x3F80, 0x3F81, 0xC020, 0]
    assert np.array_equal(rq.bf16_to_f32(bits), np.array([1.0, 1.0, 1.0078125, -2.5, 0.0], dtype=np.float32))


class _TableSrc:
    def __init__(self, t):
        self.t = t

    def get(self, n):
        return self.t[n]


def test_affine_bf16_spec_is_the_dequantized_mlx_value_rounded_to_bf16():
    rng = np.random.default_rng(11)
    q, w, s, b = synth(rng, 12, 320, 64)                                            # K = 320 like the hc "up" matrices
    src = _TableSrc({"x.weight": w, "x.scales": s, "x.biases": b})
    sp = rq.AffineBf16Spec("blk.0.hc_attn_up.weight", (320, 12), src, "x")
    assert sp.gtype == rq.GGML_BF16 and sp.rbytes == 640 and sp.nbytes == 12 * 640 and sp.kind == "affine_bf16"
    got = rq.dequant_ggml(rq.GGML_BF16, np.ascontiguousarray(sp.rows(0, 12)).tobytes(), 12, 320)
    ref = mlx_ckpt.dequant_affine(w, s, b)                                          # the engine's q*s+b in fp32, bf16 scale and bias
    assert np.array_equal(got, rq.bf16_to_f32(rq.f32_to_bf16_bits(ref)))
    assert np.abs(got - ref).max() <= np.abs(ref).max() * 2.0 ** -8                  # bf16 rounding only
    # the same values the Q4_1 path holds (to fp16 on scale and bias), so the swap changes the kernel, not the weights
    q41 = rq.dequant_q4_1(rq.AffineSpec("t", (320, 12), src, "x").rows(0, 12).reshape(12, -1, 20))
    assert np.abs(got - q41).max() <= (np.abs(ref).max() + 1.0) * 2.0 ** -8
    sub = rq.dequant_ggml(rq.GGML_BF16, np.ascontiguousarray(sp.rows(3, 7)).tobytes(), 4, 320)
    assert np.array_equal(sub, got[3:7])


def test_hyper_connection_matrices_are_written_as_bf16_and_nothing_else_is():
    specs, _ = real_specs()
    bf = [s for s in specs if isinstance(s, rq.AffineBf16Spec)]
    assert len(bf) == 48 * 4 + 2 and {s.name for s in bf} == {s.name for s in specs if rq.BF16_AFFINE_RE.match(s.name)}
    assert all(s.gtype == rq.GGML_BF16 for s in bf)
    assert all(s.gtype == rq.GGML_Q4_1 for s in specs if isinstance(s, rq.AffineSpec) and not isinstance(s, rq.AffineBf16Spec))
    by = {s.name: s for s in specs}
    assert by["blk.5.hc_attn_up.weight"].sources[0].endswith("attn_hyper_connection.input_mix_weight_up.weight")
    assert by["blk.5.hc_ffn_down.weight"].sources[0].endswith("mlp_hyper_connection.input_mix_weight_down.weight")
    assert by["output_hc_up.weight"].sources[0].endswith("hyper_connection_mixer.input_mix_weight_up.weight")


def test_build_writes_bf16_tensors_and_lists_them_in_the_manifest(tmp_path):
    w = make_world(tmp_path)

    def rules(src, cfg):
        return [(r"lin\.weight$", lambda m, t: rq.AffineBf16Spec(t.name, t.ne, src, "m.lin")), rules_ok(src, cfg)[1]]
    out = tmp_path / "out"
    man = build(w, out, rules)
    assert man["dequantized_bf16"] == ["lin.weight"] and man["type_overrides"] == []
    g = rq.read_gguf([out / "model.gguf"])
    lin = g.tensors[0]
    assert lin.gtype == rq.GGML_BF16 and lin.ne == (128, 16)
    data = (out / "model.gguf").read_bytes()[lin.offset:lin.offset + 16 * 128 * 2]
    got = rq.bf16_to_f32(np.frombuffer(data, dtype="<u2")).reshape(16, 128)
    assert np.array_equal(got, rq.bf16_to_f32(rq.f32_to_bf16_bits(mlx_ckpt.dequant_affine(w["w"], w["s"], w["b"]))))
    assert man["tensors"][0]["corr"] > 0.999
    assert build(w, tmp_path / "out2")["dequantized_bf16"] == []


class _Rows:
    kind, ne, sources, sample_k = "q4_1", (64,), ("src",), 40
    gtype = rq.GGML_F32

    def __init__(self, ours):
        self.name, self.ours, self.nrows = "blk.9.attn_qkv.weight", ours, ours.shape[0]

    def rows(self, i0, i1):
        return self.ours[i0:i1].astype("<f4").view(np.uint8)


def test_one_mislaid_row_among_many_passes_the_pooled_floor_but_not_the_row_check(monkeypatch):
    rng = np.random.default_rng(21)
    theirs = rng.standard_normal((40, 64)).astype(np.float32)
    ours = theirs + 0.1 * rng.standard_normal((40, 64)).astype(np.float32)           # 4-bit-like noise: correlation about 0.995
    monkeypatch.setattr(rq, "read_template_rows", lambda t, rows: theirs[list(rows)])
    ok = rq.check_against_template(_Rows(ours), None, rows=list(range(40)))
    assert ok["ok"] and ok["rows_below_090"] == 0 and ok["rows_below_050"] == 0 and ok["row_corr_min"] > 0.97
    bad = ours.copy()
    bad[17] = theirs[3]                                                              # a head or an expert taken from the wrong place
    r = rq.check_against_template(_Rows(bad), None, rows=list(range(40)), strict=False)
    assert r["corr"] >= rq.MIN_CORRELATION                                            # the pooled correlation does not see it
    assert r["rows_below_050"] == 1 and r["ok"] is False and "sampled rows correlate below" in r["reason"]
    with pytest.raises(rq.MappingError, match="sampled rows correlate below"):
        rq.check_against_template(_Rows(bad), None, rows=list(range(40)))


def test_row_correlations_ignore_constant_rows_and_tiny_vectors():
    a = np.vstack([np.ones(16), np.arange(16.0)])
    r = rq.row_correlations(a, a)
    assert r["rows_scored"] == 1 and r["row_corr_min"] == pytest.approx(1.0)
    r = rq.row_correlations(np.ones((2, 4)), np.ones((2, 4)))
    assert r["rows_scored"] == 0 and r["row_corr_min"] is None
