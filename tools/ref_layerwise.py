#!/usr/bin/env python3
# SPDX-License-Identifier: MIT AND Apache-2.0
# Portions ported from helgard-orlm/qwen-flash-next-8gb@958d22e7ba0317750ccc38862b9667b1034a4dd1 tools/ref_check.py:96-224 (Apache-2.0, section 4); modified.
"""Layer-by-layer numerical oracle: transformers v5.19.0 qwen4_exp modules, one decoder layer at a time,
cacheless (the whole sequence at once), weights read from the MLX checkpoint through mlx_ckpt (O_DIRECT).

Only the weight source is replaced; the module algorithms are untouched:
  - routed experts: the loop of Qwen4ExpTextExperts.forward, with each hit expert dequantized from its 9 pieces;
  - n-gram table: rows dequantized from the checkpoint shards instead of an nn.Embedding of 320 M rows.
GDN and the short convs use the torch implementations (fla, causal_conv1d and kernels are blocked), so the
oracle does not share code with the engine's FLA kernels.

Modes:
  --check-names               CPU, meta device, headers only: name, shape and norm coverage
  --parity31 DIR [--limits FILE]   layers 0-3 on the six parity31 tokens vs the MLX goldens, per-layer limits from
                              tests/gates/parity31_limits.json
  --parity31-calibrate DIR    bf16/fp32/noexp/swap forwards on the same tokens; writes result.json (no verdict)
  --generate N --prompt TEXT --state DIR     greedy, cacheless, resumable
  --teacher-force IDS_JSON --out DIR         dump hidden states, router picks, n-gram ids and logits
Library: teacher_force(tokens, dump_dir=None, experts_mode="correct") -> logits (fp32 [T, vocab]).
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Keep the oracle independent of the engine's kernels: transformers falls back to its torch code.
for _m in ("fla", "causal_conv1d", "kernels"):
    sys.modules[_m] = None

import argparse
import hashlib
import importlib.util
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from slotstream_spark.model import mlx_ckpt  # noqa: E402
from slotstream_spark.model.mlx_ckpt import CENTERED_NORMS, Checkpoint, dequant_affine, hf_tensor  # noqa: E402

LM = "language_model."
P = "language_model.model."
N_SHARDS = 128

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from transformers.models.qwen4_exp import modeling_qwen4_exp as M  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402

torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEV = "cuda"
# Debug knob: ORACLE_DTYPE=float32 runs the whole oracle in fp32 (default bfloat16, the checkpoint dtype).
DTYPE = getattr(torch, os.environ.get("ORACLE_DTYPE", "bfloat16"))


def model_dir():
    d = os.environ.get("SLOTSTREAM_MODEL")
    if not d:
        d = os.path.expanduser("~/slotstream-spark/model/qwen38-flash-next-mlx-4bit")
    return d


def load_cfg(mdir):
    cfgj = json.load(open(os.path.join(mdir, "config.json")))["text_config"]
    cfg = Qwen4ExpTextConfig(**cfgj)
    cfg._attn_implementation = "sdpa"
    return cfg


def to_tensor(arr):
    """numpy array from hf_tensor -> torch (raw uint16 is BF16 bits)."""
    arr = np.array(arr)
    if arr.dtype == np.uint16:
        return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(arr)


# ---------------------------------------------------------------- name coverage

def _quant_shape(T, name):
    """Logical (dequantized) shape of a checkpoint tensor, plus the names it consumes."""
    if name.endswith(".weight"):
        base = name[: -len(".weight")]
        if base + ".scales" in T and base + ".biases" in T:
            s = T[name].shape
            return tuple(s[:-1]) + (s[-1] * 8,), [name, base + ".scales", base + ".biases"]
    s = T[name].shape
    if "conv1d.weight" in name and len(s) == 3 and s[-1] == 1:
        s = (s[0], s[2], s[1])
    return tuple(s), [name]


def _meta_layer(cfg, L):
    with torch.device("meta"):
        return M.Qwen4ExpTextDecoderLayer(cfg, L)


def check_names(ck, cfg):
    T = ck.tensors()
    consumed = set()
    problems = []
    mapped = 0
    norm_hf = set()

    def cover(mod, prefix, exempt=()):
        nonlocal mapped
        for n, v in mod.state_dict().items():
            if any(n.endswith(e) for e in exempt):
                continue
            name = prefix + n
            if name not in T:
                problems.append(f"unmapped: {name}")
                continue
            shape, used = _quant_shape(T, name)
            if shape != tuple(v.shape):
                problems.append(f"shape: {name} checkpoint {shape} vs transformers {tuple(v.shape)}")
                continue
            consumed.update(used)
            mapped += 1
        for mn, sm in mod.named_modules():
            if isinstance(sm, M.Qwen4ExpTextRMSNorm):
                norm_hf.add(prefix + (mn + "." if mn else "") + "weight")

    exempt = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj", "ple_embedding.ngram_embedding.weight")
    for L in range(cfg.num_hidden_layers):
        cover(_meta_layer(cfg, L), f"{P}layers.{L}.", exempt)
    with torch.device("meta"):
        mix = M.Qwen4ExpTextGatedResidual(cfg, use_combine=False)
    cover(mix, P + "hyper_connection_mixer.")
    for name, shp in ((P + "embed_tokens.weight", (cfg.vocab_size, cfg.hidden_size)),
                      (LM + "lm_head.weight", (cfg.vocab_size, cfg.hidden_size))):
        if name not in T:
            problems.append(f"unmapped: {name}")
            continue
        shape, used = _quant_shape(T, name)
        if shape != shp:
            problems.append(f"shape: {name} checkpoint {shape} vs expected {shp}")
        consumed.update(used)
        mapped += 1

    # n-gram shards: all 128 present with consistent geometry
    pe = _meta_layer(cfg, cfg.ple_layer_ids[0] - 1).ple.ple_embedding
    rps = math.ceil(pe.total_vocab_size / N_SHARDS)
    for s in range(N_SHARDS):
        nm = f"{P}layers.{cfg.ple_layer_ids[0] - 1}.ple.ple_embedding.ngram_embedding.shard_{s}.weight"
        if nm not in T:
            problems.append(f"unmapped: {nm}")
        elif T[nm].shape[0] != rps:
            problems.append(f"shape: {nm} has {T[nm].shape[0]} rows, expected {rps}")

    left = [k for k in T if k.startswith(LM) and k not in consumed
            and ".switch_mlp." not in k and ".ngram_embedding.shard_" not in k]
    for k in left:
        problems.append(f"unconsumed: {k}")
    norm_ck = {k for k in T if k.startswith(LM) and mlx_ckpt.hf_key(k).endswith(CENTERED_NORMS)
               and ".switch_mlp." not in k}
    if norm_hf != norm_ck:
        problems.append(f"norm set differs: transformers-only {sorted(norm_hf - norm_ck)[:4]}, "
                        f"CENTERED_NORMS-only {sorted(norm_ck - norm_hf)[:4]}")
    if importlib.util.find_spec("fla") is not None:
        problems.append("fla is importable inside the oracle process")
    return mapped, problems


# ---------------------------------------------------------------- n-gram table rows

class TableEmb(nn.Module):
    """Stands in for nn.Embedding(320 M, 160): rows come dequantized from the checkpoint shards."""

    def __init__(self, ck, layer, rps, rec):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1), requires_grad=False)  # the module looks at .weight.device
        self.ck, self.layer, self.rps, self.rec = ck, layer, rps, rec

    def row(self, gid):
        s, r = divmod(int(gid), self.rps)
        b = f"{P}layers.{self.layer}.ple.ple_embedding.ngram_embedding.shard_{s}"
        w = self.ck.read_rows(b + ".weight", [r])
        sc = self.ck.read_rows(b + ".scales", [r])
        bi = self.ck.read_rows(b + ".biases", [r])
        return dequant_affine(w, sc, bi)[0]

    def forward(self, idx):
        ids = idx.detach().cpu()
        self.rec["ngram_ids"] = ids.numpy().reshape(-1, ids.shape[-1])
        flat = ids.reshape(-1).tolist()
        cache = {g: self.row(g) for g in sorted(set(flat))}
        out = np.stack([cache[g] for g in flat]).reshape(tuple(ids.shape) + (-1,))
        self.rec["ngram_rows"] = out.reshape(-1, out.shape[-2], out.shape[-1])
        return torch.from_numpy(out).to(DEV, DTYPE)


# ---------------------------------------------------------------- experts

def swap_nibbles(w):
    w = np.asarray(w, dtype=np.uint32)
    out = np.zeros_like(w)
    for j in range(8):
        out |= ((w >> np.uint32(4 * j)) & np.uint32(0xF)) << np.uint32(4 * (7 - j))
    return out


def make_experts_forward(ck, L, mode):
    names = [f"{P}layers.{L}.mlp.switch_mlp.{p}_proj.{k}"
             for p in ("gate", "up", "down") for k in ("weight", "scales", "biases")]
    pool = ThreadPoolExecutor(8)

    def read_expert(e):
        return [ck.read_rows(n, [e])[0] for n in names]

    def deq(w, s, b):
        if mode == "swap":
            w = swap_nibbles(w)
        return mlx_ckpt.dequant_affine_torch(w, s, b, device=DEV).to(DTYPE)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        if mode == "noexp":
            return final
        with torch.no_grad():
            mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        hits = [int(h[0]) for h in hit if int(h[0]) != self.num_experts]
        for c0 in range(0, len(hits), 32):
            chunk = hits[c0:c0 + 32]
            for e, pieces in zip(chunk, pool.map(read_expert, chunk)):
                g = deq(*pieces[0:3])
                u = deq(*pieces[3:6])
                d = deq(*pieces[6:9])
                gate_up = torch.cat([g, u], 0)  # gate first (qwen4_exp.py sanitize)
                pos, tok = torch.where(mask[e])
                cur = hidden_states[tok]
                gt, up = nn.functional.linear(cur, gate_up).chunk(2, dim=-1)
                cur = self.act_fn(gt) * up
                cur = nn.functional.linear(cur, d)
                cur = cur * top_k_weights[tok, pos, None]
                final.index_add_(0, tok, cur.to(final.dtype))
        return final

    return forward


# ---------------------------------------------------------------- loading modules

def load_module(mod, ck, prefix, skip=()):
    T = ck.tensors()
    missing = []
    persistent = set(mod.state_dict().keys())
    with torch.no_grad():
        for n, tgt in list(mod.named_parameters()) + list(mod.named_buffers()):
            if any(n.endswith(s) for s in skip) or n not in persistent:
                continue
            name = prefix + n
            if name not in T:
                missing.append(n)
                continue
            _, arr = hf_tensor(name, ck)
            t = to_tensor(arr)
            if tuple(t.shape) != tuple(tgt.shape):
                raise ValueError(f"{name}: checkpoint {tuple(t.shape)} vs module {tuple(tgt.shape)}")
            tgt.copy_(t.to(device=tgt.device, dtype=tgt.dtype))
    return missing


def build_layer(cfg, L, ck, mode, rec):
    layer = _meta_layer(cfg, L)
    if layer.ple is not None:
        pe = layer.ple.ple_embedding
        rps = math.ceil(pe.total_vocab_size / N_SHARDS)
        pe.ngram_embedding = TableEmb(ck, L, rps, rec)
    layer.mlp.experts.gate_up_proj = nn.Parameter(torch.empty(0, device="meta"), requires_grad=False)
    layer.mlp.experts.down_proj = nn.Parameter(torch.empty(0, device="meta"), requires_grad=False)
    layer = layer.to_empty(device=DEV).to(DTYPE)
    miss = load_module(layer, ck, f"{P}layers.{L}.", skip=("gate_up_proj", "down_proj", "ngram_embedding.weight"))
    miss = [m for m in miss if not m.startswith("mlp.experts.") and "ngram_embedding" not in m]
    if miss:
        raise RuntimeError(f"layer {L}: missing {miss}")
    layer.mlp.experts.forward = make_experts_forward(ck, L, mode).__get__(layer.mlp.experts)
    if layer.ple is not None:
        pe = layer.ple.ple_embedding
        assert torch.equal(pe.layer_multipliers.cpu(), M._build_layer_multipliers(
            pe.unigram_vocab_size, pe.ngram_size, pe.ple_layer_index, pe.seed)), "layer_multipliers differ"
        assert pe.ngram_heads_vocab_sizes.tolist() == pe.head_vocab_sizes
        assert pe.ngram_heads_offsets.tolist() == pe.head_offsets
    return layer.eval()


def embed_rows(ck, ids):
    b = P + "embed_tokens"
    rows = [int(i) for i in ids]
    w = dequant_affine(ck.read_rows(b + ".weight", rows), ck.read_rows(b + ".scales", rows),
                       ck.read_rows(b + ".biases", rows))
    return torch.from_numpy(w).to(DEV, DTYPE)


def lm_head_logits(ck, x, block=32768):
    b = LM + "lm_head"
    n = ck.info(b + ".weight").shape[0]
    out = torch.empty(x.shape[0], n, dtype=torch.float32, device=DEV)
    for r0 in range(0, n, block):
        rows = list(range(r0, min(n, r0 + block)))
        w = mlx_ckpt.dequant_affine_torch(ck.read_rows(b + ".weight", rows), ck.read_rows(b + ".scales", rows),
                                          ck.read_rows(b + ".biases", rows), device=DEV).to(DTYPE)
        out[:, r0:r0 + len(rows)] = nn.functional.linear(x, w).float()
    return out


# ---------------------------------------------------------------- forward

def forward_layers(tokens, ck, cfg, mode="correct", n_layers=None, want_logits=True, last_only=False, dump=None):
    """Returns dict(h_layers=[fp32 [T, hc*hidden]], ngram_ids, ngram_rows, logits)."""
    T = len(tokens)
    n_layers = cfg.num_hidden_layers if n_layers is None else n_layers
    res = {"h_layers": []}
    rec = {}
    with torch.no_grad():
        idt = torch.tensor([list(tokens)], device=DEV)
        emb = embed_rows(ck, tokens).unsqueeze(0)
        rot = M.Qwen4ExpTextRotaryEmbedding(cfg).to(DEV)
        pos = torch.arange(T, device=DEV).view(1, 1, -1).expand(3, 1, -1)
        cos, sin = rot(emb, pos)
        mask = torch.ones(T, T, dtype=torch.bool, device=DEV).tril()[None, None]
        h = emb.repeat(1, 1, cfg.hc_count)
        if dump is not None:
            np.save(f"{dump}/h_in.npy", h[0].float().cpu().numpy())
        for L in range(n_layers):
            layer = build_layer(cfg, L, ck, mode, rec)
            picks = []
            if dump is not None:
                layer.mlp.gate.register_forward_hook(
                    lambda m, i, o, picks=picks: picks.append((o[2].cpu().numpy(), o[1].float().cpu().numpy())))
            h = layer(h, position_embeddings=(cos, sin), attention_mask=mask, conv_mask=None,
                      past_key_values=None, ple_input_ids=idt)
            hf = h[0].float().cpu().numpy()
            res["h_layers"].append(hf)
            if dump is not None:
                np.save(f"{dump}/layer_{L}.npy", hf)
                if picks:
                    np.save(f"{dump}/router_ids_{L}.npy", picks[0][0])
                    np.save(f"{dump}/router_weights_{L}.npy", picks[0][1])
            del layer
            torch.cuda.empty_cache()
        if "ngram_ids" in rec:
            res["ngram_ids"] = rec["ngram_ids"]
            res["ngram_rows"] = rec["ngram_rows"]
            if dump is not None:
                np.save(f"{dump}/ngram_ids.npy", rec["ngram_ids"])
                np.save(f"{dump}/ngram_rows.npy", rec["ngram_rows"].astype(np.float32))
        if want_logits and n_layers == cfg.num_hidden_layers:
            with torch.device("meta"):
                mix = M.Qwen4ExpTextGatedResidual(cfg, use_combine=False)
            mix = mix.to_empty(device=DEV).to(DTYPE)
            miss = load_module(mix, ck, P + "hyper_connection_mixer.")
            if miss:
                raise RuntimeError(f"mixer: missing {miss}")
            hm = mix(h)[0]
            if last_only:
                hm = hm[-1:]
            logits = lm_head_logits(ck, hm)
            res["logits"] = logits
            if dump is not None:
                np.save(f"{dump}/logits.npy", logits.cpu().numpy())
    return res


def teacher_force(tokens, dump_dir=None, experts_mode="correct", ck=None, cfg=None, last_only=False):
    mdir = model_dir()
    cfg = cfg or load_cfg(mdir)
    own = ck is None
    ck = ck or Checkpoint(mdir)
    try:
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
        return forward_layers(list(tokens), ck, cfg, experts_mode, dump=dump_dir, last_only=last_only)["logits"]
    finally:
        if own:
            ck.close()


# ---------------------------------------------------------------- modes

def rel_l2(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def emit(obj):
    s = json.dumps(obj)
    cap = os.environ.get("CAP")
    if cap:
        with open(os.path.join(cap, "result.json"), "w") as f:
            f.write(s + "\n")
    print(s)


def mode_check_names(a):
    mdir = model_dir()
    cfg = load_cfg(mdir)
    ck = Checkpoint(mdir)
    mapped, problems = check_names(ck, cfg)
    if problems:
        for p in problems[:50]:
            print(p)
        print(f"names FAILED: {mapped} mapped, {len(problems)} unmapped")
        return 1
    print(f"names OK: {mapped} mapped, 0 unmapped")
    return 0


def _parity31_inputs(d):
    mdir = model_dir()
    cfg = load_cfg(mdir)
    tokens = [int(x) for x in (Path(d) / "tokens.txt").read_text().strip().split(",")]
    return mdir, cfg, tokens


def _goldens(d, tokens):
    return [np.fromfile(Path(d) / f"layer_{k}.bin", dtype=np.float32).reshape(len(tokens), -1) for k in range(4)]


def _bf16_text(x):
    """8-decimal text of float32 values rounded to bf16 (RNE), as parity_ref.py writes them."""
    t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(torch.bfloat16)
    return [f"{float(v):.8f}" for v in t]


def mode_parity31_calibrate(a):
    global DTYPE
    d = Path(a.parity31_calibrate)
    mdir, cfg, tokens = _parity31_inputs(d)
    t0 = time.time()
    ck = Checkpoint(mdir)
    gold = _goldens(d, tokens)
    runs = {}
    for name, dt, mode in (("bf16", torch.bfloat16, "correct"), ("fp32", torch.float32, "correct"),
                           ("noexp", torch.bfloat16, "noexp"), ("swap", torch.bfloat16, "swap")):
        DTYPE = dt
        try:
            runs[name] = forward_layers(tokens, ck, cfg, mode, n_layers=4, want_logits=False)["h_layers"]
        finally:
            DTYPE = torch.bfloat16
    layers = []
    for k in range(4):
        layers.append({"layer": k,
                       "spread_bf16_fp32": rel_l2(runs["bf16"][k], runs["fp32"][k]),
                       "control_noexp": rel_l2(runs["noexp"][k], gold[k]),
                       "control_swap": rel_l2(runs["swap"][k], gold[k]),
                       "bf16_vs_golden": rel_l2(runs["bf16"][k], gold[k]),
                       "fp32_vs_golden": rel_l2(runs["fp32"][k], gold[k])})
    emit({"tokens": tokens, "layers": layers, "gdn_impl": "torch", "seconds": round(time.time() - t0, 1)})
    return 0


def mode_parity31(a):
    d = Path(a.parity31)
    root = Path(__file__).resolve().parent.parent
    lim_path = Path(a.limits) if a.limits else root / "tests" / "gates" / "parity31_limits.json"
    lim_bytes = lim_path.read_bytes()
    lim = json.loads(lim_bytes)["layers"]
    mdir, cfg, tokens = _parity31_inputs(d)
    t0 = time.time()
    ck = Checkpoint(mdir)
    res = forward_layers(tokens, ck, cfg, "correct", n_layers=4, want_logits=False)
    gold = _goldens(d, tokens)
    layers = []
    for k in range(4):
        layers.append({"layer": k, "rel_l2": rel_l2(res["h_layers"][k], gold[k]), "limit": lim[k]["limit"],
                       "separable": bool(lim[k]["separable"])})
    ref_ids = np.array([[int(x) for x in ln.split(",") if x] for ln in (d / "ngram_ids.txt").read_text().split("\n")
                        if ln.strip()], dtype=np.int64)
    ids_equal = bool(res["ngram_ids"].shape == ref_ids.shape and (res["ngram_ids"] == ref_ids).all())
    row_lines = (d / "dequant_row.txt").read_text().strip().split("\n")
    gid = int(row_lines[0])  # first line: global row id; second line: its 160 dequantized values
    ref_txt = [x.strip() for x in row_lines[1].split(",")]
    rps = math.ceil(_meta_layer(cfg, cfg.ple_layer_ids[0] - 1).ple.ple_embedding.total_vocab_size / N_SHARDS)
    layer = cfg.ple_layer_ids[0] - 1
    sh, r = divmod(gid, rps)
    base = f"{P}layers.{layer}.ple.ple_embedding.ngram_embedding.shard_{sh}"
    w = np.asarray(ck.read_rows(base + ".weight", [r]), dtype=np.uint32)
    sc = mlx_ckpt.bf16_to_f32(ck.read_rows(base + ".scales", [r]))
    bi = mlx_ckpt.bf16_to_f32(ck.read_rows(base + ".biases", [r]))
    q = ((w[..., None] >> (np.arange(8, dtype=np.uint32) * 4)) & np.uint32(15)).reshape(w.shape[0], -1).astype(np.float32)
    grp = q.shape[-1] // sc.shape[-1]
    s_ = np.repeat(sc, grp, axis=-1)
    b_ = np.repeat(bi, grp, axis=-1)
    single = _bf16_text((q * s_ + b_)[0])
    qs = torch.from_numpy(np.ascontiguousarray(q * s_, dtype=np.float32)).to(torch.bfloat16).float().numpy()
    perop = _bf16_text((qs + b_)[0])
    n_single = sum(1 for x, y in zip(single, ref_txt) if x == y)
    row_exact = len(ref_txt) == len(single) and all(ref_txt[i] in (single[i], perop[i]) for i in range(len(ref_txt)))
    ok = bool(ids_equal and row_exact and all(x["separable"] and x["rel_l2"] <= x["limit"] for x in layers))
    emit({"layers": layers, "ngram_ids_equal": ids_equal, "dequant_row_exact": row_exact,
          "dequant_row_single_rounding_matches": n_single, "dequant_row_gid": gid, "gdn_impl": "torch",
          "limits_sha256": hashlib.sha256(lim_bytes).hexdigest(), "pass": ok,
          "seconds": round(time.time() - t0, 1)})
    if not ok:
        print("DESIGN GATE FAILED: oracle does not reproduce parity31 — stop and ask the user", file=sys.stderr)
    return 0 if ok else 1


def mode_generate(a):
    from transformers import AutoTokenizer
    mdir = model_dir()
    cfg = load_cfg(mdir)
    state = Path(a.state)
    state.mkdir(parents=True, exist_ok=True)
    prog = state / "progress.json"
    tok = AutoTokenizer.from_pretrained(mdir)
    if prog.exists():
        st = json.loads(prog.read_text())
    else:
        text = tok.apply_chat_template([{"role": "user", "content": a.prompt}], add_generation_prompt=True,
                                       enable_thinking=False, tokenize=False)
        st = {"prompt": a.prompt, "ids": tok(text, add_special_tokens=False)["input_ids"], "gen": []}
    ck = Checkpoint(mdir)
    stop = (248046, 248044)
    while len(st["gen"]) < a.generate and not (st["gen"] and st["gen"][-1] in stop):
        t0 = time.time()
        lg = teacher_force(st["ids"] + st["gen"], ck=ck, cfg=cfg, last_only=True)
        st["gen"].append(int(lg[-1].argmax()))
        tmp = state / "progress.json.tmp"
        tmp.write_text(json.dumps(st))
        os.replace(tmp, prog)
        print(f"token {len(st['gen'])}: {st['gen'][-1]} ({time.time() - t0:.0f} s)", flush=True)
    emit({"ids": st["ids"], "gen": st["gen"], "text": tok.decode(st["gen"])})
    return 0


def mode_teacher_force(a):
    ids = json.load(open(a.teacher_force))
    if isinstance(ids, dict):
        ids = ids["ids"]
    lg = teacher_force(ids, dump_dir=a.out)
    emit({"tokens": len(ids), "out": a.out, "argmax_last": int(lg[-1].argmax())})
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-names", action="store_true")
    ap.add_argument("--parity31")
    ap.add_argument("--parity31-calibrate", dest="parity31_calibrate")
    ap.add_argument("--limits")
    ap.add_argument("--generate", type=int)
    ap.add_argument("--prompt", default="Explain in three sentences why the sky is blue.")
    ap.add_argument("--state")
    ap.add_argument("--teacher-force")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.check_names:
        return mode_check_names(a)
    if a.parity31_calibrate:
        return mode_parity31_calibrate(a)
    if a.parity31:
        return mode_parity31(a)
    if a.generate:
        if not a.state:
            ap.error("--generate needs --state DIR")
        return mode_generate(a)
    if a.teacher_force:
        if not a.out:
            ap.error("--teacher-force needs --out DIR")
        return mode_teacher_force(a)
    ap.error("choose one of --check-names, --parity31, --parity31-calibrate, --generate, --teacher-force")


if __name__ == "__main__":
    sys.exit(main())
