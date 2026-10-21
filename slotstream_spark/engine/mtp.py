# SPDX-License-Identifier: MIT
# Port of MTP.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""MTP draft head, scheme A, experts resident (plan step 75; slotstream MTP.swift, Tools/reference/mtp_ref.py).

One extra full-attention decoder layer predicts the token after next. Per step, over `S` rows:

    e = fc_embedding(rmsnorm(embed(token)))                       [S, H]
    h = fc_hidden(rmsnorm_fullwidth(multi)) per hc branch        [S, hc, H]   (one [H, H] weight shared by the branches)
    x = (e broadcast over branches + h) flattened                [S, hc*H]
    layer: attn HC mix -> QSA (own KV + indexer cache) -> HC residual -> MLP HC mix -> router -> resident MoE + shared expert
           -> HC residual; the pre-mixer stream is the next step's `multi`
    out = hyper_connection_mixer(x)  [S, H] -> the shared lm_head

Positions are 0-based in the head's own cache, like the main model's (the +1 basis change of the training convention cancels in
RoPE, MTP.swift:21-25). The head's cache holds one entry per consumed main token: entry i fuses the main model's multi of
position i - 1 with the embedding of token i, so token 0 gets no entry and `state.pos == consumed - 1` always.

The 512 routed experts (1,415,577,600 B) sit in one device allocation addressed by a private slot-pointer table and a map row 48
of a 49 x 512 table, so `moe_decode` is reused as it is and the head never touches the slot pool or the SSD after loading.
"""
import json
import os
import tomllib
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F

    from slotstream_spark.model import reference_torch as R
    from slotstream_spark.model.layers import Segments, lin
    from slotstream_spark.model.weights import accessor, load_trunk
except ModuleNotFoundError as e:             # the Mac venv has no torch: policy(), draft_depth() and the constants still work
    if e.name != "torch":
        raise
    torch = None

MAP_ROW = 48                 # the head's layer index in the 49-row map
RESIDENT_BYTES = 1_470_946_816   # 55,369,216 B of trunk tensors + 512 records of 2,764,800 B (the ledger's MTP term)
DEFAULT_TRUNK = "model/packed/mtp_trunk.safetensors"
DEFAULT_EXPERTS = "model/packed/mtp_experts.bin"


POLICY_PATH = Path(__file__).resolve().parents[1] / "config" / "policy.toml"
POLICY_VALUES = ("off", "resident")
MODES = ("D1", "D2", "D5")
STREAMING_MSG = "MTP streaming is not implemented in v1"
DEPTH = 2                    # draft tokens per round (fixed in v1; the verify pass is 1 + DEPTH rows)


def policy(mode, env=None, path=None):
    """MTP setting of a memory mode: `SLOTSTREAM_MTP` (off|resident) when set, else policy.toml `[mtp] <mode>`. `streaming`
    raises in both places; any other value is an error."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {list(MODES)}")
    env = os.environ if env is None else env
    with open(path or POLICY_PATH, "rb") as f:
        table = tomllib.load(f).get("mtp", {})
    for m in MODES:
        if m not in table:
            raise ValueError(f"policy.toml: [mtp] {m} is missing")
    for m, v in table.items():
        _check_value(f"policy.toml [mtp] {m}", v)
    v = env.get("SLOTSTREAM_MTP")
    if v is not None:
        _check_value("SLOTSTREAM_MTP", v)
        return v
    return table[mode]


def _check_value(what, v):
    if v == "streaming":
        raise NotImplementedError(STREAMING_MSG)
    if v not in POLICY_VALUES:
        raise ValueError(f"{what}={v!r}; expected one of {list(POLICY_VALUES)}")


def draft_depth(max_ctx, pos, depth=DEPTH):
    """Drafts a round may verify: min(depth, max_ctx - pos - 1) (the round writes 1 + drafts rows); below `depth` the
    engine takes a plain step."""
    return max(0, min(depth, max_ctx - pos - 1))


class MTPState:
    """Own attention and indexer cache of the head. `attn` is keyed by layer 0, the shape Segments.qsa expects."""

    def __init__(self, device, max_ctx):
        from slotstream_spark.kernels.vendored import QSACache
        self.device, self.max_ctx = device, max_ctx
        self.attn = {0: QSACache(max_ctx, device)}
        self.pos = 0

    @property
    def offset(self):
        return self.pos

    def aligned(self, consumed):
        """The invariant of the head's cache: one entry per consumed token except the first."""
        return consumed > 0 and self.pos == consumed - 1

    def fork(self):
        """An independent copy (drafting runs on it so the committed cache is never touched)."""
        s = object.__new__(MTPState)
        s.device, s.max_ctx, s.pos = self.device, self.max_ctx, self.pos
        s.attn = {0: self.attn[0].clone()}
        return s

    def reset(self):
        c = self.attn[0]
        c.main_kv.zero_()
        c.ring.zero_()
        c.comp.zero_()
        c.pos = 0
        c.last = None
        self.pos = 0


def _read_experts(path, device, chunk=64 << 20):
    """experts file -> one uint8 device tensor, read in pieces; pages are dropped from the page cache after each piece."""
    size = os.path.getsize(path)
    out = torch.empty(size, dtype=torch.uint8, device=device)
    buf = torch.empty(chunk, dtype=torch.uint8).pin_memory()
    fd = os.open(str(path), os.O_RDONLY)
    try:
        off = 0
        while off < size:
            n = min(chunk, size - off)
            mv = memoryview(buf.numpy())[:n]
            got = 0
            while got < n:
                r = os.preadv(fd, [mv[got:]], off + got)
                if r <= 0:
                    raise OSError(f"short read of {path} at {off + got}")
                got += r
            out[off:off + n].copy_(buf[:n], non_blocking=False)
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)
            off += n
    finally:
        os.close(fd)
    return out


class MTPHead:
    """`MTPHead(cfg, trunk, experts, embed, lm_head)`: `trunk`/`experts` are the packed files (defaults under the model root),
    `embed` the main model's Embedder (bf16 rows of the shared embedding), `lm_head` the main model's lm_head weight."""

    def __init__(self, cfg, trunk=DEFAULT_TRUNK, experts=DEFAULT_EXPERTS, embed=None, lm_head=None, device="cuda",
                 max_ctx=4096, resident=True):
        if not resident:
            raise NotImplementedError(STREAMING_MSG)
        self.cfg, self.device, self.embed, self.lm_head = cfg, device, embed, lm_head
        self.H, self.hc, self.eps = cfg["hidden_size"], cfg["hc_count"], cfg["rms_norm_eps"]
        self.topk = cfg["num_experts_per_tok"]
        self.max_ctx = max_ctx
        trunk, experts = Path(trunk), Path(experts)
        self.weights = load_trunk(trunk, device=device)
        names = accessor(self.weights)
        self.W = lambda n: names("mtp." + n)
        self.seg = Segments(cfg, self.W, None, None, None, device, True, "fused", max_rows=8)
        man = json.loads(experts.with_name("mtp_experts.manifest.json").read_text())
        self.record_bytes, self.n_experts = int(man["record_bytes"]), int(man["experts"])
        self.experts = _read_experts(experts, device)
        if self.experts.numel() != self.record_bytes * self.n_experts:
            raise RuntimeError(f"{experts}: {self.experts.numel()} B, expected {self.record_bytes * self.n_experts}")
        base = self.experts.data_ptr()
        self.slot_ptrs = (base + torch.arange(self.n_experts, dtype=torch.int64, device=device) * self.record_bytes)
        self.map = torch.full(((MAP_ROW + 1) * self.n_experts,), -1, dtype=torch.int32, device=device)
        self.map[MAP_ROW * self.n_experts:] = torch.arange(self.n_experts, dtype=torch.int32, device=device)
        tensor_bytes = sum(self._nbytes(w) for w in self.weights.values())
        self.trunk_bytes = tensor_bytes
        self.resident_bytes = tensor_bytes + self.experts.numel()
        self.state = MTPState(device, max_ctx)
        self.prev_multi = None      # multi of the last consumed position [1, hc*H]
        self.consumed = 0

    @staticmethod
    def _nbytes(w):
        if isinstance(w, torch.Tensor):
            return w.numel() * w.element_size()
        return sum(t.numel() * t.element_size() for t in (w.q, w.s, w.b))

    # ------------------------------------------------------------------ one head step
    def fuse(self, embedded, multi):
        """embedded [S, H] bf16, multi [S, hc*H] bf16 -> [S, hc*H] bf16 (every op is row-independent: lin is the
        batch-invariant gemv, the norms reduce per row)."""
        W, H, hc = self.W, self.H, self.hc
        S = embedded.shape[0]
        e = lin(R.rmsnorm(embedded, W("pre_fc_norm_embedding.weight"), self.eps), W("fc_embedding.weight"))
        h = R.rmsnorm(multi, W("pre_fc_norm_hidden.weight"), self.eps).reshape(S * hc, H)
        h = lin(h.contiguous(), W("fc_hidden.weight")).reshape(S, hc, H)
        return (e.unsqueeze(-2) + h).reshape(S, hc * H)

    def _moe(self, mixed, ids, w):
        from slotstream_spark.kernels.moe_gemv_affine import moe_decode
        W, seg = self.W, self.seg
        r = mixed.shape[0]
        h_buf = torch.empty(r, self.topk, self.cfg["moe_intermediate_size"], dtype=torch.float32, device=mixed.device)
        o_buf = torch.empty(r, self.topk, self.H, dtype=torch.float32, device=mixed.device)
        routed = torch.zeros(r, self.H, dtype=torch.float32, device=mixed.device)
        moe_decode(mixed.contiguous(), ids.to(torch.int32).contiguous(), w.contiguous(), MAP_ROW, self.map, self.slot_ptrs,
                   h_buf, o_buf, routed)
        p = "layers.0.mlp."
        sh = lin(F.silu(lin(mixed, W(p + "shared_expert.gate_proj.weight"))) *
                 lin(mixed, W(p + "shared_expert.up_proj.weight")), W(p + "shared_expert.down_proj.weight"))
        sh = torch.sigmoid(lin(mixed, W(p + "shared_expert_gate.weight"))) * sh
        return routed.to(mixed.dtype) + sh

    def step(self, embedded, multi, state, state_only=False):
        """One head step over S rows, advancing `state`: -> (out [S, H] for the lm_head, multi [S, hc*H] for chaining). With
        `state_only` the step stops after the attention has written its cache (out is None, multi is the fused stream)."""
        seg = self.seg
        S = embedded.shape[0]
        if state.pos + S > state.max_ctx:
            raise ValueError(f"head context {state.pos + S} exceeds {state.max_ctx}")
        with torch.no_grad():
            h = self.fuse(embedded, multi)
            x1, inj1 = seg.hc_mix("layers.0.attn_hyper_connection", h)
            a = seg.qsa(0, state, x1)
            state.pos += S
            if state_only:
                return None, h
            h = h + (a.unsqueeze(-2) * inj1.unsqueeze(-1)).flatten(-2)
            mixed, inj, ids, w = seg.seg_router(0, h)
            m = self._moe(mixed, ids, w)
            h = h + (m.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
            return self._mixer(h), h

    def _mixer(self, h):
        return self.seg.hc_mix("hyper_connection_mixer", h, inject=False)[0]

    def logits(self, out):
        """lm_head over the head's output rows -> fp32 [S, V]."""
        if self.lm_head is None:
            raise RuntimeError("this head was built without the main model's lm_head")
        return lin(out, self.lm_head).float()

    # ------------------------------------------------------------------ the engine-facing calls
    def reset(self):
        self.state.reset()
        self.prev_multi = None
        self.consumed = 0

    def consume(self, ids, multi_rows):
        """Feed consumed main-model tokens `ids` with the main model's pre-mixer multi `multi_rows` [len(ids), hc*H] of the same
        positions. The entry of ids[i] fuses the multi of the previous position with ids[i]; at sequence start ids[0] has none.
        Keeps the last multi row for the next call and for `draft`. Invariant afterwards: state.pos == consumed - 1."""
        ids = [int(t) for t in ids]
        S = len(ids)
        if S == 0 or multi_rows.shape != (S, self.hc * self.H):
            raise ValueError("consume needs a nonempty chunk and one multi row per token")
        if self.embed is None:
            raise RuntimeError("this head was built without an embedder")
        start = 1 if self.prev_multi is None else 0
        if S - start > 0:
            e = self.embed(ids[start:]).to(torch.bfloat16)
            prev = multi_rows[:S - 1] if self.prev_multi is None else torch.cat([self.prev_multi, multi_rows[:S - 1]], 0)
            self.step(e, prev.contiguous(), self.state, state_only=True)
        self.prev_multi = multi_rows[-1:].clone()
        self.consumed += S
        if not self.state.aligned(self.consumed):
            raise RuntimeError(f"mtp.offset {self.state.pos} != token_count {self.consumed} - 1")
        return self.prev_multi

    def draft(self, token, k=2):
        """Greedy draft ids after the pending token `token` (the last sampled one, not consumed yet), chained from the last
        consumed multi. Runs on a fork of the cache: the committed state is unchanged. -> list of k ints."""
        if self.prev_multi is None:
            raise RuntimeError("draft before any consume")
        st = self.state.fork()
        multi, tok, out = self.prev_multi, int(token), []
        for _ in range(k):
            e = self.embed([tok]).to(torch.bfloat16)
            o, multi = self.step(e, multi, st)
            tok = int(torch.argmax(self.logits(o)[0]))
            out.append(tok)
        return out
