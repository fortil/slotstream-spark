# SPDX-License-Identifier: MIT
# Port of PrefixCache.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""Extend-only prefix cache of conversation states (port of PrefixCache.swift, step 71).

A bounded set (4) of reusable states plus the exact ids that produced them. Matching is by exact prefix, and under
the resume rule (always on here) an entry is offered only if every one of its tokens was read in the chronological
prefill passes a fresh read of the same ids runs (`fresh_equivalent`, same `CheckpointKey`) and its length is one of
the prompt's own pass boundaries or the whole prompt (PrefixCache.swift:61-74, :430-439). A state that consumed
generated tokens (decode-end) is stored with `fresh_equivalent=False` and can never be continued; it is the first
thing evicted.

Memory is charged in token units like the Swift cache: `max(tokens, ceil(seq_bytes / 27,648))` plus
`ceil((logits_bytes + 16,384) / 27,648)` for retained logits, and 113,246,208 B of GDN state per entry on top
(`owned_bytes`). A miss evicts before the caller allocates its own state (PrefixCache.swift:381-387).

The cache is state-agnostic: an entry's `state` is whatever the caller hands in. `snapshot_state` / `restore_state`
copy a `model.layers.SeqState` into and out of a frozen copy in place, so the addresses CUDA graphs are bound to
never change.
"""
import json
import threading
from dataclasses import dataclass
from typing import Optional

BYTES_PER_TOKEN = 27_648
FIXED_BYTES_PER_ENTRY = 36 * 48 * 128 * 128 * 4      # 113,246,208: 36 linear layers x 48 value heads x 128 x 128 fp32
LOGIT_SLACK_BYTES = 16_384
MAX_ENTRIES = 4
CONTEXT_ARITH_VERSION = 1


@dataclass(frozen=True)
class CheckpointKey:
    """What a state was built under; a different key can never resume it (PrefixCache.swift:41-59)."""
    model_identity: str
    optimizations_json: str
    prefill_pass: int
    mtp: bool
    context_arith_version: int = CONTEXT_ARITH_VERSION


@dataclass
class Hit:
    state: object
    reused: int
    logits: object                      # the retained logits when the prompt equals the entry, else None
    fresh_equivalent: bool
    decision: str


@dataclass
class Miss:
    decision: str


@dataclass
class Entry:
    state: object
    tokens: list
    seq_bytes: int
    used: int
    reusable: bool = False
    was_used_or_returned: bool = False
    last_logits: object = None
    logits_bytes: int = 0
    prompt_key: Optional[CheckpointKey] = None
    fresh_equivalent: bool = False
    produced_key: Optional[CheckpointKey] = None
    input_tokens: Optional[int] = None


def token_units(nbytes):
    nbytes = max(0, int(nbytes))
    return -(-nbytes // BYTES_PER_TOKEN)


def charge(e):
    """Token units one entry costs (PrefixCache.swift:494-503)."""
    lg = token_units(e.logits_bytes + LOGIT_SLACK_BYTES) if e.last_logits is not None else 0
    return max(len(e.tokens), token_units(e.seq_bytes)) + lg


class PrefixCache:
    def __init__(self, max_entries=MAX_ENTRIES, max_tokens=0):
        self.max_entries = int(max_entries)
        self._max_tokens = max(0, int(max_tokens))
        self._entries = []
        self._clock = 0
        self._lock = threading.RLock()
        self.enabled = True
        self.resume_rule_in_force = True
        self.hits = self.misses = self.evictions = 0
        self.checkpoint_hits = self.checkpoint_stores = 0
        self.decision = "not queried"

    # ------------------------------------------------------------------ views
    @property
    def max_tokens(self):
        return self._max_tokens

    def entries(self):
        with self._lock:
            return list(self._entries)

    def held_tokens(self):
        with self._lock:
            return sum(len(e.tokens) for e in self._entries)

    def charged_tokens(self):
        with self._lock:
            return sum(charge(e) for e in self._entries)

    def owned_bytes(self):
        """Bytes the held entries own: sequence bytes, GDN state and retained logits with their slack."""
        with self._lock:
            return sum(e.seq_bytes + FIXED_BYTES_PER_ENTRY +
                       (e.logits_bytes + LOGIT_SLACK_BYTES if e.last_logits is not None else 0)
                       for e in self._entries)

    def stats(self):
        with self._lock:
            return {"entries": len(self._entries), "max_entries": self.max_entries, "held_tokens": self.held_tokens(),
                    "charged_tokens": self.charged_tokens(), "max_tokens": self._max_tokens, "hits": self.hits,
                    "misses": self.misses, "evictions": self.evictions, "checkpoint_hits": self.checkpoint_hits,
                    "checkpoint_stores": self.checkpoint_stores, "owned_bytes": self.owned_bytes()}

    # ------------------------------------------------------------------ eviction
    def _candidates(self):
        """PrefixCache.swift:710-760: under the resume rule the unresumable states first, then the checkpoints nothing
        has used, else everything."""
        allr = list(enumerate(self._entries))
        if self.resume_rule_in_force:
            unres = [r for r in allr if not r[1].fresh_equivalent and r[1].last_logits is None]
            if unres:
                return unres
        spec = [r for r in allr if r[1].reusable and not r[1].was_used_or_returned]
        return spec or allr

    def _evict_lru(self):
        c = self._candidates()
        if not c:
            return
        i = min(c, key=lambda r: r[1].used)[0]
        del self._entries[i]
        self.evictions += 1

    def _reserve_active(self, reserve):
        allowance = self._max_tokens - min(self._max_tokens, max(0, int(reserve)))
        while self._entries and (len(self._entries) >= self.max_entries or
                                 sum(charge(e) for e in self._entries) > allowance):
            self._evict_lru()

    def _keep_prefixes_current(self, tokens):
        for e in self._entries:
            if e.reusable and len(e.tokens) < len(tokens) and tokens[:len(e.tokens)] == e.tokens:
                e.used = max(e.used, self._clock)

    def drop(self):
        """Release every entry (the governor calls this before it shrinks the pool)."""
        with self._lock:
            self.evictions += len(self._entries)
            self._entries.clear()

    def configure(self, max_tokens):
        with self._lock:
            self._max_tokens = max(0, int(max_tokens))
            while sum(charge(e) for e in self._entries) > self._max_tokens and self._entries:
                self._evict_lru()

    # ------------------------------------------------------------------ take
    @staticmethod
    def _resumable(e, n, key, boundaries):
        if not e.fresh_equivalent or e.produced_key != key:
            return False
        return n == len(e.tokens) or len(e.tokens) in boundaries

    def _best(self, prompt, key, boundaries):
        best = None
        n = len(prompt)
        for i, e in enumerate(self._entries):
            m = len(e.tokens)
            longer = n > m
            same = n == m and e.prompt_key == key and e.last_logits is not None
            if (longer or same) and prompt[:m] == e.tokens and self._resumable(e, n, key, boundaries):
                if best is None or m > len(self._entries[best].tokens):
                    best = i
        return best

    def take(self, prompt, key, boundaries, reserve_tokens=None):
        """The longest retained state `prompt` extends (or equals, with its logits), if it may be resumed; else a Miss.

        A checkpoint stays in the cache (the caller restores from it by copy); an ordinary entry is handed over and
        removed. Either way room for the caller's own state is made before returning (a miss evicts first)."""
        prompt = [int(t) for t in prompt]
        boundaries = set(boundaries)
        with self._lock:
            if not self.enabled:
                self._entries.clear()
                self.decision = "disabled"
                return Miss("disabled")
            reserve = max(len(prompt), reserve_tokens if reserve_tokens is not None else len(prompt))
            i = self._best(prompt, key, boundaries)
            if i is None:
                self.decision = self._miss_reason(prompt, key, boundaries)
                self.misses += 1
                self._reserve_active(reserve)
                return Miss(self.decision)
            sel = self._entries[i]
            self.decision = f"reusing {len(sel.tokens)} tokens"
            logits = sel.last_logits if len(prompt) == len(sel.tokens) else None
            self._clock += 1
            if sel.reusable:
                sel.used = self._clock
                sel.was_used_or_returned = True
                self._keep_prefixes_current_prompt(prompt)
                self._reserve_active(max(reserve, charge(sel)))
                self.hits += 1
                self.checkpoint_hits += 1
            else:
                del self._entries[i]
                self._keep_prefixes_current_prompt(prompt)
                self._reserve_active(max(reserve, charge(sel)))
                self.hits += 1
            return Hit(sel.state, len(sel.tokens), logits, sel.fresh_equivalent, self.decision)

    def _keep_prefixes_current_prompt(self, prompt):
        self._keep_prefixes_current(prompt)

    def _miss_reason(self, prompt, key, boundaries):
        if not self._entries:
            return f"no retained state ({self.evictions} prior evictions)"
        matching = [e for e in self._entries if prompt[:len(e.tokens)] == e.tokens]
        if not matching:
            return "retained states are not exact prefixes of this prompt"
        if not any(self._resumable(e, len(prompt), key, boundaries) for e in matching):
            return "matching tokens lack a compatible prefill-boundary state"
        return "equal-length state has no compatible next-token logits"

    def retained_match_length(self, prompt, key, boundaries):
        """Tokens `take` would reuse, without taking, touching or evicting anything."""
        with self._lock:
            if not self.enabled:
                return 0
            i = self._best([int(t) for t in prompt], key, set(boundaries))
            return 0 if i is None else len(self._entries[i].tokens)

    # ------------------------------------------------------------------ store
    def store(self, state, tokens, seq_bytes, *, fresh_equivalent=False, key=None, input_tokens=None):
        """An ordinary (non-reusable) entry, e.g. the decode-end state of a finished request."""
        tokens = [int(t) for t in tokens]
        with self._lock:
            if not self.enabled or not tokens or len(tokens) > self._max_tokens:
                return False
            self._clock += 1
            dup = next((i for i, e in enumerate(self._entries) if e.tokens == tokens), None)
            if dup is not None:
                e = self._entries[dup]
                if e.reusable:
                    e.used = self._clock
                    e.was_used_or_returned = True
                    return True
                self._entries[dup] = Entry(state, tokens, int(seq_bytes), self._clock, fresh_equivalent=fresh_equivalent,
                                           produced_key=key, input_tokens=input_tokens)
            else:
                self._entries.append(Entry(state, tokens, int(seq_bytes), self._clock,
                                           fresh_equivalent=fresh_equivalent, produced_key=key,
                                           input_tokens=input_tokens))
            self._keep_prefixes_current(tokens)
            while self._entries and (len(self._entries) > self.max_entries or
                                     sum(charge(e) for e in self._entries) > self._max_tokens):
                self._evict_lru()
            return True

    def store_checkpoint(self, state, tokens, seq_bytes, *, logits=None, logits_bytes=0, prompt_key=None,
                         retention="optional", fresh_equivalent=False, produced_key=None, reserve_tokens=None):
        """A reusable snapshot (a pass boundary, or the whole prompt with its raw logits). Plans its room before it
        changes anything: it may replace unresumable states, unused checkpoints and shallower snapshots of the same
        conversation, and a `retention="conversation"` one also the LRU entry. -> True if it was stored."""
        tokens = [int(t) for t in tokens]
        if not tokens:
            raise ValueError("a reusable checkpoint needs its exact committed tokens")
        with self._lock:
            if not self.enabled:
                return False
            chg = max(len(tokens), token_units(seq_bytes)) + (token_units(logits_bytes + LOGIT_SLACK_BYTES)
                                                             if logits is not None else 0)
            active = max(len(tokens), reserve_tokens if reserve_tokens is not None else 0, chg)
            allowance = self._max_tokens - min(self._max_tokens, active)
            if chg > allowance:
                return False

            def same(e):
                return (e.tokens == tokens and e.reusable and e.prompt_key == prompt_key and
                        e.fresh_equivalent == fresh_equivalent and e.produced_key == produced_key)

            existing = next((i for i, e in enumerate(self._entries) if same(e)), None)
            if existing is not None:
                e = self._entries[existing]
                self._clock += 1
                e.used = self._clock
                if retention == "conversation":
                    e.was_used_or_returned = True
                self._reserve_active(active)
                if any(x is e for x in self._entries):
                    return True
            self._clock += 1
            victims = set()
            inherited = False
            dup = next((i for i, e in enumerate(self._entries) if e.tokens == tokens), None)
            if dup is not None:
                d = self._entries[dup]
                inherited = (not d.reusable) or d.was_used_or_returned
                victims.add(dup)
            remaining_count = len(self._entries) - len(victims)
            remaining_charge = sum(charge(e) for i, e in enumerate(self._entries) if i not in victims)
            rule = self.resume_rule_in_force
            unres = sorted([(i, e) for i, e in enumerate(self._entries)
                            if rule and not e.fresh_equivalent and e.last_logits is None and i not in victims],
                           key=lambda r: r[1].used)
            unused = sorted([(i, e) for i, e in enumerate(self._entries)
                             if e.reusable and not e.was_used_or_returned and i not in victims and
                             not (rule and e.fresh_equivalent and e.last_logits is None)],
                            key=lambda r: r[1].used)
            superseded = sorted([(i, e) for i, e in enumerate(self._entries)
                                 if e.reusable and e.was_used_or_returned and i not in victims and
                                 len(e.tokens) < len(tokens) and e.produced_key == produced_key and
                                 tokens[:len(e.tokens)] == e.tokens], key=lambda r: -len(r[1].tokens))
            order, seen = [], set()

            def add(rows):
                for i, e in rows:
                    if i not in seen:
                        seen.add(i)
                        order.append((i, e))

            add(unres)
            add(unused)
            add(superseded)
            if retention == "conversation":
                add(sorted([(i, e) for i, e in enumerate(self._entries) if i not in victims], key=lambda r: r[1].used))
            for i, e in order:
                if i in victims:
                    continue
                if remaining_count < self.max_entries - 1 and remaining_charge <= allowance - chg:
                    break
                victims.add(i)
                remaining_count -= 1
                remaining_charge -= charge(e)
            if not (remaining_count < self.max_entries - 1 and remaining_charge <= allowance - chg):
                return False
            for i in sorted(victims, reverse=True):
                del self._entries[i]
                self.evictions += 1
            self._clock += 1
            self._entries.append(Entry(state, tokens, int(seq_bytes), self._clock, reusable=True,
                                       was_used_or_returned=inherited or retention == "conversation",
                                       last_logits=logits, logits_bytes=int(logits_bytes), prompt_key=prompt_key,
                                       fresh_equivalent=fresh_equivalent, produced_key=produced_key))
            self.checkpoint_stores += 1
            return True


# ---------------------------------------------------------------------------------------------- boundaries
def snapshot_target(ids, im_start_id):
    """Index of the last `<|im_start|>` in `ids` (the snapshot is the last pass end at or before it), else None."""
    for i in range(len(ids) - 1, -1, -1):
        if int(ids[i]) == int(im_start_id):
            return i
    return None


def snapshot_end(ids, target, base, plan_pass, boundaries, P=None):
    """Where to snapshot while reading `ids[base:]`: the last pass end at or before `target` (qwen_server.py:164-177
    plus the resume rule), only if it is strictly inside the new tokens and is one of the prompt's own boundaries."""
    from slotstream_spark.engine import prefill_schedule as ps
    if target is None or target <= base:
        return None
    end = ps.last_pass_end(int(target), base, remaining=len(ids) - base, max_chunk=plan_pass, P=P)
    if end is None or end <= base or end >= len(ids) or end not in boundaries:
        return None
    return end


# ---------------------------------------------------------------------------------------------- state copies
@dataclass
class StateSnapshot:
    pos: int
    tokens: list
    S: object
    conv: object
    ple_conv: object
    attn: dict                           # layer -> dict(pos, main_kv, comp, ring)
    nbytes: int
    model_identity: str = ""


def _used_slices(c):
    s = c.s
    pages = min(c.main_kv.shape[0], -(-c.pos // s.page))
    rows = -(-c.pos // s.ratio)
    comp = min(c.comp.shape[0], -(-rows // s.comp_page))
    return pages, comp


def snapshot_state(st):
    """A frozen copy of a `SeqState` (only the used pages of the attention caches)."""
    attn = {}
    nbytes = 0
    for L, c in st.attn.items():
        if isinstance(c, dict):
            raise TypeError("snapshot_state needs the CUDA attention caches")
        pages, comp = _used_slices(c)
        d = {"pos": c.pos, "main_kv": c.main_kv[:pages].clone(), "comp": c.comp[:comp].clone(), "ring": c.ring.clone()}
        attn[L] = d
        nbytes += sum(t.numel() * t.element_size() for t in (d["main_kv"], d["comp"], d["ring"]))
    conv = None if st.conv is None else st.conv.clone()
    ple = None if st.ple_conv is None else st.ple_conv.clone()
    return StateSnapshot(pos=int(st.pos), tokens=list(st.tokens), S=st.S.clone(), conv=conv, ple_conv=ple, attn=attn,
                         nbytes=nbytes)


def restore_state(st, snap):
    """Back to `snap` in place: every tensor keeps its address."""
    st.reset_inplace()
    st.S.copy_(snap.S)
    if snap.conv is not None:
        st.ensure_act(snap.conv.dtype)
        st.conv.copy_(snap.conv)
        st.ple_conv.copy_(snap.ple_conv)
    for L, d in snap.attn.items():
        c = st.attn[L]
        n, m = d["main_kv"].shape[0], d["comp"].shape[0]
        c.main_kv[:n].copy_(d["main_kv"])
        c.comp[:m].copy_(d["comp"])
        c.ring.copy_(d["ring"])
        c.pos = d["pos"]
        c.last = None
    st.pos = snap.pos
    st.tokens = list(snap.tokens)


# ---------------------------------------------------------------------------------------------- persistent tier glue
# The disk tier (engine.persistent_prefix) stores numpy arrays under fixed names. These functions map a frozen
# `StateSnapshot` to that view and a restored view back into a live `SeqState` (in place, addresses kept).
#   linear.<L>.ssm   fp32 [nv, dk, dv]            GDN state of linear layer L
#   linear.<L>.conv  bf16 [1, ck, C]              GDN conv state (raw uint16)
#   linear.<L0>.ple  bf16 [hist, hc*H]            PLE conv state, on the first linear layer only
#   kv.<L>.keys / .values  bf16 [1, kv_heads, rows, head_dim], rows along axis 2, `pos` live rows
#   indexer.<L>.raw  bf16 [1, 16, 128]            the raw-key ring (2 blocks x 8) as a compact window ending at `pos`
#   indexer.<L>.pooled  bf16 [1, comp_rows, 128]  compressed rows, pos // ratio live
# The ring is circular, not positional, so no row of it is ever shared with a parent: states are saved without lineage.
RING_ROWS = 16
POOLED_RATIO = 4


def _to_np(t):
    import numpy as np
    import torch
    t = t.detach().contiguous().cpu()
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(np.uint16), "bfloat16"
    if t.dtype == torch.float32:
        return t.numpy(), "float32"
    raise TypeError(f"cannot persist a {t.dtype} tensor")


def _from_np(a, device):
    import numpy as np
    import torch
    if a.dtype == np.uint16:
        return torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).view(torch.bfloat16).to(device)
    return torch.from_numpy(np.ascontiguousarray(a)).to(device)


def state_layers(st):
    """-> (linear layers in S order, full-attention layers)."""
    return sorted(st.lin_idx, key=st.lin_idx.get), list(st.full_layers)


def snapshot_to_view(snap, lin_layers, attn_layers):
    """Host StateView of a snapshot (the D2H copy). Raises ValueError for a position the format cannot hold."""
    from slotstream_spark.engine import persistent_prefix as pp
    pos = int(snap.pos)
    if pos != len(snap.tokens) or pos < RING_ROWS or pos % POOLED_RATIO:
        raise ValueError(f"state at {pos} tokens cannot be persisted (needs pos == tokens, >= {RING_ROWS}, multiple of 4)")
    ctx = [int(t) for t in snap.tokens[-2:]]
    view = pp.StateView(pos, ngram_context=ctx, compact_state_windows=False)
    for j, L in enumerate(lin_layers):
        view.linear.append(pp.LinearRecord(L, ctx))
        a, dt = _to_np(snap.S[j])
        view.fixed[f"linear.{L}.ssm"] = pp.FixedArray(dt, a)
        if snap.conv is not None:
            a, dt = _to_np(snap.conv[j][None])
            view.fixed[f"linear.{L}.conv"] = pp.FixedArray(dt, a)
    if snap.ple_conv is not None and lin_layers:
        a, dt = _to_np(snap.ple_conv)
        view.fixed[f"linear.{lin_layers[0]}.ple"] = pp.FixedArray(dt, a)
    for L in attn_layers:
        d = snap.attn[L]
        if int(d["pos"]) != pos:
            raise ValueError(f"attention layer {L} is at {d['pos']}, the state at {pos}")
        mk = d["main_kv"]
        pages, page, nkv, w = mk.shape
        hd = w // 2
        rows = pages * page
        x = mk.reshape(rows, nkv, w)
        for part, sl in (("keys", x[..., :hd]), ("values", x[..., hd:])):
            a, dt = _to_np(sl.permute(1, 0, 2).unsqueeze(0))
            view.sequences[f"kv.{L}.{part}"] = pp.SeqArray(dt, a, 2, 0, pos)
        a, dt = _to_np(d["ring"].reshape(1, RING_ROWS, -1))
        view.sequences[f"indexer.{L}.raw"] = pp.SeqArray(dt, a, 1, pos - RING_ROWS, RING_ROWS)
        comp = d["comp"]
        a, dt = _to_np(comp.reshape(1, comp.shape[0] * comp.shape[1], -1))
        pooled = pos // POOLED_RATIO
        if a.shape[1] < pooled:
            raise ValueError(f"layer {L}: {a.shape[1]} compressed rows for {pooled} needed")
        view.sequences[f"indexer.{L}.pooled"] = pp.SeqArray(dt, a, 1, 0, pooled)
        view.attention.append(pp.AttentionRecord(L, pos, pp.IndexerRecord(pos, pos - RING_ROWS, True, pooled,
                                                                           POOLED_RATIO)))
    return view


def view_into_state(st, view, tokens):
    """Load a restored StateView into `st` in place (every tensor keeps its address). `tokens` are the ids the view
    consumed. Raises ValueError when the view does not fit the live caches."""
    import torch
    lin_layers, attn_layers = state_layers(st)
    pos = int(view.token_count)
    if pos != len(tokens):
        raise ValueError("view and tokens disagree")
    dev = st.S.device
    st.reset_inplace()
    for j, L in enumerate(lin_layers):
        st.S[j].copy_(_from_np(view.fixed[f"linear.{L}.ssm"].data, dev))
    conv = view.fixed.get(f"linear.{lin_layers[0]}.conv") if lin_layers else None
    if conv is not None:
        st.ensure_act(torch.bfloat16)
        for j, L in enumerate(lin_layers):
            st.conv[j].copy_(_from_np(view.fixed[f"linear.{L}.conv"].data, dev)[0])
        ple = view.fixed.get(f"linear.{lin_layers[0]}.ple")
        if ple is not None:
            st.ple_conv.copy_(_from_np(ple.data, dev))
    for L in attn_layers:
        c = st.attn[L]
        k = _from_np(view.sequences[f"kv.{L}.keys"].data, dev)
        v = _from_np(view.sequences[f"kv.{L}.values"].data, dev)
        kv = torch.cat([k, v], dim=-1)[0].permute(1, 0, 2)               # [rows, kv_heads, 2*head_dim]
        page = c.main_kv.shape[1]
        pages = kv.shape[0] // page
        if pages > c.main_kv.shape[0]:
            raise ValueError(f"layer {L}: {pages} pages do not fit {c.main_kv.shape[0]}")
        c.main_kv[:pages].copy_(kv.reshape(pages, page, kv.shape[1], kv.shape[2]))
        c.ring.copy_(_from_np(view.sequences[f"indexer.{L}.raw"].data, dev).reshape(c.ring.shape))
        comp = _from_np(view.sequences[f"indexer.{L}.pooled"].data, dev)
        n = comp.shape[1] // c.comp.shape[1]
        if n > c.comp.shape[0]:
            raise ValueError(f"layer {L}: {n} compressed pages do not fit {c.comp.shape[0]}")
        c.comp[:n].copy_(comp.reshape(n, c.comp.shape[1], 1, comp.shape[-1]))
        c.pos = pos
        c.last = None
    st.pos = pos
    st.tokens = [int(t) for t in tokens]


class PersistentTier:
    """The disk tier as the engine uses it: lookups under the gate, D2H copies under the gate, file writes on a
    background thread that starts only after the gate is released (`release()`), a bounded flush at exit.

    `prefix` is an engine.persistent_prefix.PersistentPrefix; `prefill_chunk` the prefill pass size the states were
    produced under (a state is offered only to a prompt that prefills with the same pass)."""
    MAX_QUEUED = 4

    def __init__(self, prefix, prefill_chunk, lin_layers, attn_layers, on_event=None):
        from slotstream_spark.engine import persistent_prefix as pp
        self.prefix, self.prefill_chunk = prefix, int(prefill_chunk)
        self.lin_layers, self.attn_layers = list(lin_layers), list(attn_layers)
        self.layout = pp.Layout3(frozenset(self.lin_layers), frozenset(self.attn_layers), True)
        self.on_event = on_event or (lambda line: None)
        self._staged, self._queue = [], []
        self._cv = threading.Condition()
        self._busy = False
        self._closed = False
        self.counters = {"staged": 0, "dropped": 0, "written": 0, "restores": 0, "restore_failures": 0}
        self.last_results = []
        self._thread = threading.Thread(target=self._writer, name="prefix-writer", daemon=True)
        self._thread.start()

    # ---- lookup / restore (under the gate)
    def match(self, ids, longer_than, boundaries):
        try:
            return self.prefix.best_match([int(t) for t in ids], longer_than=int(longer_than),
                                          boundaries=set(boundaries), prefill_chunk=self.prefill_chunk)
        except Exception as e:  # noqa: BLE001 - a lookup failure is a miss
            self.on_event(f"prefix lookup failed: {e}")
            return None

    def peek(self, ids, boundaries):
        e = self.match(ids, 0, boundaries)
        return 0 if e is None else len(e.tokens)

    def restore_into(self, st, entry):
        """Read `entry` from disk into `st`; -> tokens restored. Raises on a failed read or a view that does not fit."""
        try:
            res = self.prefix.restore(entry, layout=self.layout)
            view_into_state(st, res.state, entry.tokens)
        except Exception:
            self.counters["restore_failures"] += 1
            raise
        self.counters["restores"] += 1
        return len(entry.tokens)

    # ---- save
    def stage(self, snap, tokens, shared=True):
        """D2H copy now (the caller holds the gate); the write waits for `release()`."""
        try:
            view = snapshot_to_view(snap, self.lin_layers, self.attn_layers)
        except Exception as e:  # noqa: BLE001
            self.counters["dropped"] += 1
            self.on_event(f"prefix not staged: {e}")
            return False
        with self._cv:
            self._staged.append((view, [int(t) for t in tokens], bool(shared)))
            self.counters["staged"] += 1
        return True

    def release(self):
        """The gate is released: staged states may be written now."""
        with self._cv:
            for job in self._staged:
                if len(self._queue) >= self.MAX_QUEUED:
                    self.counters["dropped"] += 1
                else:
                    self._queue.append(job)
            self._staged = []
            self._cv.notify_all()

    def _writer(self):
        while True:
            with self._cv:
                while not self._queue and not self._closed:
                    self._cv.wait()
                if not self._queue:
                    return
                view, tokens, shared = self._queue.pop(0)
                self._busy = True
            try:
                res = self.prefix.save(view, tokens, shared=shared, prefill_chunk=self.prefill_chunk)
                self.last_results.append(res)
                self.last_results = self.last_results[-8:]
                if res.outcome == "saved":
                    self.counters["written"] += 1
                self.on_event(f"prefix save {len(tokens)} tokens: {res.outcome}")
            except Exception as e:  # noqa: BLE001 - never takes the engine down
                self.on_event(f"prefix save failed: {e}")
            finally:
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()

    def flush(self, timeout=20.0):
        """Release anything staged and wait (<= timeout s) for the queue to drain. -> True when idle."""
        import time
        self.release()
        end = time.monotonic() + timeout
        with self._cv:
            while self._queue or self._busy:
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(left)
        return True

    def close(self, timeout=20.0):
        ok = self.flush(timeout)
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._thread.join(1.0)
        self.prefix.close()
        return ok


def build_identity(*, package_dir, extra_files, model_dir, packed_dir, geometry, optimizations, envelope, backend=None):
    """The identity of this engine (persistent_prefix.identity_for): code hash, config.json sha, sampled weights of
    every packed file, cache geometry, optimizations, envelope id, context arithmetic and the GPU backend."""
    import hashlib
    import json
    from pathlib import Path
    from slotstream_spark.engine import persistent_prefix as pp
    cfg_sha = hashlib.sha256((Path(model_dir) / "config.json").read_bytes()).hexdigest()
    env_id = hashlib.sha256(json.dumps(envelope, sort_keys=True).encode()).hexdigest()
    return pp.identity_for(
        code=pp.code_digest(package_dir, extra_files), config_sha=cfg_sha,
        weights=[pp.sampled_digest(p) for p in pp.packed_files(packed_dir)], geometry=geometry,
        optimizations=optimizations, envelope_id=env_id, arithmetic_version=CONTEXT_ARITH_VERSION,
        backend=backend if backend is not None else pp.backend_components())


def prefix_settings():
    """-> (directory, quota bytes). `SLOTSTREAM_PREFIX_DIR` overrides `[prefix] dir` (cold cells point it at an empty
    directory so a cold sample can never restore from disk)."""
    import os
    from slotstream_spark import config
    d = os.environ.get("SLOTSTREAM_PREFIX_DIR") or config.get("prefix.dir")
    return os.path.expanduser(d), int(config.get("prefix.quota_bytes"))


def open_tier(st, key, prefill_chunk, *, model_dir, packed_dir, envelope, extra_files=(), directory=None, quota=None,
              identity=None, on_event=None, minimum_tokens=None):
    """Open the persistent tier for a loaded engine: -> PersistentTier."""
    import atexit
    from pathlib import Path
    from slotstream_spark.engine import persistent_prefix as pp
    d0, q0 = prefix_settings()
    if identity is None:
        from slotstream_spark.kernels.vendored import QSAShapes
        from dataclasses import astuple
        identity = build_identity(
            package_dir=Path(__file__).resolve().parents[1], extra_files=extra_files, model_dir=model_dir,
            packed_dir=packed_dir, geometry=[key.model_identity, int(prefill_chunk)] + [str(x) for x in astuple(QSAShapes())],
            optimizations=json.loads(key.optimizations_json) if isinstance(key.optimizations_json, str) else {},
            envelope=envelope)
    kw = {} if minimum_tokens is None else {"minimum_tokens": int(minimum_tokens)}
    cfg = pp.Configuration(directory or d0, quota if quota is not None else q0, **kw)
    prefix = pp.PersistentPrefix(cfg, identity, on_event=on_event)
    lin, attn = state_layers(st)
    tier = PersistentTier(prefix, prefill_chunk, lin, attn, on_event=on_event)
    atexit.register(tier.close)                 # idle-exit and SIGTERM (uvicorn returns) flush in-flight writes <= 20 s
    return tier
