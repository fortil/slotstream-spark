"""Prefill driver: a prompt through the 48 layers in passes, the MoE swept over the experts (plan step 70).

`PrefillDriver.run` cuts the new tokens into passes with the schedule of `prefill_schedule` (pass = `plan.prefill_pass`
clipped by `chunk()`), groups identical small passes into a read scope (`group_scopes`), and for each scope runs layer
by layer: the attention mixers in chronological tiles (one tile = one pass, so the shapes of every tile-level op are the
same with the scope on or off), the router per tile, then one `SweepMoE.layer` over all rows of the scope. The final
pass is its own scope and is the only one that writes the pool (the `admission_quota` hottest experts per layer).

`control.check(nbytes)` runs before each pass with `pass x bytes_per_token + (scope - pass) x 32,768 + attn_transient(q,k)`
(ContextMemory.swift:248-249; bytes_per_token and the per-pair term come from the envelope record of step 69) and
`should_continue()` before every pass and after every layer.
"""
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from slotstream_spark.engine import prefill_schedule as ps
from slotstream_spark.engine.pressure import Cancelled
from slotstream_spark.memory import ledger
from slotstream_spark.model import layers
from slotstream_spark.model.moe_sweep import SweepMoE, shared_expert
from slotstream_spark.model.prefill_mixers import attn_block

KV_PER_SCOPE_TOKEN = 32_768          # ContextMemory.swift:248-249: bytes a scope token holds beyond the running pass


class PrefillCancelled(Cancelled):
    pass


class PassBudgetExceeded(RuntimeError):
    pass


class PassBudget:
    """control.check(nbytes) -> raises PassBudgetExceeded when a pass would not fit `limit_bytes`; keeps the maximum."""

    def __init__(self, limit_bytes):
        self.limit = int(limit_bytes)
        self.max_seen = 0
        self.calls = 0

    def check(self, nbytes):
        self.calls += 1
        self.max_seen = max(self.max_seen, int(nbytes))
        if nbytes > self.limit:
            raise PassBudgetExceeded(f"pass needs {int(nbytes):,} B, the budget is {self.limit:,} B")


def d5_budget():
    """The D5 prefill line of the ledger plus its read-scope term."""
    led = ledger.Ledger("D5")
    return PassBudget(led.pass_bytes + led.read_scope)


@dataclass
class PrefillPlan:
    prefill_pass: int = 4096            # D5 (plan section 3.2)
    read_scope: int = 8192              # tokens a scope may hold at most; 0 = no scope
    max_rows_per_launch: int = 81_920   # overwritten from the envelope by `from_envelope`

    @classmethod
    def from_envelope(cls, env=None, **kw):
        env = env or ps.envelope()
        if "max_rows_per_launch" in env:
            kw.setdefault("max_rows_per_launch", int(env["max_rows_per_launch"]))
        return cls(**kw)


@dataclass
class PrefillResult:
    last_logits: torch.Tensor
    passes: list
    bytes_requested: int
    records_read: int
    admitted: int
    seconds: float
    scopes: list = field(default_factory=list)
    pool_writes_nonfinal: int = 0
    checkpoint_ends: list = field(default_factory=list)
    sweep: dict = field(default_factory=dict)
    admit: bool = True


class PrefillDriver:
    def __init__(self, model, moe: SweepMoE, envelope: Optional[dict] = None):
        self.model, self.moe = model, moe
        self.env = envelope if envelope is not None else ps.envelope()
        self.P = int(self.env["qk_product"]) if "qk_product" in self.env else ps.qk_product()

    # ------------------------------------------------------------------ helpers
    def _bytes_per_token(self):
        return float(self.env.get("bytes_per_pass_token", ledger.PASS_PER_TOKEN))

    def attn_transient(self, q, k):
        return float(self.env.get("bytes_per_pair", 0.0)) * q * k

    def pass_bytes(self, size, scope_tokens, pos):
        return int(size * self._bytes_per_token() + (scope_tokens - size) * KV_PER_SCOPE_TOKEN +
                   self.attn_transient(size, pos + size))

    def _map_copy(self):
        return np.array(self.moe.pool.pool.map_numpy(), copy=True)

    # ------------------------------------------------------------------ run
    def run(self, ids, state, *, plan, control, should_continue: Callable[[], bool], checkpoints: Sequence[int] = (),
            on_rows: Optional[Callable[[int, torch.Tensor], None]] = None, dump_dir=None, admit: bool = True,
            on_multi: Optional[Callable[[list, torch.Tensor], None]] = None):
        model, seg, moe = self.model, self.model.seg, self.moe
        ids = [int(t) for t in ids]
        n = len(ids)
        if n == 0:
            raise ValueError("prefill needs at least one token")
        base0 = int(state.pos)
        if base0 + n > model.max_ctx:
            raise ValueError(f"context {base0 + n} exceeds max_ctx {model.max_ctx}")
        t0 = time.perf_counter()
        prior = list(state.tokens)
        full = prior + ids                                    # tokens by absolute position (len(prior) == base0)
        if len(prior) != base0:
            full = [0] * base0 + ids                          # state without a token list: only the n-gram history needs it
        sizes = ps.passes(n, base0, plan.prefill_pass, True, self.P)
        scopes = ps.group_scopes(sizes, base0, plan.read_scope, tuple(c for c in checkpoints))
        stats0 = dict(moe.stats.as_dict())
        gemm0 = dict(layers.GEMM_STATS)
        sink = None
        if dump_dir:
            from slotstream_spark.model.forward import _Dump
            sink = _Dump(dump_dir)
        res = PrefillResult(last_logits=None, passes=sizes, bytes_requested=0, records_read=0, admitted=0,
                            seconds=0.0, scopes=[list(s) for s in scopes], admit=admit)
        pos = base0
        ends = set()
        try:
            with torch.no_grad():
                for si, scope in enumerate(scopes):
                    final = si == len(scopes) - 1
                    pos = self._scope(scope, pos, full, final, state, plan, control, should_continue, on_rows, sink, res,
                                       admit, on_multi)
                    p = pos
                    ends.add(p)
        finally:
            if sink:
                sink.finish(seg.gdn)
                sink.close()
        state.pos = base0 + n
        state.tokens = full
        res.checkpoint_ends = sorted(c for c in checkpoints if c in ends)
        s1 = moe.stats.as_dict()
        res.bytes_requested = s1["bytes_requested"] - stats0["bytes_requested"]
        res.records_read = s1["records_read"] - stats0["records_read"]
        res.admitted = s1["admitted"] - stats0["admitted"]
        res.sweep = {k: s1[k] - stats0[k] for k in s1 if k != "max_launch_rows"}
        res.sweep["max_launch_rows"] = s1["max_launch_rows"]
        res.sweep["gemm_calls"] = layers.GEMM_STATS["gemm_calls"] - gemm0["gemm_calls"]
        res.sweep["gemv_fallbacks"] = layers.GEMM_STATS["gemv_fallbacks"] - gemm0["gemv_fallbacks"]
        res.seconds = time.perf_counter() - t0
        return res

    def _scope(self, scope, pos0, full, final, state, plan, control, should_continue, on_rows, sink, res,
               admit=True, on_multi=None):
        with layers.prefill_gemm():                           # every dense lin of the driver runs on the GEMM
            return self._scope_body(scope, pos0, full, final, state, plan, control, should_continue, on_rows, sink,
                                    res, admit, on_multi)

    def _scope_body(self, scope, pos0, full, final, state, plan, control, should_continue, on_rows, sink, res,
                    admit=True, on_multi=None):
        model, seg, moe = self.model, self.model.seg, self.moe
        scope_tokens = sum(scope)
        tiles, p, a = [], pos0, 0
        for size in scope:
            if not should_continue():
                raise PrefillCancelled("prefill cancelled before a pass")
            control.check(self.pass_bytes(size, scope_tokens, p))
            tiles.append((p, full[p:p + size], a, a + size))
            p += size
            a += size
        map_before = None if final else self._map_copy()
        hyp = torch.cat([seg.seg_begin(t[1]) for t in tiles], 0)
        if sink:
            sink.h_in(hyp)
        for L in range(model.NL):
            attn, mixed, inj, rids, rw = [], [], [], [], []
            for pos, idt, lo, hi in tiles:
                ht = attn_block(seg, L, state, hyp[lo:hi], idt, base=pos, history=full[max(0, pos - 2):pos])
                m, ij, ri, w = seg.seg_router(L, ht)
                attn.append(ht)
                mixed.append(m)
                inj.append(ij)
                rids.append(ri)
                rw.append(w)
                if sink and L in seg.ple_layers:
                    sink.ngram(seg.last_ngram_ids)
            cat = lambda xs: xs[0] if len(xs) == 1 else torch.cat(xs, 0)
            mixed_c, inj_c, rids_c, rw_c, hyp_a = cat(mixed), cat(inj), cat(rids), cat(rw), cat(attn)
            ids_host = rids_c.detach().cpu().numpy().astype(np.int32)
            routed = moe.layer(L, mixed_c, ids_host, rw_c, admit=final and admit)
            h = cat([routed[lo:hi].to(mixed_c.dtype) + shared_expert(seg, L, mixed_c[lo:hi]) for _, _, lo, hi in tiles])
            hyp = hyp_a + (h.unsqueeze(-2) * inj_c.unsqueeze(-1)).flatten(-2)
            if sink:
                sink.layer(L, hyp, ids_host, rw_c.detach().float().cpu().numpy())
            if not should_continue():
                raise PrefillCancelled(f"prefill cancelled after layer {L}")
        for pos, idt, lo, hi in tiles:
            if on_multi is not None:
                on_multi(idt, hyp[lo:hi])
            fin = seg.seg_final(hyp[lo:hi])
            if on_rows is not None:
                on_rows(pos, fin)
            if sink:
                sink.logits(model._logits(fin).cpu())
            if final and hi == tiles[-1][3]:
                res.last_logits = model._logits(fin[-1:])[0]
        if map_before is not None:
            res.pool_writes_nonfinal += int(np.count_nonzero(self._map_copy() != map_before))
        return pos0 + scope_tokens
