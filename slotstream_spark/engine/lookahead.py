# SPDX-License-Identifier: MIT
# Port of DecodeLookahead.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""Forecast tap, host side (plan step 73; Expert Lookahead of slotstream, DecodeLookahead.swift).

`ForecastTap` is what `runtime.graphs.GraphSet(..., forecast=tap)` captures into G(l): the router of layer l+1 on the streams
after layer l's attention, in FP32, optionally with the rank-128 correction, the top-24 ids and logits written to the pinned
readback slot before layer l's signal. The scheduler (step 74) reads them with `Readback.forecast(l, rows)`.

This module also holds what the gate and the scheduler share: the pure-numpy float64 reference of the tap, the tie-aware set
check (an id present in one top-10 set and not in the other is excused only if the reference logit lies within
`margin = [ops] forecast_tie_rel x max|corrected - raw|` of the reference's 10th logit), recall, and the status file of the
optional sidecar.
"""
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TOPK = 10
STATUS_RE = re.compile(r"^tap_correction: (present (\d+) ([0-9a-f]{64})|absent \((.+)\))$")


class ForecastTap:
    """`correction` None = the uncorrected (plain attention) tap."""

    def __init__(self, correction=None):
        self.correction = correction

    def write(self, gs, L, hyp):
        """Called inside G(L) (and in eager runs): forecast of layer L + 1 into gs.rb.f_ids/f_logits[L]."""
        from slotstream_spark.kernels import forecast as fc      # torch only where the tap runs
        R = gs.rows
        fc.forecast_into(gs.seg, L + 1, hyp, gs.rb.f_ids[L, :R], gs.rb.f_logits[L, :R], self.correction)
        if gs.tap_probe is not None:
            mixed, raw, used = fc.tap(gs.seg, L + 1, hyp, self.correction)
            gs.tap_probe(L, mixed, raw, used)


def tap_status(model_root):
    """-> dict(present bool, size, sha256, reason, line) from `<model_root>/tap-correction.status` (the file written by
    tools/fetch_tap_correction.sh next to the model directory); absent with reason 'no status file' when missing."""
    p = Path(model_root) / "tap-correction.status"
    if not p.is_file():
        return {"present": False, "reason": "no status file", "line": None}
    line = p.read_text().strip()
    m = STATUS_RE.match(line)
    if not m:
        return {"present": False, "reason": f"unparsable status: {line}", "line": line}
    if m.group(2):
        return {"present": True, "size": int(m.group(2)), "sha256": m.group(3), "reason": None, "line": line}
    return {"present": False, "reason": m.group(4), "line": line}


def top_ids(logits, k=TOPK):
    """Ids of the k largest logits of one row, descending by logit, ties by lower index (the router's rule)."""
    logits = np.asarray(logits)
    order = np.lexsort((np.arange(logits.shape[-1]), -logits))
    return order[:k]


def forecast_top(ids24, logits24, k=TOPK):
    """The forecast's top-k ids of one row from the tap's top-24 (ids, logits): by logit descending, ties by lower id."""
    ids24, logits24 = np.asarray(ids24), np.asarray(logits24)
    keep = ids24 >= 0
    order = np.lexsort((ids24[keep], -logits24[keep]))
    return ids24[keep][order][:k]


def recall(forecast_ids, actual_ids):
    """Share of the actual routed experts (one row) that the forecast set contains."""
    a = set(int(x) for x in actual_ids)
    return len(a & set(int(x) for x in forecast_ids)) / len(a)


def reference_logits(mixed, gate, a=None, b=None, mu=None, delta=None):
    """float64 numpy: (raw, corrected) logits [R, experts] of the tap from the engine's mixed input [R, hidden] and the
    router matrix [experts, hidden]; corrected = raw + ((mixed - mu) a) b + delta, all in float64."""
    m = np.asarray(mixed, dtype=np.float64)
    raw = m @ np.asarray(gate, dtype=np.float64).T
    if a is None:
        return raw, raw
    corr = ((m - np.asarray(mu, dtype=np.float64)) @ np.asarray(a, dtype=np.float64)) @ np.asarray(b, dtype=np.float64)
    return raw, raw + corr + np.asarray(delta, dtype=np.float64)


def set_check(engine_ids, ref_corrected, ref_raw, tie_rel, k=TOPK):
    """One row. `engine_ids`: the engine's top-k set; `ref_corrected`/`ref_raw`: the float64 reference logits [experts].
    -> dict(equal, excused, hard, margin): ids in the symmetric difference of the two sets, `excused` when the reference
    logit lies within margin of the reference's k-th logit, `hard` otherwise."""
    ref_ids = top_ids(ref_corrected, k)
    kth = float(ref_corrected[ref_ids[-1]])
    margin = float(tie_rel) * float(np.max(np.abs(np.asarray(ref_corrected) - np.asarray(ref_raw))))
    a, b = set(int(x) for x in engine_ids), set(int(x) for x in ref_ids)
    diff = (a ^ b)
    excused = [i for i in diff if abs(float(ref_corrected[i]) - kth) <= margin]
    return {"equal": not diff, "excused": len(excused), "hard": len(diff) - len(excused), "margin": margin,
            "diff": sorted(diff)}


# ----------------------------------------------------------------------------------------------------------- scheduler (74)
MODES = ("off", "attention", "attention-corrected", "boundary2")


@dataclass(frozen=True)
class LookaheadConfig:
    """`platform.toml [lookahead]`; the env variable SLOTSTREAM_LOOKAHEAD overrides `mode`."""
    mode: str = "off"
    threshold: float = 0.062
    top: int = 24
    issue_cap: int = 32
    slot_cap: int = 64
    min_evictable: int = 256

    @classmethod
    def load(cls, cfg=None, env=None):
        if cfg is None:
            from slotstream_spark import config
            cfg = config.load()
        sec = dict(cfg.get("lookahead", {}))
        env = os.environ if env is None else env
        if env.get("SLOTSTREAM_LOOKAHEAD"):
            sec["mode"] = env["SLOTSTREAM_LOOKAHEAD"]
        c = cls(**sec)
        if c.mode not in MODES:
            raise ValueError(f"lookahead mode {c.mode!r} is not one of {MODES}")
        return c

    @property
    def corrected(self):
        return self.mode == "attention-corrected"

    @property
    def on(self):
        return self.mode != "off"


def candidates(ids, logits, threshold, top=24, k=TOPK, cap=32):
    """Speculative candidates of one forecast hand-over: per row, the first `top` of the tap's top-24 by logit (ties by
    lower id) with margin = logit - the row's k-th logit >= threshold (ExpertPrefetch.swift:951). Rows merge, an id keeps its
    best margin, order is margin descending then id, at most `cap` ids. -> int32 array."""
    ids, logits = np.atleast_2d(np.asarray(ids)), np.atleast_2d(np.asarray(logits, dtype=np.float32))
    best = {}
    for r in range(ids.shape[0]):
        keep = ids[r] >= 0
        i_r, l_r = ids[r][keep], logits[r][keep]
        order = np.lexsort((i_r, -l_r))[:top]
        i_r, l_r = i_r[order], l_r[order]
        if len(l_r) < k:
            continue
        ref = l_r[k - 1]
        for e, v in zip(i_r.tolist(), l_r.tolist()):
            m = v - float(ref)
            if m >= threshold and m > best.get(e, -np.inf):
                best[e] = m
    order = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))[:cap]
    return np.array([e for e, _ in order], dtype=np.int32)


class Scheduler:
    """Speculative prefetch scheduling around the decode loop (plan step 74). `pool` offers issue(layer, ids) -> n,
    expire_layer(layer) and expire_all() (runtime.pool.ExpertPool, or a fake in the T0 tests).

    Window 1 (`attention`, `attention-corrected`): after layer L's routing has been ensured, the forecast written by G(L) for
    layer L + 1 is turned into tickets. `boundary2` (stride 2): only after even layers. Resident or already-ticketed keys are
    skipped by the pool; after each ensure(L) the tickets of layers <= L that nobody claimed are returned."""

    def __init__(self, pool, cfg):
        self.pool, self.cfg = pool, cfg
        self.issued_calls = 0
        self.candidates_seen = 0
        self.issued = 0
        self.targets = 0

    def target(self, layer, n_layers):
        """Target layer of the forecast made after `layer`, or None when this layer issues nothing."""
        if not self.cfg.on or layer + 1 >= n_layers:
            return None
        if self.cfg.mode == "boundary2" and layer % 2:
            return None
        return layer + 1

    def on_layer(self, layer, ids24, logits24, n_layers):
        """After ensure(layer). -> number of tickets issued for the target layer."""
        self.pool.expire_layer(layer)
        t = self.target(layer, n_layers)
        if t is None:
            return 0
        cand = candidates(ids24, logits24, self.cfg.threshold, self.cfg.top, cap=self.cfg.issue_cap)
        self.targets += 1
        self.candidates_seen += len(cand)
        n = int(self.pool.issue(t, cand)) if len(cand) else 0
        self.issued += n
        return n

    def end_request(self):
        """-> tickets returned at the end of a request (0 when every ticket was claimed or expired in the loop)."""
        return int(self.pool.expire_all())

    def stats(self):
        return {"mode": self.cfg.mode, "targets": self.targets, "candidates": self.candidates_seen, "issued": self.issued}
