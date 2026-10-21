# SPDX-License-Identifier: MIT
# Port of Tools/sampler_ref.py and Sampler.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""Token sampler, a bit-exact port of third_party/slotstream/Tools/sampler_ref.py (numpy float32).

Order matches the reference and Sampler.next(): presence penalty on raw logits, temperature,
top-k, top-p, min-p, inverse-CDF draw scaled by the unnormalized total. The generator is a
splitmix64 stream, so equal seeds give equal tokens on every machine.
"""

import math
from dataclasses import dataclass, field

import numpy as np

_M = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
MAX_TOKEN_CEILING = 32_768  # Generate.swift SampleParams.maxTokenCeiling


def splitmix(v):
    v = (v + _GAMMA) & _M
    v = ((v ^ (v >> 30)) * 0xBF58476D1CE4E5B9) & _M
    v = ((v ^ (v >> 27)) * 0x94D049BB133111EB) & _M
    return v ^ (v >> 31)


def make_logits(vocab, seed):
    st, out = seed, []
    for _ in range(vocab):
        st = splitmix(st)
        out.append(np.float32(st >> 40) / np.float32(1 << 24) * np.float32(8.0) - np.float32(4.0))
    return np.array(out, dtype=np.float32)


@dataclass
class SampleParams:
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 40
    min_p: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 512
    stop: list = field(default_factory=list)

    def sanitized(self):
        """Clamp every knob into the range the sampler is defined on (Generate.swift:26-44)."""
        p = SampleParams(self.temperature, self.top_p, self.top_k, self.min_p,
                         self.presence_penalty, self.max_tokens, list(self.stop))
        if not math.isfinite(p.temperature):
            p.temperature = 0.0
        p.temperature = max(0.0, p.temperature)
        if not math.isfinite(p.top_p) or p.top_p <= 0 or p.top_p > 1:
            p.top_p = 1.0
        if not math.isfinite(p.min_p):
            p.min_p = 0.0
        p.min_p = min(max(0.0, p.min_p), 1.0)
        if not math.isfinite(p.presence_penalty):
            p.presence_penalty = 0.0
        p.top_k = max(0, int(p.top_k))
        if p.max_tokens <= 0:
            p.max_tokens = MAX_TOKEN_CEILING
        p.max_tokens = min(p.max_tokens, MAX_TOKEN_CEILING)
        p.stop = [s for s in p.stop if s]
        return p


def _softmax(x):
    e = np.exp((x - x.max()).astype(np.float32))
    return (e / e.sum()).astype(np.float32)


class Sampler:
    def __init__(self, seed=0):
        self.state = seed if seed != 0 else 0xDEADBEEF

    def next(self, logits, p, generated):
        """logits: float32 [vocab]; p: SampleParams (sanitized here); generated: set of token ids."""
        p = p.sanitized()
        l = np.asarray(logits, dtype=np.float32).copy()
        if p.presence_penalty != 0 and generated:
            idx = np.array(sorted(generated), dtype=np.int64)
            l[idx] = (l[idx] - np.float32(p.presence_penalty)).astype(np.float32)
        if p.temperature <= 0:
            return int(np.argmax(l))
        l = (l / np.float32(p.temperature)).astype(np.float32)
        if 0 < p.top_k < l.shape[0]:
            kth = np.sort(l)[::-1][p.top_k - 1]
            l = np.where(l < kth, np.float32(-np.inf), l).astype(np.float32)
        probs = _softmax(l)
        if p.top_p < 1:
            order = np.argsort(-probs, kind="stable")
            srt = probs[order]
            cum = np.cumsum(srt, dtype=np.float32)
            keep_sorted = (cum - srt) < np.float32(p.top_p)
            keep = np.zeros_like(probs, dtype=bool)
            keep[order] = keep_sorted
            probs = np.where(keep, probs, np.float32(0)).astype(np.float32)
        if p.min_p > 0:
            cutoff = (probs.max() * np.float32(p.min_p)).astype(np.float32)
            probs = np.where(probs < cutoff, np.float32(0), probs).astype(np.float32)
        self.state = splitmix((self.state + 1) & _M)
        u = np.float32((self.state >> 11) / float(1 << 53))
        cdf = np.cumsum(probs, dtype=np.float32)
        total = cdf[-1]
        if not np.isfinite(total) or total <= 0:
            return int(np.argmax(np.asarray(logits, dtype=np.float32)))
        pick = int(((cdf <= np.float32(0)) if u == 0 else (cdf < (u * total).astype(np.float32))).sum())
        return min(pick, probs.shape[0] - 1)
