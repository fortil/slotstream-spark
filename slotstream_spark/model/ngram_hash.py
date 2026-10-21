# SPDX-License-Identifier: MIT AND Apache-2.0
# Portions ported from helgard-orlm/qwen-flash-next-8gb@958d22e7ba0317750ccc38862b9667b1034a4dd1 qwen_engine.py:178-258 (Apache-2.0, section 4); modified.
"""N-gram row ids for the PLE table (numpy, int64 wraparound like torch).

Mirrors Qwen4ExpTextNGramEmbedding (transformers v5.19.0 modeling:1040-1160, qwen4_exp.py:552-591):
the ids of token t depend on the two previous tokens of the same segment (EOS resets the segment),
mixed with the per-layer multipliers by xor, reduced modulo a prime per head and offset into one table.
"""

import math

import numpy as np

_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB
_P1 = 10007
N_SHARDS = 128  # split_ngram_parts; the table is padded to a multiple of this


def _splitmix64(v):
    v = (v + _GAMMA) & _MASK64
    v = ((v ^ (v >> 30)) * _M1) & _MASK64
    v = ((v ^ (v >> 27)) * _M2) & _MASK64
    return (v ^ (v >> 31)) & _MASK64


def _is_prime(v):
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    for d in range(3, math.isqrt(v) + 1, 2):
        if v % d == 0:
            return False
    return True


def _nth_prime_after(start, count):
    p = start
    for _ in range(count):
        p += 1
        while not _is_prime(p):
            p += 1
    return p


class NgramHasher:
    def __init__(self, multipliers, head_sizes, head_offsets, eos, ngram_size=3, heads_per_ngram=8):
        self.ngram_size = int(ngram_size)
        self.ctx = self.ngram_size - 1
        self.heads_per_ngram = int(heads_per_ngram)
        self.heads = self.ctx * self.heads_per_ngram
        self.eos = int(eos)
        self.mult = np.asarray(multipliers, dtype=np.int64)
        self.sizes = np.asarray(head_sizes, dtype=np.int64)
        self.offs = np.asarray(head_offsets, dtype=np.int64)
        if len(self.mult) != self.ngram_size or len(self.sizes) != self.heads or len(self.offs) != self.heads:
            raise ValueError("multipliers/head sizes/offsets do not match ngram_size and heads_per_ngram")

    @classmethod
    def from_seed(cls, vocab_size=248320, seed=1234, ple_layer_index=0,
                  ngram_vocab_size_base=20_000_000, eos=248044, ngram_size=3, heads_per_ngram=8):
        heads = (ngram_size - 1) * heads_per_ngram
        sizes, offs, tot = [], [], 0
        for h in range(heads):
            s = _nth_prime_after(ngram_vocab_size_base - 1, ple_layer_index * heads + h + 1)
            sizes.append(s)
            offs.append(tot)
            tot += s
        half = max(1, (((1 << 63) - 1) // max(vocab_size, 1)) // 2)
        base = seed + _P1 * ple_layer_index
        mult = [2 * (_splitmix64((base + _GAMMA * (i + 1)) & _MASK64) % half) + 1 for i in range(ngram_size)]
        return cls(mult, sizes, offs, eos, ngram_size, heads_per_ngram)

    @classmethod
    def from_buffers(cls, layer_multipliers, ngram_heads_vocab_sizes, ngram_heads_offsets, eos):
        n = len(layer_multipliers)
        hpn = len(ngram_heads_vocab_sizes) // (n - 1)
        return cls(layer_multipliers, ngram_heads_vocab_sizes, ngram_heads_offsets, eos, n, hpn)

    # ---- table geometry
    @property
    def total_rows(self):
        return int(self.sizes.sum())

    @property
    def padded_rows(self):
        return math.ceil(self.total_rows / N_SHARDS) * N_SHARDS

    @property
    def rows_per_shard(self):
        return math.ceil(self.padded_rows / N_SHARDS)

    def shard_row(self, gid):
        rps = self.rows_per_shard
        return int(gid) // rps, int(gid) % rps

    # ---- ids
    def _shift(self, hist, shift):
        if shift == 0:
            return hist
        T = hist.shape[0]
        pos = np.arange(T, dtype=np.int64)
        eosp = np.where(hist == self.eos, pos, -1)
        prev_incl = np.maximum.accumulate(eosp)
        prev = np.concatenate([np.array([-1], dtype=np.int64), prev_incl[:-1]])
        in_seg = pos - (prev + 1)
        src = pos - shift
        gathered = hist[np.maximum(src, 0)]
        valid = (in_seg >= shift) & (src >= 0)
        return np.where(valid, gathered, np.int64(self.eos))

    def ids(self, prev_ctx, new_ids):
        """prev_ctx: the last ngram_size-1 tokens before new_ids (EOS at the start) -> int64 [len(new_ids), heads]."""
        prev_ctx = list(prev_ctx)
        if len(prev_ctx) != self.ctx:
            raise ValueError(f"prev_ctx must hold {self.ctx} tokens")
        n_new = len(new_ids)
        hist = np.asarray(prev_ctx + list(new_ids), dtype=np.int64)
        sh = [self._shift(hist, s) for s in range(self.ngram_size)]
        blocks = []
        for ng in range(2, self.ngram_size + 1):
            a, b = (ng - 2) * self.heads_per_ngram, (ng - 1) * self.heads_per_ngram
            mixed = np.multiply(sh[0], self.mult[0], dtype=np.int64)
            for p in range(1, ng):
                mixed = np.bitwise_xor(mixed, np.multiply(sh[p], self.mult[p], dtype=np.int64))
            r = np.remainder(mixed[:, None], self.sizes[a:b][None, :])
            blocks.append(r + self.offs[a:b][None, :])
        return np.concatenate(blocks, axis=-1)[-n_new:] if n_new else np.zeros((0, self.heads), dtype=np.int64)
