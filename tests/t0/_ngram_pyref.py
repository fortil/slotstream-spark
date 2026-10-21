"""Pure-Python big-int transliteration of qwen4_exp.py:552-591 with explicit int64 wraparound."""
import math

MASK = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
M1 = 0xBF58476D1CE4E5B9
M2 = 0x94D049BB133111EB
P1 = 10007


def wrap(x):
    """Two's-complement int64 of an arbitrary Python int."""
    return ((x + (1 << 63)) & MASK) - (1 << 63)


def splitmix(v):
    v = (v + GAMMA) & MASK
    v = ((v ^ (v >> 30)) * M1) & MASK
    v = ((v ^ (v >> 27)) * M2) & MASK
    return (v ^ (v >> 31)) & MASK


def is_prime(v):
    if v < 2:
        return False
    if v % 2 == 0:
        return v == 2
    return all(v % d for d in range(3, math.isqrt(v) + 1, 2))


def nth_prime_after(start, count):
    p = start
    for _ in range(count):
        p += 1
        while not is_prime(p):
            p += 1
    return p


def build(vocab=248320, seed=1234, layer=0, base=20_000_000, ngram_size=3, hpn=8):
    heads = (ngram_size - 1) * hpn
    sizes = [nth_prime_after(base - 1, layer * heads + h + 1) for h in range(heads)]
    offs = [sum(sizes[:h]) for h in range(heads)]
    half = max(1, ((1 << 63) - 1) // max(vocab, 1) // 2)
    b = seed + P1 * layer
    mult = [2 * (splitmix((b + GAMMA * (i + 1)) & MASK) % half) + 1 for i in range(ngram_size)]
    return mult, sizes, offs


def ngram_ids(prev_ctx, ids, mult, sizes, offs, eos, ngram_size=3, hpn=8):
    hist = list(prev_ctx) + list(ids)
    n_prev = len(prev_ctx)

    def shifted(t, s):
        """Token s positions back, unless an EOS lies in between (then EOS)."""
        if s == 0:
            return hist[t]
        # previous EOS strictly before t
        j = t - 1
        while j >= 0 and hist[j] != eos:
            j -= 1
        in_seg = t - (j + 1)
        if in_seg >= s and t - s >= 0:
            return hist[t - s]
        return eos

    out = []
    for t in range(n_prev, len(hist)):
        row = []
        for ng in range(2, ngram_size + 1):
            for h in range((ng - 2) * hpn, (ng - 1) * hpn):
                mixed = wrap(shifted(t, 0) * mult[0])
                for p in range(1, ng):
                    mixed = mixed ^ wrap(shifted(t, p) * mult[p])
                row.append(mixed % sizes[h] + offs[h])
        out.append(row)
    return out
