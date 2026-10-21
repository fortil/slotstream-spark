# SPDX-License-Identifier: MIT
# Port of Context.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""Prefill pass schedule: how a prompt is cut into passes (port of Context.swift:61-320, `PrefillSchedule`).

A pass is faster the bigger it is, but the sparse attention scores every query of the pass against every key already
in the context, so the transient grows with chunk x context. The schedule halves the pass until the query-by-key
product fits `qk_product`, the largest product the CUDA transient envelope (tools/prefill_envelope.py) measured to fit
the D5 prefill ledger line. The Mac constant 4096 x 8016 = 32,833,536 is only a bootstrap value: it is used when
`SLOTSTREAM_ENVELOPE=bootstrap`, otherwise the product is read from the envelope JSON and its absence is an error.

Pure Python, no torch.
"""
import builtins
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENVELOPE_JSON = ROOT / "db" / "records" / "measurements" / "f5-prefill-envelope.json"
BOOTSTRAP_QK_PRODUCT = 4096 * 8016          # Context.swift:74
MODEL_LIMIT = 262_144                       # Context.swift:8
MIN_CHUNK = 64                              # Context.swift:76
MIN_SWEEP_TOKENS = 256                      # SweepTuning.minTokens: a short pass keeps its own dispatch
SCOPE_SIZES = (256, 512, 1024)
SCOPE_MIN_PASSES = 4
SCOPE_CAP_256 = 8192                        # Context.swift:127
SCOPE_CAP_OTHER = 4096


class EnvelopeMissing(RuntimeError):
    pass


def qk_product(path=None):
    """The query-by-key bound: the bootstrap constant under SLOTSTREAM_ENVELOPE=bootstrap, else the envelope JSON."""
    if os.environ.get("SLOTSTREAM_ENVELOPE") == "bootstrap":
        return BOOTSTRAP_QK_PRODUCT
    p = Path(path) if path else ENVELOPE_JSON
    if not p.is_file():
        raise EnvelopeMissing(f"{p} is missing; run tools/prefill_envelope.py --write or set SLOTSTREAM_ENVELOPE=bootstrap")
    v = json.loads(p.read_text()).get("qk_product")
    if not isinstance(v, int) or v <= 0:
        raise EnvelopeMissing(f"{p} has no integer qk_product")
    return v


def fits(count, position, P=None):
    P = qk_product() if P is None else P
    return count > 0 and position >= 0 and position <= P // count - count


def chunk(at, max_chunk, P=None):
    """The pass to run when the state holds `at` tokens: halve from `max_chunk` until it fits, floor 256 while 256
    fits, else 64."""
    P = qk_product() if P is None else P
    if at < 0 or at >= MODEL_LIMIT:
        return 0
    floor = 256 if fits(256, at, P) else MIN_CHUNK
    c = min(4096, max(floor, max_chunk))
    while c > floor and not fits(c, at, P):
        c = max(floor, c // 2)
    while c > 1 and not fits(c, at, P):
        c //= 2
    return c if fits(c, at, P) else 0


def next(remaining, at, max_chunk, tail_aware=True, P=None):  # noqa: A001 (the Swift name)
    """Size of the next pass. With `tail_aware` the actual remaining rows are tried before shrinking a hypothetical
    full pass: a 3,864-row tail after 4,096 fits."""
    P = qk_product() if P is None else P
    if remaining <= 0 or at < 0 or at >= MODEL_LIMIT or remaining > MODEL_LIMIT - at:
        return 0
    candidate = min(remaining, min(4096, max(1, max_chunk)))
    if tail_aware and fits(candidate, at, P):
        return candidate
    return min(remaining, chunk(at, max_chunk, P))


def passes(tokens, base=0, max_chunk=4096, tail_aware=True, P=None):
    """Sizes of the passes that read `tokens` new tokens from position `base`."""
    out, pos, left = [], base, tokens
    while left > 0:
        c = next(left, pos, max_chunk, tail_aware, P)
        if c <= 0:
            raise ValueError(f"no pass fits at position {pos}")
        out.append(c)
        pos += c
        left -= c
    return out


def scope_passes(remaining, at, max_chunk, max_scope, tail_aware=True, P=None, experimental_maximum=8192):
    """Group existing chronological compute passes without enlarging any product (a scope shares reads, it is not a
    compute pass). Context.swift:141-160."""
    P = qk_product() if P is None else P
    if remaining <= 0 or at < 0 or at >= MODEL_LIMIT or remaining > MODEL_LIMIT - at:
        return []
    out, count = [], 0
    bound = max(MIN_CHUNK, min(min(16384, max(8192, experimental_maximum)), max_scope))
    while count < remaining:
        n = next(remaining - count, at + count, max_chunk, tail_aware, P)
        if n == 0 or (count > 0 and (n < MIN_SWEEP_TOKENS or n > bound - count)):
            break
        out.append(n)
        count += n
        if count >= bound:
            break
    return out


def automatic_scope(remaining, at, max_chunk, max_scope=SCOPE_CAP_256, P=None):
    """Read scope of the passes starting at `at`: a list of >= 4 identical passes of 256, 512 or 1024 whose tokens
    stay within 8192 (256-token passes) or 4096 (others) and within `max_scope`; None otherwise (one pass, no scope)."""
    P = qk_product() if P is None else P
    if not 256 <= max_chunk <= 4096:
        return None
    scheduled = chunk(at, max_chunk, P)
    if scheduled not in SCOPE_SIZES or scheduled < MIN_SWEEP_TOKENS:
        return None
    cap = min(max_scope, SCOPE_CAP_256 if scheduled == 256 else SCOPE_CAP_OTHER)
    proposed = scope_passes(remaining, at, max_chunk, cap, False, P, max_scope)
    run = []
    for n in proposed:
        if n != scheduled:
            break
        run.append(n)
    return run if len(run) >= SCOPE_MIN_PASSES else None


def resume_boundaries(prompt_len, max_chunk, tail_aware=True, P=None):
    """Positions inside the prompt a later request may resume from: never the prompt end, never the late regime
    (where a 256-row pass no longer fits)."""
    P = qk_product() if P is None else P
    if prompt_len <= 0 or prompt_len > MODEL_LIMIT:
        return set()
    out, pos = set(), 0
    while pos < prompt_len:
        if chunk(pos, 256, P) < 256:
            break
        c = next(prompt_len - pos, pos, max_chunk, tail_aware, P)
        if c <= 0 or c > prompt_len - pos:
            break
        pos += c
        if pos >= prompt_len:
            break
        if chunk(pos, 256, P) < 256:
            break
        out.add(pos)
    return out


def last_pass_end(target, base=0, remaining=None, max_chunk=4096, tail_aware=True, P=None):
    """The last pass end at or before `target` reading from `base`; `base` when the first pass reaches past it, None
    when the target is behind."""
    P = qk_product() if P is None else P
    if base < 0 or target < base:
        return None
    left = (target - base + MODEL_LIMIT) if remaining is None else remaining
    end = base
    while left > 0:
        c = next(left, end, max_chunk, tail_aware, P)
        if c <= 0 or end + c > target:
            break
        end += c
        left -= c
    return end


def envelope(path=None):
    """The envelope record (qk_product, bytes_per_pair, bytes_per_pass_token, max_rows_per_launch, ...); under
    SLOTSTREAM_ENVELOPE=bootstrap without a file: only the bootstrap product."""
    p = Path(path) if path else ENVELOPE_JSON
    if p.is_file():
        return json.loads(p.read_text())
    if os.environ.get("SLOTSTREAM_ENVELOPE") == "bootstrap":
        return {"qk_product": BOOTSTRAP_QK_PRODUCT}
    raise EnvelopeMissing(f"{p} is missing; run tools/prefill_envelope.py --write or set SLOTSTREAM_ENVELOPE=bootstrap")


def group_scopes(sizes, base=0, read_scope=SCOPE_CAP_256, checkpoints=()):
    """Cut the pass sizes of a prompt into read scopes -> list of lists of sizes (order kept, every pass in exactly one).

    A scope is >= 4 identical consecutive passes of 256, 512 or 1024 tokens, at most 8192 tokens for 256-token passes and
    4096 otherwise, at most `read_scope` tokens. The last pass is always alone (pool writes happen only on the prompt's
    final pass), and a scope ends at a requested checkpoint when one of its passes already ends there; no other pass
    shape changes."""
    cps = set(int(c) for c in checkpoints)
    out, i, pos, n = [], 0, base, len(sizes)
    while i < n:
        s = sizes[i]
        take = 1
        if s in SCOPE_SIZES and i < n - 1:
            run = 1
            while i + run < n - 1 and sizes[i + run] == s:
                run += 1
            cap = min(read_scope, SCOPE_CAP_256 if s == 256 else SCOPE_CAP_OTHER)
            m = min(run, cap // s)
            if m >= SCOPE_MIN_PASSES:
                take = builtins.next((j for j in range(1, m + 1) if pos + j * s in cps), m)
        out.append(list(sizes[i:i + take]))
        pos += sum(sizes[i:i + take])
        i += take
    return out
