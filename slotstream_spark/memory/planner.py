"""Pool size and pass size from the ledger: whole 1 GiB chunks, a pass that costs at most a quarter of the pool.

`plan()` is pure. The estimate of decode speed is the only float and is None when it cannot be made from
measurements (no h-curve, a missing timing, or a slot count outside the curve).
"""
from dataclasses import dataclass
from typing import Optional

from slotstream_spark.config import modes
from slotstream_spark.memory.ledger import CHUNK_BYTES, PASS_PER_TOKEN, STAGING_GROUPS, Ledger

LADDER = (256, 512, 1024, 2048, 4096)
EXPERT_BYTES_PER_TOKEN = 1.327e9
NVME_BPS = 11.1e9
LAYERS = 48
PASS_POOL_FRACTION = 4                     # a pass may cost at most pool_budget / 4


@dataclass(frozen=True)
class Plan:
    chunks: int
    slots: int
    pass_tokens: int
    window: int
    mtp_reserved: int
    mtp_enabled: bool
    lookahead: bool
    expected_peak: int
    est_tok_s: Optional[float]


def _h_at(h_curve, slots):
    """Linear interpolation of a hit-rate curve [(slots, h), ...]; None outside its range."""
    pts = sorted((int(s), float(h)) for s, h in h_curve)
    if not pts or slots < pts[0][0] or slots > pts[-1][0]:
        return None
    for (s0, h0), (s1, h1) in zip(pts, pts[1:]):
        if s0 <= slots <= s1:
            return h0 if s1 == s0 else h0 + (h1 - h0) * (slots - s0) / (s1 - s0)
    return pts[-1][1]


def estimate_tok_s(h_curve, slots, t_comp, tau, t_memcpy):
    """t_token = t_comp + 48 * tau + t_io; t_io = (1 - h) * 1.327e9 / 11.1e9 + 48 * (1 - h^10) * t_memcpy."""
    if not h_curve or t_comp is None or tau is None or t_memcpy is None:
        return None
    h = _h_at(h_curve, slots)
    if h is None:
        return None
    t_io = (1.0 - h) * EXPERT_BYTES_PER_TOKEN / NVME_BPS + LAYERS * (1.0 - h ** 10) * t_memcpy
    return 1.0 / (t_comp + LAYERS * tau + t_io)


def plan(mode, cap_bytes, h_curve=None, t_comp=None, tau=None, t_memcpy=None, window=32768, measured=None,
         staging_groups=STAGING_GROUPS, mtp=None):
    """`mtp`: None keeps the design ledger (the head is reserved in D2/D5 and the plan reports it disabled); "off" charges
    nothing for MTP; "resident" charges the head (1,470,946,816 B) and the verify record ring, reports it enabled, and raises
    ValueError when the pool would not reach the floor of FLOOR_CHUNKS chunks (MTP resident is refused below the floor)."""
    if mtp not in (None, "off", "resident"):
        raise ValueError(f"mtp must be None, 'off' or 'resident', got {mtp!r}")
    m = modes.get(mode)
    ledger = Ledger(mode, measured, window, staging_groups, None if mtp is None else mtp == "resident")
    cap_bytes = int(cap_bytes)
    chunks = max(0, min(m.ceiling_chunks, (cap_bytes - ledger.fixed_bytes()) // CHUNK_BYTES))
    if mtp == "resident" and chunks < modes.FLOOR_CHUNKS:
        raise ValueError(f"MTP resident is refused: {chunks} pool chunks fit under the cap with the head "
                         f"({ledger.mtp + ledger.mtp_record:,} B), the floor is {modes.FLOOR_CHUNKS}")
    budget = chunks * CHUNK_BYTES
    fits = [n for n in LADDER if PASS_PER_TOKEN * n <= budget // PASS_POOL_FRACTION]
    pass_tokens = min(ledger.pass_tokens, max(fits) if fits else LADDER[0])
    slots = chunks * modes.SLOTS_PER_CHUNK
    return Plan(chunks, slots, pass_tokens, int(window), ledger.mtp + ledger.mtp_record, mtp == "resident", False,
                ledger.expected_peak_bytes(chunks), estimate_tok_s(h_curve, slots, t_comp, tau, t_memcpy))
