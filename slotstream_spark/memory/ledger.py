# SPDX-License-Identifier: MIT
# Port of ContextMemory.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""Integer-byte ledger of the engine's memory (ContextMemoryLedger pattern, ContextMemory.swift:109-144).

Every term is an integer number of bytes. `fixed_bytes()` is everything except the expert pool and does not depend
on the number of chunks; `expected_peak_bytes(chunks)` adds the pool (chunks * 2^30) and is the physical envelope,
each byte counted once, that a measured `footprint.peak(...).once_bytes` is compared with. The runtime term is the
M0' measurement from platform.toml (`measured.runtime_footprint_mib`) unless the caller passes `measured`; the
design's 1,750,000,000 B is not a default.
"""
import os

from slotstream_spark import config
from slotstream_spark.config import modes

TRUNK = 2_566_424_320
KV_PER_TOKEN = 27_648
GDN_STATE = 115_642_368
PASS_PER_TOKEN = 1_300_000                  # PlannerCostModel.swift:9
READ_SCOPE = 234_881_024
STAGING_GROUPS = 4
STAGING_GROUP_BYTES = 32 * 2_764_800        # 32 records of 2,764,800 B = 88,473,600
SCRATCH_ROWS = 203_052_800                  # the step-51 line (380,000,000) minus the 2x32 staging it covered
SIDECAR = 37_540_708
MTP = 1_470_946_816
# step 76, verify record ring: 3 x (36 GDN states 113,246,208 B + conv 36 x 3 x 10240 x 2 B + PLE conv 9 x 10240 x 2 B)
MTP_RECORD = 3 * (113_246_208 + 36 * 3 * 10240 * 2 + 9 * 10240 * 2)
MARGIN = 1_000_000_000
CHUNK_BYTES = modes.CHUNK_BYTES
DEFAULT_WINDOW = 32768                      # the served context window unless SLOTSTREAM_CONTEXT_WINDOW or --context-window says otherwise
MIN_WINDOW = 8192
MAX_WINDOW = 262_144                        # the model's limit (prefill_schedule.MODEL_LIMIT); measured up to 131072
ENV_WINDOW = "SLOTSTREAM_CONTEXT_WINDOW"
PREFIX_ENTRIES = 4                          # engine.prefix_cache.MAX_ENTRIES
PREFIX_ENTRY_STATE = 113_246_208            # engine.prefix_cache.FIXED_BYTES_PER_ENTRY: the GDN state one retained entry owns

# mode -> (prefix cache bytes, pass tokens at the mode's ceiling, MTP bytes)
_MODE_TERMS = {
    "D1": (500_000_000, 1024, 0),
    "D2": (1_000_000_000, 2048, MTP),
    "D5": (1_000_000_000, 4096, MTP),
}


def context_window(flag=None, env=None):
    """The context window: `flag` (a --context-window value) over SLOTSTREAM_CONTEXT_WINDOW over DEFAULT_WINDOW.

    ValueError for a value that is not an integer, outside MIN_WINDOW..MAX_WINDOW or not a multiple of 64 (serve rounds
    the engine context up to 64, so a multiple keeps the admission window, the plan and the KV cache equal)."""
    env = os.environ if env is None else env
    raw = flag if flag is not None else env.get(ENV_WINDOW)
    if raw is None or raw == "":
        return DEFAULT_WINDOW
    try:
        w = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"context window {raw!r} is not an integer") from None
    if w < MIN_WINDOW or w > MAX_WINDOW:
        raise ValueError(f"context window {w} is outside {MIN_WINDOW}..{MAX_WINDOW}")
    if w % 64:
        raise ValueError(f"context window {w} is not a multiple of 64")
    return w


def runtime_bytes(measured=None):
    """Runtime footprint: `measured["runtime_bytes"]` when given, else platform.toml `measured.runtime_footprint_mib`."""
    if measured and "runtime_bytes" in measured:
        return int(measured["runtime_bytes"])
    mib = config.get("measured.runtime_footprint_mib")
    if not config.decided("measured.runtime_footprint_mib"):
        raise ValueError("platform.toml measured.runtime_footprint_mib is not measured (run M0', step 48)")
    return int(mib) << 20


class Ledger:
    def __init__(self, mode, measured=None, window=32768, staging_groups=STAGING_GROUPS, mtp_resident=None):
        """`mtp_resident`: None keeps the per-mode design term (MTP in D2/D5); True charges the head and the verify record
        ring, False charges neither (step 76: only a resident MTP costs memory)."""
        if mode not in _MODE_TERMS:
            raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(_MODE_TERMS)}")
        self.mode = mode
        self.window = int(window)
        self.prefix, self.pass_tokens, self.mtp = _MODE_TERMS[mode]
        self.mtp_record = 0
        if mtp_resident is not None:
            self.mtp = MTP if mtp_resident else 0
            self.mtp_record = MTP_RECORD if mtp_resident else 0
        self.runtime = runtime_bytes(measured)
        self.trunk = TRUNK
        self.kv = KV_PER_TOKEN * self.window
        # Above the default window the prefix cache is sized in tokens of the window: the prefix term holds one window of
        # retained snapshots plus their GDN states, and `snapshot` one more window of KV for the clone that exists while a
        # store decides. At the default window both stay as designed (the fixed per-mode prefix bytes, no snapshot term).
        self.prefix_scaled = self.window > DEFAULT_WINDOW
        self.snapshot = 0
        if self.prefix_scaled:
            self.prefix = KV_PER_TOKEN * self.window + PREFIX_ENTRIES * PREFIX_ENTRY_STATE
            self.snapshot = KV_PER_TOKEN * self.window
        self.gdn = GDN_STATE
        self.pass_bytes = PASS_PER_TOKEN * self.pass_tokens
        self.read_scope = READ_SCOPE
        self.staging = SCRATCH_ROWS + int(staging_groups) * STAGING_GROUP_BYTES
        self.sidecar = SIDECAR
        self.margin = MARGIN

    def terms(self):
        t = {"trunk": self.trunk, "runtime": self.runtime, "kv": self.kv, "gdn": self.gdn, "prefix": self.prefix,
                "pass": self.pass_bytes, "read_scope": self.read_scope, "staging_scratch_rows": self.staging,
                "sidecar": self.sidecar, "mtp": self.mtp, "margin": self.margin}
        if self.mtp_record:
            t["mtp_record"] = self.mtp_record
        if self.snapshot:
            t["snapshot"] = self.snapshot
        return t

    def fixed_bytes(self):
        return sum(self.terms().values())

    def prefix_cache_tokens(self):
        """Token budget of the in-memory prefix cache: the window itself when the prefix term scales with it, else the
        mode's fixed prefix bytes in KV_PER_TOKEN units (36,168 in D5)."""
        return self.window if self.prefix_scaled else self.prefix // KV_PER_TOKEN

    def pool_bytes(self, chunks):
        return int(chunks) * CHUNK_BYTES

    def expected_peak_bytes(self, chunks):
        return self.fixed_bytes() + self.pool_bytes(chunks)
