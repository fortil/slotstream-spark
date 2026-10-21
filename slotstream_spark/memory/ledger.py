# SPDX-License-Identifier: MIT
# Port of ContextMemory.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""Integer-byte ledger of the engine's memory (ContextMemoryLedger pattern, ContextMemory.swift:109-144).

Every term is an integer number of bytes. `fixed_bytes()` is everything except the expert pool and does not depend
on the number of chunks; `expected_peak_bytes(chunks)` adds the pool (chunks * 2^30) and is the physical envelope,
each byte counted once, that a measured `footprint.peak(...).once_bytes` is compared with. The runtime term is the
M0' measurement from platform.toml (`measured.runtime_footprint_mib`) unless the caller passes `measured`; the
design's 1,750,000,000 B is not a default.
"""
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

# mode -> (prefix cache bytes, pass tokens at the mode's ceiling, MTP bytes)
_MODE_TERMS = {
    "D1": (500_000_000, 1024, 0),
    "D2": (1_000_000_000, 2048, MTP),
    "D5": (1_000_000_000, 4096, MTP),
}


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
        return t

    def fixed_bytes(self):
        return sum(self.terms().values())

    def pool_bytes(self, chunks):
        return int(chunks) * CHUNK_BYTES

    def expected_peak_bytes(self, chunks):
        return self.fixed_bytes() + self.pool_bytes(chunks)
