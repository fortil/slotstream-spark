# SPDX-License-Identifier: MIT
# Port of Governor.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""Memory governor policy as pure functions (Governor.swift:33-40, :242-280; rules 1-8 of section 3.7).

`decide(GovInputs, GovConfig)` returns one of Hold, ShrinkTo, GrowTo, Exit. `may_start` is rule 1. Nothing here
reads the clock, /proc or the network: the signal thread builds `GovInputs` and applies the decision. The cap is
`min(ceiling, MemAvailable + own footprint - need - margin)`, so the same (avail + own) always gives the same target.
"""
import fnmatch
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from slotstream_spark import config
from slotstream_spark.config import modes
from slotstream_spark.memory.ledger import Ledger

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
REPO = Path(__file__).resolve().parents[2]
A1_PATH = REPO / "db" / "records" / "measurements" / "a1.md"
MIB = 1 << 20
CHUNK_MIB = modes.CHUNK_BYTES // MIB

FAILED = "failed"          # class given to a failed GET /running: non-empty, kind unknown


@dataclass(frozen=True)
class Hold:
    pass


@dataclass(frozen=True)
class ShrinkTo:
    chunks: int
    reason: str
    cancel: bool = False
    drop_prefix: bool = False


@dataclass(frozen=True)
class GrowTo:
    chunks: int


@dataclass(frozen=True)
class Exit:
    reason: str


@dataclass(frozen=True)
class Resident:
    """One /running entry; `state` is None when the endpoint does not report it."""
    name: str
    state: Optional[str] = None


@dataclass(frozen=True)
class GovConfig:
    tick_s: int
    running_poll_ms: int
    floor_chunks: int
    margin_mib: int
    emergency_mib: int
    min_avail_mib: int
    grow_after_s: int
    grow_step_chunks: int
    stable_band_mib: int
    a_stable_s: int
    memfree_total_mib: int
    memfree_a_percent: int
    window_start_min: int
    may_start_end_min: int
    window_end_min: int
    d5_exit_after_s: int
    a_transient_mib: int
    a1_load_s: int
    memfree_cap: bool
    classes: Tuple[Tuple[str, Tuple[str, ...]], ...]
    default_class: str


def a1_load_seconds(path=A1_PATH, default=180):
    """Longest observed A load in seconds from a1.md ('Max load: N s'); the default when none was observed."""
    try:
        m = re.search(r"Max load[^:\n]*:\s*(\d+)\s*s\b", Path(path).read_text())
    except OSError:
        return default
    return int(m.group(1)) if m else default


MEMTOTAL_FALLBACK_MIB = 124609      # used only when /proc/meminfo is unreadable (a development machine without it)


def _memfree_total_mib(configured, meminfo="/proc/meminfo"):
    """`memfree_total_mib` of governor.toml; 0 means the host's MemTotal read at load."""
    if configured:
        return int(configured)
    try:
        with open(meminfo) as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return MEMTOTAL_FALLBACK_MIB


def load_config(governor_path=None, residents_path=None, platform_cfg=None, a1_path=A1_PATH):
    with open(governor_path or CONFIG_DIR / "governor.toml", "rb") as f:
        g = tomllib.load(f)
    with open(residents_path or CONFIG_DIR / "residents.toml", "rb") as f:
        ls = tomllib.load(f)
    local = CONFIG_DIR / "residents.local.toml"
    if residents_path is None and local.exists():           # untracked, per host: replaces classes name by name
        with open(local, "rb") as f:
            lc = tomllib.load(f)
        ls = {**ls, **{k: v for k, v in lc.items() if k != "classes"}, "classes": {**ls["classes"], **lc.get("classes", {})}}
    cfg = platform_cfg if platform_cfg is not None else config.load()
    classes = tuple((k, tuple(v)) for k, v in ls["classes"].items())
    return GovConfig(
        tick_s=g["tick_s"], running_poll_ms=g["running_poll_ms"], floor_chunks=g["floor_chunks"],
        margin_mib=g["margin_mib"], emergency_mib=g["emergency_mib"], min_avail_mib=g["min_avail_mib"],
        grow_after_s=g["grow_after_s"], grow_step_chunks=g["grow_step_chunks"],
        stable_band_mib=g["stable_band_mib"], a_stable_s=g["a_stable_s"],
        memfree_total_mib=_memfree_total_mib(g["memfree_total_mib"]), memfree_a_percent=g["memfree_a_percent"],
        window_start_min=g["window_start_min"], may_start_end_min=g["may_start_end_min"],
        window_end_min=g["window_end_min"], d5_exit_after_s=g["d5"]["exit_after_s"],
        a_transient_mib=int(config.get("memory.a_transient_mib", cfg)),
        a1_load_s=a1_load_seconds(a1_path, g["a1_load_default_s"]),
        memfree_cap=(config.get("memory.d1_variant", cfg) == "D1prime"),
        classes=classes, default_class=ls["default_class"],
    )


def classify(name, cfg):
    for cls, patterns in cfg.classes:
        if any(fnmatch.fnmatchcase(name, p) for p in patterns):
            return cls
    return cfg.default_class


@dataclass(frozen=True)
class GovInputs:
    mode: str                                   # "D1" | "D2" | "D5"
    now_min: int                                # minutes since local midnight
    running: Optional[Tuple[Resident, ...]]     # None = the GET failed
    chunks: int                                 # current pool chunks
    footprint_mib: int                          # own footprint (rule 6)
    mem_available_mib: int
    mem_free_mib: int
    calm_s: int = 0                             # seconds since the last shrink or pressure
    busy_at_floor_s: int = 0                    # seconds /running has been non-empty with the pool at the floor
    a_stable_s: int = 0                         # seconds MemAvailable stayed inside the stable band
    a_since_s: int = 0                          # seconds since A first appeared in /running
    prev_chunks: Optional[int] = None           # size before the shrink for A, to regrow to
    fixed_bytes: Optional[int] = None           # the ledger's fixed bytes (default: Ledger(mode))
    no_windows: bool = False                    # pause mode ($R/state/PAUSED NO_WINDOWS=1): the 04:00/23:59 window does not apply


def _fixed(g):
    return g.fixed_bytes if g.fixed_bytes is not None else Ledger(g.mode).fixed_bytes()


def floor_mib(g, cfg):
    return -(-(_fixed(g) + cfg.floor_chunks * modes.CHUNK_BYTES) // MIB)


def cap_mib(g, cfg, empty=True):
    """Dynamic cap: min(ceiling, MemAvailable + own - need - margin); D1' also bounds it by MemFree.

    With something in /running the need is already inside MemAvailable (or the engine is going to the floor), so
    only the empty case charges the mode's need.
    """
    m = modes.get(g.mode)
    need = m.need_mib if empty else 0
    cap = min(m.ceiling_mib, g.mem_available_mib + g.footprint_mib - need - cfg.margin_mib)
    if empty and cfg.memfree_cap:
        a_mib = -(-cfg.memfree_total_mib * cfg.memfree_a_percent // 100)
        cap = min(cap, g.mem_free_mib + g.footprint_mib - a_mib - cfg.margin_mib)
    return cap


def target_chunks(g, cfg, cap):
    """Whole chunks under `cap` after the fixed bytes, at most the mode's ceiling."""
    t = (cap * MIB - _fixed(g)) // modes.CHUNK_BYTES
    return min(modes.get(g.mode).ceiling_chunks, t)


def _entries(g, cfg):
    if g.running is None:
        return [FAILED]
    return [classify(r.name, cfg) for r in g.running]


def a_ready(g, cfg):
    """A is ready when /running says so and MemAvailable was stable; without a state field, after load time + 60 s."""
    if g.a_stable_s < cfg.a_stable_s:
        return False
    states = [r.state for r in (g.running or ()) if classify(r.name, cfg) == "A"]
    if any(s == "ready" for s in states):
        return True
    return all(s is None for s in states) and g.a_since_s >= cfg.a1_load_s + cfg.a_stable_s


def decide(g, cfg):
    if not g.no_windows and (g.now_min < cfg.window_start_min or g.now_min >= cfg.window_end_min):
        return Exit("outside the 04:00-23:59 window")
    floor = cfg.floor_chunks
    entries = _entries(g, cfg)

    if entries:
        if g.mode == "D5":
            if g.chunks > floor:
                return ShrinkTo(floor, "running not empty", cancel=True, drop_prefix=True)
            return Exit("running still not empty at the floor") if g.busy_at_floor_s >= cfg.d5_exit_after_s else Hold()
        # D1 / D2
        if any(c not in ("A", FAILED) for c in entries):
            return Exit("resident other than A loading")
        if FAILED in entries or not a_ready(g, cfg):
            at_floor = g.mem_available_mib + (g.chunks - floor) * CHUNK_MIB
            if "A" in entries and at_floor - cfg.a_transient_mib < cfg.min_avail_mib:
                return Exit("A transient leaves less than the minimum even at the floor")
            if g.chunks > floor:
                return ShrinkTo(floor, "A starting", cancel=True, drop_prefix=True)
            return Hold()
        # A ready: slow regrow to the size it had
        cap = cap_mib(g, cfg, empty=False)
        goal = min(target_chunks(g, cfg, cap), g.prev_chunks if g.prev_chunks is not None else g.chunks)
        return _grow(g, cfg, cap, goal)

    cap = cap_mib(g, cfg)
    if cap < floor_mib(g, cfg):
        return Exit("cap below the floor")
    if g.mem_available_mib < cfg.emergency_mib:
        if g.chunks <= floor:
            return Exit("MemAvailable below the emergency line at the floor")
        return ShrinkTo(g.chunks - 1, "emergency", cancel=True, drop_prefix=True)
    tgt = max(floor, target_chunks(g, cfg, cap))
    if tgt < g.chunks:
        return ShrinkTo(tgt, "cap", drop_prefix=True)
    return _grow(g, cfg, cap, tgt)


def apply_shrink(decision, prefix_cache, resize):
    """Apply a ShrinkTo: the prefix cache is dropped BEFORE the pool shrinks when the decision says so (Governor.swift:450,
    :501), so the retained states never outlive the memory they were budgeted from. `resize(chunks)` does the shrink.
    Returns True if a cache was dropped."""
    dropped = False
    if getattr(decision, "drop_prefix", False) and prefix_cache is not None:
        prefix_cache.drop()
        dropped = True
    resize(decision.chunks)
    return dropped


def _grow(g, cfg, cap, goal):
    if g.chunks >= goal or g.calm_s < cfg.grow_after_s:
        return Hold()
    if cap - g.footprint_mib < CHUNK_MIB * cfg.grow_step_chunks + cfg.margin_mib:
        return Hold()
    return GrowTo(min(goal, g.chunks + cfg.grow_step_chunks))


def may_start(now_min, running, cap_mib_, floor_mib_, cfg, no_windows=False):
    """Rule 1: not in 00:00-04:00, not from 23:45, not with /running non-empty (or unreadable), not below the floor.
    `no_windows` (pause mode) lifts only the two time checks."""
    if not no_windows and now_min < cfg.window_start_min:
        return False, "before 04:00"
    if not no_windows and now_min >= cfg.may_start_end_min:
        return False, "after 23:45"
    if running is None:
        return False, "running unreadable"
    if len(running) > 0:
        return False, "running not empty"
    if cap_mib_ < floor_mib_:
        return False, "cap below the floor"
    return True, "ok"
