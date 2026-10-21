"""Count-once accounting of process memory counters (plan Shared conventions, 2026-10-12).

A unified-memory allocation (pool chunks, staging buffers, ...) can show in several counters at once (VmRSS and
NVML). It is subtracted once from every counter it shows in. Pure integers, stdlib only.

  nonpool_bytes  VmRSS + NVML + pinned with every unified byte removed (budgets that exclude the pool)
  once_bytes     nonpool_bytes + every unified byte counted exactly once (physical envelopes)
"""
import json
from dataclasses import dataclass, field
from typing import Sequence

Flags = dict  # name -> (in_rss, in_nvml)


@dataclass(frozen=True)
class Sample:
    tag: str
    vmrss: int
    vmhwm: int
    nvml: int
    pinned: int = 0
    unified: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Peak:
    nonpool_bytes: int
    once_bytes: int
    hwm_bound_bytes: int
    unified_hwm: dict
    argmax_tag: str
    samples: int


def shows_in(before: Sample, after: Sample, name: str):
    grown = after.unified.get(name, 0) - before.unified.get(name, 0)
    if grown <= 0:
        raise ValueError(f"shows_in: allocation '{name}' did not grow ({grown} B)")
    return (after.vmrss - before.vmrss >= 0.9 * grown, after.nvml - before.nvml >= 0.9 * grown)


def _share(s: Sample, flags: Flags, which: int) -> int:
    return sum(int(b) * int(bool(flags[n][which])) for n, b in s.unified.items())


def nonpool_bytes(s: Sample, flags: Flags) -> int:
    return int(s.vmrss + s.nvml + s.pinned - _share(s, flags, 0) - _share(s, flags, 1))


def once_bytes(s: Sample, flags: Flags) -> int:
    total = sum(int(b) for b in s.unified.values())
    return int(nonpool_bytes(s, flags) + total)


def peak(samples: Sequence[Sample], flags: Flags) -> Peak:
    if not samples:
        raise ValueError("peak: no samples")
    names = {n for s in samples for n in s.unified}
    hwm = {n: max(int(s.unified.get(n, 0)) for s in samples) for n in names}
    for n in names:
        flags[n]                                                    # KeyError, never a silent 0
    rss_hw = sum(hwm[n] * int(bool(flags[n][0])) for n in names)
    bound = (max(s.vmhwm for s in samples) - rss_hw
             + max(s.nvml - _share(s, flags, 1) for s in samples) + max(s.pinned for s in samples))
    best_tag, best = "vmhwm", int(bound)
    for s in samples:
        v = nonpool_bytes(s, flags)
        if v > best:
            best, best_tag = v, s.tag
    return Peak(int(best), int(best + sum(hwm.values())), int(bound), hwm, best_tag, len(samples))


def sample_from_dict(d: dict) -> Sample:
    return Sample(str(d["tag"]), int(d["vmrss"]), int(d["vmhwm"]), int(d["nvml"]), int(d.get("pinned", 0)),
                  {k: int(v) for k, v in d.get("unified", {}).items()})


def flags_from_dict(d: dict) -> Flags:
    return {k: (bool(v[0]), bool(v[1])) for k, v in d.items()}


def peak_to_dict(p: Peak) -> dict:
    return {"nonpool_bytes": p.nonpool_bytes, "once_bytes": p.once_bytes, "hwm_bound_bytes": p.hwm_bound_bytes,
            "unified_hwm": dict(p.unified_hwm), "argmax_tag": p.argmax_tag, "samples": p.samples}


def from_jsonl(path):
    """Read the sampler JSONL: line 1 is `{"header": true, "flags": {name: [in_rss, in_nvml]}, ...}`, the rest one
    sample per line (`t_ns`, `vmrss`, `vmhwm`, `nvml`, `pinned`, `unified`; extra keys are ignored).
    Returns (samples, flags). A sample naming an allocation absent from the header flags is an error."""
    samples, flags = [], None
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("header"):
                flags = flags_from_dict(d["flags"])
                continue
            if flags is None:
                raise ValueError(f"{path}:{n}: sample before the header line")
            d.setdefault("tag", str(d.get("t_ns", n)))
            s = sample_from_dict(d)
            for name in s.unified:
                if name not in flags:
                    raise ValueError(f"{path}:{n}: allocation '{name}' has no flags in the header")
            samples.append(s)
    if flags is None:
        raise ValueError(f"{path}: no header line")
    return samples, flags
