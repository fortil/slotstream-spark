# SPDX-License-Identifier: MIT
# Port of Layers.swift and ExpertStore.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""Sweep group planner (prefill over experts that are not resident), pure numpy.

Ports the grouping of the Mac engine: rows are stably counting-sorted by bucket = (0 if resident else 512) + expert
(Layers.swift:1712-1720), then cut into groups of at most `group_size` experts (ExpertStore.swift:274-278) that never
mix resident and non-resident experts. Resident groups read the pool; non-resident groups go through the staging
buffers (slotio.Sweep), cycling over buffers 0..buffers-1.
"""
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

import numpy as np

EXPERTS = 512
GROUP_SIZE = 32
QUOTA_DIVISOR = 48   # ContextMemory.swift:363


@dataclass(frozen=True)
class Group:
    experts: np.ndarray   # int32, ascending expert ids of the group
    resident: bool
    row_lo: int           # rows [row_lo, row_hi) of SweepPlan.order belong to this group
    row_hi: int


@dataclass(frozen=True)
class SweepPlan:
    order: np.ndarray     # int64 [R]: row ids sorted by (resident first, expert id, row id)
    groups: list


def plan_sweep(expert_ids: np.ndarray, resident: np.ndarray, group_size: int = GROUP_SIZE) -> SweepPlan:
    ids = np.asarray(expert_ids)
    res = np.asarray(resident, dtype=bool)
    if ids.ndim != 1 or ids.dtype != np.int32:
        raise ValueError("expert_ids must be int32 [R]")
    if res.shape != (EXPERTS,):
        raise ValueError(f"resident must be bool [{EXPERTS}]")
    if group_size < 1:
        raise ValueError("group_size must be positive")
    if ids.size and (ids.min() < 0 or ids.max() >= EXPERTS):
        raise ValueError("expert id out of range")
    bucket = np.where(res[ids], 0, EXPERTS).astype(np.int64) + ids
    counts = np.bincount(bucket, minlength=2 * EXPERTS)
    order = np.argsort(bucket, kind="stable")          # stable counting sort: rows keep their order inside a bucket
    ends = np.cumsum(counts)
    starts = ends - counts
    groups = []
    for lo_b, hi_b, is_res in ((0, EXPERTS, True), (EXPERTS, 2 * EXPERTS, False)):
        present = np.nonzero(counts[lo_b:hi_b])[0]
        for i in range(0, present.size, group_size):
            chunk = present[i:i + group_size]
            row_lo = int(starts[lo_b + chunk[0]])
            row_hi = int(ends[lo_b + chunk[-1]])
            groups.append(Group(chunk.astype(np.int32), is_res, row_lo, row_hi))
    return SweepPlan(order, groups)


def admission_quota(slots: int) -> int:
    """Experts admitted to the pool after a sweep: min(512, max(1, slots // 48))."""
    return min(EXPERTS, max(1, int(slots) // QUOTA_DIVISOR))


def hottest(counts: np.ndarray, quota: int) -> np.ndarray:
    """The `quota` experts with the most rows (counts > 0), ties broken by lower id; int32, hottest first."""
    c = np.asarray(counts)
    ids = np.arange(c.size)
    order = np.lexsort((ids, -c))
    order = order[c[order] > 0][:max(0, int(quota))]
    return order.astype(np.int32)


class GroupIterator:
    """Yields (buf, group): buf cycles over 0..buffers-1 for the non-resident groups, None for resident ones.

    A staging buffer is reused only after the CUDA event of the kernel that last read it: the consumer stores that
    event in `events[buf]` after launching the group's kernel; before the buffer is handed out again the iterator calls
    `wait(event)` (default: `event.synchronize()`). At most `buffers` groups are alive in staging at any time; `INFLIGHT` of them ahead of the
    running one.
    """

    def __init__(self, plan: SweepPlan, wait: Optional[Callable] = None, buffers: int = 4):
        self.plan = plan
        self.buffers = buffers
        self.events = [None] * buffers
        self._wait = wait or (lambda ev: ev.synchronize())
        self.alive = [False] * buffers
        self.max_alive = 0

    def __iter__(self) -> Iterator:
        k = 0
        for g in self.plan.groups:
            if g.resident:
                yield None, g
                continue
            buf = k % self.buffers
            k += 1
            if self.events[buf] is not None:
                self._wait(self.events[buf])
                self.events[buf] = None
            self.alive[buf] = True
            self.max_alive = max(self.max_alive, sum(self.alive))
            yield buf, g
            # the consumer has launched the kernel and set events[buf]; the buffer stays alive until it is reused
