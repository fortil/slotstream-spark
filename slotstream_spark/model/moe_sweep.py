"""MoE of one layer over a whole scope of prompt rows, swept expert group by expert group (plan step 70).

Every (token, rank) pair is a row. `plan_sweep` sorts the rows by (resident first, expert id) and cuts groups of at most
32 experts that never mix resident and non-resident ones. A resident group takes its record addresses from the pool slots
(`resident_ptr_nopin_touch`: pinned for the layer's generation, no CLOCK reference bit); a non-resident group is read by
slotio's `Sweep` into one of the four staging buffers (safe path: pread O_DIRECT into the lane scratch, CPU memcpy into
staging), two groups in flight, while the GPU computes the previous group. `fused_moe_affine` runs per group on the rows of the group's experts
(split so that no launch has more than `max_rows_per_launch` rows) and writes fp32 rows; `combine_rank_order` sums each
token's ten rows in router rank order. A row's output depends on that row and that expert's record bytes only, so the
result does not depend on residency, group composition, the scope or the buffer that held the record.

On the prompt's final pass only (`admit=True`) the `admission_quota` hottest non-resident experts of the layer are
published into the pool: each one reserves a victim slot when its group is issued, slotio's lane thread copies the record
into that slot as the read's second destination, and `wait` publishes it.
"""
import numpy as np
import torch
import torch.nn.functional as F

from slotstream_spark.engine.sweep import admission_quota, hottest, plan_sweep
from slotstream_spark.kernels import fused_moe_affine as fm
from slotstream_spark.model.layers import lin

EXPERTS = 512
STAGING_GROUPS = 4     # staging buffers of 32 records each
INFLIGHT = 2           # groups issued ahead of the one the GPU runs
WAIT_MS = 120_000


def make_sweep(pool):
    """slotio.Staging (4 x 32 records in the platform pool memory type) and the Sweep reader bound to `pool`."""
    import slotio
    staging = slotio.Staging(pool.mem_type, groups=STAGING_GROUPS)
    return staging, slotio.Sweep(pool._ensurer, pool.reader, pool.file, staging)


def shared_expert(seg, L, mixed):
    """Shared expert of layer L over `mixed` [T, 2560] (same arithmetic as Segments.seg_moe)."""
    W = seg.W
    p = f"layers.{L}.mlp."
    sh = lin(F.silu(lin(mixed, W(p + "shared_expert.gate_proj.weight"))) *
             lin(mixed, W(p + "shared_expert.up_proj.weight")), W(p + "shared_expert.down_proj.weight"))
    return torch.sigmoid(lin(mixed, W(p + "shared_expert_gate.weight"))) * sh


class SweepStats:
    def __init__(self):
        self.bytes_requested = 0      # bytes read from experts.bin into staging
        self.records_read = 0
        self.admitted = 0
        self.admit_refused = 0
        self.resident_rows = 0
        self.staged_rows = 0
        self.launches = 0
        self.max_launch_rows = 0
        self.groups = 0

    def as_dict(self):
        return dict(self.__dict__)


class SweepMoE:
    def __init__(self, seg, pool, *, max_rows_per_launch, staging=None, sweep=None, wait_ms=WAIT_MS):
        self.seg, self.pool = seg, pool
        self.max_rows = int(max_rows_per_launch)
        if staging is None or sweep is None:
            staging, sweep = make_sweep(pool)
        self.staging, self.sweep = staging, sweep
        self.wait_ms = wait_ms
        self.stats = SweepStats()
        self.last_plan = None

    def resident_mask(self, L):
        m = self.pool.pool.map_numpy()
        return np.asarray(m[L * EXPERTS:(L + 1) * EXPERTS] >= 0)

    def slots(self):
        return int(self.pool.pool.stats()["slots"])

    def layer(self, L, mixed, ids_host, w, *, admit=False):
        """mixed bf16 [T,2560] · ids_host int32 [T,10] (router ids, on the host) · w fp32 [T,10] (device) ->
        routed fp32 [T,2560] on the device, the sum of the ten weighted expert outputs per token in rank order."""
        seg, sweep, stats = self.seg, self.sweep, self.stats
        T, K = ids_host.shape
        dev = mixed.device
        flat = np.ascontiguousarray(ids_host.reshape(-1), dtype=np.int32)
        R = flat.size
        resident = self.resident_mask(L)
        plan = plan_sweep(flat, resident)
        self.last_plan = plan
        row_expert = torch.as_tensor(flat, device=dev)
        row_token = torch.div(torch.arange(R, device=dev), K, rounding_mode="floor").to(torch.int32)
        row_weight = w.reshape(-1).to(torch.float32).contiguous()
        out_rows = torch.zeros(R, fm.HID, dtype=torch.float32, device=dev)
        mixed = mixed.contiguous()
        seg.gen = max(seg.gen, int(seg.progress[0])) + 1       # past every retired generation (Model.reset restarts gen)
        gen = seg.gen
        sweep.set_generation(gen)
        admit_set = set()
        if admit:
            cnt = np.bincount(flat, minlength=EXPERTS) * (~resident)
            admit_set = set(int(e) for e in hottest(cnt, admission_quota(self.slots())))
        groups = plan.groups
        nonres = [i for i, g in enumerate(groups) if not g.resident]
        nb = STAGING_GROUPS
        buf_of = {gi: j % nb for j, gi in enumerate(nonres)}
        ahead_of = {gi: nonres[j + INFLIGHT] for j, gi in enumerate(nonres[:-INFLIGHT])}
        events, tickets = [None] * nb, {}

        def issue(gi):
            buf = buf_of[gi]
            if events[buf] is not None:                       # the kernel that last read this buffer must be done
                events[buf].synchronize()
                events[buf] = None
            g = groups[gi]
            adm = [int(e) for e in g.experts if int(e) in admit_set]
            adm = np.array(adm, dtype=np.int32) if adm else None
            tickets[gi] = sweep.read_group(L, np.ascontiguousarray(g.experts, dtype=np.int32), buf, adm)
            stats.bytes_requested += int(g.experts.size) * fm.RECORD_BYTES
            stats.records_read += int(g.experts.size)

        try:
            # Every resident group's records are located (and pinned for this generation) before the first read is
            # issued, so nothing issued later can take a resident expert of this layer as a victim.
            res_ptrs = {}
            for gi, g in enumerate(groups):
                if g.resident:
                    ptrs = []
                    for e in g.experts:
                        p = sweep.resident_ptr_nopin_touch(L, int(e))
                        if p is None:
                            raise RuntimeError(f"layer {L} expert {int(e)} left the pool during its own sweep")
                        ptrs.append(int(p))
                    res_ptrs[gi] = ptrs
            for gi in nonres[:INFLIGHT]:
                issue(gi)
            for gi, g in enumerate(groups):
                stats.groups += 1
                if g.resident:
                    ptrs = res_ptrs.pop(gi)
                    stats.resident_rows += g.row_hi - g.row_lo
                else:
                    t = tickets.pop(gi)
                    if not sweep.wait(t, self.wait_ms):
                        raise TimeoutError(f"sweep read of layer {L} timed out after {self.wait_ms} ms")
                    stats.admitted += t.admitted
                    stats.admit_refused += t.admit_refused
                    ptrs = [int(sweep.record_dev(t, i)) for i in range(g.experts.size)]
                    stats.staged_rows += g.row_hi - g.row_lo
                self._group(mixed, row_token, row_weight, row_expert, out_rows, plan.order[g.row_lo:g.row_hi],
                            g.experts, ptrs)
                if not g.resident:
                    ev = torch.cuda.Event()
                    ev.record()
                    events[buf_of[gi]] = ev
                    if gi in ahead_of:
                        issue(ahead_of[gi])                   # keep INFLIGHT groups ahead of the GPU
            torch.cuda.synchronize()
        finally:
            seg._done(gen)
        return fm.combine_rank_order(out_rows, T, K)

    def _group(self, x, row_token, row_weight, row_expert, out_rows, rows, experts, ptrs):
        dev = x.device
        emap_h = np.full(EXPERTS, -1, dtype=np.int32)
        emap_h[experts] = np.arange(experts.size, dtype=np.int32)
        emap = torch.as_tensor(emap_h, device=dev)
        gptr = torch.tensor(ptrs, dtype=torch.int64, device=dev)
        for a in range(0, rows.size, self.max_rows):
            idx = torch.as_tensor(np.ascontiguousarray(rows[a:a + self.max_rows]), device=dev)
            tmp = torch.empty(idx.numel(), fm.HID, dtype=torch.float32, device=dev)
            fm.fused_moe_affine(x, row_token[idx].contiguous(), row_weight[idx].contiguous(), emap, gptr, tmp,
                                row_expert=row_expert[idx].contiguous())
            out_rows.index_copy_(0, idx, tmp)
            self.stats.launches += 1
            self.stats.max_launch_rows = max(self.stats.max_launch_rows, int(idx.numel()))
