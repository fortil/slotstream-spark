"""Router readback signal: one tiny program at the end of every segment graph (plan section 3.3).

It copies the layer's top-10 expert ids and weights to the pinned readback slot of that layer and then publishes
`seq = base + layer` with a system-scope release exchange, so a host that observes the word with an acquire load
(slotio.wait_seq) also sees the ids. `base` is t * 49 for decode step t, kept in a device word that the first
graph of the step loads from pinned memory. Layer 48 has no ids (n = 0): it only publishes the logits.
"""
import torch
import triton
import triton.language as tl

SLOT = 32      # elements per layer slot: 3 rows x 10 experts at most


@triton.jit
def _signal(ids_ptr, w_ptr, slot_ids_ptr, slot_w_ptr, seq_ptr, ctrl_ptr, layer, n, SLOT: tl.constexpr):
    offs = tl.arange(0, SLOT)
    m = offs < n
    ids = tl.load(ids_ptr + offs, mask=m, other=0)
    w = tl.load(w_ptr + offs, mask=m, other=0.0)
    tl.store(slot_ids_ptr + layer * SLOT + offs, ids, mask=m)
    tl.store(slot_w_ptr + layer * SLOT + offs, w, mask=m)
    tl.debug_barrier()
    val = tl.load(ctrl_ptr + 1) + layer
    tl.atomic_xchg(seq_ptr + offs * 0, val + tl.zeros([SLOT], dtype=tl.int64), mask=offs == 0,
                   sem="release", scope="sys")


def signal(ids, w, slot_ids, slot_w, seq, ctrl, layer):
    """ids int32 [R, 10] and w fp32 [R, 10] (device, contiguous) or None; slot_ids/slot_w/seq are the pinned
    readback buffers (int32 [49*SLOT], float32 [49*SLOT], int64 [1]); ctrl int64 [2] on the device (pos, base)."""
    n = 0 if ids is None else ids.numel()
    if n > SLOT:
        raise ValueError(f"at most {SLOT} ids per layer slot")
    src = ids if ids is not None else slot_ids
    srcw = w if w is not None else slot_w
    _signal[(1,)](src, srcw, slot_ids, slot_w, seq, ctrl, layer, n, SLOT=SLOT, num_warps=1)
