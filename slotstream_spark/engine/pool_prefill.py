"""Pool pre-fill from the expert frequency table (step 77e, DIAG_F6 option P4).

Precondition: no GraphSet is attached (the pool's progress pointer is `seg.progress`). Call it before the graphs are
built, or after `GraphSet.detach()`.
"""
import json
import time
from pathlib import Path

import numpy as np

from slotstream_spark.memory.expert_freq import LAYERS, check_table, select

TABLE_PATH = Path(__file__).resolve().parents[1] / "config" / "expert_freq_gb10.json"


def load_table(path=None):
    d = json.loads(Path(path or TABLE_PATH).read_text())
    check_table(d)
    return np.asarray(d["counts"], dtype=np.int64)


def prefill_pool(seg, pool, counts, per_layer=None, batch=32):
    """Read the hottest `per_layer` experts of every layer into the pool; best effort, never raises on Exhausted.

    Every pin of a layer retires through the progress word (`seg._done`) after the layer, so a later `ensure` finds
    unpinned victims. Returns dict(records, layers, per_layer, batches, exhausted, seconds).
    """
    import slotio

    t0 = time.perf_counter()
    if per_layer is None:
        per_layer = int(pool.pool.slots) // LAYERS
    chosen = select(counts, per_layer)
    records = batches = exhausted = 0
    for layer in range(LAYERS):
        ids = chosen[layer]
        for i in range(0, ids.size, batch):
            seg.gen = max(seg.gen, int(seg.progress[0])) + 1
            batches += 1
            try:
                pool.ensure(layer, ids[i:i + batch], seg.gen)
            except slotio.Exhausted:
                exhausted += 1
                continue
            records += int(pool.last_timing["misses"])
        seg._done(seg.gen)
    return {"records": records, "layers": LAYERS, "per_layer": per_layer, "batches": batches,
            "exhausted": exhausted, "seconds": round(time.perf_counter() - t0, 3)}
