"""H1: layer 0 of the engine against the oracle, pool sizes, 1,000 reads, coherence and peak.

Writes $CAP/result.json (judged by tools/gates/records.py h1). Run through
`tools/spark/run.sh --heavy h1 4096 3600 -- tools/spark/py.sh -m pytest -q -m gate tests/gates/test_layer0.py`.
"""
import gc
import hashlib
import json
import mmap
import os
import random
import time
from pathlib import Path

import numpy as np
import pytest

import contract as ct
from slotstream_spark.memory import footprint

pytestmark = pytest.mark.gate

POOLS = (13, 88, 181, 512)
FIRST_TOKENS = 64
ENSURES = 1000
READ_LIMIT_S = 2.0
LAYERS = 48
EXPERTS = 512
RES = {"samples": []}
_LIVE = []                                        # ExpertPool objects alive, for the live pool bytes


def _sample(tag):
    import torch
    torch.cuda.synchronize()
    RES["samples"].append({"tag": tag, "vmrss": vm_kb("VmRSS"), "vmhwm": vm_kb("VmHWM"),
                           "nvml": nvml_used(os.getpid())[0], "pinned": 0,
                           "unified": {"pool": sum(p.owned_bytes() for p in _LIVE)}})


def vm_kb(field):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    return 0


def nvml_used(pid):
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
            if p.pid == pid and p.usedGpuMemory is not None:
                return int(p.usedGpuMemory), "pynvml"
    except Exception:
        pass
    return 0, "unavailable"


class Sink:
    """Model._chunk sink: layer-0 hyper-width output and routing of every chunk."""

    def __init__(self):
        self.out, self.ids = [], []

    def h_in(self, hyp):
        pass

    def layer(self, L, hyp, ids, w):
        self.out.append(hyp.float().cpu().numpy())
        self.ids.append(np.asarray(ids))

    def ngram(self, gids):
        pass

    def result(self):
        return np.concatenate(self.out, 0), np.concatenate(self.ids, 0)


@pytest.fixture(scope="module")
def engine(model_dir, packed_dir):
    import slotio
    import torch
    from slotstream_spark.model.config import TextConfig
    from slotstream_spark.model.forward import Model
    from slotstream_spark.model.weights import Embedder, NgramSource, accessor, load_trunk
    from slotstream_spark.runtime.pool import ExpertPool

    cfg = TextConfig.from_checkpoint(model_dir)
    weights = load_trunk(packed_dir / "trunk.safetensors", layers=(0, 1), device="cuda")   # layer 1 holds the n-gram tables
    reader = slotio.Reader()
    embed = Embedder(reader, packed_dir / "trunk.safetensors", device="cuda")
    ngram = NgramSource(reader, packed_dir, weights, cfg["eos_token_id"], device="cuda")
    torch.cuda.synchronize()
    _sample("base")
    pool = ExpertPool(packed_dir, POOLS[-1], reader=reader)
    _LIVE.append(pool)
    model = Model(cfg, accessor(weights), embed, pool, ngram, device="cuda", gdn="fused", max_ctx=128)
    model.NL = 1                                   # layer 0 only
    _sample("fixture")
    smp = [footprint.sample_from_dict(x) for x in RES["samples"]]
    RES["flags"] = {"pool": list(footprint.shows_in(smp[0], smp[1], "pool"))}
    assert tuple(RES["flags"]["pool"]) != (False, False), "the pool shows in neither counter: broken counter read"
    yield {"model": model, "pool": pool, "reader": reader, "cfg": cfg}
    out = os.environ.get("CAP")
    if out:
        os.makedirs(out, exist_ok=True)
        Path(out, "result.json").write_text(ct.dumps(RES))


def prefill_layer0(model, toks):
    model.reset()
    sink = Sink()
    with __import__("torch").no_grad():
        model._chunk(toks, sink)
    return sink.result()


def decode_layer0(model, toks):
    model.reset()
    sink = Sink()
    for t in toks:
        model._chunk([t], sink)
    return sink.result()


def test_a_layer0_vs_oracle(engine):
    import torch
    from calibrate_contract import rel_l2
    c = ct.load()
    model = engine["model"]
    g = __import__("make_goldens").load_golden("dev0")
    toks = [int(t) for t in g["tokens"]][:FIRST_TOKENS]
    T = len(toks)
    gold = g["layer_00_out"][:T]
    gold_ids = g["router_00_ids"][:T]
    out, ids = prefill_layer0(model, toks)
    v, lim, sep, ok = ct.layer_check(0, out, gold, c)
    rv, rlim, rsep, rok = ct.routing_check(0, ids, gold_ids, c)
    model.reset()
    gids = model.seg.ngram.ids([], toks)
    ngram_ok = ct.ngram_check(np.asarray(gids), g["ngram_ids"][:T])
    dout, _ = decode_layer0(model, toks)
    dv = rel_l2(dout, gold)
    print(f"layer 0: rel_l2 {v:.5g} (limit {lim:.5g}, spread {c.layer['spread'][0]:.5g}, control "
          f"{c.layer['control'][0]:.5g}, separable {sep}); decode path {dv:.5g}; routing {rv:.5g} "
          f"(limit {rlim:.5g}, separable {rsep}); ngram exact {ngram_ok}")
    RES["a"] = {"positions": T, "rel_l2": v, "limit": lim, "separable": sep, "ok": bool(ok),
                "spread": c.layer["spread"][0], "control": c.layer["control"][0], "decode_rel_l2": dv,
                "routing_disagreement": rv, "routing_limit": rlim, "routing_separable": rsep,
                "routing_ok": bool(rok), "ngram_ok": ngram_ok}
    assert sep, f"layer 0 is not separable in the numeric contract (control {c.layer['control'][0]}, spread {c.layer['spread'][0]}): contract must be recalibrated"
    assert v <= lim, f"layer 0 rel_l2 {v:.5g} > calibrated limits[0] {lim:.5g}"
    assert ngram_ok, "n-gram ids differ from the oracle"
    torch.cuda.synchronize()
    _sample("a")


def test_b_pools_bitwise_equal(engine):
    import torch
    model, pool = engine["model"], engine["pool"]
    toks = [int(t) for t in __import__("make_goldens").load_golden("dev0")["tokens"]][:FIRST_TOKENS]
    digests, outs = {}, {}
    for n in POOLS:
        gc.collect()
        pool.resize_slots(n)
        out, _ = decode_layer0(model, toks)             # one token per step: 10 unique experts fit a pool of 13
        digests[n] = hashlib.sha256(np.ascontiguousarray(out).tobytes()).hexdigest()
        outs[n] = out
        _sample(f"b-{n}")
    same = all(np.array_equal(outs[n], outs[POOLS[-1]]) for n in POOLS)
    print("pool digests:", json.dumps(digests))
    RES["b"] = {"pools": list(POOLS), "bitwise_equal": bool(same), "sha256": digests[POOLS[-1]],
                "digests": {str(k): v for k, v in digests.items()}}
    assert same, f"outputs differ between pools {POOLS}: {digests}"
    torch.cuda.synchronize()


class _CAI:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (int(ptr), False),
                                         "version": 2}


def _direct_read(fd, offset, nbytes, buf):
    os.preadv(fd, [buf], offset)
    return bytes(buf[:nbytes])


def test_c_thousand_reads(engine, packed_dir):
    import torch
    from slotstream_spark.runtime.pool import ExpertPool
    man = json.loads((packed_dir / "experts.manifest.json").read_text())
    rb, align = int(man["record_bytes"]), int(man.get("alignment", 4096))
    assert rb % align == 0 and int(man["layers"]) == LAYERS and int(man["experts"]) == EXPERTS
    pool = ExpertPool(packed_dir, 64, reader=engine["reader"])         # small: forces evictions and slot overwrites
    _LIVE.append(pool)
    progress = np.zeros(1, dtype=np.uint64)
    pool.set_progress_ptr(progress.ctypes.data)
    fd = os.open(str(packed_dir / "experts.bin"), os.O_RDONLY | os.O_DIRECT)
    buf = mmap.mmap(-1, rb)
    rnd = random.Random(1234)
    errors = bytes_equal = hits = misses = gpu_ok = overwritten_ok = 0
    max_read = 0.0
    slot_last = {}
    try:
        for i in range(ENSURES):
            L = rnd.randrange(LAYERS)
            ids = np.array(rnd.sample(range(EXPERTS), rnd.choice((1, 2, 4, 8))), dtype=np.int32)
            t0 = time.perf_counter()
            try:
                slots = pool.ensure(L, ids, i + 1)
            except Exception as e:                                      # recorded, not hidden
                errors += 1
                print(f"ensure {i} layer {L} ids {ids.tolist()}: {type(e).__name__}: {e}")
                progress[0] = i + 2
                continue
            max_read = max(max_read, time.perf_counter() - t0)
            hits += int(pool.last_timing["hits"])
            misses += int(pool.last_timing["misses"])
            good = True
            for e, s in zip(ids.tolist(), [int(x) for x in slots]):
                want = _direct_read(fd, (L * EXPERTS + e) * rb, rb, buf)
                got = bytes(pool.pool.slot_view(s))
                good = good and got == want
                if i % 10 == 0 or s in slot_last and slot_last[s] != (L, e):
                    gpu = torch.as_tensor(_CAI(pool.pool.slot_dev(s), rb), device="cuda")
                    gsum = int(gpu.to(torch.int64).sum().item())
                    if gsum == int(np.frombuffer(want, dtype=np.uint8).astype(np.int64).sum()):
                        gpu_ok += 1
                        if s in slot_last and slot_last[s] != (L, e):
                            overwritten_ok += 1
                    else:
                        good = False
                slot_last[s] = (L, e)
            bytes_equal += int(good)
            progress[0] = i + 2
    finally:
        os.close(fd)
    torch.cuda.synchronize()
    RES["c"] = {"ensures": ENSURES, "errors": errors, "bytes_equal": bytes_equal, "gpu_checksum_ok": bool(
        errors == 0 and bytes_equal == ENSURES and overwritten_ok > 0), "gpu_checks": gpu_ok,
        "gpu_checks_after_overwrite": overwritten_ok, "max_read_s": max_read, "pool_slots": 64,
        "hits": hits, "misses": misses}
    print("reads:", json.dumps(RES["c"]))
    assert errors == 0 and bytes_equal == ENSURES
    assert overwritten_ok > 0, "no slot was overwritten and re-checked on the GPU"
    assert max_read < READ_LIMIT_S
    _sample("c")
    _LIVE.remove(pool)
    del pool
    gc.collect()


def test_d_peak(engine):
    import torch
    torch.cuda.synchronize()
    _sample("d")
    _, src = nvml_used(os.getpid())
    smp = [footprint.sample_from_dict(x) for x in RES["samples"]]
    p = footprint.peak(smp, footprint.flags_from_dict(RES["flags"]))
    RES["d"] = footprint.peak_to_dict(p) | {
        "peak_bytes": p.nonpool_bytes, "vmhwm": max(s.vmhwm for s in smp), "nvml_max": max(s.nvml for s in smp),
        "nvml_source": src, "pinned": 0,
        "rule": "VmHWM + NVML + pinned with the pool subtracted once per counter where it shows (flags from the "
                "base->fixture growth, 90 % rule); gate on peak_bytes; once_bytes = peak_bytes + pool high-water, "
                "informational"}
    print("peak:", json.dumps(RES["d"]), f"= {p.nonpool_bytes / (1 << 20):.0f} MiB (once {p.once_bytes / (1 << 20):.0f})")
    assert src == "pynvml", "NVML did not report this process"
    assert p.unified_hwm["pool"] == 3 << 30, f"pool high-water {p.unified_hwm['pool']} != 3 GiB: (c) pool not tracked"
    assert p.nonpool_bytes < 4096 << 20, f"peak {p.nonpool_bytes / (1 << 20):.0f} MiB >= 4096 MiB (pool subtracted once per counter)"
