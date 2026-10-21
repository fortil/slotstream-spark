"""H2: the 48-layer model over several pool sizes, resizing, a read fault and the memory peak.

8 dev prompts x 64 greedy tokens. Writes $CAP/h2_model.json (merged into result.json by h2_job.sh and judged by
tools/gates/records.py h2) and $CAP/h2_ids/dev<k>.json (prompt + generated ids at the D5 maximum pool, read by
h2_oracle_compare.py). Run through tests/gates/h2_job.sh under the guard (SLOTSTREAM_FOOTPRINT_MIB is the cap).
"""
import gc
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

import contract as ct
from slotstream_spark.memory import footprint

pytestmark = pytest.mark.gate

PROMPTS = 8
NEW_TOKENS = 64
SIZES = (624, 2304, 3880, 5044, 19012)          # 13/layer, 48, D1 max, D2 max, D5 max (slots)
RESIZE_SEQ = (1440, 19012, 1440, 2400)
SLOTS_PER_CHUNK = 388
FAULT_AFTER_JOBS = 37
FAULT_SLOTS = 2304                               # 48 per layer
RES = {"samples": [], "flags": {}}
_STATE = {}


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


def _sample(tag):
    import torch
    torch.cuda.synchronize()
    pool = _STATE.get("pool")
    RES["samples"].append({"tag": tag, "vmrss": vm_kb("VmRSS"), "vmhwm": vm_kb("VmHWM"),
                           "nvml": nvml_used(os.getpid())[0], "pinned": 0,
                           "unified": {"pool": int(pool.owned_bytes()) if pool is not None else 0}})


class Rec:
    """Model proxy for generate(): hashes every logits vector the model returns."""

    def __init__(self, model):
        self.m, self.h = model, hashlib.sha256()

    def prefill(self, ids):
        lg = self.m.prefill(ids)
        self.h.update(lg.cpu().numpy().tobytes())
        return lg

    def decode(self, tok):
        lg = self.m.decode(tok)
        self.h.update(lg.cpu().numpy().tobytes())
        return lg


class Spy:
    """pool.ensure wrapper: remembers whether a failing ensure left the map or the pins changed."""

    def __init__(self, pool):
        self.pool, self.orig, self.fail = pool, pool.ensure, None
        pool.ensure = self

    def release(self):
        self.pool.ensure = self.orig

    def __call__(self, layer, ids, gen=None):
        before = np.array(self.pool.pool.map_numpy(), copy=True)
        pins = int(self.pool.stats()["pinned"])
        try:
            return self.orig(layer, ids, gen)
        except Exception as e:
            after = np.array(self.pool.pool.map_numpy(), copy=True)
            # victims are unmapped before the reads by design; the invariant is that nothing was published
            published = int(np.count_nonzero((before < 0) & (after >= 0)))
            changed_mapped = int(np.count_nonzero((before >= 0) & (after >= 0) & (before != after)))
            self.fail = {"exc": type(e).__name__, "layer": int(layer), "ids": [int(x) for x in ids],
                         "map_unchanged": bool(published == 0 and changed_mapped == 0),
                         "published": published, "remapped": changed_mapped,
                         "pins_before": pins, "pins_after": int(self.pool.stats()["pinned"])}
            raise


def prompt_ids(k):
    import make_goldens as mg
    d = mg.find_prompt_dir(mg.DEFAULT_DATA, f"dev{k}")
    return [int(t) for t in json.loads((d / "tokens.json").read_text())["ids"]]


def run_prompt(model, k):
    from slotstream_spark.engine.generate import generate
    model.reset()
    rec = Rec(model)
    ids = prompt_ids(k)
    out = generate(rec, ids, max_tokens=NEW_TOKENS, greedy=True)
    assert not out["nan"], f"prompt {k}: non-finite logits"
    assert len(out["tokens"]) == NEW_TOKENS, f"prompt {k}: {len(out['tokens'])} tokens"
    return ids, [int(t) for t in out["tokens"]], rec.h.hexdigest()


@pytest.fixture(scope="module")
def engine(model_dir, packed_dir):
    import slotio
    import torch
    from slotstream_spark.model.config import TextConfig
    from slotstream_spark.model.forward import Model
    from slotstream_spark.model.weights import Embedder, NgramSource, accessor, load_trunk
    from slotstream_spark.runtime.pool import ExpertPool

    cfg = TextConfig.from_checkpoint(model_dir)
    weights = load_trunk(packed_dir / "trunk.safetensors", device="cuda")
    reader = slotio.Reader()
    embed = Embedder(reader, packed_dir / "trunk.safetensors", device="cuda")
    ngram = NgramSource(reader, packed_dir, weights, cfg["eos_token_id"], device="cuda")
    torch.cuda.synchronize()
    _sample("base")
    pool = ExpertPool(packed_dir, SIZES[0], reader=reader)
    _STATE["pool"] = pool
    model = Model(cfg, accessor(weights), embed, pool, ngram, device="cuda", gdn="fused", max_ctx=256)
    _sample("pool-first")
    smp = [footprint.sample_from_dict(x) for x in RES["samples"]]
    RES["flags"]["pool"] = list(footprint.shows_in(smp[0], smp[1], "pool"))
    assert tuple(RES["flags"]["pool"]) != (False, False), "the pool shows in neither counter: broken counter read"
    yield {"model": model, "pool": pool, "reader": reader, "slotio": slotio}
    out = os.environ.get("CAP")
    if out:
        os.makedirs(out, exist_ok=True)
        Path(out, "h2_model.json").write_text(ct.dumps(RES))
    _STATE.clear()
    del model, pool, ngram, embed, weights, reader
    gc.collect()
    torch.cuda.empty_cache()


def test_a_pool_sizes_identical(engine):
    model, pool = engine["model"], engine["pool"]
    cap = os.environ.get("CAP")
    ref, seqs = {}, {}
    sizes = {}
    for n in SIZES:
        pool.resize_slots(n)
        _sample(f"size-{n}")
        digests, toks = [], []
        for k in range(PROMPTS):
            ids, gen, dg = run_prompt(model, k)
            digests.append(dg)
            toks.append(gen)
            if n == SIZES[-1]:
                seqs[k] = ids + gen
        sizes[str(n)] = {"slots": n, "chunks": int(pool.pool.chunks), "tokens_sha256": hashlib.sha256(
            json.dumps(toks).encode()).hexdigest(), "logits_sha256": hashlib.sha256("".join(digests).encode()).hexdigest()}
        ref[n] = (toks, digests)
        _sample(f"after-size-{n}")
        print(f"pool {n} slots ({pool.pool.chunks} chunks): tokens {sizes[str(n)]['tokens_sha256'][:12]} "
              f"logits {sizes[str(n)]['logits_sha256'][:12]}", flush=True)
    same_tokens = all(ref[n][0] == ref[SIZES[-1]][0] for n in SIZES)
    same_logits = all(ref[n][1] == ref[SIZES[-1]][1] for n in SIZES)
    RES["sizes"] = {"prompts": PROMPTS, "new_tokens": NEW_TOKENS, "per_size": sizes,
                    "tokens_equal": bool(same_tokens), "logits_equal": bool(same_logits)}
    RES["chunks_max"] = int(sizes[str(SIZES[-1])]["chunks"])
    RES["reference_digest_prompt0"] = ref[SIZES[-1]][1][0]
    RES["reference_tokens"] = ref[SIZES[-1]][0]
    RES["reference_digests"] = ref[SIZES[-1]][1]
    if cap:
        d = Path(cap) / "h2_ids"
        d.mkdir(parents=True, exist_ok=True)
        for k, s in seqs.items():
            (d / f"dev{k}.json").write_text(json.dumps({"id": f"dev{k}", "prompt_len": len(s) - NEW_TOKENS, "ids": s}))
    assert RES["chunks_max"] == math.ceil(SIZES[-1] / SLOTS_PER_CHUNK)
    assert same_tokens, "greedy tokens differ across pool sizes"
    assert same_logits, "logits differ across pool sizes (SHA-256)"


def test_b_resize_sequence(engine):
    model, pool = engine["model"], engine["pool"]
    want = RES["reference_digests"][0]
    want_tokens = RES["reference_tokens"][0]
    steps = []
    for n in RESIZE_SEQ:
        pool.resize_slots(n)
        _, gen, dg = run_prompt(model, 0)
        steps.append({"slots": n, "identical": bool(dg == want and gen == want_tokens)})
        _sample(f"resize-{n}")
        print(f"resize -> {n}: prompt 0 identical {steps[-1]['identical']}", flush=True)
    RES["resize"] = {"sequence": list(RESIZE_SEQ), "steps": steps, "all_identical": all(s["identical"] for s in steps)}
    assert RES["resize"]["all_identical"], f"prompt 0 changed after a resize: {steps}"


def test_c_read_fault(engine):
    import slotio
    model, pool = engine["model"], engine["pool"]
    k = 1
    pool.resize_slots(FAULT_SLOTS)
    _, ref_gen, ref_dg = run_prompt(model, k)
    assert (ref_gen, ref_dg) == (RES["reference_tokens"][k], RES["reference_digests"][k]), "prompt 1 differs at 48/layer"
    pool.resize_slots(624)                     # drop most entries so the faulted run has to read
    pool.resize_slots(FAULT_SLOTS)
    spy = Spy(pool)
    flt = slotio.ReadFault(FAULT_AFTER_JOBS)
    pool.reader.set_fault(flt)
    raised = None
    try:
        run_prompt(model, k)
    except slotio.IoError as e:
        raised = f"{type(e).__name__}: {e}"
    finally:
        pool.reader.set_fault(None)
        spy.release()
    fired = bool(flt.fired)
    model.reset()
    _, gen, dg = run_prompt(model, k)          # the retry
    retry_ok = bool(gen == ref_gen and dg == ref_dg)
    f = spy.fail or {}
    RES["fault"] = {"after_jobs": FAULT_AFTER_JOBS, "slots": FAULT_SLOTS, "prompt": k, "fired": fired,
                    "raised": raised, "failing_ensure": f, "map_unchanged": bool(f.get("map_unchanged", False)),
                    "pins_restored": bool(f and f["pins_after"] <= f["pins_before"]), "retry_identical": retry_ok}
    _sample("after-fault")
    print("fault:", json.dumps(RES["fault"]), flush=True)
    assert fired and raised, "the injected fault never fired (the faulted run did no reads?)"
    assert RES["fault"]["map_unchanged"], "the failing ensure published map entries"
    assert RES["fault"]["pins_restored"], "the failing ensure left pins behind"
    assert retry_ok, "the retried request differs from the unfaulted one"


def test_d_peak(engine):
    import torch
    torch.cuda.synchronize()
    _sample("end")
    _, src = nvml_used(os.getpid())
    cap_mib = int(os.environ["SLOTSTREAM_FOOTPRINT_MIB"]) if os.environ.get("SLOTSTREAM_FOOTPRINT_MIB") else 0
    RES["cap_bytes"] = cap_mib << 20
    smp = [footprint.sample_from_dict(x) for x in RES["samples"]]
    p = footprint.peak(smp, footprint.flags_from_dict(RES["flags"]))
    RES["peak"] = footprint.peak_to_dict(p)
    RES["nvml_source"] = src
    print(f"peak: nonpool {p.nonpool_bytes / (1 << 20):.0f} MiB, once {p.once_bytes / (1 << 20):.0f} MiB, "
          f"cap {cap_mib} MiB, pool high-water {p.unified_hwm.get('pool', 0) / (1 << 20):.0f} MiB", flush=True)
    assert src == "pynvml", "NVML did not report this process"
    assert cap_mib > 0, "SLOTSTREAM_FOOTPRINT_MIB is not set (run under the guard)"
    assert p.once_bytes <= RES["cap_bytes"], f"peak once_bytes {p.once_bytes} > cap {RES['cap_bytes']}"
