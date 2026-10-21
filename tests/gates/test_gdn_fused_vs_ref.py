"""H2: the fused GDN against the reference recurrence on the whole 48-layer model, 512 positions.

The fused run generates 448 greedy tokens after dev0's prompt; both GDN implementations are then teacher-forced
on those 512 ids. Limits are the calibrated numeric contract (no literal thresholds). Writes $CAP/h2_gdn.json.
"""
import gc
import json
import os
from pathlib import Path

import numpy as np
import pytest

import contract as ct

pytestmark = pytest.mark.gate

POSITIONS = 512
POOL_SLOTS = 2304
RES = {}


class Store:
    """Model._chunk sink that keeps every layer's hyper-width output on the host."""

    def __init__(self):
        self.out = {}
        self.ids = {}

    def h_in(self, hyp):
        pass

    def layer(self, L, hyp, ids, w):
        self.out[L] = hyp.float().cpu().numpy()
        self.ids[L] = np.asarray(ids)

    def ngram(self, gids):
        pass


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
    pool = ExpertPool(packed_dir, POOL_SLOTS, reader=reader)
    model = Model(cfg, accessor(weights), embed, pool, ngram, device="cuda", gdn="fused", max_ctx=POSITIONS + 64)
    yield model
    out = os.environ.get("CAP")
    if out:
        os.makedirs(out, exist_ok=True)
        Path(out, "h2_gdn.json").write_text(ct.dumps(RES))
    del model, pool, ngram, embed, weights, reader
    gc.collect()
    torch.cuda.empty_cache()


def _teacher(model, ids, gdn):
    model.set_gdn(gdn)
    model.reset()
    sink = Store()
    final = model._chunk(ids, sink)
    logits = model._logits(final).cpu().numpy()
    return sink, logits


def test_fused_vs_reference_full_model(engine):
    import make_goldens as mg
    from slotstream_spark.engine.generate import generate
    c = ct.load()
    model = engine
    d = mg.find_prompt_dir(mg.DEFAULT_DATA, "dev0")
    prompt = [int(t) for t in json.loads((d / "tokens.json").read_text())["ids"]]
    model.set_gdn("fused")
    model.reset()
    gen = generate(model, prompt, max_tokens=POSITIONS - len(prompt), greedy=True)
    assert not gen["nan"]
    ids = prompt + [int(t) for t in gen["tokens"]]
    assert len(ids) == POSITIONS, f"{len(ids)} positions, wanted {POSITIONS}"
    fused, lg_fused = _teacher(model, ids, "fused")
    ref, lg_ref = _teacher(model, ids, "ref")
    model.set_gdn("fused")

    layers, bad = [], []
    for L in range(len(fused.out)):
        v, lim, sep, ok = ct.layer_check(L, fused.out[L], ref.out[L], c)
        rv, rlim, rsep, rok = ct.routing_check(L, fused.ids[L], ref.ids[L], c)
        layers.append({"layer": L, "rel_l2": v, "limit": lim, "separable": sep, "ok": bool(ok),
                       "routing": rv, "routing_limit": rlim, "routing_separable": rsep, "routing_ok": bool(rok)})
        if not (ok and rok):
            bad.append(L)
    lc = ct.logits_check(lg_fused, lg_ref, c)
    argmax_equal = int((lg_fused.argmax(-1) == lg_ref.argmax(-1)).sum())
    RES.update({"positions": POSITIONS, "layers": layers, "logits": lc, "argmax_equal": argmax_equal,
                "failing_layers": bad, "separable_layers": sum(1 for x in layers if x["separable"])})
    print(f"fused vs ref: logits {json.dumps(lc)}; argmax equal {argmax_equal}/{POSITIONS}; failing layers {bad}",
          flush=True)
    if not lc["ok"] or bad:
        print("DECISION: fused GDN fails the contract at full model; reference/chunk recurrence costs ~4.3 ms/token "
              "(the technology decision)", flush=True)
    assert not bad, f"separable layers over their limits (or routing over its limit): {bad}"
    assert lc["ok"], f"logits outside the calibrated [logits] limits: {lc}"
