"""Step 65, H3 (graph mode): the pool size never changes the tokens.

Pools of 13 and 48 experts per layer and the D5 maximum give bitwise-equal tokens (and logits) over 2 prompts x 32 tokens,
and a 3-row verify equals three 1-row steps at each pool size. One engine at a time (the pool is the big allocation).
Writes $CAP/h3.json when CAP is set. Run: tools/spark/run.sh --heavy h3 auto 5400 -- tools/spark/py.sh -m pytest -q -m gate
tests/gates/test_h3.py
"""
import gc
import json
import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.gate

LAYERS = 48
NEW_TOKENS = 32
RES = {"pools": {}}


def prompt_ids(k):
    import make_goldens as mg
    d = mg.find_prompt_dir(mg.DEFAULT_DATA, f"dev{k}")
    return [int(t) for t in json.loads((d / "tokens.json").read_text())["ids"]]


def run_pool(slots):
    """Tokens, logits and 3-row-vs-1-row equality for both prompts at one pool size."""
    import torch
    from slotstream_spark import cli
    from slotstream_spark.runtime.graphs import GraphSet, restore_state, snapshot_state
    model, pool, _cfg, _info = cli.build_engine(Namespace(gdn="fused"), slots, 256)
    model.st.ensure_act(torch.bfloat16)
    g1 = GraphSet(model, 1)
    g3 = GraphSet(model, 3, readback=g1.rb, pool_handle=g1.pool_handle)
    out = {"tokens": [], "logits": [], "rows3_equal": []}
    try:
        for k in (0, 1):
            g1.detach()
            model.st.reset_inplace()
            lg = model.prefill(prompt_ids(k)).cpu().numpy()
            toks, lgs = [], [lg]
            for _ in range(NEW_TOKENS):
                toks.append(int(np.argmax(lg)))
                lg = g1.step([toks[-1]])[0].numpy()
                lgs.append(lg)
            out["tokens"].append(toks)
            out["logits"].append(lgs)
            snap = snapshot_state(model.st)                  # after 32 tokens: the 3-row check starts here
            x = [int(np.argmax(lg))]
            rows1 = []
            for _ in range(3):
                r = g1.step([x[-1]])[0].numpy()
                rows1.append(r)
                x.append(int(np.argmax(r)))
            restore_state(model.st, snap)
            rows3 = g3.step(x[:3]).numpy()
            out["rows3_equal"].append([bool(np.array_equal(rows1[i], rows3[i])) for i in range(3)])
    finally:
        del g3, g1, model, pool
        gc.collect()
        torch.cuda.empty_cache()
    return out


def test_pool_size_does_not_change_tokens(model_dir, packed_dir):
    from slotstream_spark import cli
    d5 = cli._slots(Namespace(pool_slots=0, pool_epl=0, mode="D5"))
    sizes = {"epl13": 13 * LAYERS, "epl48": 48 * LAYERS, "d5max": d5}
    runs = {name: run_pool(s) for name, s in sizes.items()}
    base = runs["epl48"]
    for name, r in runs.items():
        eq_tok = r["tokens"] == base["tokens"]
        eq_log = all(np.array_equal(a, b) for pa, pb in zip(r["logits"], base["logits"]) for a, b in zip(pa, pb))
        RES["pools"][name] = {"slots": sizes[name], "tokens_equal_to_epl48": eq_tok, "logits_equal_to_epl48": eq_log,
                              "rows3_equal": r["rows3_equal"]}
        print(f"{name}: {json.dumps(RES['pools'][name])}", flush=True)
    out = os.environ.get("CAP")
    if out:
        os.makedirs(out, exist_ok=True)
        Path(out, "h3.json").write_text(json.dumps(RES, indent=1, sort_keys=True))
    for name, v in RES["pools"].items():
        assert v["tokens_equal_to_epl48"], f"{name}: tokens differ from the 48-per-layer pool"
        assert v["logits_equal_to_epl48"], f"{name}: logits differ from the 48-per-layer pool"
        assert all(all(x) for x in v["rows3_equal"]), f"{name}: a 3-row verify differs from three 1-row steps"
