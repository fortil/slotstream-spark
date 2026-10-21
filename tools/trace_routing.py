#!/usr/bin/env python3
"""Routing traces of the engine in D5 graph mode over the lookahead corpus (input of tools/h_curve.py, step 65).

  trace_routing.py --out DIR [--n 48] [--tokens 128] [--corpus FILE] [--mode D5]

Takes the first N requests of the corpus train split that carry inline `messages` (130 do; the validation split has
17 and the rest need `git cat-file`, unavailable on the Spark), renders each as one user turn, prefills, then decodes
`--tokens` greedy tokens through the CUDA-graph path and writes DIR/trace-NN.npz with `ids` int16 (steps, 48, 10),
the cachesim.py `trace_to_keys` input, plus DIR/index.json. Without a GPU, `select_requests` and `check_trace` still work.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "third_party/slotstream/Tools/fixtures/expert-lookahead/corpus.json"
LAYERS, TOPK = 48, 10


def select_requests(corpus, n=48, split="train"):
    """The first `n` requests of `split` with inline messages, in corpus order. Fewer than n is an error."""
    out = [r for r in corpus["requests"] if r.get("split") == split and r.get("messages")]
    if len(out) < n:
        raise ValueError(f"corpus has {len(out)} inline {split} requests, {n} asked")
    return out[:n]


def request_text(req):
    """The single user turn of a request."""
    msgs = req["messages"]
    if len(msgs) != 1 or msgs[0]["role"] != "user":
        raise ValueError(f"request {req.get('id')}: expected one user message")
    return msgs[0]["content"]


def check_trace(arr, steps=None):
    """Shape (steps, 48, 10), int16, ids in [0, 512)."""
    import numpy as np
    if arr.ndim != 3 or arr.shape[1:] != (LAYERS, TOPK) or arr.dtype != np.int16:
        raise ValueError(f"trace {arr.shape} {arr.dtype}, expected (steps, {LAYERS}, {TOPK}) int16")
    if steps is not None and arr.shape[0] != steps:
        raise ValueError(f"trace has {arr.shape[0]} steps, expected {steps}")
    if arr.size and (arr.min() < 0 or arr.max() >= 512):
        raise ValueError("expert id outside [0, 512)")
    return arr


def trace_one(model, graphs, sampler, params, ids, tokens, eos, slotio_rb):
    """Prefill `ids`, decode up to `tokens` greedy tokens; -> (np.int16 [steps, 48, 10], generated tokens)."""
    import numpy as np
    graphs.detach()
    model.st.reset_inplace()
    logits = model.prefill(ids).cpu().numpy()
    seen, out, rows = set(), [], []
    for i in range(tokens):
        tok = sampler.next(logits, params, seen)
        out.append(tok)
        seen.add(tok)
        if tok == eos or i == tokens - 1:
            break
        logits = graphs.step([tok])[0].numpy()
        rows.append(np.stack([slotio_rb.layer_ids(L, TOPK) for L in range(LAYERS)]).astype(np.int16))
    arr = np.stack(rows) if rows else np.zeros((0, LAYERS, TOPK), dtype=np.int16)
    return arr, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--mode", default="D5", choices=("D1", "D2", "D5"))
    a = ap.parse_args(argv)
    import numpy as np
    sys.path.insert(0, str(ROOT))
    import torch
    from slotstream_spark import cli
    from slotstream_spark.engine.sampler import SampleParams, Sampler
    from slotstream_spark.runtime.graphs import GraphSet

    reqs = select_requests(json.loads(Path(a.corpus).read_text()), a.n)
    tok = cli._tokenizer()
    texts = [tok.encode(tok.render_user_turn(request_text(r))) for r in reqs]
    ns = argparse.Namespace(mode=a.mode, gdn="fused", pool_slots=None, pool_epl=None)
    slots = cli._slots(ns)
    max_ctx = -(-(max(len(t) for t in texts) + a.tokens + 16) // 64) * 64
    model, pool, cfg, info = cli.build_engine(ns, slots, max_ctx)
    model.st.ensure_act(torch.bfloat16)
    g = GraphSet(model, 1)
    params = SampleParams(temperature=0.0)
    params.max_tokens = a.tokens
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    t0 = time.time()
    for i, (r, ids) in enumerate(zip(reqs, texts)):
        arr, gen = trace_one(model, g, Sampler(0), params, ids, a.tokens, cfg["eos_token_id"], g.rb)
        check_trace(arr)
        np.savez(out / f"trace-{i:02d}.npz", ids=arr)
        index.append({"i": i, "id": r["id"], "prompt_tokens": len(ids), "steps": int(arr.shape[0]), "generated": len(gen)})
        print(f"{i + 1}/{len(reqs)} {r['id']}: {len(ids)} prompt tokens, {arr.shape[0]} steps", flush=True)
    (out / "index.json").write_text(json.dumps({"mode": a.mode, "slots": slots, "tokens": a.tokens, "n": len(reqs),
                                                "layers": LAYERS, "topk": TOPK, "seconds": round(time.time() - t0, 1),
                                                "requests": index}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
