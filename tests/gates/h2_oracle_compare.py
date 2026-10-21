#!/usr/bin/env python3
"""H2: the engine against the transformers oracle on the engine's own D5-max sequences (teacher forcing).

  h2_oracle_compare.py [all]     engine dumps, oracle dumps (bf16, then fp32), comparison (the first three in
                                 subprocesses so the 49 GiB pool is gone before the oracle runs)
  h2_oracle_compare.py engine    Model.teacher_force(ids, dump_dir) for the 8 sequences of $CAP/h2_ids
  h2_oracle_compare.py oracle    tools/ref_layerwise.py teacher_force + make_goldens.convert_dump, same layout
  h2_oracle_compare.py oracle32  the same oracle in fp32 on the same ids (ORACLE_DTYPE=float32 before the import)
  h2_oracle_compare.py compare   ngram_check exact; routing_check_seq and layer_check_seq per layer; logits_check over
                                 the 64 generated positions; writes $CAP/h2_oracle.json, exit 1 on a failed gate

Per-sequence rule (design decision 16, user approval 2026-10-12): the engine's own sequences are not the calibration
prompts, so each limit is L_seq = min(max(L_k, 2*d_k + margin), c_k/2) with d_k the oracle's own bf16-vs-fp32 metric on
that sequence (contract.seq_limit). The calibrated arrays are unchanged.

The per-prompt count of generated tokens equal to the oracle argmax is reported, not gated (a greedy divergence at
a position inside bf16 noise is expected). Limits come from the calibrated numeric contract.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools", ROOT / "tests" / "gates"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import contract as ct  # noqa: E402

PROMPTS = 8
NEW_TOKENS = 64
POOL_SLOTS = 19012
LAYERS = 48
HIDDEN = 10240
TOPK = 10
HEADS = 16
VOCAB = 248320


def cap_dir():
    v = os.environ.get("CAP")
    if not v:
        sys.exit("CAP is not set (run through tests/gates/h2_job.sh under the guard)")
    return Path(v)


def work_dir():
    r = os.environ.get("SLOTSTREAM_R")
    if not r:
        sys.exit("SLOTSTREAM_R is not set (run through tools/spark/py.sh)")
    return Path(r) / "tmp" / ("h2-oracle-" + cap_dir().name)


def load_ids():
    out = {}
    for k in range(PROMPTS):
        p = cap_dir() / "h2_ids" / f"dev{k}.json"
        if not p.exists():
            sys.exit(f"missing {p}: test_h2_model.py did not write the D5-max sequences")
        d = json.loads(p.read_text())
        out[k] = ([int(t) for t in d["ids"]], int(d["prompt_len"]))
    return out


def step_engine():
    import slotio
    import torch
    from slotstream_spark.model.config import TextConfig
    from slotstream_spark.model.forward import Model
    from slotstream_spark.model.weights import Embedder, NgramSource, accessor, load_trunk
    from slotstream_spark.runtime.pool import ExpertPool

    model_dir = Path(os.environ["SLOTSTREAM_MODEL"])
    packed = Path(os.environ["SLOTSTREAM_PACKED"])
    cfg = TextConfig.from_checkpoint(model_dir)
    weights = load_trunk(packed / "trunk.safetensors", device="cuda")
    reader = slotio.Reader()
    embed = Embedder(reader, packed / "trunk.safetensors", device="cuda")
    ngram = NgramSource(reader, packed, weights, cfg["eos_token_id"], device="cuda")
    pool = ExpertPool(packed, POOL_SLOTS, reader=reader)
    ids_by = load_ids()
    model = Model(cfg, accessor(weights), embed, pool, ngram, device="cuda", gdn="fused",
                  max_ctx=max(len(v[0]) for v in ids_by.values()) + 16)
    base = work_dir() / "engine"
    shutil.rmtree(base, ignore_errors=True)
    for k, (ids, plen) in ids_by.items():
        d = base / f"dev{k}"
        with torch.no_grad():
            model.teacher_force(ids, dump_dir=d)
        (d / "tokens.json").write_text(json.dumps({"id": f"dev{k}", "prompt_len": plen, "ids": ids}))
        print(f"engine dump dev{k}: {len(ids)} tokens", flush=True)


def step_oracle(fp32=False):
    if fp32:
        os.environ["ORACLE_DTYPE"] = "float32"      # read by ref_layerwise at import: set it before the import
    import make_goldens as mg
    import ref_layerwise as ref
    from slotstream_spark.model.mlx_ckpt import Checkpoint

    mdir = ref.model_dir()
    cfg = ref.load_cfg(mdir)
    ck = Checkpoint(mdir)
    ids_by = load_ids()
    name = "oracle32" if fp32 else "oracle"
    base = work_dir() / name
    shutil.rmtree(base, ignore_errors=True)
    try:
        for k, (ids, plen) in ids_by.items():
            tmp = work_dir() / (name + "-tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            dst = base / f"dev{k}"
            dst.mkdir(parents=True)
            ref.teacher_force(ids, dump_dir=str(tmp), ck=ck, cfg=cfg)
            mg.convert_dump(tmp, dst, len(ids))
            shutil.rmtree(tmp, ignore_errors=True)
            (dst / "tokens.json").write_text(json.dumps({"id": f"dev{k}", "prompt_len": plen, "ids": ids}))
            print(f"{name} dump dev{k}: {len(ids)} tokens", flush=True)
    finally:
        ck.close()


def _arr(d, name, dtype, shape):
    return np.fromfile(d / name, dtype=dtype).reshape(shape)


def step_compare():
    c = ct.load()
    ids_by = load_ids()
    per, ok_all = {}, True
    for k, (ids, plen) in ids_by.items():
        T = len(ids)
        e, o = work_dir() / "engine" / f"dev{k}", work_dir() / "oracle" / f"dev{k}"
        o32 = work_dir() / "oracle32" / f"dev{k}"
        ngram_ok = ct.ngram_check(_arr(e, "ngram_ids.i64", np.int64, (T, HEADS)),
                                  _arr(o, "ngram_ids.i64", np.int64, (T, HEADS)))
        layers, bad = [], []
        worst = (0.0, -1, "")
        for L in range(LAYERS):
            ly = ct.layer_check_seq(L, _arr(e, f"layer_{L:02d}_out.f32", np.float32, (T, HIDDEN)),
                                    _arr(o, f"layer_{L:02d}_out.f32", np.float32, (T, HIDDEN)),
                                    _arr(o32, f"layer_{L:02d}_out.f32", np.float32, (T, HIDDEN)), c)
            rt = ct.routing_check_seq(L, _arr(e, f"router_{L:02d}_ids.i32", np.int32, (T, TOPK)),
                                      _arr(o, f"router_{L:02d}_ids.i32", np.int32, (T, TOPK)),
                                      _arr(o32, f"router_{L:02d}_ids.i32", np.int32, (T, TOPK)), c)
            layers.append({"layer": L, "rel_l2": ly["value"], "limit": ly["limit_seq"], "separable": ly["separable"],
                           "ok": ly["ok"], "routing": rt["value"], "routing_limit": rt["limit_seq"],
                           "routing_separable": rt["separable"], "routing_ok": rt["ok"],
                           "rel_l2_pair": ly["pair"], "limit_cal": ly["limit_cal"], "limit_src": ly["limit_src"],
                           "routing_pair": rt["pair"], "routing_limit_cal": rt["limit_cal"],
                           "routing_limit_src": rt["limit_src"]})
            if rt["separable"] and rt["value"] / rt["limit_seq"] > worst[0]:
                worst = (rt["value"] / rt["limit_seq"], L, rt["limit_src"])
            if not (ly["ok"] and rt["ok"]):
                bad.append(L)
        a, b = plen - 1, plen - 1 + NEW_TOKENS         # logits at t predict token t+1
        le = _arr(e, "logits.f32", np.float32, (T, VOCAB))[a:b]
        lo = _arr(o, "logits.f32", np.float32, (T, VOCAB))[a:b]
        lc = ct.logits_check(le, lo, c)
        gen = np.asarray(ids[plen:plen + NEW_TOKENS])
        argmax_oracle_equal = int((lo.argmax(-1) == gen).sum())
        argmax_engine_equal = int((le.argmax(-1) == gen).sum())
        ok = bool(ngram_ok and not bad and lc["ok"])
        ok_all = ok_all and ok
        per[f"dev{k}"] = {"tokens": T, "prompt_len": plen, "ngram_ok": bool(ngram_ok), "failing_layers": bad,
                          "layers": layers, "logits": lc, "worst_routing_ratio": worst[0],
                          "worst_routing_layer": worst[1], "worst_routing_src": worst[2],
                          "bound_layers": [x["layer"] for x in layers
                                           if x["limit_src"] != "calibrated" or x["routing_limit_src"] != "calibrated"], "generated_equal_oracle_argmax": argmax_oracle_equal,
                          "generated_equal_engine_argmax": argmax_engine_equal, "ok": ok}
        print(f"dev{k}: ngram exact {ngram_ok}; failing layers {bad}; logits {json.dumps(lc)}; generated tokens equal "
              f"to the oracle argmax {argmax_oracle_equal}/{NEW_TOKENS}; worst routing ratio {worst[0]:.3f} at layer "
              f"{worst[1]} ({worst[2]})", flush=True)
    res = {"prompts": per, "ok": bool(ok_all), "positions_per_prompt_logits": NEW_TOKENS}
    cap_dir().joinpath("h2_oracle.json").write_text(ct.dumps(res))
    if ok_all:
        shutil.rmtree(work_dir(), ignore_errors=True)
    else:
        print(f"dumps kept in {work_dir()}", flush=True)
    return 0 if ok_all else 1


def main(argv):
    mode = argv[1] if len(argv) > 1 else "all"
    if mode == "engine":
        step_engine()
        return 0
    if mode == "oracle":
        step_oracle()
        return 0
    if mode == "oracle32":
        step_oracle(fp32=True)
        return 0
    if mode == "compare":
        return step_compare()
    if mode != "all":
        sys.exit(f"unknown mode {mode}")
    load_ids()
    for sub in ("engine", "oracle", "oracle32"):
        rc = subprocess.call([sys.executable, str(Path(__file__).resolve()), sub])
        if rc != 0:
            print(f"h2_oracle_compare: step {sub} failed ({rc})", flush=True)
            return rc
    return step_compare()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
