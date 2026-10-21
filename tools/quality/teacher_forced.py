#!/usr/bin/env python3
"""F9 teacher-forced comparison on the 8 reserved prompts (plan step 93; judged by step 95).

  teacher_forced.py [all]   engine dumps, fp32 oracle dumps, comparison (the first two in subprocesses, one model process
                            at a time, so the engine's pool is gone before the oracle runs)
  teacher_forced.py engine  Model.teacher_force(golden ids, dump_dir) for res0..res7
  teacher_forced.py oracle32 tests/gates/oracle_pair.fp32_dump on each reserved prompt's golden ids
  teacher_forced.py compare ngram_check exact; routing_check_seq and layer_check_seq per layer; logits_check_seq over every
                            position; writes $CAP/quality/teacher_forced.json

The bf16 side of every pair is the prompt's golden itself (step 36, the transformers oracle in the checkpoint dtype); the
fp32 side is the same oracle in fp32 on the same ids (design decisions 16 and 18: the reserved prompts are the hold-out,
the calibrated arrays came from dev prompts only, and their noise is measured on themselves). Limits come from
tests/gates/contract.py; nothing here has a literal limit. Top-1 is judged once, pooled over the prompts
(`top1_pooled`, design decision 21); a prompt's own `ok` covers n-gram ids, layers, routing and the KL limits. The json carries per prompt: `logits` (logits_check_seq:
metrics, `pair`, `limits_seq`, `limit_src`, `judgeable`, `fixed_limits_ok`), `layers` (per layer rel_l2, routing, their
pairs, limits and `limit_src`), `worst_routing_ratio`, `bound_layers`, `ngram_ok`. Exit 1 when a prompt fails.
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

PROMPTS = [f"res{k}" for k in range(8)]
LAYERS, HIDDEN, TOPK, HEADS, VOCAB = 48, 10240, 10, 16, 248320


def cap_dir():
    v = os.environ.get("CAP")
    if not v:
        sys.exit("CAP is not set (run through the guard)")
    d = Path(v) / "quality"
    d.mkdir(parents=True, exist_ok=True)
    return d


def work_dir():
    r = os.environ.get("SLOTSTREAM_R")
    if not r:
        sys.exit("SLOTSTREAM_R is not set (run through tools/spark/py.sh)")
    return Path(r) / "tmp" / "f9-teacher-forced"


def oracle_root():
    return Path(os.environ["SLOTSTREAM_R"]) / "tmp" / "oracle-pair-f9"


def golden_ids(pid):
    import make_goldens as mg
    return [int(t) for t in mg.load_golden(pid)["tokens"]]


def step_engine(mode="D5"):
    import torch
    from bench import ttft
    ids_by = {pid: golden_ids(pid) for pid in PROMPTS}
    eng = ttft.bring_up(mode, -(-(max(len(v) for v in ids_by.values()) + 16) // 64) * 64)
    model = eng["model"]
    base = work_dir() / "engine"
    shutil.rmtree(base, ignore_errors=True)
    for pid, ids in ids_by.items():
        d = base / pid
        with torch.no_grad():
            model.teacher_force(ids, dump_dir=d)
        (d / "tokens.json").write_text(json.dumps({"id": pid, "ids": ids}))
        print(f"engine dump {pid}: {len(ids)} tokens", flush=True)
    eng["meter"].stop()


def step_oracle32():
    import oracle_pair
    for pid in PROMPTS:
        ids = golden_ids(pid)
        oracle_pair.fp32_dump(ids, oracle_root() / pid, tag=pid)
        print(f"oracle fp32 dump {pid}: {len(ids)} tokens", flush=True)


def _arr(d, name, dtype, shape):
    return np.fromfile(Path(d) / name, dtype=dtype).reshape(shape)


def compare_prompt(pid, c, engine_dir, fp32_dir):
    import make_goldens as mg
    g = mg.load_golden(pid)
    ids = [int(t) for t in g["tokens"]]
    T = len(ids)
    for d in (engine_dir, fp32_dir):
        if [int(t) for t in json.loads((Path(d) / "tokens.json").read_text())["ids"]] != ids:
            raise SystemExit(f"{pid}: {d}/tokens.json holds other ids than the golden")
    ngram_ok = ct.ngram_check(_arr(engine_dir, "ngram_ids.i64", np.int64, (T, HEADS)), g["ngram_ids"])
    layers, bad = [], []
    worst = (0.0, -1, "")
    for L in range(LAYERS):
        ly = ct.layer_check_seq(L, _arr(engine_dir, f"layer_{L:02d}_out.f32", np.float32, (T, HIDDEN)),
                                g[f"layer_{L:02d}_out"], _arr(fp32_dir, f"layer_{L:02d}_out.f32", np.float32, (T, HIDDEN)), c)
        rt = ct.routing_check_seq(L, _arr(engine_dir, f"router_{L:02d}_ids.i32", np.int32, (T, TOPK)),
                                  g[f"router_{L:02d}_ids"], _arr(fp32_dir, f"router_{L:02d}_ids.i32", np.int32, (T, TOPK)), c)
        layers.append({"layer": L, "rel_l2": ly["value"], "limit": ly["limit_seq"], "separable": ly["separable"],
                       "ok": ly["ok"], "rel_l2_pair": ly["pair"], "limit_cal": ly["limit_cal"],
                       "limit_src": ly["limit_src"], "routing": rt["value"], "routing_limit": rt["limit_seq"],
                       "routing_separable": rt["separable"], "routing_ok": rt["ok"], "routing_pair": rt["pair"],
                       "routing_limit_cal": rt["limit_cal"], "routing_limit_src": rt["limit_src"]})
        if rt["separable"] and rt["value"] / rt["limit_seq"] > worst[0]:
            worst = (rt["value"] / rt["limit_seq"], L, rt["limit_src"])
        if not (ly["ok"] and rt["ok"]):
            bad.append(L)
    le = _arr(engine_dir, "logits.f32", np.float32, (T, VOCAB))
    l32 = _arr(fp32_dir, "logits.f32", np.float32, (T, VOCAB))
    lc = ct.logits_check_seq(le, g["logits"], l32, c)
    ok = bool(ngram_ok and not bad and ct.logits_kl_ok(lc))
    return {"tokens": T, "ngram_ok": bool(ngram_ok), "failing_layers": bad, "layers": layers, "logits": lc,
            "worst_routing_ratio": worst[0], "worst_routing_layer": worst[1], "worst_routing_src": worst[2],
            "bound_layers": [x["layer"] for x in layers
                             if x["limit_src"] != "calibrated" or x["routing_limit_src"] != "calibrated"],
            "positions": T, "ok": ok}


def step_compare():
    c = ct.load()
    per, ok_all = {}, True
    for pid in PROMPTS:
        r = compare_prompt(pid, c, work_dir() / "engine" / pid, oracle_root() / pid)
        per[pid] = r
        ok_all = ok_all and r["ok"]
        print(f"{pid}: ngram exact {r['ngram_ok']}; failing layers {r['failing_layers']}; logits {json.dumps(r['logits'])}; "
              f"worst routing ratio {r['worst_routing_ratio']:.3f} at layer {r['worst_routing_layer']} "
              f"({r['worst_routing_src']}); bound layers {r['bound_layers']}", flush=True)
    pooled = ct.top1_pooled([(pid, per[pid]["logits"]["top1_agree"], per[pid]["positions"]) for pid in PROMPTS], c)
    ok_all = bool(ok_all and pooled["ok"])
    print(f"top-1 pooled {pooled['top1']:.4f} ({pooled['agree']}/{pooled['positions']})", flush=True)
    res = {"prompts": per, "top1_pooled": pooled, "ok": bool(ok_all)}
    (cap_dir() / "teacher_forced.json").write_text(ct.dumps(res))
    if ok_all:
        shutil.rmtree(work_dir(), ignore_errors=True)
    else:
        print(f"dumps kept in {work_dir()} and {oracle_root()}", flush=True)
    return 0 if ok_all else 1


def main(argv):
    mode = argv[1] if len(argv) > 1 else "all"
    if mode == "engine":
        step_engine()
        return 0
    if mode == "oracle32":
        step_oracle32()
        return 0
    if mode == "compare":
        return step_compare()
    if mode != "all":
        sys.exit(f"unknown mode {mode}")
    cap_dir()
    for sub in ("engine", "oracle32"):
        rc = subprocess.call([sys.executable, str(Path(__file__).resolve()), sub])
        if rc != 0:
            print(f"teacher_forced: step {sub} failed ({rc})", flush=True)
            return rc
    return step_compare()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
