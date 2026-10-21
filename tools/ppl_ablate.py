#!/usr/bin/env python3
# SPDX-License-Identifier: MIT AND Apache-2.0
# Portions ported from helgard-orlm/qwen-flash-next-8gb@958d22e7ba0317750ccc38862b9667b1034a4dd1 tools/ppl_ablate.py:1-26 (Apache-2.0, section 4); modified.
"""Independent check of the expert unpacking: teacher-forced perplexity of known text with the correct experts,
with the nibbles swapped and with no experts. The correct unpacking must be far better than both ablations.

  ppl_ablate.py --texts F --modes correct,swap,noexp --state DIR --out JSON

Criterion (numeric_contract.toml [ablation]): ppl_correct <= ppl_correct_max, and swap and noexp perplexity
>= swap_min_ratio / noexp_min_ratio x ppl_correct, for every text. Resumable: each (mode, text) result is
stored in DIR/progress.json as soon as it exists.
"""
import argparse
import json
import math
import os
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "slotstream_spark" / "config" / "numeric_contract.toml"


def evaluate(ppl, ablation):
    """ppl: {mode: {text: value}} -> (pass, reasons)."""
    reasons = []
    for text, c in ppl["correct"].items():
        if not c <= ablation["ppl_correct_max"]:
            reasons.append(f"{text}: ppl_correct {c:.2f} > {ablation['ppl_correct_max']}")
        for mode, key in (("swap", "swap_min_ratio"), ("noexp", "noexp_min_ratio")):
            if mode in ppl and not ppl[mode][text] >= ablation[key] * c:
                reasons.append(f"{text}: {mode}/correct = {ppl[mode][text] / c:.2f} < {ablation[key]}")
    return not reasons, reasons


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True)
    ap.add_argument("--modes", default="correct,swap,noexp")
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    import ref_layerwise as ref
    from transformers import AutoTokenizer
    from slotstream_spark.model.mlx_ckpt import Checkpoint

    texts = json.load(open(a.texts))
    modes = a.modes.split(",")
    state = Path(a.state)
    state.mkdir(parents=True, exist_ok=True)
    prog_path = state / "progress.json"
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else {}
    mdir = ref.model_dir()
    tok = AutoTokenizer.from_pretrained(mdir)
    cfg = ref.load_cfg(mdir)
    ck = Checkpoint(mdir)
    for mode in modes:
        for name, txt in texts.items():
            key = f"{mode}/{name}"
            if key in prog:
                continue
            ids = tok(txt, add_special_tokens=False)["input_ids"]
            t0 = time.time()
            lg = ref.teacher_force(ids, experts_mode=mode, ck=ck, cfg=cfg)
            lp = torch.log_softmax(lg[:-1], -1).gather(1, torch.tensor(ids[1:], device=lg.device)[:, None])
            nll = float(-lp.mean().item())
            prog[key] = {"nll": nll, "ppl": math.exp(nll), "tokens": len(ids), "seconds": round(time.time() - t0, 1)}
            tmp = state / "progress.json.tmp"
            tmp.write_text(json.dumps(prog))
            os.replace(tmp, prog_path)
            print(f"{mode:8s} {name}: ppl {prog[key]['ppl']:10.2f} ({len(ids)} tokens)", flush=True)
    ck.close()
    ppl = {m: {n: prog[f"{m}/{n}"]["ppl"] for n in texts} for m in modes}
    ablation = tomllib.load(open(CONTRACT, "rb"))["ablation"]
    ok, reasons = evaluate(ppl, ablation) if "correct" in ppl else (False, ["no correct mode"])
    res = {"ppl": ppl, "detail": prog, "ablation": ablation, "pass": ok, "reasons": reasons}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"pass": ok, "reasons": reasons}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
