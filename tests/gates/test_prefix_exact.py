"""Step 71 gate: resuming from the prefix cache is bit-identical to a cold prefill (D5).

Run through `tools/spark/run.sh --heavy f5-prefix-exact auto 2400 -- tools/spark/py.sh -m pytest -q -m "cuda or gate"
tests/t0/test_prefill_driver_control.py tests/gates/test_prefix_exact.py`.

Three-turn conversations rendered with the checkpoint's own chat template, thinking off and on. The system message is
long enough for the first prompt to need two passes, so the snapshot taken at the last pass end before the final
`<|im_start|>` is a real boundary (4096). Turns 2 and 3 add at most 1,024 new tokens. Per turn the resumed run (through
`generate(..., prefix_cache=...)`) is compared with a cold run of the same ids (empty cache, same driver): the logits of
the last prompt position and 64 greedy ids bit for bit, the reused length equal to the snapshot boundary, and the
decode-end state of the previous turn never the one resumed. A repeat of the last prompt resumes from the retained
complete-prompt state (equal length) with the retained logits and the same ids. This compares the engine with itself
([engine_self] prefix_resume, bit-identity): the per-sequence logits rule of design decision 18 does not apply.
"""
import gc
import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.gate, pytest.mark.cuda]

SLOTS = 19012
MAX_CTX = 9216
NEW_TOKENS = 64
SYSTEM_TOKENS = 4600
MAX_NEW_PER_TURN = 1024
BOUNDARY = 4096
S = {}


def filler(tok, n, tag):
    """About n tokens of real text (the two perplexity texts), each repetition numbered; trimmed at a sentence end."""
    texts = json.loads((Path(__file__).resolve().parents[1] / "goldens" / "ppl_texts.json").read_text())
    out, k = "", 0
    while len(tok.encode(out)) < n:
        out += f"Section {tag}-{k}.\n" + texts["en" if k % 2 == 0 else "es"] + "\n"
        k += 1
    return out


@pytest.fixture(scope="module")
def engine(model_dir, packed_dir):
    import slotio
    import torch
    from slotstream_spark.engine import prefill_schedule as ps
    from slotstream_spark.engine.prefill import PrefillDriver
    from slotstream_spark.engine.tokenizer import Tokenizer
    from slotstream_spark.model.config import TextConfig
    from slotstream_spark.model.forward import Model
    from slotstream_spark.model.moe_sweep import SweepMoE
    from slotstream_spark.model.weights import Embedder, NgramSource, accessor, load_trunk
    from slotstream_spark.runtime.pool import ExpertPool

    cfg = TextConfig.from_checkpoint(model_dir)
    weights = load_trunk(packed_dir / "trunk.safetensors", device="cuda")
    reader = slotio.Reader()
    embed = Embedder(reader, packed_dir / "trunk.safetensors", device="cuda")
    ngram = NgramSource(reader, packed_dir, weights, cfg["eos_token_id"], device="cuda")
    pool = ExpertPool(packed_dir, SLOTS, reader=reader)
    model = Model(cfg, accessor(weights), embed, pool, ngram, device="cuda", gdn="fused", max_ctx=MAX_CTX)
    env = ps.envelope()
    moe = SweepMoE(model.seg, pool, max_rows_per_launch=int(env["max_rows_per_launch"]))
    driver = PrefillDriver(model, moe, env)
    tok = Tokenizer.from_checkpoint(model_dir)
    eng = {"model": model, "pool": pool, "moe": moe, "driver": driver, "env": env, "torch": torch, "tok": tok,
           "model_dir": Path(model_dir), "parts": (weights, embed, ngram, reader)}
    S["eng"] = eng
    del weights, reader, embed, ngram, pool, model, moe, driver
    yield eng
    eng = S.pop("eng", None)
    if eng is not None:
        t = eng["torch"]
        eng.clear()
        gc.collect()
        t.cuda.empty_cache()


def run_cold(eng, ids):
    from slotstream_spark.engine.generate import generate
    from slotstream_spark.engine.prefill import PrefillPlan
    model = eng["model"]
    model.st.reset_inplace()
    plan = PrefillPlan.from_envelope(eng["env"], prefill_pass=4096, read_scope=8192)
    return generate(model, ids, NEW_TOKENS, driver=eng["driver"], plan=plan)


def run_cached(eng, ids, cache, key, target):
    from slotstream_spark.engine.generate import generate
    from slotstream_spark.engine.prefill import PrefillPlan
    plan = PrefillPlan.from_envelope(eng["env"], prefill_pass=4096, read_scope=8192)
    return generate(eng["model"], ids, NEW_TOKENS, driver=eng["driver"], plan=plan, prefix_cache=cache, key=key,
                    snapshot_target=target)


def same_bits(a, b):
    return a["first_logits"].tobytes() == b["first_logits"].tobytes() and a["tokens"] == b["tokens"]


@pytest.mark.parametrize("thinking", [False, True], ids=["think-off", "think-on"])
def test_three_turn_conversation_resumes_bit_identical(engine, thinking):
    from slotstream_spark.engine import prefill_schedule as ps
    from slotstream_spark.engine import prefix_cache as pc
    from slotstream_spark.memory.ledger import Ledger
    tok = engine["tok"]
    im_start = tok.encode("<|im_start|>")
    assert len(im_start) == 1
    im_start = im_start[0]
    system = filler(tok, SYSTEM_TOKENS, "sys")
    users = ["Summarize the first section in two sentences.", "Now the second one, and name its tone.",
             "Which of the two was longer, and by how much?"]
    replies = [filler(tok, 200, "a1")[:900], filler(tok, 200, "a2")[:900]]
    key = pc.CheckpointKey("qwen38-flash-next-mlx-4bit", json.dumps({"gdn": "fused", "thinking": thinking}),
                           4096, False)
    max_tokens = Ledger("D5").prefix // pc.BYTES_PER_TOKEN
    cache = pc.PrefixCache(max_tokens=max_tokens)
    msgs = [{"role": "system", "content": system}]
    prev_len = 0
    report = []
    for turn in range(3):
        msgs.append({"role": "user", "content": users[turn]})
        ids = tok.encode(tok.render_chat(msgs, enable_thinking=thinking))
        assert len(ids) + NEW_TOKENS <= MAX_CTX, len(ids)
        if turn:
            assert len(ids) - (prev_len + len(tok.encode(replies[turn - 1]))) <= MAX_NEW_PER_TURN, "turn adds too much"
        target = pc.snapshot_target(ids, im_start)
        boundaries = ps.resume_boundaries(len(ids), 4096, True, engine["driver"].P)
        assert BOUNDARY in boundaries, (len(ids), boundaries)
        cold = run_cold(engine, ids)
        res = run_cached(engine, ids, cache, key, target)
        info = res["prefix"]
        print(f"thinking={thinking} turn {turn + 1}: {len(ids)} tokens, reused {info['reused']}, "
              f"decision {info['decision']!r}, snapshot_end {info.get('snapshot_end')}, "
              f"entries {[(len(e.tokens), e.fresh_equivalent) for e in cache.entries()]}", flush=True)
        if turn == 0:
            assert info["reused"] == 0 and info.get("snapshot_end") == BOUNDARY
            assert info["snapshot_admitted"] == 0, "the snapshot run must not admit (step 71b)"
            assert info["admitted"] > 0, "the final run of a cached prefill must admit"
            print(f"thinking={thinking} turn 1 sweep stats: {res['prefill'].sweep}", flush=True)
        else:
            assert info["reused"] == BOUNDARY, "the resume length must be the snapshot boundary"
            assert info["reused"] > 0
        assert same_bits(cold, res), f"turn {turn + 1}: resumed prompt logits or greedy ids differ from the cold prefill"
        assert all(not e.fresh_equivalent or e.reusable for e in cache.entries())
        report.append({"turn": turn + 1, "tokens": len(ids), "reused": info["reused"]})
        if turn < 2:
            msgs.append({"role": "assistant", "content": replies[turn]})
        prev_len = len(ids)
    # equal length: the retained complete-prompt state and its logits
    again = run_cached(engine, ids, cache, key, target)
    cold = run_cold(engine, ids)
    print(f"thinking={thinking} repeat of turn 3: reused {again['prefix']['reused']} of {len(ids)}", flush=True)
    assert again["prefix"]["reused"] == len(ids), "an identical prompt must resume from the retained complete-prompt state"
    assert same_bits(cold, again), "equal-length resume differs from the cold prefill"
    # a decode-end state is retained but never the one resumed
    assert any(not e.fresh_equivalent for e in cache.entries()) or cache.stats()["evictions"] > 0
    print(json.dumps({"thinking": thinking, "turns": report, "cache": cache.stats()}), flush=True)
