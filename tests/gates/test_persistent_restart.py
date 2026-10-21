"""Step 86 gate: a conversation survives a process restart through the persistent prefix tier (D5).

Run through `tools/spark/run.sh --heavy f7-persist auto 2400 -- tools/spark/py.sh -m pytest -q -m gate
tests/gates/test_persistent_restart.py`.

Two engine processes share one empty prefix directory (the test points `SLOTSTREAM_PREFIX_DIR` at it, so nothing from an
earlier run can be restored):
- process A answers turn 1 (a 4,600-token system message, so the first prompt takes two passes and the snapshot at the
  pass end 4096 is a real boundary), flushes its writes and, still in memory, runs turn 2 without the disk tier: that is
  A2, the in-memory continuation;
- process B is fresh (empty memory cache), runs turn 2 with the disk tier and must restore the 4096-token snapshot from
  disk (reused > 0). It then runs the same turn 2 cold (no cache, no disk).
The prompt logits (bytes) and the 64 greedy ids of B must equal those of A2 and of the cold run. An engine-vs-itself
check ([engine_self] prefix_resume, bit-identity): the per-sequence logits rule of design decision 18 does not apply.
A changed identity then opens the same directory and must delete every old file.

The module also runs as the worker of those two processes: `python test_persistent_restart.py --worker A|B ...`.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gate, pytest.mark.cuda]

ROOT = Path(__file__).resolve().parents[2]
SLOTS = 19012
MAX_CTX = 9216
NEW_TOKENS = 64
SYSTEM_TOKENS = 4600
BOUNDARY = 4096
USERS = ["Summarize the first section in two sentences.", "Now the second one, and name its tone."]
REPLY = "The first section describes the opening scene and sets a calm, observational tone."


def filler(tok, n, tag):
    texts = json.loads((ROOT / "tests" / "goldens" / "ppl_texts.json").read_text())
    out, k = "", 0
    while len(tok.encode(out)) < n:
        out += f"Section {tag}-{k}.\n" + texts["en" if k % 2 == 0 else "es"] + "\n"
        k += 1
    return out


def conversation_ids(tok):
    system = filler(tok, SYSTEM_TOKENS, "sys")
    m1 = [{"role": "system", "content": system}, {"role": "user", "content": USERS[0]}]
    m2 = m1 + [{"role": "assistant", "content": REPLY}, {"role": "user", "content": USERS[1]}]
    return (tok.encode(tok.render_chat(m1, enable_thinking=False)),
            tok.encode(tok.render_chat(m2, enable_thinking=False)))


def sha(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


# ===================================================================================================== worker
def worker(which, prefix_dir, out_path):
    import gc

    import slotio
    import torch
    from slotstream_spark.engine import prefill_schedule as ps
    from slotstream_spark.engine import prefix_cache as pc
    from slotstream_spark.engine.generate import generate
    from slotstream_spark.engine.prefill import PrefillDriver, PrefillPlan
    from slotstream_spark.engine.tokenizer import Tokenizer
    from slotstream_spark.memory.ledger import Ledger
    from slotstream_spark.model.config import TextConfig
    from slotstream_spark.model.forward import Model
    from slotstream_spark.model.moe_sweep import SweepMoE
    from slotstream_spark.model.weights import Embedder, NgramSource, accessor, load_trunk
    from slotstream_spark.runtime.pool import ExpertPool

    model_dir, packed = Path(os.environ["SLOTSTREAM_MODEL"]), Path(os.environ["SLOTSTREAM_PACKED"])
    cfg = TextConfig.from_checkpoint(model_dir)
    weights = load_trunk(packed / "trunk.safetensors", device="cuda")
    reader = slotio.Reader()
    embed = Embedder(reader, packed / "trunk.safetensors", device="cuda")
    ngram = NgramSource(reader, packed, weights, cfg["eos_token_id"], device="cuda")
    pool = ExpertPool(packed, SLOTS, reader=reader)
    model = Model(cfg, accessor(weights), embed, pool, ngram, device="cuda", gdn="fused", max_ctx=MAX_CTX)
    model.st.ensure_act(torch.bfloat16)
    env = ps.envelope()
    moe = SweepMoE(model.seg, pool, max_rows_per_launch=int(env["max_rows_per_launch"]))
    driver = PrefillDriver(model, moe, env)
    tok = Tokenizer.from_checkpoint(model_dir)
    im_start = tok.encode("<|im_start|>")[0]
    ids1, ids2 = conversation_ids(tok)
    plan = PrefillPlan.from_envelope(env, prefill_pass=4096, read_scope=8192)
    key = pc.CheckpointKey("qwen38-flash-next-mlx-4bit", json.dumps({"gdn": "fused", "sweep": True}), 4096, False)
    max_tokens = Ledger("D5").prefix // pc.BYTES_PER_TOKEN
    boundaries2 = ps.resume_boundaries(len(ids2), 4096, True, driver.P)
    events = []
    tier = pc.open_tier(model.st, key, 4096, model_dir=model_dir, packed_dir=packed, envelope=env,
                        extra_files=[slotio.__file__], directory=prefix_dir, on_event=events.append)

    def run(ids, cache, persistent):
        res = generate(model, ids, NEW_TOKENS, driver=driver, plan=plan, prefix_cache=cache, key=key,
                       snapshot_target=pc.snapshot_target(ids, im_start), persistent=persistent)
        if persistent is not None:
            persistent.release()
        return {"n": len(ids), "logits_sha": sha(res["first_logits"]), "tokens": res["tokens"],
                "info": {k: v for k, v in res["prefix"].items() if isinstance(v, (int, str, bool, type(None)))}}

    out = {"which": which, "len1": len(ids1), "len2": len(ids2), "boundary_in_turn2": BOUNDARY in boundaries2,
           "identity": tier.prefix.identity.components, "identity_digest": tier.prefix.identity.digest}
    if which == "A":
        cache_a = pc.PrefixCache(max_tokens=max_tokens)
        out["turn1"] = run(ids1, cache_a, tier)
        out["flushed"] = tier.flush(20.0)
        out["files"] = sorted(p.name for p in Path(prefix_dir).iterdir())
        out["writes"] = [r.outcome for r in tier.last_results]
        out["A2"] = run(ids2, cache_a, None)                                       # in-memory continuation
    else:
        out["files_at_start"] = sorted(p.name for p in Path(prefix_dir).iterdir())
        out["B"] = run(ids2, pc.PrefixCache(max_tokens=max_tokens), tier)
        out["restores"] = tier.counters["restores"]
        out["cold"] = run(ids2, pc.PrefixCache(max_tokens=max_tokens), None)       # no cache, no disk
    out["events"] = events
    out["ok_close"] = tier.close(20.0)
    Path(out_path).write_text(json.dumps(out, indent=1))
    del tier, driver, moe, model, pool, weights, embed, ngram, reader
    gc.collect()
    torch.cuda.empty_cache()


# ===================================================================================================== the gate
def run_worker(which, prefix_dir, out):
    env = dict(os.environ, SLOTSTREAM_PREFIX_DIR=str(prefix_dir))
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", which, str(prefix_dir), str(out)],
                       env=env, cwd=str(ROOT), timeout=1500)
    assert r.returncode == 0, f"worker {which} exited {r.returncode}"
    return json.loads(Path(out).read_text())


def test_restart_restores_from_disk_bit_identical(tmp_path):
    pdir = tmp_path / "prefix"
    a = run_worker("A", pdir, tmp_path / "a.json")
    print(json.dumps({k: a[k] for k in ("len1", "len2", "turn1", "A2", "files", "writes", "flushed")}), flush=True)
    assert a["boundary_in_turn2"], "turn 2 must have the 4096 boundary"
    assert a["turn1"]["info"]["snapshot_end"] == BOUNDARY and a["turn1"]["info"].get("persist_staged") is True
    assert a["flushed"] and "saved" in a["writes"], a["writes"]
    assert any(f.endswith(".slotprefix") for f in a["files"]) and any(f.endswith(".slotseg") for f in a["files"])
    assert a["A2"]["info"]["reused"] == BOUNDARY, "A2 must continue from memory"
    assert "disk_reused" not in a["A2"]["info"]

    b = run_worker("B", pdir, tmp_path / "b.json")
    print(json.dumps({k: b[k] for k in ("B", "cold", "files_at_start", "restores")}), flush=True)
    assert b["identity_digest"] == a["identity_digest"], "the two processes must compute the same identity"
    assert b["B"]["info"]["reused"] > 0, "B re-read the whole prompt"
    assert b["B"]["info"]["reused"] == BOUNDARY and b["B"]["info"].get("disk_reused") == BOUNDARY, b["B"]["info"]
    assert b["restores"] == 1
    assert b["cold"]["info"]["reused"] == 0
    assert b["B"]["logits_sha"] == a["A2"]["logits_sha"], "prompt logits differ from the in-memory continuation"
    assert b["B"]["tokens"] == a["A2"]["tokens"] and len(b["B"]["tokens"]) == NEW_TOKENS
    assert b["B"]["logits_sha"] == b["cold"]["logits_sha"] and b["B"]["tokens"] == b["cold"]["tokens"]

    # a changed identity ignores and deletes the old files
    from slotstream_spark.engine import persistent_prefix as pp
    changed = dict(b["identity"], config="0" * 64)
    with pp.PersistentPrefix(pp.Configuration(pdir), pp.Identity(changed)) as tier:
        assert len(tier.heads) == 0 and not tier.segments
    left = sorted(p.name for p in pdir.iterdir() if p.name.endswith((".slotprefix", ".slotseg")))
    assert left == [], f"old files survived an identity change: {left}"


if __name__ == "__main__":
    if len(sys.argv) == 5 and sys.argv[1] == "--worker":
        sys.path.insert(0, str(ROOT))
        worker(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        sys.exit("usage: test_persistent_restart.py --worker A|B <prefix dir> <out json>")
