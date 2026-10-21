"""Step 75 gate: the MTP head (scheme A, resident experts) against the MLX reference fixture.

Run through `tools/spark/run.sh --heavy f6-mtp-head auto 1500 -- tools/spark/py.sh -m pytest -q -m gate
tests/gates/test_mtp_head.py`.

  a  the fixture files match tests/fixtures/mtp/SHA256SUMS, and the installed mtp.safetensors equals the converted head the
     fixture was made with (model.lock.json mtp.sha256); a different head FAILS here and says so
  b  on the real inputs of the fixture, the 4-entry prefill step and the 1-entry cached decode step, lm_head over both sides:
     KL(fixture -> ours) <= [fixture] kl_max_nats at every one of the 5 positions; argmax match and rel_l2 of the outputs are
     printed per position and never asserted (two bf16 implementations)
  c  the ledger term: the head's resident bytes equal 1,470,946,816
  d  engine-facing calls: consume keeps `state.pos == consumed - 1`, draft(k=2) returns 2 ids without moving the committed state,
     and the fusion is row-invariant (4 rows at once == 4 single rows, bit for bit)
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.gate, pytest.mark.cuda]

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "mtp"
RES = {}


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def say(request, text):
    """A line on the real terminal (pytest -q hides the output of tests that pass)."""
    tr = request.config.pluginmanager.get_plugin("terminalreporter")
    if tr is not None:
        tr.write_line(text)
    else:
        print(text, flush=True)


def load_st(path):
    import torch
    from slotstream_spark.model.weights import read_header, _DT
    base, header = read_header(path)
    out = {}
    with open(path, "rb") as f:
        for k, m in header.items():
            if k == "__metadata__":
                continue
            b, e = m["data_offsets"]
            f.seek(base + b)
            out[k] = torch.frombuffer(bytearray(f.read(e - b)), dtype=_DT[m["dtype"]]).reshape(m["shape"])
    return out


@pytest.fixture(scope="module")
def engine(model_dir, packed_dir):
    import gc
    import slotio
    import torch
    from slotstream_spark.engine.mtp import MTPHead
    from slotstream_spark.model.config import TextConfig
    from slotstream_spark.model.weights import Embedder, load_trunk

    cfg = TextConfig.from_checkpoint(model_dir)
    main = load_trunk(packed_dir / "trunk.safetensors", layers=[], device="cuda")      # lm_head and the final mixer only
    reader = slotio.Reader()
    embed = Embedder(reader, packed_dir / "trunk.safetensors", device="cuda")
    head = MTPHead(cfg, packed_dir / "mtp_trunk.safetensors", packed_dir / "mtp_experts.bin", embed=embed,
                   lm_head=main["lm_head.weight"], device="cuda", max_ctx=256)
    e = {"head": head, "cfg": cfg, "embed": embed, "model_dir": Path(model_dir)}
    yield e
    out = os.environ.get("CAP")
    if out:
        os.makedirs(out, exist_ok=True)
        Path(out, "mtp_head.json").write_text(json.dumps(RES, indent=1, sort_keys=True))
    del head, embed, main, reader
    gc.collect()
    torch.cuda.empty_cache()


def test_a_fixture_and_head_provenance(engine):
    for line in (FIX / "SHA256SUMS").read_text().splitlines():
        want, name = line.split()
        assert sha256(FIX / name) == want, f"fixture {name} differs from SHA256SUMS"
    lock = json.loads((ROOT / "model.lock.json").read_text())["mtp"]
    got = sha256(engine["model_dir"] / "mtp.safetensors")
    RES["mtp_safetensors_sha256"] = got
    assert got == lock["sha256"], (f"our mtp.safetensors sha {got} differs from the converted head the fixture was made with "
                                   f"({lock['sha256']}): the parity numbers below would compare different weights")


def test_c_resident_bytes_match_the_ledger(engine):
    from slotstream_spark.engine.mtp import RESIDENT_BYTES
    from slotstream_spark.memory.ledger import MTP
    h = engine["head"]
    RES["resident_bytes"] = h.resident_bytes
    assert h.resident_bytes == RESIDENT_BYTES == MTP, (h.resident_bytes, RESIDENT_BYTES, MTP)


def test_b_fixture_parity(engine, request):
    import torch
    from slotstream_spark.engine.mtp import MTPState
    from slotstream_spark.model.layers import lin
    import contract
    from calibrate_contract import logits_metrics, rel_l2

    c = contract.load()
    limit = float(c.fixture["kl_max_nats"])
    head = engine["head"]
    fx = load_st(FIX / "mtp_parity.safetensors")
    dev = "cuda"
    state = MTPState(dev, 256)
    rows = []
    for step, (emb, hid, ref) in enumerate((("embedded", "hidden", "out1"), ("embedded2", "hidden2", "out2")), start=1):
        e, h = fx[emb][0].to(dev), fx[hid][0].to(dev)
        out, _ = head.step(e, h, state)
        ours = head.logits(out).cpu().numpy()
        theirs = head.logits(fx[ref][0].to(dev)).cpu().numpy()      # lm_head over the fixture side too
        kl, agree = logits_metrics(theirs, ours)                   # KL(fixture -> ours), full vocabulary, per position
        for i in range(ours.shape[0]):
            rl2 = rel_l2(out[i].float().cpu().numpy(), fx[ref][0][i].float().numpy())
            rows.append({"step": step, "pos": i, "kl": float(kl[i]), "argmax_match": bool(agree[i]), "rel_l2_out": float(rl2)})
    assert len(rows) == 5
    RES["positions"], RES["kl_limit"] = rows, limit
    say(request, f"mtp fixture parity (limit KL <= {limit:.5f} nats)")
    for r in rows:
        say(request, f"  step {r['step']} pos {r['pos']}: KL {r['kl']:.5f}  argmax {'match' if r['argmax_match'] else 'DIFFERS'}"
                     f"  rel_l2(out) {r['rel_l2_out']:.5f}")
    bad = [r for r in rows if r["kl"] > limit]
    assert not bad, f"KL above {limit} at {[(r['step'], r['pos'], r['kl']) for r in bad]}"
    assert state.pos == 5


def test_d_engine_facing_calls(engine, request):
    import torch
    head = engine["head"]
    fx = load_st(FIX / "mtp_parity.safetensors")
    H, hc = head.H, head.hc
    # row invariance of the fusion: 4 rows at once == 4 single rows
    e, h = fx["embedded"][0].cuda(), fx["hidden"][0].cuda()
    whole = head.fuse(e, h)
    single = torch.cat([head.fuse(e[i:i + 1], h[i:i + 1]) for i in range(4)], 0)
    assert torch.equal(whole, single), "the fusion is not row-invariant"
    # consume / draft on the fixture ids: 5 tokens, multi rows of the main model are the fixture's hidden (4) + hidden2
    ids = [int(x) for x in fx["ids"]]
    multi = torch.cat([fx["hidden"][0], fx["hidden"][0][:1]], 0).cuda()          # any 5 real rows will do for the invariant
    head.reset()
    head.consume(ids[:3], multi[:3])
    assert head.state.pos == 2 and head.consumed == 3
    head.consume(ids[3:], multi[3:])
    assert head.state.pos == 4 and head.consumed == 5
    d = head.draft(int(fx["step2_id"][0]), k=2)
    assert len(d) == 2 and all(isinstance(t, int) and 0 <= t < engine["cfg"]["vocab_size"] for t in d)
    assert head.state.pos == 4, "draft moved the committed head state"
    say(request, f"consume/draft: state.pos {head.state.pos} == consumed {head.consumed} - 1; draft ids {d}")
    with pytest.raises(NotImplementedError, match="MTP streaming is not implemented in v1"):
        type(head)(engine["cfg"], resident=False)
