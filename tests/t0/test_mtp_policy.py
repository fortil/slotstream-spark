"""Step 76, T0: the MTP policy per mode, the planner/ledger terms, the step record and the rollback, and the speculative loop
against a fake model whose logits depend only on the state (so the plain loop is the oracle for every draft schedule)."""
import tomllib
from types import SimpleNamespace

import numpy as np
import pytest

from slotstream_spark.config import modes
from slotstream_spark.engine import mtp as mtp_mod
from slotstream_spark.engine import rollback as rb
from slotstream_spark.engine.generate import _mtp_decode
from slotstream_spark.engine.sampler import SampleParams, Sampler
from slotstream_spark.memory import ledger as L
from slotstream_spark.memory.planner import plan

try:                                           # the tensor tests run on the Spark (the Mac venv has no torch)
    import torch
    from slotstream_spark.kernels import gdn_step_record as gsr
except ModuleNotFoundError:
    torch = gsr = None
needs_torch = pytest.mark.skipif(torch is None, reason="needs torch: run on the Spark")


# ------------------------------------------------------------------ policy
def test_policy_file_has_a_valid_value_for_every_mode():
    t = tomllib.loads(mtp_mod.POLICY_PATH.read_text())
    assert set(t) == {"mtp"} and set(t["mtp"]) == set(mtp_mod.MODES)
    for m in mtp_mod.MODES:
        assert mtp_mod.policy(m, env={}) in mtp_mod.POLICY_VALUES


def write_policy(tmp_path, body):
    p = tmp_path / "policy.toml"
    p.write_text(body)
    return p


def test_policy_env_overrides_and_values_are_checked(tmp_path):
    p = write_policy(tmp_path, '[mtp]\nD1 = "off"\nD2 = "off"\nD5 = "resident"\n')
    assert mtp_mod.policy("D5", env={}, path=p) == "resident"
    assert mtp_mod.policy("D1", env={}, path=p) == "off"
    assert mtp_mod.policy("D5", env={"SLOTSTREAM_MTP": "off"}, path=p) == "off"
    assert mtp_mod.policy("D1", env={"SLOTSTREAM_MTP": "resident"}, path=p) == "resident"
    with pytest.raises(ValueError, match="maybe"):
        mtp_mod.policy("D5", env={"SLOTSTREAM_MTP": "maybe"}, path=p)
    with pytest.raises(ValueError, match="unknown mode"):
        mtp_mod.policy("D3", env={}, path=p)


def test_policy_streaming_raises_from_file_and_env(tmp_path):
    p = write_policy(tmp_path, '[mtp]\nD1 = "off"\nD2 = "streaming"\nD5 = "off"\n')
    with pytest.raises(NotImplementedError, match="MTP streaming is not implemented in v1"):
        mtp_mod.policy("D5", env={}, path=p)
    ok = write_policy(tmp_path, '[mtp]\nD1 = "off"\nD2 = "off"\nD5 = "off"\n')
    with pytest.raises(NotImplementedError, match="MTP streaming is not implemented in v1"):
        mtp_mod.policy("D5", env={"SLOTSTREAM_MTP": "streaming"}, path=ok)
    if torch is not None:
        with pytest.raises(NotImplementedError, match="MTP streaming is not implemented in v1"):
            mtp_mod.MTPHead({"hidden_size": 1, "hc_count": 1, "rms_norm_eps": 1e-6, "num_experts_per_tok": 1}, resident=False)


def test_policy_missing_mode_is_an_error(tmp_path):
    p = write_policy(tmp_path, '[mtp]\nD1 = "off"\nD5 = "off"\n')
    with pytest.raises(ValueError, match="D2 is missing"):
        mtp_mod.policy("D5", env={}, path=p)


@pytest.mark.parametrize("pos,max_ctx,want", [(0, 100, 2), (97, 100, 2), (98, 100, 1), (99, 100, 0), (100, 100, 0)])
def test_draft_depth_is_limited_by_the_context(pos, max_ctx, want):
    assert mtp_mod.draft_depth(max_ctx, pos) == want


# ------------------------------------------------------------------ ledger and planner
CAP = modes.get("D5").ceiling_mib << 20


def test_ledger_charges_the_head_only_when_resident():
    assert L.Ledger("D5").mtp == L.MTP and L.Ledger("D5").mtp_record == 0          # design term, unchanged
    off, res = L.Ledger("D5", mtp_resident=False), L.Ledger("D5", mtp_resident=True)
    assert off.mtp == 0 and off.mtp_record == 0 and "mtp_record" not in off.terms()
    assert res.mtp == 1_470_946_816 and res.mtp_record == L.MTP_RECORD == 346_927_104
    assert res.fixed_bytes() - off.fixed_bytes() == 1_470_946_816 + 346_927_104
    assert L.Ledger("D1", mtp_resident=True).mtp == 1_470_946_816


def test_planner_mtp_modes():
    base = plan("D5", CAP)
    assert (base.mtp_enabled, base.mtp_reserved) == (False, 1_470_946_816)       # legacy call, unchanged
    off, res = plan("D5", CAP, mtp="off"), plan("D5", CAP, mtp="resident")
    assert (off.mtp_enabled, off.mtp_reserved) == (False, 0)
    assert (res.mtp_enabled, res.mtp_reserved) == (True, 1_470_946_816 + L.MTP_RECORD)
    assert off.chunks >= res.chunks and off.expected_peak == L.Ledger("D5", mtp_resident=False).expected_peak_bytes(off.chunks)
    assert res.expected_peak == L.Ledger("D5", mtp_resident=True).expected_peak_bytes(res.chunks)
    with pytest.raises(ValueError, match="mtp must be"):
        plan("D5", CAP, mtp="streaming")


def test_planner_refuses_resident_below_the_floor():
    fixed_off = L.Ledger("D5", mtp_resident=False).fixed_bytes()
    cap = fixed_off + 2 * (1 << 30)
    ok = plan("D5", cap, mtp="off")
    assert ok.chunks == 2
    with pytest.raises(ValueError, match="refused"):
        plan("D5", cap, mtp="resident")                       # the head and the ring take the room of the two chunks
    cap2 = L.Ledger("D5", mtp_resident=True).fixed_bytes() + 2 * (1 << 30)
    assert plan("D5", cap2, mtp="resident").chunks == 2


# ------------------------------------------------------------------ step record and rollback on CPU tensors
def fake_state(n_lin=3, nv=2, dk=4, dv=4, ck=3, ch=5, hist=9, w=6):
    return SimpleNamespace(S=torch.zeros(n_lin, nv, dk, dv), conv=torch.zeros(n_lin, ck, ch, dtype=torch.bfloat16),
                           ple_conv=torch.zeros(hist, w, dtype=torch.bfloat16), pos=0, tokens=[],
                           attn={0: SimpleNamespace(pos=0, last="x")})


@needs_torch
def test_gdn_step_record_equals_three_one_row_steps(monkeypatch):
    calls = []

    def fake_decode(q, k, v, g, beta, state):
        calls.append(q.shape[1])
        assert q.shape[1] == 1                                   # always the one-row kernel
        s = state * 0.5 + (q.sum() + 3 * k.sum() + v.sum() + g.sum() + beta.sum())
        return v * 2 + s.mean(), s

    monkeypatch.setattr(gsr.vendored, "gdn_decode", fake_decode)
    g = torch.Generator().manual_seed(1)
    T = 3
    q, k = torch.randn(1, T, 2, 4, generator=g), torch.randn(1, T, 2, 4, generator=g)
    v, gg, bt = torch.randn(1, T, 4, 4, generator=g), torch.randn(1, T, 4, generator=g), torch.rand(1, T, 4, generator=g)
    s0 = torch.randn(1, 4, 4, 4, generator=g)
    ring = torch.zeros(T, 4, 4, 4)
    o, s = gsr.gdn_step_record(q, k, v, gg, bt, s0, ring)
    ref_s, ref_o = s0, []
    for i in range(T):
        oi, ref_s = fake_decode(q[:, i:i + 1], k[:, i:i + 1], v[:, i:i + 1], gg[:, i:i + 1], bt[:, i:i + 1], ref_s)
        ref_o.append(oi)
        assert torch.equal(ring[i], ref_s[0])
    assert torch.equal(o, torch.cat(ref_o, 1)) and torch.equal(s, ref_s) and calls == [1] * 6


@needs_torch
def test_record_restore_and_release_accounting():
    st = fake_state()
    st.conv = torch.zeros_like(st.conv)
    rec = gsr.StepRecord(st, rows=3)
    for i in range(3):
        rec.S[i].fill_(i + 1)
        rec.conv[i].fill_(10 + i)
        rec.ple[i].fill_(20 + i)
    with pytest.raises(RuntimeError):
        rec.release()                                          # nothing outstanding
    rec.begin()
    rec.restore(st, 2)
    assert torch.all(st.S == 2) and torch.all(st.conv == 11) and torch.all(st.ple_conv == 21) and rec.outstanding == 0
    rec.begin()
    with pytest.raises(ValueError):
        rec.restore(st, 4)
    assert rec.outstanding == 1


@needs_torch
def test_rollback_trims_position_tokens_and_caches():
    st = fake_state()
    rec = gsr.StepRecord(st, rows=3)
    model = SimpleNamespace(st=st)
    st.pos, st.tokens = 5, [1, 2, 3, 4, 5]
    ck = rb.checkpoint(model)
    rec.begin()
    st.pos, st.tokens = 8, st.tokens + [6, 7, 8]
    st.attn[0].pos = 8
    for i in range(3):
        rec.S[i].fill_(i)
    rb.rollback(model, rec, ck, 1)
    assert (st.pos, st.tokens, st.attn[0].pos, st.attn[0].last) == (6, [1, 2, 3, 4, 5, 6], 6, None)
    assert torch.all(st.S == 0) and rec.outstanding == 0
    rec.begin()
    st.pos, st.tokens = 9, st.tokens + [7, 8, 9]
    rb.rollback(model, rec, rb.Checkpoint(6, 6), 3)           # keep all: nothing restored, record released
    assert st.pos == 9 and len(st.tokens) == 9 and rec.outstanding == 0
    rec.begin()
    with pytest.raises(RuntimeError, match="expected the checkpoint"):
        rb.rollback(model, rec, rb.Checkpoint(0, 0), 1)
    with pytest.raises(ValueError):
        rb.rollback(model, rec, rb.Checkpoint(6, 6), 0)


# ------------------------------------------------------------------ the loop against a fake model
V = 53


def nxt(h):
    return (h * 7 + 3) % V


class FakeGraphs:
    """State h (a hash of the consumed tokens) lives in model.st.S[0,0,0,0]; logits one-hot of nxt(h) after each row."""

    def __init__(self, model, rows, record=None):
        self.model, self.rows, self.record = model, rows, record
        self.multi_on = record is not None
        self.lookahead = None
        self.last_multi = None
        self.log = []

    @staticmethod
    def upd(h, tok):
        return (h * 31 + tok + 1) % 100003

    def step(self, toks):
        st = self.model.st
        h = int(st.S[0, 0, 0, 0])
        out = np.zeros((self.rows, V), dtype=np.float32)
        multi = torch.zeros(self.rows, 4)
        for i, t in enumerate(toks):
            h = self.upd(h, int(t))
            out[i, nxt(h)] = 5.0
            out[i, (nxt(h) + 1) % V] = 1.0
            multi[i, 0] = h
            if self.record is not None:
                self.record.S[i].zero_()
                self.record.S[i, 0, 0, 0, 0] = h
        self.log.append(list(toks))
        if self.record is not None:
            self.record.begin()
        st.S[0, 0, 0, 0] = h
        st.pos += len(toks)
        st.tokens += [int(t) for t in toks]
        st.attn[0].pos = st.pos
        self.last_multi = multi
        return torch.from_numpy(out)

    def detach(self):
        pass


class FakeHead:
    def __init__(self, model, schedule):
        self.model, self.schedule = model, schedule     # schedule(pending, n_out) -> 'good' | 'half' | 'bad'
        self.state = SimpleNamespace(pos=0, max_ctx=10 ** 6)
        self.consumed_ids = []

    def consume(self, ids, multi):
        assert multi.shape[0] == len(ids)
        self.consumed_ids += list(ids)
        self.state.pos += len(ids)


def make(rec_rows=3):
    st = fake_state()
    st.pos = 0
    model = SimpleNamespace(st=st, max_ctx=10 ** 6)
    rec = gsr.StepRecord(st, 3)
    return model, FakeGraphs(model, 1), FakeGraphs(model, 3, rec), rec


def prefill(model, ids):
    h = 0
    for t in ids:
        h = FakeGraphs.upd(h, t)
    model.st.S[0, 0, 0, 0] = h
    model.st.pos, model.st.tokens = len(ids), list(ids)
    lg = np.zeros(V, dtype=np.float32)
    lg[nxt(h)] = 5.0
    return lg


def plain(ids, n, eos=None, params=None, seed=0):
    model, g1, _, _ = make()
    lg = prefill(model, ids)
    s, seen, out = Sampler(seed), set(), []
    p = params or SampleParams(temperature=0.0)
    p.max_tokens = n
    for _ in range(n):
        t = s.next(lg, p, seen)
        out.append(t)
        seen.add(t)
        if (eos is not None and t == eos) or len(out) == n:
            break
        lg = g1.step([t])[0].numpy()
    return out, model.st.S[0, 0, 0, 0].item()


def spec(ids, n, draft_fn, eos=None, params=None, seed=0):
    model, g1, g3, rec = make()
    lg = prefill(model, ids)
    head = FakeHead(model, None)
    p = params or SampleParams(temperature=0.0)
    p.max_tokens = n
    out, nan, _, st = _mtp_decode(model, g1, g3, head, Sampler(seed), p, set(), lg, eos, None, n, draft_fn)
    return out, nan, st, model, rec, head, g3


def oracle_next(ids, n):
    """The plain continuation, to build correct drafts."""
    return plain(ids, n)[0]


@pytest.mark.parametrize("mode", ["good", "bad", "half", "head"])
@pytest.mark.parametrize("n", [1, 2, 3, 7, 40])
@needs_torch
def test_speculative_output_equals_plain_for_every_draft_schedule(mode, n):
    ids = [5, 9, 11, 2]
    ref, ref_h = plain(ids, n + 3)
    ref = ref[:n]
    ref_n, _ = plain(ids, n)
    assert ref == ref_n

    def fn(pending, k, n_out):
        want = oracle_next(ids, n_out + k + 1)[n_out:n_out + k]       # the plain tokens after the pending one
        if mode == "good":
            return want
        if mode == "bad":
            return [(w + 1) % V for w in want]
        if mode == "half":
            return [want[0], (want[1] + 1) % V] if n_out % 2 else [(want[0] + 1) % V, want[1]]
        return [3, 4]                                                  # whatever the "head" says

    out, nan, st, model, rec, head, g3 = spec(ids, n, fn)
    assert out == ref and not nan
    assert rec.outstanding == 0                                         # no leaked record
    if mode == "good" and n >= 3:
        assert st["accepted"] >= 1 and st["rounds"] >= 1 and st["accepted_by_depth"][0] == st["attempts_by_depth"][0]
    if mode == "bad":
        assert st["accepted"] == 0
    # the state after the run is the plain state of the tokens consumed (nothing left from a discarded position)
    assert len(model.st.tokens) == model.st.pos == len(ids) + len(out) - 1


@needs_torch
def test_state_after_each_round_equals_the_plain_state():
    ids = [5, 9, 11, 2]
    n = 30
    model, g1, g3, rec = make()
    lg = prefill(model, ids)
    long = plain(ids, n + 4)[0]
    ref = long[:n]
    wrong = lambda pending, k, n_out: [(w + 1) % V for w in long[n_out:n_out + k]]
    head = FakeHead(model, None)
    p = SampleParams(temperature=0.0)
    p.max_tokens = n
    out, *_ = _mtp_decode(model, g1, g3, head, Sampler(0), p, set(), lg, None, None, n, wrong)
    assert out == ref
    # every kept token went through the head exactly once, the pending one first
    assert head.consumed_ids == ([ids[-1]] * 0 + ref[:len(head.consumed_ids)])
    # state: consumed tokens are prompt + out[:-1]
    assert model.st.tokens == ids + out[:-1] and model.st.pos == len(ids) + len(out) - 1
    m2, g12, _, _ = make()
    prefill(m2, ids)
    for t in out[:-1]:
        g12.step([t])
    assert model.st.S[0, 0, 0, 0].item() == m2.st.S[0, 0, 0, 0].item()


@needs_torch
def test_eos_stops_like_the_plain_loop_and_keeps_the_eos_token():
    ids = [1, 2, 3]
    full = plain(ids, 40)[0]
    eos = full[9]
    ref, _ = plain(ids, 40, eos=eos)
    assert ref[-1] == eos and len(ref) <= 10
    good = lambda pending, k, n_out: plain(ids, n_out + k + 1)[0][n_out:n_out + k]
    out, nan, st, model, rec, *_ = spec(ids, 40, good, eos=eos)
    assert out == ref and rec.outstanding == 0


@needs_torch
def test_sampled_run_draws_the_same_random_stream_as_the_plain_loop():
    ids = [4, 4, 8]
    prm = lambda: SampleParams(temperature=0.9, top_p=0.95, top_k=20, presence_penalty=0.3)
    ref, _ = plain(ids, 50, params=prm(), seed=7)
    for fn in (lambda pending, k, n_out: [1, 2], lambda pending, k, n_out: plain(ids, n_out + k + 1, params=prm(), seed=7)[0][n_out:n_out + k]):
        out, *_ = spec(ids, 50, fn, params=prm(), seed=7)
        assert out == ref


@needs_torch
def test_context_end_switches_to_one_row_steps():
    ids = [1, 2, 3]
    model, g1, g3, rec = make()
    model.max_ctx = 11
    lg = prefill(model, ids)
    p = SampleParams(temperature=0.0)
    p.max_tokens = 9
    good = lambda pending, k, n_out: plain(ids, n_out + k + 1)[0][n_out:n_out + k]
    out, nan, _, st = _mtp_decode(model, g1, g3, FakeHead(model, None), Sampler(0), p, set(), lg, None, None, 9, good)
    assert out == plain(ids, 9)[0] and st["plain_steps"] >= 1 and model.st.pos <= 11 and rec.outstanding == 0
    assert g3.log and all(len(r) == 3 for r in g3.log)
