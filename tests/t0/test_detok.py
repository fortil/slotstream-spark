"""Incremental detokenizing (step 84) and the engine adapter's request order, on stubs: no model, no GPU."""
import random
import threading
from types import SimpleNamespace

import pytest

from slotstream_spark.engine.pressure import GenerationGate, PressureBoundary
from slotstream_spark.engine.sampler import SampleParams
from slotstream_spark.server import detok
from slotstream_spark.server.engine_adapter import EngineAdapter
from slotstream_spark.server.engine_api import EngineFailure, GenParams, PromptRequest

EOS = 9999


def byte_decode(ids):
    """Byte-level decoding: every id below 256 is one byte; a partial UTF-8 sequence becomes U+FFFD."""
    return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


def to_ids(text):
    return list(text.encode("utf-8"))


def stream(text, stop=(), decode=byte_decode):
    d = detok.IncrementalDecoder(decode, stop)
    parts = []
    for t in to_ids(text):
        parts.append(d.push(t))
        if d.stopped is not None:
            break
    parts.append(d.finish())
    return parts, d


# ---- the decoder ------------------------------------------------------------------------------------------
def test_ascii_streams_one_piece_per_token():
    parts, d = stream("hello")
    assert parts[:-1] == list("hello") and d.text == "hello"


def test_a_split_character_is_held_until_complete():
    text = "a\U0001F600bé"        # a, a 4-byte emoji, b, a 2-byte letter
    parts, _ = stream(text)
    assert "".join(parts) == text
    assert not any("�" in p for p in parts)
    # the four bytes of the emoji produce nothing until the last one
    assert parts[1:4] == ["", "", ""] and parts[4] == "\U0001F600"


def test_stop_sequence_split_across_tokens_is_cut_and_not_emitted():
    parts, d = stream("one two END three", stop=["END"])
    assert "".join(parts) == "one two " and d.stopped == "END"
    assert d.push(ord("x")) == ""            # nothing after the stop


def test_held_prefix_of_a_stop_is_flushed_when_it_never_completes():
    parts, d = stream("ab EN", stop=["END"])
    assert "".join(parts) == "ab EN" and d.stopped is None
    assert "EN" not in "".join(parts[:-1])    # EN was held back until finish


def test_earliest_of_several_stops_wins():
    parts, d = stream("x STOP2 y STOP1", stop=["STOP1", "STOP2"])
    assert "".join(parts) == "x " and d.stopped == "STOP2"


def test_incomplete_character_at_the_end_is_the_replacement_character():
    d = detok.IncrementalDecoder(byte_decode)
    out = "".join(d.push(t) for t in to_ids("ok")) + d.push(0xF0) + d.finish()
    assert out == "ok�"


def test_streaming_equals_decoding_once():
    rng = random.Random(7)
    alphabet = "ab é\U0001F600日本 END\n"
    for _ in range(200):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        stop = rng.choice([(), ("END",), ("b é", "日")])
        parts, d = stream(text, stop)
        once, matched = detok.decode_once(byte_decode, to_ids(text), stop)
        assert "".join(parts) == once, (text, stop)
        assert d.stopped == matched


def test_decode_once_cuts_at_the_earliest_stop():
    assert detok.decode_once(byte_decode, to_ids("abcdef"), ["de", "cd"]) == ("ab", "cd")
    assert detok.decode_once(byte_decode, to_ids("abc"), []) == ("abc", None)


# ---- the adapter ------------------------------------------------------------------------------------------
class StubTokenizer:
    def encode(self, text):
        return to_ids(text) or [0]

    decode = staticmethod(byte_decode)


class StubTemplate:
    pass


class StubLedger:
    trunk = 1000

    def expected_peak_bytes(self, chunks):
        return 5000 + int(chunks)


class StubGenerator:
    def __init__(self, tokens, reused=0, boom=None):
        self.tokens, self.reused, self.boom = list(tokens), reused, boom
        self.calls = []
        self.order = []

    def reusable(self, ids):
        self.order.append("reusable")
        return self.reused

    def run(self, ids, sample, seed, max_tokens, on_token, should_continue):
        self.order.append("run")
        self.calls.append((list(ids), sample, seed, max_tokens))
        out = []
        for t in self.tokens:
            if len(out) >= max_tokens:
                break
            if self.boom:
                raise self.boom
            on_token(t)
            out.append(t)
            if t == EOS:
                break
        return {"tokens": out, "nan": False, "timings": {"prefill_s": 0.5, "decode_s": [0.1] * len(out)},
                "prefix": {"reused": self.reused}}


def make(tokens, *, reused=0, window=64, cap=10 ** 9, boom=None, max_ctx=None):
    gov = SimpleNamespace(gate=GenerationGate(), boundary=PressureBoundary())
    plan = SimpleNamespace(window=window, chunks=4)
    gen = StubGenerator(tokens, reused, boom)
    ad = EngineAdapter(gen, StubTokenizer(), StubTemplate(), StubLedger(), gov, plan=plan, window=window,
                       max_ctx=max_ctx, cap_fn=lambda: cap, eos=EOS)
    return ad, gen, gov


def run(ad, ids=(1, 2, 3), *, max_tokens=50, stop=(), cont=lambda: True):
    events = []
    p = GenParams(SampleParams(temperature=0.0, max_tokens=max_tokens, stop=list(stop)), seed=5)
    res = ad.generate(list(ids), p, on_admitted=lambda r: events.append(("admitted", r)),
                      on_token=lambda t: events.append(("token", t)), should_continue=cont)
    return res, events


def test_admission_fires_before_the_first_token_and_carries_reused():
    ad, gen, _ = make(to_ids("hi") + [EOS], reused=2)
    res, ev = run(ad)
    assert ev[0] == ("admitted", 2) and [e[1] for e in ev[1:]] == ["h", "i"]
    assert res.finish_reason == "stop" and res.text == "hi" and res.reused_prefix_tokens == 2
    assert res.prompt_tokens == 3 and res.decode_tokens == 3 and res.prefill_seconds == 0.5


def test_length_finish_when_max_tokens_runs_out():
    ad, _, _ = make(to_ids("abcdef"))
    res, ev = run(ad, max_tokens=3)
    assert res.finish_reason == "length" and res.text == "abc"


def test_stop_sequence_ends_the_loop_and_is_reported():
    ad, _, _ = make(to_ids("abc END more"))
    res, ev = run(ad, stop=["END"])
    assert res.finish_reason == "stop" and res.stop_sequence == "END" and res.text == "abc "
    assert "".join(e[1] for e in ev if e[0] == "token") == "abc "


def test_sampler_params_are_sanitized_copies_and_the_seed_travels():
    ad, gen, _ = make(to_ids("x"))
    p = GenParams(SampleParams(temperature=0.5, max_tokens=7), seed=123)
    ad.generate([1], p, on_admitted=lambda r: None, on_token=lambda t: None, should_continue=lambda: True)
    ids, sample, seed, max_tokens = gen.calls[0]
    assert seed == 123 and max_tokens == 7 and sample is not p.sample


def test_max_tokens_is_clamped_to_the_context():
    ad, gen, _ = make(to_ids("abcdefgh"), max_ctx=8)
    run(ad, ids=(1, 2, 3, 4, 5), max_tokens=100)
    assert gen.calls[0][3] == 3


def test_prompt_over_the_window_is_refused_before_anything_runs():
    ad, gen, _ = make(to_ids("x"), window=4)
    with pytest.raises(EngineFailure) as e:
        run(ad, ids=(1, 2, 3, 4))
    assert e.value.code == "context_length_exceeded" and gen.calls == []


def test_memory_admission_refusal_fires_no_admission():
    ad, gen, _ = make(to_ids("x"), cap=10)
    seen = []
    with pytest.raises(EngineFailure) as e:
        ad.generate([1], GenParams(SampleParams(), 1), on_admitted=seen.append, on_token=lambda t: None,
                    should_continue=lambda: True)
    assert e.value.code == "insufficient_memory" and seen == [] and gen.calls == []


def test_gate_is_released_after_success_and_after_failure():
    ad, gen, gov = make(to_ids("a"))
    run(ad)
    assert not gov.gate.held()
    ad2, _, gov2 = make([], boom=RuntimeError("x"))
    ad2.generator.tokens = [1]
    with pytest.raises(RuntimeError):
        run(ad2)
    assert not gov2.gate.held()


def test_client_cancel_stops_between_tokens():
    ad, _, _ = make(to_ids("abcdef"))
    count = {"n": 0}

    def cont():
        count["n"] += 1
        return count["n"] <= 2

    res, ev = run(ad, cont=cont)
    assert res.finish_reason == "length" and res.decode_tokens < 6


def test_memory_pressure_raises_insufficient_memory():
    ad, gen, gov = make(to_ids("abcdef"))
    gov.boundary.request("rule 6")
    with pytest.raises(EngineFailure) as e:
        run(ad)
    assert e.value.code == "insufficient_memory" and not gov.gate.held()


def test_snapshot_never_waits_for_the_gate():
    ad, _, gov = make(to_ids("a"))
    assert gov.gate.acquire(0)                      # a generation is running
    got = []
    t = threading.Thread(target=lambda: got.append(ad.snapshot()))
    t.start()
    t.join(1.0)
    assert got and got[0].footprint_bytes == 5004
    gov.gate.release()


def test_render_and_count_use_the_engine_tokenizer():
    ad, _, _ = make(to_ids("x"))
    req = PromptRequest(lambda tok: "abc def")
    assert ad.render_and_tokenize(req) == to_ids("abc def") and ad.count_tokens(req) == 7
