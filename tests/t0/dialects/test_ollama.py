"""Ollama CLI shapes (chat, generate, show, tags, ps) on the fake engine, plus the rules the server learned from
the CLI's own requests."""
import datetime
import json

import pytest

from slotstream_spark.server.common import DialectError
from slotstream_spark.server.dialects import DIALECTS, ollama as ol

INFO = ol.ModelInfo("qwen38", digest="sha256:abc", weights_bytes=104_000, resident_bytes=9_000, max_context=32768)
FIXED = datetime.datetime(2026, 10, 14, 12, 0, 0, tzinfo=datetime.timezone.utc)


def now():
    return FIXED


def refused(fn, *args, status=400):
    with pytest.raises(DialectError) as err:
        fn(*args)
    assert err.value.status == status
    return err.value.message


def test_dialect_registry():
    assert DIALECTS["ollama"] is ol


# ---- chat --------------------------------------------------------------------------------------------

def test_chat_defaults_stream_and_instruct_sampling():
    p = ol.prepare_chat({"model": "qwen38", "messages": [{"role": "user", "content": "hi"}], "options": None,
                         "keep_alive": "5m"}, INFO)
    assert p.kind == "chat" and p.stream and not p.thinking
    assert (p.params.temperature, p.params.top_k, p.params.presence_penalty) == (0.7, 20, 1.5)


def test_chat_options_map_to_sampling():
    p = ol.prepare_chat({"messages": [{"role": "user", "content": "hi"}], "think": True, "stream": False, "options": {
        "temperature": 0.1, "top_p": 0.5, "top_k": 3, "min_p": 0.05, "presence_penalty": 0.5, "num_predict": 64,
        "seed": 9, "stop": "END"}}, INFO)
    assert p.thinking and not p.stream and p.seed == 9
    assert (p.params.temperature, p.params.top_p, p.params.top_k, p.params.min_p, p.params.max_tokens) \
        == (0.1, 0.5, 3, 0.05, 64)
    assert p.params.stop == ["END"]


def test_seed_minus_one_is_random_and_num_predict_minus_one_is_the_ceiling():
    p = ol.prepare_chat({"messages": [{"role": "user", "content": "hi"}], "options": {"seed": -1, "num_predict": -1}},
                        INFO)
    assert p.seed > 0 and p.params.max_tokens == 32_768


def test_empty_messages_is_the_load_request():
    r = ol.prepare_chat({"model": "qwen38", "messages": []}, INFO, now)
    assert isinstance(r, ol.Load) and r.body["done_reason"] == "load" and r.body["created_at"] == "2026-10-14T12:00:00Z"


@pytest.mark.parametrize("body,needle", [
    ({"model": "other", "messages": [{"role": "user", "content": "x"}]}, "not loaded"),
    ({"messages": [{"role": "user", "content": "x"}], "bogus": 1}, "unsupported request field(s): bogus"),
    ({"messages": [{"role": "user", "content": "x"}], "options": {"num_ctx": 4096}}, "num_ctx"),
    ({"messages": [{"role": "user", "content": "x"}], "options": {"repeat_penalty": 1.1}}, "repeat_penalty"),
    ({"messages": [{"role": "user", "content": "x"}], "options": {"top_k": 1.5}}, "options.top_k must be an integer"),
    ({"messages": [{"role": "user", "content": "x"}], "think": "high"}, "named reasoning levels"),
    ({"messages": [{"role": "user", "content": "x"}], "stream": 1}, "stream must be true or false"),
    ({"messages": [{"role": "user", "content": "x", "extra": 1}]}, "unsupported field(s): extra"),
    ({"messages": [{"role": "tool", "content": "x"}]}, "/v1/chat/completions"),
    ({"messages": [{"role": "user", "content": "x"}], "tools": []}, "/v1/chat/completions"),
    ({"messages": "x"}, "messages must be an array"),
    ({"messages": [{"role": "robot", "content": "x"}]}, "role must be"),
    ({"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]}, "must end with a user"),
])
def test_chat_refusals(body, needle):
    assert needle in refused(ol.prepare_chat, body, INFO)


def test_chat_images_are_a_400():
    with pytest.raises(DialectError) as err:
        ol.prepare_chat({"messages": [{"role": "user", "content": "x", "images": ["abc"]}]}, INFO)
    assert err.value.status == 400 and err.value.code == "invalid_image" and "--vision off" in err.value.message
    with pytest.raises(DialectError) as err:
        ol.prepare_chat({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "u"}}]}]}, INFO)
    assert err.value.code == "invalid_image"


def test_chat_content_parts_join():
    p = ol.prepare_chat({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]}, INFO)
    assert p.messages[0].content == "ab"


def test_chat_prompt_renders_with_thinking_state(tokenizer):
    off = ol.prepare_chat({"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]}, INFO)
    on = ol.prepare_chat({"messages": [{"role": "user", "content": "hi"}], "think": True}, INFO)
    assert off.prompt_text(tokenizer).endswith("<think>\n\n</think>\n\n") and "S" in off.prompt_text(tokenizer)
    assert on.prompt_text(tokenizer).endswith("<think>\n")


# ---- generate ----------------------------------------------------------------------------------------

def test_generate_prompt_and_system_become_messages():
    p = ol.prepare_generate({"model": "qwen38", "prompt": "Why?", "system": "Be brief.", "suffix": "",
                             "template": ""}, INFO)
    assert [m.role for m in p.messages] == ["system", "user"] and p.raw_prompt is None and p.stream


def test_generate_raw_goes_to_the_tokenizer_untouched(tokenizer):
    p = ol.prepare_generate({"prompt": "<|im_start|>user\nhi", "raw": True}, INFO)
    assert p.raw_prompt == "<|im_start|>user\nhi" and p.messages == []
    assert p.prompt_text(tokenizer) == "<|im_start|>user\nhi"


def test_generate_empty_prompt_is_the_load_request():
    r = ol.prepare_generate({"model": "qwen38", "prompt": ""}, INFO, now)
    assert isinstance(r, ol.Load) and r.body == {"model": "qwen38", "created_at": "2026-10-14T12:00:00Z",
                                                  "response": "", "done": True, "done_reason": "load"}
    assert isinstance(ol.prepare_generate({"model": "qwen38"}, INFO, now), ol.Load)


@pytest.mark.parametrize("body,needle", [
    ({"prompt": "x", "suffix": "tail"}, "fill-in-the-middle"),
    ({"prompt": "x", "template": "{{ .Prompt }}"}, "template overrides"),
    ({"prompt": 1}, "prompt must be text"),
    ({"prompt": "x", "system": 1}, "system must be text"),
    ({"prompt": "x", "raw": "yes"}, "raw must be true or false"),
    ({"prompt": "x", "raw": True, "system": "s"}, "raw generation cannot apply"),
    ({"prompt": "x", "raw": True, "think": True}, "raw generation cannot apply"),
    ({"prompt": "x", "images": "abc"}, "images must be an array"),
    ({"prompt": "x", "options": {"num_ctx": 1}}, "num_ctx"),
    ({"prompt": "x", "model": "nope"}, "not loaded"),
])
def test_generate_refusals(body, needle):
    assert needle in refused(ol.prepare_generate, body, INFO)


def test_generate_images_are_a_400():
    with pytest.raises(DialectError) as err:
        ol.prepare_generate({"prompt": "x", "images": ["abc"]}, INFO)
    assert err.value.status == 400 and err.value.code == "invalid_image"
    ol.prepare_generate({"prompt": "x", "images": []}, INFO)


# ---- replies on the fake engine ----------------------------------------------------------------------

def drive(fx, prepared, text, chunk=3):
    reply = ol.OllamaReply(prepared, "qwen38", now)
    frames = []
    engine = fx.FakeEngine(text, chunk=chunk, reused_prefix_tokens=12)

    def on_token(_, delta):
        f = reply.delta(delta)
        if f is not None and prepared.stream:
            frames.append(f)
        return True
    out, _, stats = engine.generate("p", on_token=on_token)
    stats.prefill_seconds, stats.decode_seconds = 0.5, 2.0
    final = reply.final(stats, out, 3.0)
    return frames, final


def test_streamed_chat_frames_and_final(fx):
    p = ol.prepare_chat({"messages": [{"role": "user", "content": "hi"}]}, INFO)
    frames, final = drive(fx, p, "Hello there.")
    assert "".join(f["message"]["content"] for f in frames) == "Hello there."
    assert all(f["done"] is False and f["message"]["role"] == "assistant" for f in frames)
    assert final["done"] and final["message"]["content"] == "" and final["done_reason"] == "stop"
    assert (final["prompt_eval_count"], final["eval_count"]) == (len("p"), len(frames))
    assert final["prompt_eval_duration"] == 500_000_000 and final["eval_duration"] == 2_000_000_000
    assert final["total_duration"] == 3_000_000_000 and final["cached_tokens"] == 12
    line = ol.ndjson(final)
    assert line.endswith("\n") and json.loads(line) == final


def test_non_streamed_chat_is_one_body(fx):
    p = ol.prepare_chat({"messages": [{"role": "user", "content": "hi"}], "stream": False}, INFO)
    frames, final = drive(fx, p, "Whole reply.")
    assert frames == [] and final["message"]["content"] == "Whole reply." and final["done"]


def test_thinking_goes_in_its_own_field_streamed_and_not(fx):
    text = "Sum: 391.</think>\n\nAnswer 391."
    p = ol.prepare_chat({"messages": [{"role": "user", "content": "hi"}], "think": True}, INFO)
    frames, final = drive(fx, p, text, chunk=2)
    think = "".join(f["message"].get("thinking", "") for f in frames) + final["message"].get("thinking", "")
    answer = "".join(f["message"]["content"] for f in frames) + final["message"]["content"]
    assert (think, answer) == ("Sum: 391.", "Answer 391.")
    p = ol.prepare_chat({"messages": [{"role": "user", "content": "hi"}], "think": True, "stream": False}, INFO)
    _, final = drive(fx, p, text)
    assert final["message"]["thinking"] == "Sum: 391." and final["message"]["content"] == "Answer 391."


def test_generate_frames_use_response_and_thinking_at_the_top(fx):
    p = ol.prepare_generate({"prompt": "hi", "think": True}, INFO)
    frames, final = drive(fx, p, "why</think>\n\nbecause", chunk=2)
    think = "".join(f.get("thinking", "") for f in frames) + final.get("thinking", "")
    answer = "".join(f["response"] for f in frames) + final["response"]
    assert (think, answer) == ("why", "because") and "message" not in final


def test_unclosed_reasoning_is_all_thinking(fx):
    p = ol.prepare_generate({"prompt": "hi", "think": True, "stream": False}, INFO)
    _, final = drive(fx, p, "still thinking")
    assert final["thinking"] == "still thinking" and final["response"] == ""


# ---- metadata ----------------------------------------------------------------------------------------

def test_show_reports_completion_only():
    body = ol.show({"model": "qwen38", "name": "", "system": "", "template": "", "options": {}, "verbose": False},
                   INFO)
    assert body["capabilities"] == ["completion"] and body["parameters"] == "num_ctx 32768"
    assert body["model_info"]["qwen4_exp.context_length"] == 32768


def test_show_accepts_the_deprecated_name_alias_in_both_directions():
    assert ol.show({"name": "qwen38"}, INFO)["capabilities"] == ["completion"]
    assert ol.show({"model": "", "name": "qwen38"}, INFO)["capabilities"] == ["completion"]
    assert ol.show({"model": "qwen38", "name": ""}, INFO)["capabilities"] == ["completion"]


@pytest.mark.parametrize("body,status,needle", [
    ({"model": "nope"}, 404, "not loaded"),
    ({"model": "qwen38", "system": "x"}, 400, "system overrides"),
    ({"model": "qwen38", "template": "x"}, 400, "template overrides"),
    ({"model": "qwen38", "options": {"temperature": 1}}, 400, "options overrides"),
    ({"model": "qwen38", "verbose": "yes"}, 400, "verbose"),
    ({"model": "qwen38", "bogus": 1}, 400, "unsupported request field(s): bogus"),
])
def test_show_refusals(body, status, needle):
    assert needle in refused(ol.show, body, INFO, status=status)


def test_tags_and_ps():
    t = ol.tags(INFO, now)["models"][0]
    assert t["name"] == t["model"] == "qwen38" and t["size"] == 104_000 and "expires_at" not in t
    assert t["modified_at"] == "2026-10-14T12:00:00Z" and t["digest"] == "sha256:abc"
    p = ol.ps(INFO, now)["models"][0]
    assert p["size"] == p["size_vram"] == 9_000 and p["expires_at"] == "2026-10-14T13:00:00Z"
    assert ol.version_body("0.2.28") == {"version": "0.2.28"}
    assert ol.error_body(DialectError("boom")) == {"error": "boom"}


# ---- scalar validation audit ------------------------------------------------------------------------

CHAT = {"model": "qwen38", "messages": [{"role": "user", "content": "hi"}]}


@pytest.mark.parametrize("key", ["temperature", "top_p", "min_p", "presence_penalty"])
@pytest.mark.parametrize("bad", ["abc", True, [1], {"a": 1}, 1e39], ids=repr)
def test_bad_option_numbers_are_a_400(key, bad):
    assert refused(ol.prepare_chat, {**CHAT, "options": {key: bad}}, INFO) == f"options.{key} must be a number"
    assert refused(ol.prepare_generate, {"prompt": "x", "options": {key: bad}}, INFO) \
        == f"options.{key} must be a number"


@pytest.mark.parametrize("key", ["top_k", "num_predict", "seed"])
@pytest.mark.parametrize("bad", ["abc", True, 1.5, [1], {"a": 1}, 1e39], ids=repr)
def test_bad_option_integers_are_a_400(key, bad):
    assert refused(ol.prepare_chat, {**CHAT, "options": {key: bad}}, INFO) == f"options.{key} must be an integer"


@pytest.mark.parametrize("key", ["stream", "think"])
def test_bad_booleans_are_a_400(key):
    refused(ol.prepare_chat, {**CHAT, key: "yes"}, INFO)
    refused(ol.prepare_generate, {"prompt": "x", key: "yes"}, INFO)


@pytest.mark.parametrize("key", ["raw"])
def test_bad_generate_boolean_is_a_400(key):
    refused(ol.prepare_generate, {"prompt": "x", key: 1}, INFO)


def test_bad_show_verbose_is_a_400():
    refused(ol.show, {"model": "qwen38", "verbose": "yes"}, INFO)


def test_min_p_option_is_applied():
    p = ol.prepare_chat({**CHAT, "options": {"min_p": 0.05}}, INFO)
    assert p.params.min_p == 0.05
