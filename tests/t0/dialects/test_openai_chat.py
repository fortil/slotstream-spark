"""Ports of openai-conversation and openai-tool-output (OpenAIChecks.swift), plus the request/reply flow of the
Chat Completions dialect on the fake engine."""
import json

import pytest

from slotstream_spark.server import common, template
from slotstream_spark.server import tool_calls as tc
from slotstream_spark.server.common import DialectError, ToolChoice
from slotstream_spark.server.dialects import DIALECTS, openai_chat as oc
from slotstream_spark.server.output import OutputAccumulator

READ_TOOL = {"type": "function", "function": {"name": "read_file", "description": "Read a file.", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}},
    "required": ["path"]}}}
USER = {"role": "user", "content": "Read hello.txt"}
CALL = {"id": "call_42", "type": "function",
        "function": {"name": "read_file", "arguments": '{"path":"hello.txt","start_line":1}'}}
ASSISTANT = {"role": "assistant", "content": None, "reasoning_content": "I need the file.", "tool_calls": [CALL]}
RESULT = {"role": "tool", "tool_call_id": "call_42", "content": "fixture42"}
LIMIT = 65_536


def parse(**extra):
    body = {"messages": [USER], "tools": [READ_TOOL], **extra}
    return oc.conversation(body, LIMIT)


def rejected(**extra):
    with pytest.raises(DialectError):
        parse(**extra)


def test_dialect_registry():
    assert DIALECTS["openai_chat"] is oc


# ---- openai-conversation ----------------------------------------------------------------------------

def test_stock_sdk_no_op_defaults_are_accepted():
    assert oc.no_op_error({"n": 1, "frequency_penalty": 0, "logprobs": False, "logit_bias": {},
                           "response_format": {"type": "text"}, "user": "local-user"}) is None
    for bad in ({"n": 2}, {"frequency_penalty": 1}, {"logprobs": True}, {"top_logprobs": 1},
                {"logit_bias": {"42": 1}}, {"user": 123}, {"n": True}):
        assert oc.no_op_error(bad) is not None, bad


def test_stateless_fields_are_read_and_checked():
    assert {"store", "metadata", "prompt_cache_key", "prompt_cache_retention", "safety_identifier",
            "service_tier", "min_p"} <= oc.FIELDS
    assert oc.no_op_error({"store": False}) is None
    assert oc.no_op_error({"metadata": {"session": "a"}, "prompt_cache_key": "k", "prompt_cache_retention": "24h",
                           "safety_identifier": "s", "service_tier": "auto"}) is None
    assert "store: true" in oc.no_op_error({"store": True})
    for bad in ({"store": "no"}, {"store": 0}, {"metadata": {"n": 1}}, {"metadata": "x"}, {"prompt_cache_key": 1},
                {"prompt_cache_retention": 24}, {"safety_identifier": False}, {"service_tier": ["auto"]}):
        assert next(iter(bad)) in oc.no_op_error(bad), bad
    assert "previous_response_id" not in oc.FIELDS


def test_constrained_output_error_names_the_field():
    assert "response_format is not supported" in oc.no_op_error({"response_format": {"type": "json_schema"}})


def test_json_object_accepted_adds_instruction():
    assert oc.no_op_error({"response_format": {"type": "json_object"}}) is None
    p = oc.prepare({"messages": [USER], "response_format": {"type": "json_object"}}, LIMIT)
    assert p.messages[0].role == "system" and p.messages[0].content == common.JSON_OBJECT_INSTRUCTION
    assert p.messages[1].content == "Read hello.txt"
    withsys = oc.prepare({"messages": [{"role": "system", "content": "Be brief."}, USER],
                          "response_format": {"type": "json_object"}}, LIMIT)
    assert withsys.messages[0].content == common.JSON_OBJECT_INSTRUCTION + "\n\nBe brief."
    plain = oc.prepare({"messages": [USER], "response_format": {"type": "text"}}, LIMIT)
    assert all(m.role != "system" for m in plain.messages)


def test_json_object_forces_thinking_off():
    p = oc.prepare({"messages": [USER], "reasoning_effort": "high", "response_format": {"type": "json_object"}},
                   LIMIT)
    assert p.conversation.thinking is False and p.conversation.effort is None
    assert p.params.temperature == 0.7


def test_json_schema_still_rejected():
    assert oc.validation_error({"messages": [USER], "response_format": {"type": "json_schema"}},
                               "slotstream-test", LIMIT) is not None


def test_json_object_max_tokens_null_is_unset():
    p = oc.prepare({"messages": [USER], "max_tokens": None, "response_format": {"type": "json_object"}}, LIMIT)
    assert p.params.max_tokens == common.output_budget(LIMIT)


def test_unknown_fields_are_refused_by_name():
    err = oc.validation_error({"model": "m", "messages": [USER], "bogus": 1, "other": 2}, "m", LIMIT)
    assert err == "unsupported request field(s): bogus, other"


def test_a_conversation_with_options_and_reasoning():
    c = parse(reasoning_effort="none", think=False, options={"num_ctx": 64_000}, parallel_tool_calls=False)
    assert c.context_limit == 64_000 and not c.thinking and not c.parallel
    assert c.tools[0].name == "read_file"


def test_history_keeps_call_ids_arguments_and_reasoning():
    c = parse(messages=[USER, ASSISTANT, RESULT, USER])
    assert c.messages[1].tool_calls[0].id == "call_42"
    assert c.messages[1].tool_calls[0].arguments["start_line"] == 1
    assert type(c.messages[1].tool_calls[0].arguments["start_line"]) is int
    assert c.messages[2].tool_name == "read_file"
    assert c.messages[1].reasoning == "I need the file."


def test_out_of_order_results_follow_call_order():
    second = dict(CALL, id="call_43")
    two = dict(ASSISTANT, tool_calls=[CALL, second])
    second_result = {"role": "tool", "tool_call_id": "call_43", "content": "second fixture"}
    c = parse(messages=[USER, two, second_result, RESULT, USER])
    assert c.messages[2].tool_call_id == "call_42" and c.messages[2].content == "fixture42"
    assert c.messages[3].content == "second fixture"


def test_history_survives_removed_current_tools():
    parse(messages=[USER, ASSISTANT, RESULT], tools=[])


@pytest.mark.parametrize("extra", [
    {"messages": [USER, ASSISTANT]},                                  # missing result
    {"messages": [USER, RESULT]},                                     # orphan result
    {"messages": [USER, ASSISTANT, RESULT, RESULT]},                  # duplicate result
    {"messages": [USER, ASSISTANT, USER, RESULT]},                    # intervening user
    {"messages": [USER, ASSISTANT, dict(RESULT, name="different")]},  # wrong function name
    {"parallel_tool_calls": 1},                                       # numeric boolean
    {"reasoning_effort": "none", "think": True},                      # conflicting reasoning
    {"options": {"num_ctx": 1_000_000}},                              # inflated context
    {"options": {"num_ctx": 100.5}},                                  # fractional context
    {"options": {"num_ctx": True}},
    {"options": {"num_predict": 10}},                                 # unknown option
    {"tool_choice": "required", "tools": []},
    {"tool_choice": {"type": "function", "function": {"name": "missing"}}},
    {"messages": [dict(USER, name="extra")]},
])
def test_refusals(extra):
    rejected(**extra)


def test_strict_schema_is_not_invented():
    strict = {"type": "function", "function": dict(READ_TOOL["function"], strict=True)}
    rejected(tools=[strict])
    loose = {"type": "function", "function": dict(READ_TOOL["function"], strict=False)}
    parse(tools=[loose])


def test_system_and_developer_join_at_the_front():
    c = parse(messages=[{"role": "system", "content": "One"}, {"role": "developer", "content": "Two"}, USER])
    assert c.messages[0].content == "One\n\nTwo" and c.messages[0].role == "system"


def test_late_system_message_is_refused():
    rejected(messages=[USER, {"role": "system", "content": "late"}])


def test_null_fields_mean_not_set():
    c = oc.conversation({"messages": [USER], "tools": None, "max_tokens": None, "tool_choice": None}, LIMIT)
    assert c.tools == []


# ---- images -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("content", [
    [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}],
    [{"type": "text", "text": "x"}, {"type": "image_url", "image_url": "u"}],
    [{"image": "u"}],
])
def test_image_parts_are_a_400_invalid_image(content):
    with pytest.raises(DialectError) as err:
        oc.conversation({"messages": [{"role": "user", "content": content}]}, LIMIT)
    assert err.value.status == 400 and err.value.code == "invalid_image" and err.value.message == common.VISION_OFF
    body = oc.refusal_body(err.value)
    assert body["error"]["code"] == "invalid_image" and body["error"]["type"] == "invalid_request_error"


def test_images_field_is_refused_too():
    with pytest.raises(DialectError) as err:
        oc.conversation({"messages": [dict(USER, images=["abc"])]}, LIMIT)
    assert err.value.code == "invalid_image"
    oc.conversation({"messages": [dict(USER, images=[])]}, LIMIT)


def test_text_parts_are_joined_in_order():
    c = oc.conversation({"messages": [{"role": "user", "content": [
        {"type": "text", "text": "Hello, "}, {"type": "text", "text": "world."}]}]}, LIMIT)
    assert c.messages[0].content == "Hello, world."


# ---- prepare ----------------------------------------------------------------------------------------

def test_prepare_presets_and_overrides():
    plain = oc.prepare({"messages": [USER]}, LIMIT)
    assert (plain.params.temperature, plain.params.presence_penalty) == (0.7, 1.5)
    assert plain.params.max_tokens == common.output_budget(LIMIT) == 8192
    agent = oc.prepare({"messages": [USER], "tools": [READ_TOOL], "temperature": 0.5}, LIMIT)
    assert (agent.params.temperature, agent.params.presence_penalty) == (0.5, 0.0)
    thinking = oc.prepare({"messages": [USER], "think": True}, LIMIT)
    assert (thinking.params.temperature, thinking.params.top_p) == (1.0, 0.95)
    limited = oc.prepare({"messages": [USER], "max_tokens": 100, "max_completion_tokens": 50, "seed": 7,
                          "stop": ["END"], "stream": True, "stream_options": {"include_usage": True}}, LIMIT)
    assert limited.params.max_tokens == 50 and limited.seed == 7 and limited.params.stop == ["END"]
    assert limited.stream and limited.want_usage


def test_unseeded_requests_get_a_random_seed_at_the_boundary():
    seeds = {oc.prepare({"messages": [USER]}, LIMIT).seed for _ in range(5)}
    assert len(seeds) > 1 and all(s > 0 for s in seeds)


def test_instructions_follow_tool_choice_and_parallelism():
    p = oc.prepare({"messages": [USER], "tools": [READ_TOOL], "tool_choice": "required",
                    "parallel_tool_calls": False}, LIMIT)
    system = p.messages[0].content
    assert "You must call one of the available tools now." in system and "Call at most one tool" in system
    named = oc.prepare({"messages": [USER], "tools": [READ_TOOL], "tool_choice": {
        "type": "function", "function": {"name": "read_file"}}}, LIMIT)
    assert "You must call the read_file tool now." in named.messages[0].content
    off = oc.prepare({"messages": [USER], "tools": [READ_TOOL], "tool_choice": "none"}, LIMIT)
    assert off.render_tools == []


def test_prompt_room_is_checked_against_the_requested_context():
    p = oc.prepare({"messages": [USER], "options": {"num_ctx": 1000}}, LIMIT)
    p.check_prompt(900)
    assert p.params.max_tokens == 100
    with pytest.raises(DialectError) as err:
        p.check_prompt(1000)
    assert err.value.code == "context_length_exceeded"


def test_render_prompt_uses_the_template_layer(tokenizer):
    p = oc.prepare({"messages": [{"role": "system", "content": "Be brief."}, USER], "tools": [READ_TOOL]}, LIMIT)
    text = oc.render_prompt(p, tokenizer)
    assert "Be brief." in text and '"name": "read_file"' in text and text.endswith("<think>\n\n</think>\n\n")
    thinking = oc.render_prompt(oc.prepare({"messages": [USER], "reasoning_effort": "low"}, LIMIT), tokenizer)
    assert "Reasoning effort is set to low" in thinking and thinking.endswith("<think>\n")


# ---- openai-tool-output -----------------------------------------------------------------------------

@pytest.fixture
def tools():
    return parse().tools


def output(tools, parallel=True, choice=common.AUTO, **kw):
    return OutputAccumulator(tools, choice, parallel, **kw)


def canonical(message):
    return json.dumps(message, sort_keys=True)


def test_a_complete_call_finishes_with_tool_calls(fx, tools):
    complete = output(tools)
    complete.consume(tc.ToolCallSplitter.parse_all(fx.prose_then_call, [t.schema for t in tools], fx.ids()))
    assert complete.finish_reason("stop") == "tool_calls"
    assert complete.calls[0].input_json == '{"path":"hello.txt","start_line":1}'


def test_every_split_gives_the_same_message(fx, tools):
    schemas = [t.schema for t in tools]
    complete = output(tools)
    complete.consume(tc.ToolCallSplitter.parse_all(fx.prose_then_call, schemas, fx.ids()))
    expected = canonical(complete.message)
    text = fx.prose_then_call
    for cut in range(len(text) + 1):
        parser = tc.ToolCallSplitter(schemas, fx.ids())
        streamed = output(tools)
        streamed.consume(parser.push(text[:cut]))
        streamed.consume(parser.push(text[cut:]))
        streamed.consume(parser.flush())
        assert canonical(streamed.message) == expected, cut


def test_wire_arguments_reassemble_at_every_split(fx, tools):
    schemas = [t.schema for t in tools]
    complete = output(tools)
    complete.consume(tc.ToolCallSplitter.parse_all(fx.prose_then_call, schemas, fx.ids()))
    text = fx.prose_then_call
    for cut in range(len(text) + 1):
        parser = tc.ToolCallSplitter(schemas, fx.ids())
        out = output(tools, choice=common.REQUIRED, stream_tool_arguments=True, allow_length_truncation=True)
        events = out.consume(parser.push(text[:cut])) + out.consume(parser.push(text[cut:])) + out.consume(parser.flush())
        wire = [c for e in events for c in e.get("tool_calls", [])]
        assert "".join(c["function"]["arguments"] for c in wire if "arguments" in c.get("function", {})) \
            == complete.calls[0].input_json, cut
        assert sum(1 for c in wire if "id" in c) == 1, cut


def test_a_partial_call_is_never_published(fx, tools):
    parser = tc.ToolCallSplitter([t.schema for t in tools], fx.ids())
    partial = output(tools)
    prefix = "<tool_call>\n<function=read_file>\n<parameter=path>\nhello.txt\n</parameter>"
    provisional = partial.consume(parser.push(prefix))
    assert not partial.calls and all("tool_calls" not in d for d in provisional)
    partial.consume(parser.flush())
    assert partial.error is not None and not partial.calls


def test_parallel_false_exposes_exactly_one_call(fx, tools):
    schemas = [t.schema for t in tools]
    single = output(tools, parallel=False)
    single.consume(tc.ToolCallSplitter.parse_all(fx.one_call + fx.one_call, schemas, fx.ids()))
    assert len(single.calls) == 1 and single.finished_single_call


def test_parallel_calls_get_indices_and_distinct_ids(fx, tools):
    multiple = output(tools)
    deltas = multiple.consume(tc.ToolCallSplitter.parse_all(fx.one_call + fx.one_call,
                                                            [t.schema for t in tools], fx.ids()))
    assert [d["tool_calls"][0]["index"] for d in deltas if "tool_calls" in d] == [0, 1]
    assert multiple.calls[0].id != multiple.calls[1].id


def test_required_cannot_silently_degrade_to_text(tools):
    required = output(tools, choice=common.REQUIRED)
    required.consume([tc.Text("I forgot to call it.")])
    required.finish_reason("stop")
    assert required.error is not None


def test_undeclared_tool_is_not_exposed(tools):
    unknown = output(tools)
    unknown.consume([tc.ToolCall(tc.ParsedToolCall("bad", "undeclared", {}))])
    assert unknown.error is not None and not unknown.calls


def test_reasoning_is_separate_from_the_answer(tools):
    r = output(tools)
    r.reasoning_delta("Private deliberation.")
    r.consume([tc.Text("Answer.")])
    assert r.message["content"] == "Answer." and r.message["reasoning_content"] == "Private deliberation."


@pytest.mark.parametrize("shape", [
    {"type": "string"},
    {"anyOf": [{"type": "string"}, {"type": "null"}]},
    {"type": ["string", "null"]},
    {"type": ["null", "string"]},
])
def test_long_string_arguments_stream_and_truncate_as_length(fx, shape):
    definition = tc.ToolDefinition("read_file", "", {"type": "object", "properties": {"path": shape}})
    parser = tc.ToolCallSplitter([definition.schema], fx.ids())
    out = OutputAccumulator([definition], common.REQUIRED, True, stream_tool_arguments=True,
                            allow_length_truncation=True)
    out.consume(parser.push("<tool_call><function=read_file><parameter=path>\n"))
    span, fragments = 'a\\"\t🙂\n', 0
    for _ in range(2000):
        fragments += len(out.consume(parser.push(span)))
    assert fragments > 1900 and not out.calls
    out.consume(parser.flush())
    assert out.finish_reason("length") == "length"
    assert out.error is None and not out.calls
    pending = out.message["tool_calls"][0]["function"]
    assert pending["arguments"] == '{"path":"' + tc.quote(span * 2000)[1:-1]


def test_budget_before_a_required_call_is_length_not_refusal(tools):
    capped = output(tools, choice=common.REQUIRED, stream_tool_arguments=True, allow_length_truncation=True)
    assert capped.finish_reason("length") == "length" and capped.error is None
    invalid = output(tools, choice=common.REQUIRED, stream_tool_arguments=True, allow_length_truncation=True)
    assert invalid.consume([tc.ToolInputStart("bad", "absent")]) == []
    invalid.finish_reason("length")
    assert invalid.error is not None


# ---- reply flow on the fake engine ------------------------------------------------------------------

def run_reply(fx, body, text, chunk=3, finish="stop"):
    p = oc.prepare(body, LIMIT)
    reply = oc.ChatReply(p, "qwen", rid="chatcmpl-1", created=1, id_factory=fx.ids())
    engine = fx.FakeEngine(text, chunk=chunk, finish_reason=finish, reused_prefix_tokens=5)
    wire = []

    def on_token(_, delta):
        wire.extend(reply.consume(delta))
        return not reply.should_stop
    out, _, stats = engine.generate("prompt", p.params, on_token=on_token if reply.incremental else None,
                                    should_continue=lambda: not reply.should_stop)
    tail, finish_reason = reply.end(stats.finish_reason, text=out)
    return reply, wire + tail, finish_reason, stats


def test_streamed_text_reply(fx):
    reply, deltas, finish, stats = run_reply(fx, {"messages": [USER], "stream": True,
                                                  "stream_options": {"include_usage": True}}, "Hello there.")
    assert "".join(d.get("content", "") for d in deltas) == "Hello there." and finish == "stop"
    first = reply.chunk(deltas[0])
    assert first["choices"][0]["delta"]["role"] == "assistant"
    assert "role" not in reply.chunk(deltas[1])["choices"][0]["delta"]
    usage = oc.usage_body(stats.prompt_tokens, stats.decode_tokens, stats.reused_prefix_tokens)
    final = reply.final_chunk(finish, usage)
    assert final["usage"]["prompt_tokens_details"]["cached_tokens"] == 5
    assert final["choices"][0]["finish_reason"] == "stop" and final["choices"][0]["delta"] == {}
    assert oc.sse(final).startswith("data: {") and oc.sse(final).endswith("\n\n")
    assert oc.DONE == "data: [DONE]\n\n" and oc.KEEPALIVE == ": keepalive\n\n"


def test_final_chunk_omits_usage_unless_asked(fx):
    reply, _, finish, _ = run_reply(fx, {"messages": [USER], "stream": True}, "ok")
    assert "usage" not in reply.final_chunk(finish, oc.usage_body(1, 1, 0))


def test_streamed_reasoning_and_tool_call(fx):
    text = "plan it</think>\n\n" + fx.prose_then_call
    reply, deltas, finish, stats = run_reply(fx, {"messages": [USER], "stream": True, "tools": [READ_TOOL],
                                                  "think": True}, text, chunk=4)
    assert finish == "tool_calls"
    assert "".join(d.get("reasoning_content", "") for d in deltas) == "plan it"
    msg = reply.acc.message
    assert msg["content"] == "Let me read it.\n" and msg["tool_calls"][0]["function"]["arguments"] == \
        '{"path":"hello.txt","start_line":1}'
    body = reply.completion(finish, oc.usage_body(3, 4, 0))
    assert body["object"] == "chat.completion" and body["choices"][0]["finish_reason"] == "tool_calls"


def test_non_streamed_text_reply_reads_the_whole_text(fx):
    reply, deltas, finish, _ = run_reply(fx, {"messages": [USER]}, "Whole reply.")
    assert reply.acc.message["content"] == "Whole reply." and finish == "stop"


def test_single_call_turn_stops_the_engine_after_the_first_call(fx):
    two = fx.one_call + "\n" + fx.one_call
    reply, _, finish, stats = run_reply(fx, {"messages": [USER], "tools": [READ_TOOL],
                                             "parallel_tool_calls": False}, two, chunk=2)
    assert len(reply.acc.calls) == 1 and finish == "tool_calls" and stats.decode_tokens < len(two) // 2


def test_required_without_a_call_is_a_server_error(fx):
    reply, _, finish, _ = run_reply(fx, {"messages": [USER], "tools": [READ_TOOL], "tool_choice": "required"},
                                    "I will not.")
    assert reply.acc.error and "tool_choice: required" in reply.acc.error
    assert oc.server_error_body(reply.acc.error)["error"]["type"] == "server_error"


def test_error_bodies():
    assert oc.error_body("bad") == {"error": {"message": "bad", "type": "invalid_request_error",
                                              "code": "invalid_request_error"}}


# ---- scalar validation audit ------------------------------------------------------------------------

BAD_NUMBERS = ["abc", True, [1], {"a": 1}, 1e39, float("inf")]
BAD_INTS = ["abc", True, 1.5, [1], {"a": 1}, 1e39]


def checked(**extra):
    return oc.validation_error({"messages": [USER], **extra}, None, LIMIT)


@pytest.mark.parametrize("key", ["temperature", "top_p", "presence_penalty", "min_p"])
@pytest.mark.parametrize("bad", BAD_NUMBERS, ids=repr)
def test_bad_sampling_numbers_are_a_400_and_never_reach_prepare(key, bad):
    assert checked(**{key: bad}) == f"{key} must be a number"


@pytest.mark.parametrize("key", ["top_k", "max_tokens", "max_completion_tokens", "seed"])
@pytest.mark.parametrize("bad", BAD_INTS, ids=repr)
def test_bad_integers_are_a_400(key, bad):
    assert checked(**{key: bad}) == f"{key} must be an integer"


@pytest.mark.parametrize("key", ["stream", "think", "parallel_tool_calls", "store"])
def test_bad_booleans_are_a_400(key):
    assert checked(**{key: "yes"}) is not None


def test_bad_stream_options_include_usage_is_a_400():
    assert checked(stream_options={"include_usage": 1}) == "stream_options.include_usage must be true or false"


def test_min_p_is_validated_and_applied():
    assert checked(min_p=0.05) is None and checked(min_p=7) is None
    assert oc.prepare({"messages": [USER], "min_p": 0.05}, LIMIT).params.min_p == 0.05
    # out of range is clamped by the sampler, as the Swift server does
    assert oc.prepare({"messages": [USER], "min_p": 7}, LIMIT).params.min_p == 1.0


@pytest.mark.parametrize("key", ["temperature", "top_p", "presence_penalty", "min_p", "top_k", "seed"])
@pytest.mark.parametrize("bad", ["abc", True, [1]], ids=repr)
def test_prepare_never_raises_on_a_bad_scalar(key, bad):
    oc.prepare({"messages": [USER], key: bad}, LIMIT)


def test_integers_outside_64_bits_are_refused():
    assert common.as_int(2**63) is None and common.as_int(-2**63 - 1) is None
    assert common.as_int(2**63 - 1) == 2**63 - 1 and common.as_int(64000.0) == 64000
