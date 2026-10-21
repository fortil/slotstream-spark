"""Ports of anthropic-request, anthropic-logical-turns, anthropic-events and AnthropicTurnChecks
(AnthropicChecks.swift, AnthropicTurnChecks.swift), plus the reply flow on the fake engine and the count_tokens
contract on a fake tokenizer.

Pictures differ from the Swift checks on purpose: this server runs text-only, so every image block is a 400 with
the prefix Claude Code retries on, and the tests that carried a picture through in Swift assert that refusal."""
import itertools
import json
import math
from types import SimpleNamespace

import pytest

from slotstream_spark.server import common, template
from slotstream_spark.server.common import AUTO, DISABLED, REQUIRED, VISION_OFF, ChatMessage, named_tool
from slotstream_spark.server.dialects import DIALECTS, anthropic as an
from slotstream_spark.server.tool_calls import ParsedToolCall

IMAGE = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}
IMAGE_MESSAGE = "Could not process image: " + VISION_OFF


def claude_code_tools():
    return [
        {"name": "Bash", "description": "Executes a given bash command.",
         "input_schema": {"type": "object", "additionalProperties": False, "required": ["command"],
                          "properties": {"command": {"type": "string", "description": "The command to execute"},
                                         "timeout": {"type": "number"}}}},
        {"name": "Read", "description": "Reads a file from the local filesystem.",
         "input_schema": {"type": "object", "required": ["file_path"],
                          "properties": {"file_path": {"type": "string"}, "offset": {"type": "integer"}}}},
        {"name": "Write", "description": "Writes a file to the local filesystem.",
         "input_schema": {"type": "object", "required": ["file_path", "content"],
                          "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}}},
    ]


def claude_code_body(**extra):
    """A Claude Code first turn, reduced to the shapes that matter."""
    body = {
        "model": "qwen3.8-flash-next:4bit", "max_tokens": 32000, "stream": True,
        "system": [
            {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.270.f7f; cc_entrypoint=sdk-cli;"},
            {"type": "text", "text": "You are a Claude agent.", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Use the tools to help the user.", "cache_control": {"type": "ephemeral"}},
        ],
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "<system-reminder>\nToday is a test.\n</system-reminder>"},
                {"type": "text", "text": "Create hello.txt"}]},
            {"role": "system", "content": [
                {"type": "text", "text": "# Environment\nPrimary working directory: /tmp/work",
                 "cache_control": {"type": "ephemeral"}}]},
        ],
        "tools": claude_code_tools(),
        "thinking": {"type": "adaptive", "display": "omitted"},
        "output_config": {"effort": "high"},
        "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        "metadata": {"user_id": "{\"session_id\":\"s\"}"},
    }
    body.update(extra)
    return body


def claude_code_loop(result_image=False):
    thought = "I will write the file, then list it."
    assistant = {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "", "signature": an.signature_for(thought)},
        {"type": "text", "text": "Writing it now."},
        {"type": "tool_use", "id": "toolu_1", "name": "Write",
         "input": {"file_path": "hello.txt", "content": "SLOTSTREAM OK"}},
        {"type": "tool_use", "id": "toolu_2", "name": "Bash", "input": {"command": "ls", "timeout": 5000}}]}
    first = [{"type": "text", "text": "hello.txt"}] + ([IMAGE] if result_image else [])
    results = {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_2", "content": first},
        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "permission denied", "is_error": True},
        {"type": "text", "text": "Keep going."}]}
    body = claude_code_body()
    body["messages"] = body["messages"] + [assistant, results]
    return body


def refused(body, fragment, counting=False, type_="invalid_request_error"):
    with pytest.raises(an.Failure) as e:
        an.parse(body, counting=counting)
    assert fragment in e.value.message, e.value.message
    assert e.value.type == type_ and e.value.status == 400
    return e.value


def test_dialect_registry():
    assert DIALECTS["anthropic"] is an


# ---- anthropic-request ---------------------------------------------------------------------------------

def test_claude_code_first_turn():
    r = an.parse(claude_code_body())
    assert r.messages[0].content == "You are a Claude agent.\n\nUse the tools to help the user."
    assert [m.role for m in r.messages] == ["system", "user", "user"]
    assert r.messages[1].content == "<system-reminder>\nToday is a test.\n</system-reminder>\n\nCreate hello.txt"
    assert r.messages[2].content == "# Environment\nPrimary working directory: /tmp/work"
    assert [t.name for t in r.tools] == ["Bash", "Read", "Write"]
    assert r.tools[0].schema.params["command"] == "string"
    assert r.thinking and r.effort == "xhigh" and not r.show_thinking
    assert r.max_tokens == 32000 and r.stream and r.choice == AUTO and r.parallel
    assert r.ignored == []
    later = an.parse(claude_code_body(speed="fast", inference_geo="us"))
    assert later.ignored == ["inference_geo", "speed"]


def test_tool_loop_in_call_order():
    r = an.parse(claude_code_loop())
    assert [m.role for m in r.messages] == ["system", "user", "user", "assistant", "tool", "tool", "user"]
    a = r.messages[3]
    assert a.content == "Writing it now."
    assert a.reasoning == "I will write the file, then list it."
    assert [c.id for c in a.tool_calls] == ["toolu_1", "toolu_2"]
    assert a.tool_calls[1].arguments["timeout"] == 5000
    assert r.messages[4].tool_call_id == "toolu_1" and r.messages[4].tool_name == "Write"
    assert r.messages[4].content == "Error: permission denied"
    assert r.messages[5].content == "hello.txt"
    assert r.messages[6].content == "Keep going."


def test_thinking_and_effort():
    def parse(**extra):
        body = claude_code_body(**extra)
        for k, v in extra.items():
            if v is None:
                body.pop(k)
        return an.parse(body)
    assert not parse(thinking=None).thinking and parse(thinking=None).show_thinking
    assert not parse(thinking={"type": "disabled"}).thinking
    small = parse(thinking={"type": "enabled", "budget_tokens": 2000}, output_config=None)
    assert small.thinking and small.effort == "low" and small.show_thinking
    assert parse(thinking={"type": "enabled", "budget_tokens": 4095}, output_config=None).effort == "low"
    assert parse(thinking={"type": "enabled", "budget_tokens": 4096}, output_config=None).effort == "medium"
    assert parse(thinking={"type": "enabled", "budget_tokens": 8000}, output_config=None).effort == "medium"
    assert parse(thinking={"type": "enabled", "budget_tokens": 16383}, output_config=None).effort == "medium"
    assert parse(thinking={"type": "enabled", "budget_tokens": 16384}, output_config=None).effort == "xhigh"
    assert parse(thinking={"type": "enabled", "budget_tokens": 2000}, output_config={"effort": "max"}).effort == "xhigh"
    low = parse(output_config={"effort": "low"})
    assert low.thinking and low.effort == "low"
    assert parse(thinking={"type": "adaptive"}, output_config=None).effort == "xhigh"
    assert parse(thinking={"type": "adaptive", "display": "summarized"}).show_thinking
    newer = parse(thinking={"type": "adaptive", "display": "updates"})
    assert newer.show_thinking and newer.thinking


def test_tool_choice_and_tool_types():
    any_ = an.parse(claude_code_body(tool_choice={"type": "any", "disable_parallel_tool_use": True}))
    assert any_.choice == REQUIRED and not any_.parallel
    assert an.parse(claude_code_body(tool_choice={"type": "tool", "name": "Read"})).choice == named_tool("Read")
    assert an.parse(claude_code_body(tool_choice={"type": "none"})).choice == DISABLED
    hosted = an.parse(claude_code_body(tools=claude_code_tools() + [
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
        {"type": "custom", "name": "Note", "input_schema": {"type": "object"}, "strict": True,
         "defer_loading": False, "cache_control": {"type": "ephemeral"}}]))
    assert [t.name for t in hosted.tools] == ["Bash", "Read", "Write", "Note"]
    advisor = an.parse(claude_code_body(tools=claude_code_tools() + [{"type": "advisor_20260301", "name": "advisor"}]))
    assert len(advisor.tools) == 3
    counted = an.parse({"model": "m", "messages": [{"role": "user", "content": "hi"}]}, counting=True)
    assert counted.max_tokens == 0 and len(counted.messages) == 1
    prefilled = an.parse({"model": "m", "messages": [{"role": "user", "content": "hi"},
                                                      {"role": "assistant", "content": "Hello"}]}, counting=True)
    assert [m.role for m in prefilled.messages] == ["user", "assistant"]
    s = an.parse(claude_code_body(stop_sequences=["END", ""], temperature=0.2, top_p=0.9, top_k=10))
    assert s.stop_sequences == ["END"] and (s.temperature, s.top_p, s.top_k) == (0.2, 0.9, 10)


def test_content_the_api_allows():
    document = {"type": "document", "title": "Notes",
                "source": {"type": "text", "media_type": "text/plain", "data": "gate code MAPLE"}}
    body = {"model": "m", "max_tokens": 10, "system": "x-anthropic-billing-header: cc_version=1;\nYou are brief.",
            "messages": [
                {"role": "system", "content": "Answer in English."},
                {"role": "user", "content": [document, {"type": "text", "text": "What is the code?"}]},
                {"role": "assistant", "content": "It is"},
                {"role": "assistant", "content": [{"type": "redacted_thinking", "data": "abc"},
                                                  {"type": "text", "text": " MAPLE."}]},
                {"role": "user", "content": "Thanks."}]}
    r = an.parse(body)
    assert r.messages[0].content == "You are brief.\n\nAnswer in English."
    assert r.messages[1].content == "Notes\n\ngate code MAPLE\n\nWhat is the code?"
    assert r.messages[2].content == "It is MAPLE."
    assert [m.role for m in r.messages] == ["system", "user", "assistant", "user"]
    bare = an.parse({"model": "m", "max_tokens": 1, "system": "x-anthropic-billing-header: cc_version=1;",
                     "messages": [{"role": "user", "content": "hi"}]})
    assert [m.role for m in bare.messages] == ["user"]


def test_refusals_name_their_cause():
    body = claude_code_body()
    no_max = dict(body)
    no_max.pop("max_tokens")
    refused(no_max, "max_tokens: field required")
    no_model = dict(body)
    no_model.pop("model")
    refused(no_model, "model: field required")
    refused(claude_code_body(model=7), "model: must be text")
    refused(claude_code_body(container="c1"), "container")
    refused(claude_code_body(mcp_servers=[{"type": "url"}]), "mcp_servers")
    refused(claude_code_body(output_config={"format": {"type": "json_schema", "schema": {}}}), "output_config.format")
    refused(claude_code_body(output_config={"effort": "extreme"}), "output_config.effort")
    refused(claude_code_body(output_config="high"), "output_config")
    refused(claude_code_body(thinking={"type": "sometimes"}), "thinking.type")
    refused(claude_code_body(thinking="on"), "thinking")
    refused(claude_code_body(thinking={"type": "enabled"}), "budget_tokens")
    refused(claude_code_body(tools=[{"type": "bash_20250124", "name": "bash"}]), "bash_20250124")
    refused(claude_code_body(tools=claude_code_tools() + [claude_code_tools()[0]]), "unique")
    refused(claude_code_body(tools=[{"name": "x"}]), "input_schema")
    refused(claude_code_body(tool_choice={"type": "tool", "name": "Nope"}), "tool_choice.name")
    refused(claude_code_body(tool_choice="auto"), "tool_choice")
    refused(claude_code_body(tools=[{"type": "web_search_20250305", "name": "web_search"}],
                             tool_choice={"type": "tool", "name": "web_search"}), "runs on Anthropic's servers")
    refused(claude_code_body(tools=[], tool_choice={"type": "any"}), "needs at least one tool")
    refused(claude_code_body(stop_sequences=[1]), "stop_sequences")
    refused(claude_code_body(top_p=1.5), "top_p")
    refused(claude_code_body(max_tokens=0), "max_tokens")
    refused(claude_code_body(metadata="x"), "metadata")
    refused(claude_code_body(context_management=[]), "context_management")
    refused(claude_code_body(service_tier=1), "service_tier")


@pytest.mark.parametrize("key, value, fragment", [
    ("max_tokens", True, "max_tokens"), ("max_tokens", 1.5, "max_tokens"), ("max_tokens", "9", "max_tokens"),
    ("max_tokens", float("nan"), "max_tokens"), ("max_tokens", float("inf"), "max_tokens"),
    ("max_tokens", 1 << 70, "max_tokens"), ("max_tokens", -3, "max_tokens"),
    ("temperature", True, "temperature"), ("temperature", float("nan"), "temperature"),
    ("temperature", float("inf"), "temperature"), ("temperature", -0.1, "temperature"),
    ("temperature", 1e39, "temperature"), ("temperature", "0.5", "temperature"),
    ("top_p", True, "top_p"), ("top_p", float("nan"), "top_p"), ("top_p", -0.01, "top_p"),
    ("top_p", "x", "top_p"),
    ("top_k", True, "top_k"), ("top_k", 2.5, "top_k"), ("top_k", -1, "top_k"), ("top_k", float("inf"), "top_k"),
    ("top_k", 1 << 70, "top_k"), ("top_k", "3", "top_k"),
    ("stream", 1, "stream"), ("stream", "true", "stream"),
])
def test_no_unvalidated_scalar_reaches_the_sampler(key, value, fragment):
    refused(claude_code_body(**{key: value}), fragment)


@pytest.mark.parametrize("budget", [True, 0, -5, 2.5, float("nan"), float("inf"), "8000", 1 << 70])
def test_thinking_budget_is_a_positive_integer(budget):
    refused(claude_code_body(thinking={"type": "enabled", "budget_tokens": budget}), "budget_tokens")


def test_accepted_scalars_are_exactly_what_the_sampler_gets():
    r = an.parse(claude_code_body(temperature=0, top_p=1, top_k=0, max_tokens=64000.0))
    assert (r.temperature, r.top_p, r.top_k, r.max_tokens) == (0.0, 1.0, 0, 64000)
    p = an.prepare(r).params
    assert math.isfinite(p.temperature) and p.top_k == 0 and p.max_tokens == 32768 and 0 < p.top_p <= 1
    assert isinstance(p.top_k, int) and isinstance(p.max_tokens, int)


def test_non_finite_values_in_replayed_tool_input_are_refused():
    base = [{"role": "user", "content": "go"}]
    for bad in (float("nan"), float("inf"), {"a": [float("-inf")]}):
        call = {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash",
                                                  "input": {"command": bad}}]}
        refused(claude_code_body(messages=base + [call, {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "x"}]}]), "input")
    refused(claude_code_body(tools=[{"name": "x", "input_schema": {"type": "object", "x": float("nan")}}]),
            "input_schema")


def test_images_are_refused_with_the_prefix():
    for body in (
        claude_code_body(messages=[{"role": "user", "content": [IMAGE, {"type": "text", "text": "what"}]}]),
        claude_code_loop(result_image=True),
        claude_code_body(messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}]}]),
        claude_code_body(messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/tiff", "data": "x"}}]}]),
    ):
        for counting in (False, True):
            e = refused(body, "Could not process image: ", counting=counting)
            assert e.message == IMAGE_MESSAGE


def test_pdf_becomes_a_note():
    pdf = {"type": "document", "title": "Spec",
           "source": {"type": "base64", "media_type": "application/pdf", "data": "x"}}
    result = {"type": "tool_result", "tool_use_id": "t1", "content": [pdf]}
    read = {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read",
                                              "input": {"file_path": "a.pdf"}}]}
    r = an.parse(claude_code_body(messages=[
        {"role": "user", "content": [pdf, {"type": "text", "text": "Summarize"}]}, read,
        {"role": "user", "content": [result]}]))
    assert "'Spec' is not shown" in r.messages[1].content and "pdftotext" in r.messages[1].content
    assert "is not shown" in r.messages[-1].content


def test_late_tool_announcements_and_search_results():
    announced = {"role": "system", "content": [
        {"type": "text", "text": "New tools are available."},
        {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "mcp__docs__search"}},
        {"type": "tool_removal", "tool": {"type": "tool_reference", "name": "mcp__old__x"}}]}
    search = {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "ToolSearch",
                                                "input": {"query": "docs"}}]}
    found = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": [
        {"type": "tool_reference", "tool_name": "mcp__docs__search"},
        {"type": "search_result", "title": "Guide", "source": "https://example.com/g",
         "content": [{"type": "text", "text": "Use launch."}]}]}]}
    r = an.parse(claude_code_body(messages=[{"role": "user", "content": "go"}, announced, search, found]))
    assert r.messages[2].content == "New tools are available."
    assert r.messages[-1].content == "Tool available: mcp__docs__search\nGuide\nSource: https://example.com/g\nUse launch."


def test_message_level_refusals():
    go = {"role": "user", "content": "go"}
    call = {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash",
                                              "input": {"command": "ls"}}]}

    def msgs(*m):
        return claude_code_body(messages=list(m))
    refused(msgs(go, {"role": "system", "content": [{"type": "image", "source": {}}]}), "not 'image'")
    refused(msgs(go, call, {"role": "user", "content": "next"}), "without tool_result")
    refused(msgs(go, call), "no tool_result")
    stray = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t9", "content": "x"}]}
    refused(msgs(go, call, stray), "tool_result must answer")
    refused(msgs(go, {"role": "assistant", "content": "Sure"}), "last message")
    refused(msgs({"role": "tool", "content": "x"}), "role")
    refused(msgs({"content": "x"}), "role")
    refused(msgs(), "at least one message")
    refused(msgs({"role": "system", "content": "x"}), "at least one user message")
    refused(msgs(go, {"role": "assistant", "content": [
        {"type": "server_tool_use", "id": "s", "name": "web_search", "input": {}}]}, {"role": "user", "content": "x"}),
        "server_tool_use")
    refused(msgs(go, {"role": "assistant", "content": [
        {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}},
        {"type": "tool_use", "id": "t1", "name": "Read", "input": {}}]}), "used twice")
    refused(msgs(go, {"role": "assistant", "content": [
        {"type": "tool_use", "id": "t1", "name": "Bash", "input": "ls"}]}), "input")
    refused(msgs(go, {"role": "assistant", "content": [
        {"type": "tool_use", "id": "", "name": "Bash", "input": {}}]}), ".id: field required")
    refused(msgs(go, "not an object"), "messages.1: must be an object")
    refused(msgs(go, {"role": "assistant", "content": [{"type": "thinking", "thinking": 5}]}), "thinking: must be text")
    refused(claude_code_body(messages="hi"), "messages: must be an array")
    refused(msgs(go, call, {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": "x", "is_error": "yes"}]}), "is_error")


def test_error_surface():
    assert an.Failure("bad").body == {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}
    assert an.prompt_too_long(40000, 32767) == "prompt is too long: 40000 tokens > 32767 maximum"
    assert [an.error_type(s) for s in (400, 404, 413, 431, 429, 503, 529, 500, 401, 403)] == [
        "invalid_request_error", "not_found_error", "request_too_large", "request_too_large", "rate_limit_error",
        "overloaded_error", "overloaded_error", "api_error", "authentication_error", "permission_error"]
    assert an.header_safe_field_names(["speed", "bad\r\nX-Evil: 1", "", "a.b-c_d", "é", "x" * 65]) == [
        "speed", "a.b-c_d"]
    assert len(an.header_safe_field_names([f"f{i}" for i in range(40)])) == 16
    assert len(an.header_safe_field_names(["a" * 64])) == 1
    e = an.model_error({"model": "other"}, "qwen3.8-flash-next:4bit")
    assert e.status == 404 and e.type == "not_found_error" and e.message.startswith("model: ")
    assert an.model_error({"model": "qwen3.8-flash-next:4bit"}, "qwen3.8-flash-next:4bit") is None
    assert an.model_error({"model": 5}, "m").status == 404


# ---- anthropic-logical-turns ---------------------------------------------------------------------------

def message(role, content):
    return {"role": role, "content": content}


def parse_msgs(messages, counting=False):
    return an.parse({"model": "m", "max_tokens": 100, "messages": messages}, counting=counting).messages


def partitions(role, blocks):
    """Every placement of message boundaries between these blocks."""
    for mask in range(1 << (len(blocks) - 1)):
        groups = [[blocks[0]]]
        for i in range(1, len(blocks)):
            if mask & (1 << (i - 1)):
                groups.append([])
            groups[-1].append(blocks[i])
        yield [message(role, g) for g in groups]


CALL_A = {"type": "tool_use", "id": "a", "name": "Read", "input": {"file": "a.txt"}}
CALL_B = {"type": "tool_use", "id": "b", "name": "Read", "input": {"file": "b.txt"}}
RESULT_A = {"type": "tool_result", "tool_use_id": "a", "content": "denied", "is_error": True}
RESULT_B = {"type": "tool_result", "tool_use_id": "b", "content": [{"type": "text", "text": "B contents"}]}
TEXT = {"type": "text", "text": "Continue."}
START = message("user", "Read both files.")
ASSISTANT = [
    {"type": "thinking", "thinking": "", "signature": an.signature_for("Read both.")},
    {"type": "text", "text": "Reading."}, CALL_A, {"type": "text", "text": "Also B."}, CALL_B]


def test_every_partition_of_a_logical_turn_is_the_same_input():
    user = [RESULT_B, RESULT_A, TEXT]
    expected = parse_msgs([START, message("assistant", ASSISTANT), message("user", user)])
    assert [m.role for m in expected] == ["user", "assistant", "tool", "tool", "user"]
    assert [m.tool_call_id for m in expected if m.role == "tool"] == ["a", "b"]
    assert next(m for m in expected if m.tool_call_id == "a").content == "Error: denied"
    assert next(m for m in expected if m.role == "assistant").reasoning == "Read both."
    n = 0
    for left, right in itertools.product(partitions("assistant", ASSISTANT), partitions("user", user)):
        split = [START] + left + right
        assert parse_msgs(split) == expected
        assert parse_msgs(split, counting=True) == expected
        n += 1
    assert n == 16 * 4


def test_empty_messages_inside_logical_turns():
    squeezed = parse_msgs([START, message("assistant", [CALL_A, CALL_B]), message("user", [RESULT_B, RESULT_A])])
    spread = parse_msgs([START, message("assistant", []), message("assistant", [CALL_A]),
                         message("assistant", []), message("assistant", [CALL_B]),
                         message("user", []), message("user", [RESULT_B]),
                         message("user", []), message("user", [RESULT_A]), message("user", [])])
    assert spread == squeezed


def test_user_text_joins_once_and_accumulators_reset():
    ordinary = [{"type": "text", "text": "One"}, {"type": "text", "text": "Two"}]
    expected = parse_msgs([message("user", ordinary)])
    assert [m.content for m in expected] == ["One\n\nTwo"]
    for split in partitions("user", ordinary):
        assert parse_msgs(split) == expected
    assert parse_msgs([message("user", "One"), message("user", [{"type": "text", "text": "Two"}])]) == expected
    cycles = [START, message("assistant", [CALL_A]), message("user", [RESULT_A]), message("user", "Next"),
              message("assistant", [CALL_B]), message("user", [RESULT_B]), message("user", "Done")]
    assert parse_msgs(cycles) == parse_msgs([
        START, message("assistant", [CALL_A]),
        message("user", [RESULT_A, {"type": "text", "text": "Next"}]),
        message("assistant", [CALL_B]), message("user", [RESULT_B, {"type": "text", "text": "Done"}])])
    assert parse_msgs([START, message("assistant", "One"), message("assistant", "Two")], counting=True) == \
        parse_msgs([START, message("assistant", "OneTwo")], counting=True)
    assert parse_msgs([message("user", "One"), message("system", "Context"), message("user", "Two")]) == [
        ChatMessage("user", "One"), ChatMessage("user", "Context"), ChatMessage("user", "Two")]


def test_split_turns_keep_validation_and_locations():
    calls = [START, message("assistant", [CALL_A, CALL_B])]

    def bad(messages, fragment, counting=False):
        with pytest.raises(an.Failure) as e:
            parse_msgs(messages, counting)
        assert fragment in e.value.message and e.value.type == "invalid_request_error", e.value.message
    bad(calls + [message("user", [RESULT_A]), message("user", [TEXT])], "messages.3: tool_use ids")
    bad(calls + [message("user", [RESULT_A]), message("assistant", "Done")], "messages.2: tool_use ids")
    bad(calls + [message("user", [RESULT_A]), message("user", [RESULT_B, RESULT_A])], "messages.3.content.1: tool_result")
    bad(calls + [message("user", [RESULT_A, RESULT_B]),
                 message("user", [{"type": "tool_result", "tool_use_id": "unknown", "content": "x"}])],
        "messages.3.content.0: tool_result")
    bad([START, message("assistant", [CALL_A]), message("assistant", [CALL_A]), message("user", [RESULT_A])],
        "messages.2.content.0: tool_use id 'a' is used twice")
    bad([START, message("assistant", [CALL_A]), message("user", [RESULT_A]), message("assistant", [CALL_A]),
         message("user", [RESULT_A])], "messages.3.content.0: tool_use id 'a' is used twice")
    bad(calls + [message("system", "Context"), message("user", [RESULT_A, RESULT_B])], "messages.2: tool_use ids")
    bad(calls + [message("user", [RESULT_A]), message("system", []), message("user", [RESULT_B])],
        "messages.2: tool_use ids")
    bad(calls + [message("user", [RESULT_A]), message("user", [RESULT_B, {"type": "text", "text": 42}])],
        "messages.3.content.1.text")
    bad(calls + [message("user", [RESULT_A]), message("user", 42)], "messages.3.content: must be text")
    bad(calls + [message("user", [RESULT_A]), message("user", [CALL_B])],
        "messages.3.content.0: content type 'tool_use'")
    bad([START, message("assistant", [CALL_A]), message("assistant", [RESULT_A])],
        "messages.2.content.0: content type 'tool_result'")
    bad([START, message("assistant", "One"), message("assistant", "Two")], "continuing a partial assistant reply")
    bad([START, message("assistant", [CALL_A]), message("assistant", [CALL_B])], "no tool_result after them",
        counting=True)


def test_user_picture_in_a_split_turn_is_still_refused():
    refused({"model": "m", "max_tokens": 1, "messages": [message("user", "One"), message("user", [IMAGE])]},
            IMAGE_MESSAGE)


# ---- anthropic-events ----------------------------------------------------------------------------------

def decode(frame):
    lines = frame.split("\n")
    assert len(lines) == 4 and lines[0].startswith("event: ") and lines[1].startswith("data: ")
    assert lines[2] == "" and lines[3] == ""
    return lines[0][7:], json.loads(lines[1][6:])


def test_stream_event_order_and_payloads():
    thought = "Plan: write, then check. ✓"
    stream = an.MessageStream("qwen3.8-flash-next:4bit", show_thinking=False, mid="msg_test")
    frames = []
    frames += stream.start(120)
    frames += stream.thinking("Plan: write, ")
    frames += stream.thinking("then check. ✓")
    frames += stream.text("Writing ")
    frames += stream.text("now.")
    frames += stream.tool_use(ParsedToolCall("toolu_a", "Write", {"file_path": "hello.txt", "content": "OK"},
                                             ("file_path", "content")))
    usage = an.usage_body(120, 100, 30)
    frames += stream.finish("stop", None, usage)
    d = [decode(f) for f in frames]
    assert all(data["type"] == ev for ev, data in d)
    assert [ev for ev, _ in d] == [
        "message_start", "ping", "content_block_start", "content_block_delta", "content_block_stop",
        "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop",
        "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
    start = d[0][1]["message"]
    assert start["id"] == "msg_test" and start["usage"]["input_tokens"] == 120 and start["stop_reason"] is None
    reused = decode(an.MessageStream("m").start(120, reused=120)[0])[1]["message"]["usage"]
    assert [reused["input_tokens"], reused["cache_read_input_tokens"]] == [0, 120]
    assert d[2][1]["content_block"]["type"] == "thinking"
    sig = d[3][1]["delta"]
    assert sig["type"] == "signature_delta" and an.reasoning_from_signature(sig["signature"]) == thought
    assert d[5][1]["index"] == 1 and d[6][1]["delta"]["text"] == "Writing "
    tool = d[9][1]["content_block"]
    assert [tool["id"], tool["name"]] == ["toolu_a", "Write"]
    assert d[10][1]["delta"]["partial_json"] == '{"file_path":"hello.txt","content":"OK"}'
    assert d[12][1]["delta"]["stop_reason"] == "tool_use" and d[12][1]["delta"]["stop_sequence"] is None
    u = d[12][1]["usage"]
    assert [u["input_tokens"], u["cache_read_input_tokens"], u["cache_creation_input_tokens"],
            u["output_tokens"]] == [20, 100, 0, 30]

    whole = stream.message("stop", None, usage)
    assert [b["type"] for b in whole["content"]] == ["thinking", "text", "tool_use"]
    assert whole["content"][0]["thinking"] == ""
    assert whole["content"][1]["text"] == "Writing now."
    assert whole["content"][2]["input"]["content"] == "OK"
    assert (whole["type"], whole["role"]) == ("message", "assistant")


def test_stop_reasons_and_priority():
    def reason(engine, stop, calls=False, window=False):
        s = an.MessageStream("m")
        s.text("x")
        if calls:
            s.tool_use(ParsedToolCall("t", "Bash", {}))
        return s.message(engine, stop, an.usage_body(5, 0, 1), window)
    assert reason("stop", None)["stop_reason"] == "end_turn"
    assert reason("length", None)["stop_reason"] == "max_tokens"
    stopped = reason("stop", "END")
    assert (stopped["stop_reason"], stopped["stop_sequence"]) == ("stop_sequence", "END")
    assert reason("length", None, calls=True)["stop_reason"] == "tool_use"
    assert reason("length", "END", calls=True)["stop_reason"] == "tool_use"
    assert reason("length", "END")["stop_reason"] == "stop_sequence"
    assert reason("length", None, window=True)["stop_reason"] == "model_context_window_exceeded"
    assert reason("stop", None, window=True)["stop_reason"] == "end_turn"
    assert reason("stop", None)["stop_sequence"] is None


def test_keepalive_is_a_ping_except_during_hidden_thinking():
    quiet = an.MessageStream("m", show_thinking=False)
    assert decode(quiet.keepalive())[0] == "ping"
    quiet.thinking("long reasoning")
    ev, data = decode(quiet.keepalive())
    assert ev == "content_block_delta" and data["delta"] == {"type": "thinking_delta", "thinking": ""}
    quiet.text("answer")
    assert decode(quiet.keepalive())[0] == "ping"
    shown = an.MessageStream("m", show_thinking=True)
    shown.thinking("visible")
    assert decode(shown.keepalive())[0] == "ping"


def test_late_failure_does_not_close_the_open_block():
    s = an.MessageStream("m", show_thinking=True)
    frames = [decode(f) for f in s.thinking("abc")]
    assert frames[-1][1]["delta"]["thinking"] == "abc"
    failed = [decode(f) for f in s.fail("api_error", "boom")]
    assert [ev for ev, _ in failed] == ["error"]
    assert failed[0][1]["error"] == {"type": "api_error", "message": "boom"}
    assert s.message("stop", None, an.usage_body(1, 0, 0))["content"] == []


def test_usage_and_signatures():
    assert an.usage_body(10, 12, 0)["cache_read_input_tokens"] == 10
    assert an.usage_body(10, 12, 0)["input_tokens"] == 0
    assert an.usage_body(10, -4, 0)["cache_read_input_tokens"] == 0
    odd = "line one\nline \"two\" é 🧠 \u0000"
    assert an.reasoning_from_signature(an.signature_for(odd)) == odd
    assert an.reasoning_from_signature("EqQBCgIYAhIM") is None
    assert an.reasoning_from_signature("slotstream.thinking.v1.!!notbase64!!") is None
    assert an.reasoning_from_signature("slotstream.thinking.v1." + "/w==") is None  # not UTF-8
    a, b = an.tool_use_id(), an.tool_use_id()
    assert a.startswith("toolu_") and len(a) == len("toolu_") + 24 and a != b
    assert an.new_id("msg_").startswith("msg_") and len(an.new_id("msg_")) == 28


# ---- prompt rendering and count_tokens -----------------------------------------------------------------

class FakeTemplate:
    def render(self, **kw):
        return FakeTokenizer.flatten(kw["messages"], kw["tools"], kw["enable_thinking"], kw.get("reasoning_effort"))


class FakeTokenizer:
    """Renders messages to text deterministically; one token per whitespace-separated word."""
    _template = FakeTemplate()

    @staticmethod
    def flatten(messages, tools, thinking, effort):
        lines = [f"<{m['role']}> {m['content']}" for m in messages]
        return "\n".join(lines + [f"tools={json.dumps(tools, sort_keys=True)}", f"think={thinking}:{effort}"])

    def render_chat(self, messages, *, enable_thinking=False, tools=None):
        return self.flatten(messages, tools, enable_thinking, None)

    def encode(self, text):
        return text.split()


def test_two_attribution_lines_render_the_same_prompt():
    def prompt_of(line):
        body = claude_code_body()
        body["system"][0]["text"] = line
        return an.render_prompt(an.parse(body), FakeTokenizer())
    a = prompt_of("x-anthropic-billing-header: cc_version=2.1.270.f7f; cc_entrypoint=sdk-cli;")
    b = prompt_of("x-anthropic-billing-header: cc_version=9.9.9.abc; cc_entrypoint=other; cch=00000;")
    assert a == b and "billing" not in a
    plain = claude_code_body()
    plain["system"] = [plain["system"][1], plain["system"][2]]
    assert an.render_prompt(an.parse(plain), FakeTokenizer()) == a


def test_count_tokens_is_the_rendered_token_count():
    tok = FakeTokenizer()
    req = an.parse({"model": "m", "messages": [{"role": "user", "content": "one two three"}],
                    "system": "be brief"}, counting=True)
    assert an.count_tokens(req, tok) == len(tok.encode(an.render_prompt(req, tok)))
    assert an.count_tokens(req, tok) == len("<system> be brief <user> one two three tools=null think=False:None"
                                           .split())
    with_tool = an.parse({"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": [
        {"name": "Read", "input_schema": {"type": "object"}}], "tool_choice": {"type": "any"}}, counting=True)
    prompt = an.render_prompt(with_tool, tok)
    assert "You must call one of the available tools now." in prompt and "Read" in prompt
    assert an.count_tokens(with_tool, tok) > an.count_tokens(
        an.parse({"model": "m", "messages": [{"role": "user", "content": "hi"}]}, counting=True), tok)
    # A tool choice of none renders no tools, and the count follows the render.
    off = an.parse({"model": "m", "messages": [{"role": "user", "content": "hi"}], "tools": [
        {"name": "Read", "input_schema": {"type": "object"}}], "tool_choice": {"type": "none"}}, counting=True)
    assert "Read" not in an.render_prompt(off, tok)
    # A count may end on an assistant turn (a prefill); rendering it is the template layer's refusal, not a crash here.
    prefilled = an.parse({"model": "m", "messages": [{"role": "user", "content": "hi"},
                                                      {"role": "assistant", "content": "Hello"}]}, counting=True)
    with pytest.raises(template.TemplateError):
        an.count_tokens(prefilled, tok)


def test_count_tokens_on_the_real_template(tokenizer):
    req = an.parse(claude_code_body(), counting=True)
    # The checkout tokenizer is template-only (no tokenizer.json), so the real template is checked for its
    # rendering and the counting arithmetic is the fake tokenizer's job above.
    prompt = an.render_prompt(req, tokenizer)
    assert "You are a Claude agent." in prompt and "billing" not in prompt and "Create hello.txt" in prompt


# ---- prepare and reply flow ----------------------------------------------------------------------------

def test_prepare_presets_and_sampling():
    plain = an.parse({"model": "m", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]})
    p = an.prepare(plain)
    assert (p.params.temperature, p.params.top_p, p.params.presence_penalty) == (0.7, 0.8, 1.5)
    assert p.params.max_tokens == 50 and p.render_tools == []
    thinking = an.parse({"model": "m", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}],
                         "thinking": {"type": "enabled", "budget_tokens": 5000}})
    assert an.prepare(thinking).params.temperature == 1.0
    agent = an.prepare(an.parse(claude_code_body()))
    assert agent.params.temperature == 0.2 and agent.params.presence_penalty == 0.0
    explicit = an.prepare(an.parse(claude_code_body(temperature=0.5, top_p=0.5, top_k=7, stop_sequences=["E"])))
    assert (explicit.params.temperature, explicit.params.top_p, explicit.params.top_k) == (0.5, 0.5, 7)
    assert explicit.params.stop == ["E"]
    assert an.prepare(plain, random_seed=5).seed == 5 and an.prepare(plain).seed > 0
    named = an.prepare(an.parse(claude_code_body(tool_choice={"type": "tool", "name": "Read"})))
    assert "You must call the Read tool now." in named.messages[0].content
    serial = an.prepare(an.parse(claude_code_body(tool_choice={"type": "auto", "disable_parallel_tool_use": True})))
    assert "Call at most one tool in this response." in serial.messages[0].content
    none = an.prepare(an.parse(claude_code_body(tool_choice={"type": "none"})))
    assert none.render_tools == []


def test_the_reply_budget_is_clamped_to_the_window():
    req = an.parse(claude_code_body(max_tokens=32000))
    p = an.prepare(req)
    p.check_prompt(1000, 32768)
    assert p.params.max_tokens == 31768 and p.window_limited
    q = an.prepare(an.parse(claude_code_body(max_tokens=100)))
    q.check_prompt(1000, 32768)
    assert q.params.max_tokens == 100 and not q.window_limited
    for tokens, expect in ((40000, "prompt is too long: 40000 tokens > 32767 maximum"),
                           (32768, "prompt is too long: 32768 tokens > 32767 maximum")):
        with pytest.raises(an.Failure) as e:
            an.prepare(req).check_prompt(tokens, 32768)
        assert e.value.message == expect
    with pytest.raises(an.Failure, match="prompt is empty"):
        an.prepare(req).check_prompt(0, 32768)
    last = an.prepare(req)
    last.check_prompt(32767, 32768)
    assert last.params.max_tokens == 1


def run_reply(fx, text, request, chunk=3, engine_reason="stop", stop_sequence=None):
    prepared = an.prepare(request)
    reply = an.MessagesReply(prepared, "m", id_factory=fx.ids(), mid="msg_t")
    frames = [] + reply.message.start(10)
    engine = fx.FakeEngine(text=text, chunk=chunk, finish_reason=engine_reason)
    produced, _, stats = engine.generate("p", on_token=lambda n, d: (frames.extend(reply.consume(d)), not reply.should_stop)[1]
                                         if reply.incremental else True,
                                         should_continue=lambda: not reply.should_stop)
    frames += reply.end(stats.finish_reason, None if reply.incremental else produced)
    return reply, frames, stats


def events(frames):
    return [decode(f) for f in frames]


def test_reply_with_thinking_text_and_a_call(fx):
    req = an.parse(claude_code_body(thinking={"type": "enabled", "budget_tokens": 5000}, output_config=None))
    text = "Let me check.</think>\n\nChecking.\n" + fx.one_call.replace("read_file", "Read").replace(
        "<parameter=path>", "<parameter=file_path>").replace("</parameter>", "</parameter>")
    reply, frames, stats = run_reply(fx, text, req)
    types = [ev for ev, _ in events(frames)]
    assert types[:2] == ["message_start", "ping"] and "content_block_stop" in types
    blocks = [d["content_block"]["type"] for ev, d in events(frames) if ev == "content_block_start"]
    assert blocks == ["thinking", "text", "tool_use"]
    assert reply.failure("stop") is None
    done = reply.message.finish("stop", None, an.usage_body(10, 0, 5))
    final = events(done)
    assert final[-2][1]["delta"]["stop_reason"] == "tool_use"
    msg = reply.message.message("stop", None, an.usage_body(10, 0, 5))
    assert msg["content"][-1]["input"] == {"file_path": "hello.txt"}
    assert msg["content"][-1]["id"].startswith("call_")
    assert msg["content"][0]["thinking"] == "Let me check."  # no display: omitted, so shown


def test_reply_chunking_does_not_change_the_message(fx):
    req = an.parse(claude_code_body(thinking={"type": "enabled", "budget_tokens": 5000}, output_config=None,
                                    stream=True))
    text = "think hard</think>\n\nHello there.\n" + fx.one_call.replace("read_file", "Read").replace(
        "<parameter=path>", "<parameter=file_path>")
    results = set()
    for chunk in (1, 2, 5, 11, 1000):
        reply, frames, _ = run_reply(fx, text, req, chunk=chunk)
        msg = reply.message.message("stop", None, an.usage_body(10, 0, 5))
        msg["content"][-1]["id"] = "x"
        results.add(json.dumps(msg, sort_keys=True))
    assert len(results) == 1


def test_a_single_call_stops_generation(fx):
    req = an.parse(claude_code_body(stream=False, tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                                    thinking={"type": "disabled"}))
    text = fx.two_calls.replace("read_file", "Read").replace("<parameter=path>", "<parameter=file_path>").replace(
        "list_dir", "Bash")
    reply, frames, stats = run_reply(fx, text, req, chunk=4)
    assert reply.message.tool_uses == 1 and reply.should_stop
    assert stats.decode_tokens < len(text) // 4 + 1


def test_a_call_cut_by_length_is_max_tokens_not_a_failure(fx):
    req = an.parse(claude_code_body(stream=True, thinking={"type": "disabled"}))
    text = "Working.\n<tool_call>\n<function=Read>\n<parameter=file_path>\nhel"
    reply, frames, _ = run_reply(fx, text, req, engine_reason="length")
    assert reply.failure("length") is None
    msg = reply.message.message("length", None, an.usage_body(10, 0, 5))
    assert msg["stop_reason"] == "max_tokens"
    assert [b["type"] for b in msg["content"]] == ["text"]
    bad_stop = an.MessagesReply(an.prepare(req), "m")
    bad_stop.acc.error = "model produced an incomplete or malformed tool call"
    assert bad_stop.failure("stop") == "model produced an incomplete or malformed tool call"


def test_an_undeclared_call_is_a_failure(fx):
    req = an.parse(claude_code_body(stream=True, thinking={"type": "disabled"}))
    text = fx.one_call.replace("read_file", "Nope")
    reply, _, _ = run_reply(fx, text, req)
    assert "undeclared tool" in reply.failure("stop")
