"""Ports of responses-request, responses-events, responses-replay, responses-pending-turn, responses-codex-tools
and responses-codex-fixture (ResponsesChecks.swift, CodexFixture.swift), plus the reply flow on the fake engine.

Pictures differ from the Swift checks on purpose: this server runs text-only, so every input_image is a 400 with
code invalid_image, and the checks that carried a picture through in Swift assert that refusal instead."""
import json
import math
from pathlib import Path

import pytest

from slotstream_spark.server import tool_calls as tc
from slotstream_spark.server.common import AUTO, DISABLED, VISION_OFF, named_tool
from slotstream_spark.server.dialects import DIALECTS, responses as rs
from slotstream_spark.server.tool_calls import ParsedToolCall, ToolCallSplitter

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "codex" / "codex_fixture.json"
IMAGE_URL = "data:image/png;base64,iVBORw0KGgo="

EXEC_COMMAND = {
    "type": "function", "name": "exec_command", "strict": False,
    "description": "Runs a command in a PTY, returning output or a session ID for ongoing interaction.",
    "parameters": {"type": "object", "properties": {
        "cmd": {"type": "string", "description": "Shell command to execute."}, "workdir": {"type": "string"},
        "tty": {"type": "boolean"}, "yield_time_ms": {"type": "number"}, "max_output_tokens": {"type": "number"},
        "shell": {"type": "string"}, "login": {"type": "boolean"}}, "required": ["cmd"],
        "additionalProperties": False}}
WRITE_STDIN = {
    "type": "function", "name": "write_stdin", "strict": False,
    "description": "Writes characters to an existing unified exec session.",
    "parameters": {"type": "object", "properties": {
        "session_id": {"type": "number"}, "chars": {"type": "string"}, "yield_time_ms": {"type": "number"},
        "max_output_tokens": {"type": "number"}}, "required": ["session_id"], "additionalProperties": False}}
UPDATE_PLAN = {
    "type": "function", "name": "update_plan", "strict": False, "description": "Updates the task plan.",
    "parameters": {"type": "object", "properties": {
        "explanation": {"type": "string"},
        "plan": {"type": "array", "items": {"type": "object", "properties": {
            "step": {"type": "string"},
            "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}},
            "required": ["step", "status"], "additionalProperties": False}}},
        "required": ["plan"], "additionalProperties": False}}
VIEW_IMAGE = {
    "type": "function", "name": "view_image", "strict": False,
    "description": "View a local image file from the filesystem when visual inspection is needed.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": "Local filesystem path to an image file."}},
        "required": ["path"], "additionalProperties": False},
    "output_schema": {"type": "object", "properties": {"image_url": {"type": "string"}},
                      "required": ["image_url"], "additionalProperties": False}}


def codex_body(**extra):
    """A Codex turn-3 body, reduced to the shapes that matter and without pictures."""
    developer = {"type": "message", "role": "developer", "content": [
        {"type": "input_text", "text": "<permissions_instructions>\nworkspace-write\n</permissions_instructions>"}]}
    user = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Read hello.txt"}]}
    reasoning = {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "I need the file."}],
                 "encrypted_content": None}
    message = {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed", "phase": "commentary",
               "content": [{"type": "output_text", "text": "Let me look.", "annotations": []}]}
    call1 = {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "exec_command",
             "arguments": '{"cmd":"cat hello.txt","yield_time_ms":10000}', "status": "completed"}
    call2 = {"type": "function_call", "id": "fc_2", "call_id": "call_2", "name": "view_image",
             "arguments": '{"path":"shape.png"}'}
    out2 = {"type": "function_call_output", "call_id": "call_2", "output": "no preview"}
    out1 = {"type": "function_call_output", "call_id": "call_1", "output": "fixture42\n"}
    later = {"type": "message", "role": "developer", "content": [
        {"type": "input_text", "text": "<model_switch>\nlow\n</model_switch>"}]}
    again = {"type": "message", "role": "user", "content": "And now?"}
    body = {
        "model": "qwen3.8-flash-next:4bit", "instructions": "You are Codex.",
        "input": [developer, user, reasoning, message, call1, call2, out2, out1, later, again],
        "tools": [EXEC_COMMAND, VIEW_IMAGE, {"type": "web_search", "external_web_access": True}],
        "tool_choice": "auto", "parallel_tool_calls": True, "reasoning": {"effort": "low", "summary": "auto"},
        "store": False, "stream": True, "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": "thread-1", "client_metadata": {"thread_id": "thread-1"}}
    body.update(extra)
    return body


def parse(**extra):
    return rs.parse(codex_body(**extra))


def refused(body, code=None, fragment=None):
    with pytest.raises(rs.Failure) as e:
        rs.parse(body)
    if code is not None:
        assert e.value.code == code, (e.value.code, e.value.message)
    if fragment is not None:
        assert fragment in e.value.message, e.value.message
    assert e.value.status == 400
    return e.value


def rejected_extra(code=None, **extra):
    return refused(codex_body(**extra), code)


def decode(frame):
    lines = frame.split("\n")
    assert len(lines) == 4 and lines[0].startswith("event: ") and lines[1].startswith("data: ")
    assert lines[2] == "" and lines[3] == ""
    return lines[0][7:], json.loads(lines[1][6:])


def usage(i=1, c=0, o=8, r=0):
    return rs.usage_body(i, c, o, r)


def test_dialect_registry():
    assert DIALECTS["responses"] is rs


# ---- responses-request ---------------------------------------------------------------------------------

def test_codex_body_parses():
    r = parse()
    assert r.messages[0].content == ("You are Codex.\n\n<permissions_instructions>\nworkspace-write\n"
                                     "</permissions_instructions>")
    assert [m.role for m in r.messages] == ["system", "user", "assistant", "tool", "tool", "user", "user"]
    assert r.messages[1].content == "Read hello.txt"
    assert r.messages[2].content == "Let me look."
    assert r.messages[2].reasoning == "I need the file."
    assert [c.id for c in r.messages[2].tool_calls] == ["call_1", "call_2"]
    assert r.messages[2].tool_calls[0].arguments["yield_time_ms"] == 10000
    assert r.messages[3].tool_call_id == "call_1" and r.messages[3].content == "fixture42\n"
    assert r.messages[4].tool_name == "view_image"
    assert r.messages[5].content == "<model_switch>\nlow\n</model_switch>"
    assert [t.name for t in r.tools] == ["exec_command", "view_image"]
    assert len(r.echo["tools"]) == 3
    assert r.echo["reasoning"]["effort"] == "low" and r.echo["store"] is False
    assert r.thinking and r.effort == "low" and r.stream and r.parallel and r.choice == AUTO


def test_defaults_and_alternatives():
    plain = rs.parse({"model": "qwen3.8-flash-next:4bit", "input": "hi"})
    assert [m.role for m in plain.messages] == ["user"]
    assert not plain.stream and plain.parallel and not plain.thinking
    assert not parse(reasoning={"effort": "none"}).thinking
    assert parse(reasoning={"effort": "high"}).effort == "xhigh"
    assert parse(tool_choice={"type": "function", "name": "exec_command"}).choice == named_tool("exec_command")
    quiet = parse(tool_choice="none", max_output_tokens=64, temperature=0.1, top_p=0.9, truncation="disabled",
                  text={"verbosity": "low", "format": {"type": "text"}}, metadata={"k": "v"}, user="u",
                  safety_identifier="s", service_tier="default")
    assert quiet.choice == DISABLED and quiet.max_output_tokens == 64 and quiet.temperature == 0.1


def test_refusals_name_their_cause():
    rejected_extra("stored_state_unsupported", previous_response_id="resp_1")
    rejected_extra("stored_state_unsupported", conversation={"id": "conv_1"})
    rejected_extra("unsupported_field", background=True)
    rejected_extra("unsupported_field", text={"format": {"type": "json_schema", "schema": {}}})
    rejected_extra("unsupported_field", truncation="auto")
    rejected_extra("unsupported_field", temperature_scale=1)
    rejected_extra("unsupported_field", prompt={"id": "p"})
    rejected_extra("unsupported_field", context_management=[{"type": "compaction"}])
    rejected_extra("unsupported_field", max_tool_calls=3)
    rejected_extra("unsupported_field", top_logprobs=2)
    rejected_extra("invalid_request", background="no")
    rejected_extra("invalid_request", truncation="sometimes")
    rejected_extra("invalid_request", text={"verbosity": "loud"})
    rejected_extra("unsupported_field", reasoning={"effort": "low", "mode": "x"})
    rejected_extra("invalid_request", reasoning="low")
    rejected_extra("invalid_request", reasoning={"effort": "lots"})
    rejected_extra("invalid_request", parallel_tool_calls=1)
    rejected_extra("invalid_request", store="false")
    rejected_extra("invalid_request", include="x")
    rejected_extra("invalid_request", metadata=[])
    rejected_extra("invalid_request", user=1)
    rejected_extra("invalid_request", client_metadata="x")
    rejected_extra("invalid_request", model=5)
    rejected_extra("invalid_request", instructions=5)
    rejected_extra("invalid_request", stream="yes")
    refused({"model": "m"}, "invalid_request", "input is required")
    refused({"model": "m", "input": 5}, "invalid_request")
    refused({"model": "m", "input": ""}, "invalid_request")
    refused({"model": "m", "input": []}, "invalid_request")
    refused([], "invalid_request")


@pytest.mark.parametrize("key, value", [
    ("max_output_tokens", True), ("max_output_tokens", 0), ("max_output_tokens", -4), ("max_output_tokens", 2.5),
    ("max_output_tokens", float("nan")), ("max_output_tokens", float("inf")), ("max_output_tokens", 1 << 70),
    ("max_output_tokens", "64"),
    ("temperature", True), ("temperature", float("nan")), ("temperature", float("inf")),
    ("temperature", float("-inf")), ("temperature", 1e39), ("temperature", "0.5"), ("temperature", [1]),
    ("top_p", True), ("top_p", float("nan")), ("top_p", 1e39), ("top_p", "x"),
    ("stream", 1), ("stream", "true"), ("parallel_tool_calls", 0),
])
def test_no_unvalidated_scalar_reaches_the_sampler(key, value):
    refused(codex_body(**{key: value}), "invalid_request")


def test_non_finite_numbers_anywhere_in_the_body_are_refused():
    # The response object echoes tools and metadata; a NaN there would be invalid JSON on the wire.
    nan_tool = dict(EXEC_COMMAND, parameters={"type": "object", "default": float("nan")})
    refused(codex_body(tools=[nan_tool]), "invalid_request", "not finite")
    refused(codex_body(metadata={"k": float("inf")}), "invalid_request", "not finite")
    deep = cur = {}
    for _ in range(200):
        cur["x"] = {}
        cur = cur["x"]
    refused(codex_body(client_metadata=deep), "invalid_request")


def test_accepted_values_are_clean_for_the_sampler():
    r = parse(temperature=0, top_p=1, max_output_tokens=64000.0)
    assert (r.temperature, r.top_p, r.max_output_tokens) == (0.0, 1.0, 64000)
    p = rs.prepare(r, 32768).params
    assert math.isfinite(p.temperature) and p.max_tokens == 32768 and 0 < p.top_p <= 1
    # Out-of-range finite values are clamped by the sampler's own rule, not passed through.
    odd = rs.prepare(parse(temperature=-1, top_p=7), 32768).params
    assert odd.temperature == 0.0 and odd.top_p == 1.0


def test_freeform_tool_renders_as_a_one_parameter_function():
    patch_tool = {"type": "custom", "name": "apply_patch", "description": "Use the `apply_patch` tool to edit files.",
                  "format": {"type": "grammar", "syntax": "lark", "definition": "start: begin_patch hunk+ end_patch"}}
    call = {"type": "custom_tool_call", "call_id": "call_p", "name": "apply_patch",
            "input": "*** Begin Patch\n*** Add File: a.txt\n+hi\n*** End Patch"}
    out = {"type": "custom_tool_call_output", "call_id": "call_p", "output": "Done!"}
    r = parse(tools=[patch_tool], input=[{"role": "user", "content": "go"}, call, out,
                                         {"role": "user", "content": "and?"}])
    assert [t.name for t in r.tools] == ["apply_patch"]
    assert r.tools[0].schema.params["input"] == "string"
    assert "lark grammar" in r.tools[0].description and "begin_patch" in r.tools[0].description
    assert r.freeform == {"apply_patch"}
    forced = parse(tools=[patch_tool], tool_choice={"type": "custom", "name": "apply_patch"})
    assert forced.choice == named_tool("apply_patch")
    a = next(m for m in r.messages if m.role == "assistant")
    assert a.tool_calls[0].arguments["input"] == "*** Begin Patch\n*** Add File: a.txt\n+hi\n*** End Patch"
    assert next(m for m in r.messages if m.role == "tool").content == "Done!"
    s = rs.ResponseStream("m", rid="resp_ff", created_at=1, freeform=r.freeform)
    frames = s.function_call(ParsedToolCall("call_p", "apply_patch", {"input": "*** Begin Patch\n*** End Patch"},
                                            ("input",)))
    assert [decode(f)[0] for f in frames] == [
        "response.output_item.added", "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done", "response.output_item.done"]
    item = decode(frames[-1])[1]["item"]
    assert item["input"] == "*** Begin Patch\n*** End Patch" and item["type"] == "custom_tool_call"


def test_namespace_bundle_flattens_and_reports_back_split():
    bundle = {"type": "namespace", "name": "multi_agent_v1",
              "description": "Tools for spawning and managing sub-agents.",
              "tools": [{"type": "function", "name": "wait_agent", "strict": False, "description": "Wait.",
                         "parameters": {"type": "object", "properties": {"timeout_ms": {"type": "number"}}}}]}
    call = {"type": "function_call", "call_id": "call_9", "name": "wait_agent", "namespace": "multi_agent_v1",
            "arguments": '{"timeout_ms":5}'}
    out = {"type": "function_call_output", "call_id": "call_9", "output": "done"}
    r = parse(tools=[bundle], input=[{"role": "user", "content": "go"}, call, out,
                                     {"role": "user", "content": "and?"}])
    assert [t.name for t in r.tools] == ["multi_agent_v1.wait_agent"]
    assert r.tools[0].description.startswith("Tools for spawning")
    assert r.namespaces == {"multi_agent_v1"}
    assert next(m for m in r.messages if m.role == "assistant").tool_calls[0].name == "multi_agent_v1.wait_agent"
    s = rs.ResponseStream("m", rid="resp_ns", created_at=1, namespaces=r.namespaces)
    item = decode(s.function_call(ParsedToolCall("call_9", "multi_agent_v1.wait_agent", {"timeout_ms": 5}))[-1])[1]["item"]
    assert (item["namespace"], item["name"]) == ("multi_agent_v1", "wait_agent")
    plain = rs.ResponseStream("m", rid="resp_plain", created_at=1)
    item = decode(plain.function_call(ParsedToolCall("c", "a.b", {}))[-1])[1]["item"]
    assert item["name"] == "a.b" and "namespace" not in item


def test_tool_and_item_refusals():
    rejected_extra("unsupported_tool", tools=[{"type": "namespace", "name": "ns", "tools": [
        {"type": "custom", "name": "x", "format": {"type": "grammar"}}]}])
    rejected_extra("unsupported_tool", tools=[dict(EXEC_COMMAND, strict=True)])
    rejected_extra("unsupported_tool", tools=[{"type": "computer_use"}])
    rejected_extra("unsupported_tool", tools=[dict(EXEC_COMMAND, strict=0)])
    rejected_extra("invalid_request", tools=[{"name": "x"}])
    rejected_extra("invalid_request", tools=[EXEC_COMMAND, EXEC_COMMAND])
    rejected_extra("invalid_request", tools=[dict(EXEC_COMMAND, parameters=[])])
    rejected_extra("invalid_request", tools="x")
    rejected_extra("invalid_request", tools=[{"type": "namespace", "name": "a.b", "tools": []}])
    rejected_extra("unsupported_field", tool_choice={"type": "allowed_tools", "mode": "auto", "tools": []})
    rejected_extra("invalid_request", tool_choice={"type": "function", "name": "missing"})
    rejected_extra("invalid_request", tool_choice="sometimes")
    rejected_extra("invalid_request", tool_choice={"type": "function", "name": "exec_command", "extra": 1})
    rejected_extra("invalid_request", tools=[], tool_choice="required")
    user = {"role": "user", "content": "hi"}
    call = {"type": "function_call", "call_id": "c1", "name": "exec_command", "arguments": "{}"}
    result = {"type": "function_call_output", "call_id": "c1", "output": "a"}
    rejected_extra("unsupported_item", input=[{"type": "local_shell_call", "call_id": "x", "action": {}}])
    rejected_extra("invalid_request", input=[{"content": "no type or role"}])
    rejected_extra("invalid_request", input=[{"type": 5}])
    rejected_extra("unsupported_field", input=[{"role": "user", "content": [
        {"type": "input_audio", "audio_url": "data:audio/wav;base64,AA=="}]}])
    rejected_extra("unsupported_field", input=[{"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}])
    rejected_extra("unsupported_field", input=[user, {"type": "reasoning", "summary": [], "encrypted_content": "gAAAA"},
                                               {"role": "user", "content": "x"}])
    rejected_extra("invalid_request", input=[user, {"type": "reasoning", "summary": [], "encrypted_content": 5},
                                             {"role": "user", "content": "x"}])
    rejected_extra("assistant_prefill_unsupported", input=[user, {"role": "assistant", "content": "prefill"}])
    rejected_extra("invalid_request", input=[user, {"type": "function_call_output", "call_id": "nope", "output": "x"}])
    rejected_extra("invalid_request", input=[user, call, {"role": "user", "content": "x"}])
    rejected_extra("invalid_request", input=[user, call, result, dict(result, output="b"), user])
    rejected_extra("invalid_request", input=[user, call, dict(call, name="other")])
    rejected_extra("invalid_request", input=[user, dict(call, arguments="[1]"), result, user])
    rejected_extra("invalid_request", input=[user, dict(call, arguments="NaN"), result, user])
    rejected_extra("invalid_request", input=[user, dict(call, arguments=5), result, user])
    rejected_extra("invalid_request", input=[user, call, dict(result, name="other"), user])
    rejected_extra("invalid_request", input=[{"role": "user", "content": "hi", "surprise": 1}])
    rejected_extra("invalid_request", input=[{"role": "tool", "content": "hi"}])
    rejected_extra("invalid_request", input=["x"])
    rejected_extra("invalid_request", input=[{"role": "user", "content": 5}])
    rejected_extra("invalid_request", input=[{"role": "user", "content": [{"type": "input_text"}]}])
    rejected_extra("invalid_request", input=[{"role": "user", "content": [{"type": "video"}]}])
    rejected_extra("invalid_request", input=[{"role": "user"}])
    rejected_extra("invalid_request", input=[user, {"type": "custom_tool_call", "call_id": "p", "name": "patch",
                                                    "input": 5}])
    rejected_extra("unsupported_field", input=[user, call, dict(result, output=[{"type": "input_file"}]), user])


def test_every_picture_is_invalid_image():
    pic = {"type": "input_image", "image_url": IMAGE_URL, "detail": "high"}
    cases = [
        [{"role": "user", "content": [{"type": "input_text", "text": "see"}, pic]}],
        [{"role": "user", "content": [{"type": "input_image", "file_id": "file_1"}]}],
        [{"role": "user", "content": [{"type": "input_image", "image_url": "https://example.com/a.png",
                                      "surprise": 1}]}],
        [{"role": "developer", "content": [pic]}, {"role": "user", "content": "hi"}],
        [{"role": "user", "content": "go"},
         {"type": "function_call", "call_id": "c1", "name": "view_image", "arguments": "{}"},
         {"type": "function_call_output", "call_id": "c1", "output": [pic]}, {"role": "user", "content": "x"}],
    ]
    for items in cases:
        e = refused(codex_body(input=items), "invalid_image")
        assert e.message == VISION_OFF
        assert e.body == {"error": {"message": VISION_OFF, "type": "invalid_request_error", "code": "invalid_image"}}


def test_content_variants():
    r = rs.parse({"input": [{"role": "user", "content": [
        {"type": "input_text", "text": "a"}, {"type": "text", "text": "b"}, {"type": "output_text", "text": "c"},
        {"type": "refusal", "refusal": "d"}]}]})
    assert r.messages[0].content == "abcd"
    out = rs.parse({"input": [{"role": "user", "content": "go"},
                              {"type": "function_call", "call_id": "c", "name": "n", "arguments": ""},
                              {"type": "function_call_output", "call_id": "c", "output": [
                                  {"type": "output_text", "text": "x"}, {"type": "input_text", "text": "y"}]},
                              {"role": "user", "content": "k"}]})
    assert out.messages[2].content == "x\ny"
    assert out.messages[1].tool_calls[0].arguments == {}


def test_error_body_and_model_error():
    assert rs.Failure("x_code", "msg").body == {"error": {"message": "msg", "type": "invalid_request_error",
                                                          "code": "x_code"}}
    e = rs.model_error({"model": "other"}, "qwen3.8-flash-next:4bit")
    assert e.code == "model_not_found" and e.status == 400
    assert rs.model_error({"model": "qwen3.8-flash-next:4bit"}, "qwen3.8-flash-next:4bit") is None
    assert rs.model_error({}, "m") is None


# ---- responses-events ----------------------------------------------------------------------------------

def test_stream_events_and_ordering():
    s = rs.ResponseStream("qwen3.8-flash-next:4bit", rid="resp_fixture000000", created_at=1_700_000_000)
    frames = []
    frames += s.created()
    frames += s.reasoning("I ")
    frames += s.reasoning("think")
    frames += s.text("Hel")
    frames += s.text("lo")
    frames += s.function_call(ParsedToolCall("call_1", "exec_command", {"cmd": "ls", "yield_time_ms": 10000},
                                             ("cmd", "yield_time_ms")))
    frames += s.text("Done.")
    frames += s.finish("stop", rs.usage_body(6136, 6000, 29, 4))
    d = [decode(f) for f in frames]
    kinds = [k for k, _ in d]
    assert all(data["type"] == ev for ev, data in d)
    assert [data["sequence_number"] for _, data in d] == list(range(len(frames)))
    assert kinds[:2] == ["response.created", "response.in_progress"]
    assert d[0][1]["response"]["id"] == "resp_fixture000000" and d[0][1]["response"]["status"] == "in_progress"

    def first(ev, typ=None):
        return next(i for i, (k, data) in enumerate(d) if k == ev and (typ is None or data["item"]["type"] == typ))
    assert first("response.output_item.added", "reasoning") < kinds.index("response.reasoning_summary_text.delta")
    assert first("response.output_item.added", "message") < kinds.index("response.output_text.delta")
    assert d[kinds.index("response.output_text.delta")][1]["logprobs"] == []
    assert first("response.output_item.done", "reasoning") < first("response.output_item.added", "message")
    reasoning_item = d[first("response.output_item.done", "reasoning")][1]["item"]
    assert reasoning_item["summary"][0]["text"] == "I think" and reasoning_item["encrypted_content"] is None
    assert first("response.output_item.done", "message") < first("response.output_item.done", "function_call")
    call = d[first("response.output_item.done", "function_call")][1]["item"]
    assert (call["call_id"], call["name"], call["status"]) == ("call_1", "exec_command", "completed")
    assert call["arguments"] == '{"cmd":"ls","yield_time_ms":10000}'
    assert d[kinds.index("response.function_call_arguments.done")][1]["arguments"] == call["arguments"]
    msg = d[first("response.output_item.done", "message")][1]["item"]
    assert msg["content"][0]["text"] == "Hello" and msg["status"] == "completed"
    assert kinds[-1] == "response.completed"
    done = d[-1][1]["response"]
    assert done["status"] == "completed" and done["id"] == "resp_fixture000000"
    u = done["usage"]
    assert (u["input_tokens"], u["output_tokens"], u["total_tokens"]) == (6136, 29, 6165)
    assert u["input_tokens_details"] == {"cached_tokens": 6000, "cache_write_tokens": 0}
    assert u["output_tokens_details"]["reasoning_tokens"] == 4
    assert [o["type"] for o in done["output"]] == ["reasoning", "message", "function_call", "message"]
    ids = [o["id"] for o in done["output"]]
    assert len(set(ids)) == len(ids) and all("fixture00000" in i for i in ids)
    assert s.text_total == "HelloDone." and s.reasoning_total == "I think"
    assert decode(s.keepalive())[0] == "response.in_progress"


def test_text_never_precedes_its_item_announcement():
    # The falsifier: a text delta before output_item.added is dropped from Codex's live view.
    s = rs.ResponseStream("m", rid="resp_order", created_at=1)
    frames = [decode(f)[0] for f in s.text("a") + s.text("b")]
    assert frames[0] == "response.output_item.added"
    assert frames.index("response.output_item.added") < frames.index("response.output_text.delta")


def test_endings_that_are_not_a_plain_completion():
    cut = rs.ResponseStream("m", rid="resp_cut", created_at=1)
    cut.text("half")
    last = decode(cut.finish("length", usage())[-1])
    assert last[0] == "response.incomplete"
    assert last[1]["response"]["incomplete_details"] == {"reason": "max_output_tokens"}
    called = rs.ResponseStream("m", rid="resp_called", created_at=1)
    called.function_call(ParsedToolCall("c", "exec_command", {}))
    assert decode(called.finish("length", usage())[-1])[0] == "response.completed"
    failed = rs.ResponseStream("m", rid="resp_failed", created_at=1)
    failed.text("partial")
    ev, data = decode(failed.fail("context_length_exceeded", "too long")[0])
    assert ev == "response.failed"
    assert data["response"]["error"]["code"] == "context_length_exceeded" and data["response"]["status"] == "failed"
    assert data["response"]["output"] == []


def test_non_streaming_object_is_the_completed_response():
    whole = rs.ResponseStream("m", rid="resp_whole", created_at=1)
    whole.text("Hi")
    o = whole.finished("stop", rs.usage_body(2, 0, 1, 0))
    assert o["status"] == "completed" and o["output"][0]["type"] == "message" and isinstance(o["usage"], dict)
    assert o["tool_choice"] == "auto" and o["parallel_tool_calls"] is True and o["tools"] == []
    assert o["store"] is False and o["instructions"] is None and o["text"]["format"]["type"] == "text"
    assert o["previous_response_id"] is None and o["background"] is False and o["truncation"] == "disabled"
    echoed = rs.ResponseStream("m", rid="resp_echo", created_at=1, echo=rs.echoed(
        {"tool_choice": "required", "temperature": 0.2, "text": {"verbosity": "low"}}))
    e = echoed.finished("stop", rs.usage_body(1, 0, 0, 0))
    assert e["tool_choice"] == "required" and e["temperature"] == 0.2
    assert e["text"] == {"format": {"type": "text"}, "verbosity": "low"}
    assert json.loads(json.dumps(e, allow_nan=False)) == e  # the whole object is valid JSON


def test_incomplete_non_streaming():
    cut = rs.ResponseStream("m", rid="resp_cut", created_at=1)
    cut.text("half")
    o = cut.finished("length", usage())
    assert o["status"] == "incomplete" and o["incomplete_details"] == {"reason": "max_output_tokens"}


# ---- responses-replay ----------------------------------------------------------------------------------

@pytest.mark.parametrize("custom", [False, True])
@pytest.mark.parametrize("reasoning", [False, True])
def test_a_complete_response_replays_as_input(custom, reasoning):
    user = {"role": "user", "content": "Inspect the fixture."}
    name = "apply_patch" if custom else "exec_command"
    tool = {"type": "custom" if custom else "function", "name": name}
    s = rs.ResponseStream("m", rid="resp_replay", created_at=1, freeform={name} if custom else ())
    s.text("Before.")
    s.function_call(ParsedToolCall("call_1", name, {"input": "first"}))
    if reasoning:
        s.reasoning("Continue thinking.")
    else:
        s.text("After.")
    s.function_call(ParsedToolCall("call_2", name, {"input": "second"}))
    response = s.finished("stop", usage())
    kind = "custom_tool_call_output" if custom else "function_call_output"
    first = {"type": kind, "call_id": "call_1", "output": "first result"}
    second = {"type": kind, "call_id": "call_2", "output": "second result"}
    # Clients append the complete output before the tool results; cross the JSON boundary so wire values replay.
    items = json.loads(json.dumps(response))["output"]
    r = rs.parse({"input": [user] + items + [second, first], "tools": [tool]})
    assert [m.role for m in r.messages] == ["user", "assistant", "tool", "tool"]
    a = next(m for m in r.messages if m.role == "assistant")
    assert a.content == ("Before." if reasoning else "Before.After.")
    assert a.reasoning == ("Continue thinking." if reasoning else None)
    assert [c.id for c in a.tool_calls] == ["call_1", "call_2"]
    if custom:
        assert [c.arguments["input"] for c in a.tool_calls] == ["first", "second"]
    results = [m for m in r.messages if m.role == "tool"]
    assert [m.tool_call_id for m in results] == ["call_1", "call_2"]
    assert [m.content for m in results] == ["first result", "second result"]


USER = {"role": "user", "content": "Inspect the fixture."}
CALL1 = {"type": "function_call", "call_id": "call_1", "name": "exec_command", "arguments": "{}"}
CALL2 = {"type": "function_call", "call_id": "call_2", "name": "exec_command", "arguments": "{}"}
RESULT1 = {"type": "function_call_output", "call_id": "call_1", "output": "first result"}
RESULT2 = {"type": "function_call_output", "call_id": "call_2", "output": "second result"}
TEXT = {"role": "assistant", "content": "Another turn."}
THOUGHT = {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Another thought."}]}


def invalid(items, code="invalid_request"):
    with pytest.raises(rs.Failure) as e:
        rs.parse({"input": items})
    assert e.value.code == code, e.value.message


def test_replay_refusals():
    invalid([USER, CALL1, TEXT])
    invalid([USER, CALL1, RESULT1, RESULT1])
    invalid([USER, CALL1, USER, RESULT1])
    for continuation in (TEXT, THOUGHT):
        context = {"role": "developer", "content": "Updated context."}
        invalid([USER, CALL1, context, continuation, RESULT1])
        invalid([USER, CALL1, CALL2, RESULT1, continuation, RESULT2])
        call3 = dict(CALL1, call_id="call_3")
        result3 = dict(RESULT1, call_id="call_3", output="third result")
        invalid([USER, CALL1, context, CALL2, continuation, RESULT1, RESULT2])
        invalid([USER, CALL1, CALL2, RESULT1, call3, continuation, RESULT2, result3])


def test_completed_results_allow_a_new_assistant_turn():
    r = rs.parse({"input": [USER, CALL1, CALL2, RESULT2, RESULT1, THOUGHT, TEXT, USER]})
    assert [m.role for m in r.messages] == ["user", "assistant", "tool", "tool", "assistant", "user"]
    assert r.messages[4].content == "Another turn." and r.messages[4].reasoning == "Another thought."


# ---- responses-pending-turn ----------------------------------------------------------------------------

def _call(cid, custom):
    return ({"type": "custom_tool_call", "call_id": cid, "name": "patch", "input": cid} if custom else
            {"type": "function_call", "call_id": cid, "name": "echo", "arguments": "{}"})


def _result(cid, custom):
    return {"type": "custom_tool_call_output" if custom else "function_call_output", "call_id": cid,
            "output": cid + " result"}


@pytest.mark.parametrize("label, a, b, d", [("function", False, False, False), ("custom", True, True, True),
                                           ("mixed", False, True, False)])
def test_pending_turn_ownership(label, a, b, d):
    user = {"role": "user", "content": "run"}
    ca, cb, cd = _call("a", a), _call("b", b), _call("c", d)
    ra, rb, rd = _result("a", a), _result("b", b), _result("c", d)
    invalid([user, ca, cb, ra, cd, rb, rd])
    for role in ("developer", "system"):
        invalid([user, ca, {"role": role, "content": "context"}, cb, ra, rb])
    for empty in ({"role": "assistant", "content": ""}, {"type": "reasoning", "summary": []}):
        invalid([user, ca, cb, ra, empty, cd, rb, rd])
    invalid([user, ca, user, ra])
    for context in (False, True):
        middle = [{"role": "developer", "content": "context"}] if context else []
        r = rs.parse({"input": [user, ca, cb, rb, ra] + middle + [cd, rd]})
        assert [m.role for m in r.messages] == (
            ["user", "assistant", "tool", "tool", "user", "assistant", "tool"] if context else
            ["user", "assistant", "tool", "tool", "assistant", "tool"])
        assert [[c.id for c in m.tool_calls] for m in r.messages if m.role == "assistant"] == [["a", "b"], ["c"]]
        results = [m for m in r.messages if m.role == "tool"]
        assert [m.tool_call_id for m in results] == ["a", "b", "c"]
        assert [m.tool_name for m in results] == ["patch" if a else "echo", "patch" if b else "echo",
                                                   "patch" if d else "echo"]
        assert [m.content for m in results] == ["a result", "b result", "c result"]
    r = rs.parse({"input": [user, ca, {"role": "assistant", "content": "between"},
                            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thought"}]},
                            cb, rb, ra]})
    assert [m.role for m in r.messages] == ["user", "assistant", "tool", "tool"]
    assert [c.id for c in r.messages[1].tool_calls] == ["a", "b"]
    assert (r.messages[1].content, r.messages[1].reasoning) == ("between", "thought")
    assert [m.tool_call_id for m in r.messages if m.role == "tool"] == ["a", "b"]


# ---- responses-codex-tools -----------------------------------------------------------------------------

GENERATED = """I'll write the file.

<tool_call>
<function=exec_command>
<parameter=cmd>
apply_patch <<'EOF'
*** Begin Patch
*** Add File: hello.txt
+SLOTSTREAM OK
*** End Patch
EOF
</parameter>
<parameter=yield_time_ms>
10000
</parameter>
<parameter=tty>
false
</parameter>
</function>
</tool_call>"""


def test_codex_tool_set_and_apply_patch_heredoc():
    r = rs.parse({"model": "qwen3.8-flash-next:4bit", "input": "hi",
                  "tools": [EXEC_COMMAND, WRITE_STDIN, UPDATE_PLAN, VIEW_IMAGE]})
    assert [t.name for t in r.tools] == ["exec_command", "write_stdin", "update_plan", "view_image"]
    schemas = [t.schema for t in r.tools]
    assert schemas[0].params["yield_time_ms"] == "number" and schemas[0].params["tty"] == "boolean"
    assert schemas[0].params["cmd"] == "string" and schemas[1].params["session_id"] == "number"
    assert schemas[2].params["plan"] == "array"
    splitter = ToolCallSplitter(schemas, lambda: "call_fixed")
    events = splitter.push(GENERATED) + splitter.flush()
    calls = [e.call for e in events if isinstance(e, tc.ToolCall)]
    assert len(calls) == 1
    call = calls[0]
    assert call.arguments["cmd"] == ("apply_patch <<'EOF'\n*** Begin Patch\n*** Add File: hello.txt\n"
                                     "+SLOTSTREAM OK\n*** End Patch\nEOF")
    assert call.arguments["yield_time_ms"] == 10000 and call.arguments["tty"] is False
    s = rs.ResponseStream("m", rid="resp_tools", created_at=1)
    item = decode(s.function_call(call)[-1])[1]["item"]
    parsed = json.loads(item["arguments"])
    assert parsed["yield_time_ms"] == 10000 and parsed["tty"] is False
    assert len(parsed["cmd"].split("\n")) == 6


# ---- responses-codex-fixture ---------------------------------------------------------------------------

def test_real_codex_first_turn_parses():
    body = json.loads(FIXTURE.read_text())
    r = rs.parse(body)
    assert [t.name for t in r.tools] == [
        "exec_command", "write_stdin", "update_plan", "request_user_input", "view_image",
        "multi_agent_v1.close_agent", "multi_agent_v1.resume_agent", "multi_agent_v1.send_input",
        "multi_agent_v1.spawn_agent", "multi_agent_v1.wait_agent"]
    assert r.namespaces == {"multi_agent_v1"}
    assert [m.role for m in r.messages] == ["system", "user", "user"]
    assert r.messages[0].content.startswith("You are a coding agent")
    assert "<permissions instructions>" in r.messages[0].content
    assert r.messages[1].content.startswith("<environment_context>")
    assert r.messages[2].content == "Say hello"
    assert not r.thinking and r.effort is None and r.stream and r.parallel and r.choice == AUTO
    assert r.max_output_tokens is None
    assert r.tools[0].schema.params["prefix_rule"] == "array"
    assert r.tools[0].schema.params["yield_time_ms"] == "number"
    assert r.tools[3].schema.params["questions"] == "array"


def test_fixture_carries_no_local_paths():
    text = FIXTURE.read_text()
    # Built from pieces so this file itself carries no such path (the push hook scans tracked files).
    for needle in ("/" + "Users/", "/" + "home/", "/" + "private/", "claude" + "-502"):
        assert needle not in text


# ---- prepare and reply flow ----------------------------------------------------------------------------

def test_prepare_budget_and_presets():
    plain = rs.parse({"input": "hi"})
    p = rs.prepare(plain, 32768)
    assert (p.params.temperature, p.params.presence_penalty) == (0.7, 1.5)
    assert p.params.max_tokens == 8192  # the advertised budget, not the 512-token chat default
    agent = rs.prepare(rs.parse({"input": "hi", "tools": [EXEC_COMMAND]}), 32768)
    assert agent.params.temperature == 0.2 and agent.params.presence_penalty == 0.0
    thinking = rs.prepare(rs.parse({"input": "hi", "reasoning": {"effort": "high"}}), 32768)
    assert thinking.params.temperature == 1.0
    named = rs.prepare(rs.parse({"input": "hi", "tools": [EXEC_COMMAND],
                                 "tool_choice": {"type": "function", "name": "exec_command"}}), 32768)
    assert "You must call the exec_command tool now." in named.messages[0].content
    serial = rs.prepare(rs.parse({"input": "hi", "tools": [EXEC_COMMAND], "parallel_tool_calls": False}), 32768)
    assert "Call at most one tool in this response." in serial.messages[0].content
    assert rs.prepare(rs.parse({"input": "hi", "tools": [EXEC_COMMAND], "tool_choice": "none"}),
                      32768).render_tools == []
    assert rs.prepare(plain, 1000, random_seed=7).seed == 7 and rs.prepare(plain, 1000).seed > 0


def test_prompt_checks_and_room():
    req = rs.parse({"input": "hi", "max_output_tokens": 20000})
    p = rs.prepare(req, 32768)
    p.check_prompt(1000, 32768)
    assert p.params.max_tokens == 20000
    q = rs.prepare(req, 32768)
    q.check_prompt(20000, 32768)
    assert q.params.max_tokens == 12768
    last = rs.prepare(req, 32768)
    last.check_prompt(32767, 32768)
    assert last.params.max_tokens == 1
    for tokens in (32768, 40000):
        with pytest.raises(rs.Failure) as e:
            rs.prepare(req, 32768).check_prompt(tokens, 32768)
        assert e.value.code == "context_length_exceeded"
        assert e.value.message == f"prompt is {tokens} tokens, leaving no reply room in the 32768-token window"
    with pytest.raises(rs.Failure) as e:
        rs.prepare(req, 32768).check_prompt(0, 32768)
    assert e.value.code == "empty_prompt"


def run_reply(fx, text, body, chunk=3, engine_reason="stop"):
    req = rs.parse(body)
    reply = rs.ResponsesReply(rs.prepare(req, 32768), "m", rid="resp_t0000000000", created_at=1, id_factory=fx.ids())
    frames = list(reply.response.created())
    engine = fx.FakeEngine(text=text, chunk=chunk, finish_reason=engine_reason)

    def on_token(n, delta):
        frames.extend(reply.consume(delta))
        return not reply.should_stop
    produced, _, stats = engine.generate("p", on_token=on_token if reply.incremental else None,
                                         should_continue=lambda: not reply.should_stop)
    frames += reply.end(stats.finish_reason, None if reply.incremental else produced)
    return reply, frames, stats


CALL_LS = "<tool_call>\n<function=exec_command>\n<parameter=cmd>\nls\n</parameter>\n</function>\n</tool_call>"


def test_reply_with_thinking_text_and_a_call(fx):
    body = {"input": "go", "stream": True, "reasoning": {"effort": "low"}, "tools": [EXEC_COMMAND]}
    text = ("Let me check.</think>\n\nChecking.\n"
            "<tool_call>\n<function=exec_command>\n<parameter=cmd>\nls\n</parameter>\n"
            "<parameter=yield_time_ms>\n10000\n</parameter>\n</function>\n</tool_call>")
    reply, frames, stats = run_reply(fx, text, body)
    assert reply.failure() is None
    frames += reply.response.finish(stats.finish_reason, reply.usage(10, 0, stats.decode_tokens))
    d = [decode(f) for f in frames]
    added = [data["item"]["type"] for k, data in d if k == "response.output_item.added"]
    assert added == ["reasoning", "message", "function_call"]
    done = d[-1][1]["response"]
    assert done["status"] == "completed" and done["usage"]["output_tokens_details"]["reasoning_tokens"] >= 1
    call = done["output"][-1]
    assert json.loads(call["arguments"]) == {"cmd": "ls", "yield_time_ms": 10000}
    assert [data["sequence_number"] for _, data in d] == list(range(len(d)))


def test_reply_chunking_does_not_change_the_response(fx):
    body = {"input": "go", "stream": True, "reasoning": {"effort": "low"}, "tools": [EXEC_COMMAND]}
    text = "think hard</think>\n\nHello there.\n" + CALL_LS
    seen = set()
    for chunk in (1, 2, 5, 11, 1000):
        reply, frames, stats = run_reply(fx, text, body, chunk=chunk)
        o = reply.response.finished(stats.finish_reason, reply.usage(1, 0, 1))
        seen.add(json.dumps([{k: v for k, v in it.items() if k != "call_id"} for it in o["output"]], sort_keys=True))
    assert len(seen) == 1


def test_a_single_call_stops_generation(fx):
    body = {"input": "go", "stream": False, "parallel_tool_calls": False, "tools": [EXEC_COMMAND]}
    text = CALL_LS + "\n" + CALL_LS.replace("ls", "pwd")
    reply, frames, stats = run_reply(fx, text, body, chunk=4)
    assert reply.response.call_count == 1 and reply.should_stop
    assert stats.decode_tokens < len(text) // 4


def test_a_non_streamed_reply_reads_the_whole_text_once(fx):
    body = {"input": "go", "tools": [EXEC_COMMAND]}
    reply, frames, stats = run_reply(fx, "Hi.\n" + CALL_LS, body)
    assert not reply.incremental and reply.response.call_count == 1
    o = reply.response.finished("stop", reply.usage(1, 0, 1))
    assert [i["type"] for i in o["output"]] == ["message", "function_call"]


def test_a_call_cut_by_length_is_incomplete_and_not_a_failure(fx):
    body = {"input": "go", "stream": True, "tools": [EXEC_COMMAND]}
    text = "Working.\n<tool_call>\n<function=exec_command>\n<parameter=cmd>\nl"
    reply, frames, stats = run_reply(fx, text, body, engine_reason="length")
    final = reply.response.finish("length", reply.usage(1, 0, 5))
    assert decode(final[-1])[0] == "response.incomplete"


def test_a_malformed_or_undeclared_call_is_a_failure(fx):
    body = {"input": "go", "stream": True, "tools": [EXEC_COMMAND]}
    reply, _, _ = run_reply(fx, "x\n<tool_call>\n<function=exec_command>\n<parameter=cmd>\nl", body)
    assert reply.failure() == "model produced an incomplete or malformed tool call"
    undeclared = run_reply(fx, "<tool_call>\n<function=nope>\n</function>\n</tool_call>", body)[0]
    assert "undeclared tool" in undeclared.failure()
