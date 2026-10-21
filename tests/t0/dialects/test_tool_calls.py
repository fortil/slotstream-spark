"""Ports of toolcall-check, toolcall-stream-check, toolcall-coercion, tool-schema-refs and tool-schema-ref-output
(third_party/slotstream/Sources/SlotstreamTestKit/ToolCallChecks.swift), plus the common and template layers."""
import json

import pytest

from slotstream_spark.server import common, template
from slotstream_spark.server import tool_calls as tc
from slotstream_spark.server.common import ChatMessage
from slotstream_spark.server.tool_calls import ToolCallSplitter, ToolDefinition


def calls_of(events):
    return [e.call for e in events if isinstance(e, tc.ToolCall)]


def test_one_call(fx):
    ev = ToolCallSplitter.parse_all(fx.one_call, fx.tools, fx.ids())
    assert fx.normalize(ev) == ["start:read_file", 'delta:{"path":"hello.txt"}', "end",
                                'call:read_file:{"path":"hello.txt"}']
    call = ev[-1].call
    assert (call.name, call.input_json, call.id) == ("read_file", '{"path":"hello.txt"}', "call_00000001")


def test_prose_then_typed_call(fx):
    ev = ToolCallSplitter.parse_all(fx.prose_then_call, fx.tools, fx.ids())
    assert ev[0] == tc.Text("Let me read it.\n")
    # start_line is declared integer, so it must not arrive quoted.
    assert ev[-1].call.input_json == '{"path":"hello.txt","start_line":1}'


def test_two_calls_and_trailing_text(fx):
    calls = calls_of(ToolCallSplitter.parse_all(fx.two_calls, fx.tools, fx.ids()))
    assert [c.input_json for c in calls] == ['{"path":"a.txt"}', '{"path":".","depth":2}']
    assert calls[0].id != calls[1].id
    ev = ToolCallSplitter.parse_all(fx.one_call + "\nDone.", fx.tools, fx.ids())
    assert ev[-1] == tc.Text("\nDone.")


def test_unterminated_block_comes_back_verbatim(fx):
    broken = "<tool_call>\n<function=read_file>\n<parameter=path>\nhello.txt"
    ev = ToolCallSplitter.parse_all(broken, fx.tools, fx.ids())
    assert [e.text for e in ev if isinstance(e, tc.Malformed)] == [broken]
    assert not calls_of(ev)


def test_call_without_parameters_is_a_whole_object(fx):
    ev = ToolCallSplitter.parse_all("<tool_call>\n<function=list_dir>\n</function>\n</tool_call>",
                                    fx.tools, fx.ids())
    assert ev[-1].call.input_json == "{}"


def test_streamed_deltas_concatenate_to_the_input(fx):
    for fixture in (fx.one_call, fx.prose_then_call, fx.two_calls):
        events = ToolCallSplitter.parse_all(fixture, fx.tools, fx.ids())
        by_id = {}
        for e in events:
            if isinstance(e, tc.ToolInputDelta):
                by_id[e.id] = by_id.get(e.id, "") + e.delta
        for call in calls_of(events):
            assert by_id[call.id] == call.input_json


def test_empty_call_followed_by_a_real_one_is_not_swallowed(fx):
    text = "<tool_call></tool_call>" + fx.one_call
    calls = calls_of(ToolCallSplitter.parse_all(text, fx.tools, fx.ids()))
    assert [c.name for c in calls] == ["read_file"]


def test_duplicate_parameter_is_malformed(fx):
    text = ("<tool_call>\n<function=read_file>\n<parameter=path>\na\n</parameter>\n"
            "<parameter=path>\nb\n</parameter>\n</function>\n</tool_call>")
    ev = ToolCallSplitter.parse_all(text, fx.tools, fx.ids())
    assert any(isinstance(e, tc.Malformed) for e in ev) and not calls_of(ev)


# ---- streaming: every split of the same text yields the same events ----------------------------------

@pytest.mark.parametrize("label", ["one call", "prose", "two calls", "trailing"])
def test_every_split_agrees_and_no_tag_leaks(fx, label):
    fixture = {"one call": fx.one_call, "prose": fx.prose_then_call, "two calls": fx.two_calls,
               "trailing": fx.one_call + "\nDone."}[label]
    whole = fx.normalize(ToolCallSplitter.parse_all(fixture, fx.tools, fx.ids()))
    mismatches = leaked = 0
    for cut in range(len(fixture) + 1):
        s = ToolCallSplitter(fx.tools, fx.ids())
        ev = s.push(fixture[:cut]) + s.push(fixture[cut:]) + s.flush()
        if fx.normalize(ev) != whole:
            mismatches += 1
        for e in ev:
            if isinstance(e, tc.Text) and any(t in e.text for t in ("<tool_call", "<function", "<parameter",
                                                                    "</function")):
                leaked += 1
    assert mismatches == 0 and leaked == 0


@pytest.mark.parametrize("label", ["one call", "two calls"])
def test_character_by_character_agrees(fx, label):
    fixture = fx.one_call if label == "one call" else fx.two_calls
    whole = fx.normalize(ToolCallSplitter.parse_all(fixture, fx.tools, fx.ids()))
    s = ToolCallSplitter(fx.tools, fx.ids())
    ev = []
    for ch in fixture:
        ev += s.push(ch)
    ev += s.flush()
    assert fx.normalize(ev) == whole


def test_holdback_is_exactly_as_long_as_it_must_be():
    assert tc.holdback("hello", "<tool_call>") == 0
    assert tc.holdback("ab<tool", "<tool_call>") == 5
    assert tc.holdback("x<", "<tool_call>") == 1
    assert tc.holdback("x<tool_call>", "<tool_call>") == 0


def test_a_long_string_argument_streams_before_its_closing_tag(fx):
    s = ToolCallSplitter([tc.ToolSchema("write_file", {"content": "string"})], fx.ids())
    s.push("<tool_call><function=write_file><parameter=content>\n")
    fragments = 0
    for _ in range(2000):
        fragments += len(s.push("a\\\"\t\n"))
    assert fragments > 1900


# ---- coercion ----------------------------------------------------------------------------------------

def jt(raw, kind):
    return tc.json_text(tc.coerce(raw, kind))


@pytest.mark.parametrize("raw,kind,want", [
    ("\nhello.txt\n", "string", '"hello.txt"'),
    ("\na\nb\n", "string", '"a\\nb"'),
    ("\nx\n\n", "string", '"x\\n"'),
    ("\n42\n", "string", '"42"'),
    ("\n42\n", "integer", "42"),
    ("\n-7\n", "integer", "-7"),
    ("\nabc\n", "integer", '"abc"'),
    ("\n2\n", "number", "2"),
    ("\n2.5\n", "number", "2.5"),
    ("\ntrue\n", "boolean", "true"),
    ("\nfalse\n", "boolean", "false"),
    ("\nyes\n", "boolean", '"yes"'),
    ("\n[1,2]\n", "array", "[1,2]"),
    ("\nnot json\n", "array", '"not json"'),
    ('{"a":1}', "object", '{"a":1}'),
    ('{"b":1,"a":2}', "object", '{"a":2,"b":1}'),
    ("\nhello\n", "unknown", '"hello"'),
    ("\n30\n", "unknown", "30"),
    ("\ntrue\n", "unknown", "true"),
    ('{"a":1}', "unknown", '{"a":1}'),
    ('\nsay "hi"\\\n', "string", '"say \\"hi\\"\\\\"'),
])
def test_coercion_rows(raw, kind, want):
    assert jt(raw, kind) == want


@pytest.mark.parametrize("value", ["1e20", "-1e20", "9223372036854775808", "1e309", "inf", "nan"])
@pytest.mark.parametrize("kind", ["integer", "number"])
def test_out_of_range_numbers_stay_serializable(value, kind):
    json.loads(jt(value, kind))


def test_bool_is_never_an_integer_in_the_schema_sense(fx):
    # `true` where an integer is declared must stay text, not become 1.
    assert tc.coerce("\ntrue\n", "integer") == "true"
    assert tc.coerce("\n1\n", "boolean") == "1"


def test_undeclared_and_missing_parameters_pass_through(fx):
    ev = ToolCallSplitter.parse_all("<tool_call>\n<function=read_file>\n<parameter=nonesuch>\nv\n</parameter>\n"
                                    "</function>\n</tool_call>", fx.tools, fx.ids())
    assert ev[-1].call.input_json == '{"nonesuch":"v"}'
    ev = ToolCallSplitter.parse_all("<tool_call>\n<function=read_file>\n<parameter=start_line>\n3\n</parameter>\n"
                                    "</function>\n</tool_call>", fx.tools, fx.ids())
    assert ev[-1].call.input_json == '{"start_line":3}'


def test_emission_order_is_kept(fx):
    text = ("<tool_call>\n<function=read_file>\n<parameter=start_line>\n3\n</parameter>\n"
            "<parameter=path>\nz\n</parameter>\n</function>\n</tool_call>")
    call = calls_of(ToolCallSplitter.parse_all(text, fx.tools, fx.ids()))[0]
    assert call.input_json == '{"start_line":3,"path":"z"}'


def test_ids_are_call_plus_eight_hex():
    assert len(tc.random_id()) == 13 and tc.random_id().startswith("call_")
    int(tc.random_id()[5:], 16)


# ---- schema inference --------------------------------------------------------------------------------

STRING = {"type": "string"}


def ref(pointer):
    return {"$ref": pointer}


def kind(field, definitions=None):
    return ToolDefinition("echo", "", {"$defs": definitions or {}, "properties": {"value": field}}).schema.params.get("value")


DEFS = {
    "Text": STRING, "Alias": ref("#/$defs/Text"), "a/b~c": STRING, "~1": STRING, "white space": STRING,
    "雪": STRING, "": STRING,
    "Nullable": {"anyOf": [ref("#/$defs/Text"), {"type": "null"}]}, "Null": {"type": "null"},
    "List": [STRING], "Self": ref("#/$defs/Self"), "A": ref("#/$defs/B"), "B": ref("#/$defs/A"),
    "Scalar": "string", "BooleanSchema": True,
    "Resource": {"$id": "other.json", "$defs": {"Text": STRING}, "properties": {"text": ref("#/$defs/Text")}},
}


@pytest.mark.parametrize("pointer", ["#/$defs/Text", "#/$defs/Alias", "#/$defs/a~1b~0c", "#/$defs/~01",
                                     "#/$defs/white%20space", "#/$defs/%E9%9B%AA", "#/$defs/", "#/$defs/List/0",
                                     "#/%24defs/Text", "#/$defs/Nullable"])
def test_local_pointers_resolve(pointer):
    assert kind(ref(pointer), DEFS) == "string"


@pytest.mark.parametrize("pointer", ["#/$defs/Missing", "#/$defs/Self", "#/$defs/A", "#anchor",
                                     "other.json#/$defs/Text", "https://example.invalid/schema", "#/$defs/Scalar",
                                     "#/$defs/BooleanSchema", "#/$defs/List/00", "#/$defs/List/-",
                                     "#/$defs/List/1", "#/$defs/a~2b", "#/$defs/white%ZZspace", "#/$defs/Resource",
                                     "#/$defs/Resource/properties/text"])
def test_unsupported_pointers_stay_unknown(pointer):
    assert kind(ref(pointer), DEFS) == "unknown"


def test_union_and_sibling_rules():
    assert kind({"anyOf": [ref("#/$defs/Text"), ref("#/$defs/Null")]}, DEFS) == "string"
    assert kind({"anyOf": [ref("#/$defs/Text"), {"type": "number"}]}, DEFS) == "unknown"
    assert kind({"anyOf": [ref("#/$defs/Text"), ref("#/$defs/Text")]}, DEFS) == "unknown"
    assert kind({"$ref": "#/$defs/Text", "description": "Text", "default": None}, DEFS) == "string"
    assert kind({"$ref": "#/$defs/Text", "type": "number"}, DEFS) == "number"
    for pointer in ["#/$defs/Text", "#/$defs/Missing", "#/$defs/Self", "other.json#/$defs/Text", "#bad~2"]:
        assert kind({"$ref": pointer, "type": "string"}, DEFS) == "string"
    assert kind({"$ref": "#/$defs/Text", "type": ["number", "null"]}, DEFS) == "number"
    assert kind({"$ref": "#/$defs/Missing", "anyOf": [STRING, {"type": "null"}]}, DEFS) == "string"
    assert kind({"$id": "other.json", "$ref": "#/$defs/Text",
                 "anyOf": [ref("#/$defs/Text"), {"type": "null"}]}, DEFS) == "unknown"
    assert kind({"$ref": "#/$defs/Text", "allOf": [STRING]}, DEFS) == "unknown"
    assert kind({"$id": "other.json", "$ref": "#/$defs/Text"}, DEFS) == "unknown"


@pytest.mark.parametrize("ident", ["$id", "id"])
def test_resource_ids(ident):
    assert kind({ident: "other.json", "type": "string"}) == "string"
    assert kind({ident: "other.json", "anyOf": [STRING, {"type": "null"}]}) == "string"
    assert kind({ident: "other.json", "anyOf": [ref("#/$defs/Text"), {"type": "null"}]}, DEFS) == "unknown"
    assert kind(ref("#/$defs/Text"), {ident: STRING, "Text": STRING}) == "string"


def test_pointer_matches_exact_code_points():
    assert kind(ref("#/$defs/é"), {"é": STRING}) == "unknown"
    assert kind(ref("#/$defs/é"), {"é": STRING}) == "string"
    for key, pointer in [("tilde~́key", "#/$defs/tilde~0́key"),
                         ("slash/́key", "#/$defs/slash~1́key"),
                         ("́leading", "#/$defs/́leading")]:
        assert kind(ref(pointer), {key: STRING}) == "string"


def test_work_is_bounded():
    deep = STRING
    for _ in range(200):
        deep = {"anyOf": [deep, {"type": "null"}]}
    assert kind(deep) == "unknown"
    chain = {"200": STRING}
    for i in range(200):
        chain[str(i)] = ref(f"#/$defs/{i + 1}")
    assert kind(ref("#/$defs/0"), chain) == "unknown"
    assert ToolDefinition("echo", "", ref("#")).schema.params == {}


def test_root_ref_keeps_its_own_properties_and_string_bytes(fx):
    root = {"$ref": "#/$defs/Other",
            "$defs": {"Text": STRING, "Other": {"properties": {"ignored": {"type": "number"}}}},
            "properties": {"direct": STRING, "referenced": ref("#/$defs/Text")}}
    tool = ToolDefinition("echo", "", root)
    assert tool.schema.params == {"direct": "string", "referenced": "string"}
    ev = ToolCallSplitter.parse_all("<tool_call><function=echo><parameter=direct>00123</parameter></function>"
                                    "</tool_call>", [tool.schema], fx.ids())
    assert calls_of(ev)[0].arguments["direct"] == "00123"


def test_legacy_definitions_and_the_original_schema_survive():
    parameters = {"$id": "https://example.invalid/tool", "$ref": "#/definitions/Params",
                  "definitions": {"Text": STRING, "Params": {"type": "object", "properties": {
                      "first": ref("#/definitions/Text"), "second": ref("#/definitions/Text"),
                      "recursive": ref("#")}}}}
    tool = ToolDefinition("echo", "", parameters)
    assert tool.schema.params["first"] == "string" and tool.schema.params["second"] == "string"
    assert tool.schema.params["recursive"] == "object"
    assert tool.template_value["function"]["parameters"] is parameters


OUTPUT_CASES = [
    ("string", "00123", "00123"), ("string", "false", "false"), ("string", '{"flag":true}', '{"flag":true}'),
    ("string", " \t雪🙂\nnext\n", " \t雪🙂\nnext\n"),
    ("number", "1.25", 1.25), ("number", "1e-3", 0.001), ("integer", "42", 42), ("boolean", "false", False),
    ("object", '{"code":"00123","flag":"false","nested":["1",null]}',
     {"code": "00123", "flag": "false", "nested": ["1", None]}),
    ("array", '["00123","false",{"code":"00123"}]', ["00123", "false", {"code": "00123"}]),
]


@pytest.mark.parametrize("type_,raw,expected", OUTPUT_CASES)
def test_referenced_type_gives_the_same_events_as_a_direct_one(fx, type_, raw, expected):
    scalar = {"type": type_}
    direct = ToolDefinition("echo", "", {"properties": {"value": scalar}})
    referenced = ToolDefinition("echo", "", {"$defs": {"Value": scalar}, "properties": {
        "value": {"anyOf": [ref("#/$defs/Value"), {"type": "null"}]}}})
    assert referenced.schema.params["value"] == type_
    xml = f"<tool_call>\n<function=echo>\n<parameter=value>\n{raw}\n</parameter>\n</function>\n</tool_call>"
    whole = fx.normalize(ToolCallSplitter.parse_all(xml, [direct.schema], fx.ids()))
    for cut in range(len(xml) + 1):
        p = ToolCallSplitter([referenced.schema], fx.ids())
        assert fx.normalize(p.push(xml[:cut]) + p.push(xml[cut:]) + p.flush()) == whole
    p = ToolCallSplitter([referenced.schema], fx.ids())
    events = [e for ch in xml for e in p.push(ch)] + p.flush()
    assert fx.normalize(events) == whole
    call = calls_of(events)[0]
    value = call.arguments["value"]
    assert value == expected and type(value) is type(expected)
    assert call.input_json == "".join(e.delta for e in events if isinstance(e, tc.ToolInputDelta))


# ---- common ------------------------------------------------------------------------------------------

def test_strict_scalars():
    assert common.as_int(True) is None and common.as_int(1) == 1 and common.as_int(64000.0) == 64000
    assert common.as_int(100.5) is None and common.as_int("1") is None
    assert common.as_num(True) is None and common.as_num(1e300) is None and common.as_num(2) == 2.0
    assert common.as_bool(1) is None and common.as_bool(False) is False
    assert common.without_nulls({"a": None, "b": 0, "c": False}) == {"b": 0, "c": False}


@pytest.mark.parametrize("label,want", [
    ("none", (False, None)), ("minimal", (False, None)), ("off", (False, None)), ("low", (True, "low")),
    ("medium", (True, "medium")), ("high", (True, "xhigh")), ("xhigh", (True, "xhigh")), ("max", (True, "xhigh")),
    ("bogus", (True, None)), (None, (False, None)), ({"effort": "low"}, (True, "low")),
])
def test_reasoning_labels(label, want):
    r = common.map_reasoning(label)
    assert (r.thinking, r.effort) == want


def test_presets_and_budget():
    i, a, t = common.instruct_params(), common.agent_params(), common.thinking_params()
    assert (i.temperature, i.top_p, i.top_k, i.presence_penalty) == (0.7, 0.8, 20, 1.5)
    assert (a.temperature, a.top_p, a.presence_penalty) == (0.2, 0.9, 0)
    assert (t.temperature, t.top_p) == (1.0, 0.95)
    assert common.output_budget(65536) == 8192 and common.output_budget(100) == 99
    assert "--vision off" in common.VISION_OFF


# ---- template layer ----------------------------------------------------------------------------------

def test_tool_results_follow_call_order():
    c1 = tc.ParsedToolCall("c1", "read_file", {"path": "a"})
    c2 = tc.ParsedToolCall("c2", "read_file", {"path": "b"})
    msgs = [ChatMessage("user", "go"), ChatMessage("assistant", tool_calls=[c1, c2]),
            ChatMessage("tool", "second", tool_call_id="c2", tool_name="read_file"),
            ChatMessage("tool", "first", tool_call_id="c1", tool_name="read_file")]
    assert [m.content for m in template.reorder_tool_results(msgs)[2:]] == ["first", "second"]


def test_conversation_must_end_in_user_or_tool():
    with pytest.raises(template.TemplateError):
        template.require_open_turn([ChatMessage("user", "x"), ChatMessage("assistant", "y")])
    template.require_open_turn([ChatMessage("user", "x")])
    with pytest.raises(template.TemplateError):
        template.require_open_turn([])


def test_images_are_refused_by_the_template_layer():
    with pytest.raises(template.TemplateError, match="vision off"):
        template.template_messages([ChatMessage("user", "x", images=["abc"])])


def test_render_with_a_replayed_call_and_the_tool_schema(tokenizer):
    tool = ToolDefinition("read_file", "Read a file.", {"type": "object", "properties": {"path": STRING}})
    call = tc.ParsedToolCall("c1", "read_file", {"path": "a.txt", "start_line": 3}, ("path", "start_line"))
    msgs = [ChatMessage("user", "go"), ChatMessage("assistant", "", reasoning="need it", tool_calls=[call]),
            ChatMessage("tool", "fixture42", tool_call_id="c1", tool_name="read_file")]
    text = template.render(tokenizer, msgs, tools=[tool], thinking=False)
    assert '"name": "read_file"' in text
    assert "<function=read_file>\n<parameter=path>\na.txt\n</parameter>\n<parameter=start_line>\n3\n</parameter>" in text
    assert "<tool_response>\nfixture42\n</tool_response>" in text
    assert text.endswith("<think>\n\n</think>\n\n")


def test_effort_reaches_the_template(tokenizer):
    text = template.render(tokenizer, [ChatMessage("user", "x")], thinking=True, effort="low")
    assert "Reasoning effort is set to low" in text
