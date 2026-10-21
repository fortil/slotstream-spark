# SPDX-License-Identifier: MIT
# Port of ResponsesDialect.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""OpenAI Responses dialect, the wire Codex speaks (port of ResponsesDialect.swift and the v1Responses handler).

Pure over values: parsing, validation, request preparation and the event builders, no socket and no engine.

Four details of Codex's consumer decide shapes here, each a way to be silently wrong:

* `response.output_item.added` must precede the first `output_text.delta` of a message; Codex streams text only for
  an active item and drops deltas that arrive before it was announced.
* Its idle timer counts SSE events, not bytes, so a comment line does not reset it. A multi-minute prefill needs
  real `response.in_progress` events as its keepalive.
* `response.completed` must carry `response.id` and `usage`, and a stream that closes without it is an error.
  `response.incomplete` is an error too, so it is used only for the case the specification names: the reply ran
  out of tokens before it finished, with no complete call to deliver.
* `response.failed` is read for `error.code`; `context_length_exceeded` is the one code Codex treats specially.

Codex echoes a reasoning item's `summary` on the next turn and drops its raw `content`, so the model's `<think>`
text streams as summary text, which is also what comes back for the template to replay.

Every accepted scalar is checked here (finite, in range, integer where one is needed, bool never an int); the
values reach the sampler and are echoed back on the response object, so an unchecked NaN is a 500 or invalid JSON.
"""
import json
import math
import random
import time
import uuid
from dataclasses import dataclass, field

from slotstream_spark.server import common, template
from slotstream_spark.server.common import (AUTO, DISABLED, REQUIRED, VISION_OFF, ChatMessage, DialectError,
                                            ToolChoice, as_bool, as_int, as_num, map_reasoning, named_tool,
                                            without_nulls)
from slotstream_spark.server.output import OutputAccumulator
from slotstream_spark.server.think import ThinkSplitter
from slotstream_spark.server.tool_calls import (ParsedToolCall, Text, ToolCallSplitter, ToolDefinition,
                                                json_text, parse_json)

KEEPALIVE_INTERVAL_S = 10
MAX_DEPTH = 64

# The API's own fields plus the two Codex adds (`client_metadata`, `access_programs`). Pinned to the specification
# rather than to what Codex sends today, so a field the API defines and a later Codex starts sending is handled by
# the rules below instead of being fatal. Anything outside the set is refused by name.
KNOWN_FIELDS = frozenset({
    "model", "input", "instructions", "tools", "tool_choice", "parallel_tool_calls", "reasoning",
    "max_output_tokens", "temperature", "top_p", "stream", "store", "include", "metadata", "prompt_cache_key",
    "prompt_cache_retention", "safety_identifier", "user", "service_tier", "stream_options", "text", "truncation",
    "previous_response_id", "conversation", "background", "max_tool_calls", "top_logprobs", "prompt",
    "context_management", "client_metadata", "access_programs",
})
# Tools the API runs on its own servers. The model cannot run them here and showing it a tool nothing executes
# invites a call that can only fail, so they are dropped before rendering.
HOSTED_TOOL_TYPES = frozenset({"web_search", "web_search_preview", "web_search_preview_2025_03_11", "file_search"})
EFFORT_LABELS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})


class Failure(DialectError):
    """A request that never reaches the model. `code` is the stable snake_case identifier the wire carries."""

    def __init__(self, code, message, status=400):
        super().__init__(message, code=code, status=status)

    @property
    def body(self):
        return error_body(self.code, self.message)


def error_body(code, message):
    return {"error": {"message": message, "type": "invalid_request_error", "code": code}}


@dataclass
class Request:
    messages: list
    tools: list
    choice: ToolChoice
    parallel: bool
    thinking: bool
    effort: str
    max_output_tokens: int = None
    temperature: float = None
    top_p: float = None
    stream: bool = False
    # Namespaces whose members were flattened to `namespace.name`; a call to one is reported with `namespace` set.
    namespaces: frozenset = frozenset()
    # Freeform (`custom`) tools, shown to the model as functions with one `input` parameter; a call to one is
    # reported as a `custom_tool_call` carrying that text.
    freeform: frozenset = frozenset()
    echo: dict = field(default_factory=dict)


def _finite_json(value, depth=0):
    if depth > MAX_DEPTH:
        return False
    if value is None or isinstance(value, (bool, str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_json(v, depth + 1) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _finite_json(v, depth + 1) for k, v in value.items())
    return False


def model_error(body, model_name):
    """A `model_not_found` Failure for a model this server does not serve; None when the name fits."""
    if not isinstance(body, dict):
        return None
    e = common.model_error(body, model_name)
    return None if e is None else Failure("model_not_found", e)


def echoed(body):
    """The request parameters a response object repeats. The API returns `tools`, `tool_choice`,
    `parallel_tool_calls` and the sampling and text settings on every response, and the SDK models declare the first
    three as required, so a client that validates the object finds them. Values are repeated as sent, with the API's
    defaults where absent; `store` is always false and `previous_response_id` null because nothing is kept."""
    text = {"format": {"type": "text"}}
    sent = body.get("text")
    if isinstance(sent, dict):
        if isinstance(sent.get("format"), dict):
            text["format"] = sent["format"]
        if isinstance(sent.get("verbosity"), str):
            text["verbosity"] = sent["verbosity"]
    return {
        "tools": body.get("tools", []), "tool_choice": body.get("tool_choice", "auto"),
        "parallel_tool_calls": body.get("parallel_tool_calls", True), "instructions": body.get("instructions"),
        "temperature": body.get("temperature"), "top_p": body.get("top_p"),
        "max_output_tokens": body.get("max_output_tokens"), "reasoning": body.get("reasoning"), "text": text,
        "truncation": body.get("truncation", "disabled"), "metadata": body.get("metadata", {}),
        "user": body.get("user"), "service_tier": body.get("service_tier"), "previous_response_id": None,
        "background": False, "store": False,
    }


def _require(item, label, allowed):
    extra = sorted(set(item) - set(allowed))
    if extra:
        raise Failure("invalid_request", f"{label} has unsupported field(s): {', '.join(extra)}")


def _image(label):
    """This server runs text-only: every picture is refused, whatever its source."""
    raise Failure("invalid_image", VISION_OFF)


# ---- tools -------------------------------------------------------------------------------------------------

def _function_tool(tool, label):
    # `output_schema` and `defer_loading` are Codex additions with no effect on a server that renders the input
    # schema only.
    _require(tool, label, {"type", "name", "description", "parameters", "strict", "output_schema", "defer_loading"})
    name = tool.get("name")
    if not isinstance(name, str) or not name:
        raise Failure("invalid_request", f"{label} needs a name")
    if "strict" in tool and as_bool(tool["strict"]) is not False:
        raise Failure("unsupported_tool", f"{label}: strict tool schemas are not supported; omit strict or use false")
    if "description" in tool and not isinstance(tool["description"], str):
        raise Failure("invalid_request", f"{label}.description must be text")
    parameters = tool["parameters"] if "parameters" in tool else {"type": "object", "properties": {}}
    if not isinstance(parameters, dict):
        raise Failure("invalid_request", f"{label}.parameters must be a JSON Schema object")
    return ToolDefinition(name, tool.get("description", ""), parameters)


def parse_tools(value):
    """Function tools, Codex's `namespace` bundles flattened to `namespace.name` functions, and freeform `custom`
    tools. Codex calls a bundle member as `name` with a separate `namespace` field, but the model's grammar has one
    flat function name, so a member renders as `namespace.name` and a call to that dotted name is reported back
    split into the two fields Codex routes on. Hosted tools are dropped; every other type is refused by name."""
    if value is None:
        return [], set(), set(), set()
    if not (isinstance(value, list) and all(isinstance(t, dict) for t in value)):
        raise Failure("invalid_request", "tools must be an array of tool objects")
    tools, names, namespaces, freeform = [], set(), set(), set()
    for i, raw in enumerate(value):
        tool = without_nulls(raw)
        kind = tool.get("type")
        if not isinstance(kind, str):
            raise Failure("invalid_request", f"tools[{i}] needs a type")
        if kind in HOSTED_TOOL_TYPES:
            continue
        if kind == "function":
            d = _function_tool(tool, f"tools[{i}]")
            if d.name in names:
                raise Failure("invalid_request", f"tools[{i}]: duplicate function name '{d.name}'")
            names.add(d.name)
            tools.append(d)
        elif kind == "custom":
            # A freeform tool takes one block of raw text. Codex's `apply_patch` is one, with a Lark grammar the
            # model is shown in the description so it knows the syntax; the receiving tool enforces it.
            _require(tool, f"tools[{i}]", {"type", "name", "description", "format", "defer_loading"})
            name = tool.get("name")
            if not isinstance(name, str) or not name:
                raise Failure("invalid_request", f"tools[{i}] needs a name")
            if name in names:
                raise Failure("invalid_request", f"tools[{i}]: duplicate tool name '{name}'")
            names.add(name)
            description = tool["description"] if isinstance(tool.get("description"), str) else ""
            fmt = tool.get("format")
            if (isinstance(fmt, dict) and fmt.get("type") == "grammar" and isinstance(fmt.get("definition"), str)
                    and fmt["definition"]):
                syntax = fmt["syntax"] if isinstance(fmt.get("syntax"), str) else "grammar"
                description += f"\n\nThe input must follow this {syntax} grammar:\n" + fmt["definition"]
            tools.append(ToolDefinition(name, description, {
                "type": "object",
                "properties": {"input": {"type": "string", "description":
                                         "The complete raw input for this tool, exactly as it should be received."}},
                "required": ["input"]}))
            freeform.add(name)
        elif kind == "namespace":
            _require(tool, f"tools[{i}]", {"type", "name", "description", "tools"})
            ns = tool.get("name")
            if not isinstance(ns, str) or not ns or "." in ns:
                raise Failure("invalid_request", f"tools[{i}]: a namespace needs a name without dots")
            members = tool.get("tools")
            if not (isinstance(members, list) and all(isinstance(m, dict) for m in members)):
                raise Failure("invalid_request", f"tools[{i}].tools must be an array of function tools")
            about = tool["description"] if isinstance(tool.get("description"), str) else ""
            for j, raw_member in enumerate(members):
                member = without_nulls(raw_member)
                if member.get("type") != "function":
                    raise Failure("unsupported_tool", f"tools[{i}].tools[{j}]: only function tools are supported "
                                                      f"inside a namespace")
                d = _function_tool(member, f"tools[{i}].tools[{j}]")
                name = ns + "." + d.name
                if name in names:
                    raise Failure("invalid_request", f"tools[{i}].tools[{j}]: duplicate function name '{name}'")
                names.add(name)
                tools.append(ToolDefinition(name, d.description if not about else about + " " + d.description,
                                            d.parameters))
            namespaces.add(ns)
        else:
            raise Failure("unsupported_tool", f"tools[{i}]: tool type '{kind}' is not supported; only function "
                                              f"tools run here")
    return tools, names, namespaces, freeform


def parse_tool_choice(value, declared):
    if value is None:
        return AUTO
    bad = "tool_choice must be auto, none, required, or a function object"
    if isinstance(value, str):
        if value == "auto":
            return AUTO
        if value == "none":
            return DISABLED
        if value == "required":
            return REQUIRED
        raise Failure("invalid_request", bad)
    if not (isinstance(value, dict) and isinstance(value.get("type"), str)):
        raise Failure("invalid_request", bad)
    if value["type"] not in ("function", "custom"):
        raise Failure("unsupported_field", f"tool_choice type '{value['type']}' is not supported; only a named "
                                           f"function or custom tool can be forced")
    name = value.get("name")
    if not (set(value) <= {"type", "name"} and isinstance(name, str) and name in declared):
        raise Failure("invalid_request", "tool_choice must name a declared tool")
    return named_tool(name)


def parse_reasoning(value):
    if value is None:
        return False, None
    if not isinstance(value, dict):
        raise Failure("invalid_request", "reasoning must be an object")
    extra = sorted(set(value) - {"effort", "summary", "generate_summary"})
    if extra:
        raise Failure("unsupported_field", f"unsupported reasoning field(s): {', '.join(extra)}")
    # `summary` and `generate_summary` ask how much of the reasoning to show; this server always streams it.
    effort = value.get("effort")
    if effort is None:
        return False, None
    if not (isinstance(effort, str) and effort in EFFORT_LABELS):
        raise Failure("invalid_request", "reasoning.effort must be none, minimal, low, medium, high, xhigh, or max")
    mapped = map_reasoning(effort)
    return mapped.thinking, mapped.effort


# ---- input items -------------------------------------------------------------------------------------------

def _parts(value, label):
    if not (isinstance(value, list) and all(isinstance(p, dict) for p in value)):
        raise Failure("invalid_request", label)
    return [without_nulls(p) for p in value]


def parse_content(value, label, allow_empty):
    """Message content: text, or typed parts."""
    if value is None:
        if allow_empty:
            return ""
        raise Failure("invalid_request", f"{label} needs content")
    if isinstance(value, str):
        return value
    parts = _parts(value, f"{label}.content must be text or an array of parts")
    text = []
    for j, part in enumerate(parts):
        kind = part.get("type", "")
        where = f"{label}.content[{j}]"
        # A picture is named before anything else: a part with extra keys would otherwise read as a malformed
        # text part and hide the real reason.
        if kind == "input_image":
            _image(where)
        if kind in ("input_text", "output_text", "text"):
            _require(part, where, {"type", "text", "annotations", "logprobs"})
            if not isinstance(part.get("text"), str):
                raise Failure("invalid_request", f"{where} needs text")
            text.append(part["text"])
        elif kind == "refusal":
            if not isinstance(part.get("refusal"), str):
                raise Failure("invalid_request", f"{where} needs refusal text")
            text.append(part["refusal"])
        elif kind == "input_file":
            raise Failure("unsupported_field", f"{where}: file inputs are not supported; send the text")
        elif kind == "input_audio":
            raise Failure("unsupported_field", f"{where}: audio inputs are not supported")
        else:
            raise Failure("invalid_request", f"{where}: unsupported content type '{kind}'")
    return "".join(text)


def parse_tool_output(value, label):
    """A tool result: plain text, or content items."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    parts = _parts(value, f"{label}.output must be text or an array of content items")
    text = []
    for j, part in enumerate(parts):
        kind = part.get("type", "")
        where = f"{label}.output[{j}]"
        if kind in ("input_text", "output_text", "text"):
            if not isinstance(part.get("text"), str):
                raise Failure("invalid_request", f"{where} needs text")
            text.append(part["text"])
        elif kind == "input_image":
            _image(where)
        elif kind in ("input_file", "input_audio"):
            raise Failure("unsupported_field", f"{where}: only text can be returned to the model")
        elif kind == "encrypted_content":
            raise Failure("unsupported_field", f"{where}: encrypted tool output from another provider cannot be "
                                               f"replayed here")
        else:
            raise Failure("invalid_request", f"{where}: unsupported content type '{kind}'")
    return "\n".join(text)


def _reasoning_text(item, label):
    """The text of a replayed reasoning item: raw `content` when a client kept it, otherwise the summary, which is
    all Codex keeps."""
    def texts(value, types, field_):
        if value is None:
            return []
        if not (isinstance(value, list) and all(isinstance(p, dict) for p in value)):
            raise Failure("invalid_request", f"{label}.{field_} must be an array")
        out = []
        for part in value:
            if not (part.get("type") in types and isinstance(part.get("text"), str)):
                raise Failure("invalid_request", f"{label}.{field_} has an unsupported part")
            out.append(part["text"])
        return out
    content = texts(item.get("content"), ("reasoning_text", "text"), "content")
    if content:
        return "\n".join(content)
    return "\n".join(texts(item.get("summary"), ("summary_text",), "summary"))


def parse_items(items, leading_systems):
    """Conversation items to template messages.

    A Responses conversation is flat: an assistant turn arrives as separate `reasoning`, `message` and
    `function_call` items, and tool results as `function_call_output` items keyed by `call_id`. The template wants one
    assistant message per turn with its calls attached, followed by one tool message per call in call order, so
    consecutive assistant items are merged and results re-ordered by their call.

    `system` and `developer` messages before the conversation starts join the instructions. After it starts they
    render as user text, which is what Codex itself does for its context fragments: the template renders one system
    turn and rejects a later one, so a mid-conversation developer message (Codex sends one after a model or mode
    switch) would otherwise fail the whole request."""
    out = []
    assistant = None
    pending = {}  # call_id -> name
    pending_order = []
    pending_results = {}
    call_ids = set()
    started = False

    def flush_assistant():
        nonlocal assistant
        if assistant is not None:
            out.append(assistant)
            assistant = None

    def open_assistant():
        return assistant or ChatMessage("assistant", "")

    def require_open_assistant(index):
        # A continuation belongs to the turn with every pending call, before any result arrives. A later call
        # cannot reopen a turn flushed by intervening context.
        held = len(assistant.tool_calls) if assistant is not None else None
        if pending and (held != len(pending) or pending_results):
            raise Failure("invalid_request", f"input[{index}] must first supply outputs for outstanding "
                                             f"function calls")

    for i, raw in enumerate(items):
        if not isinstance(raw, dict):
            raise Failure("invalid_request", f"input[{i}] must be an object")
        item = without_nulls(raw)
        kind = item.get("type")
        if kind is not None and not isinstance(kind, str):
            raise Failure("invalid_request", f"input[{i}].type must be text")
        if kind is None:
            kind = "message" if "role" in item else ""
        label = f"input[{i}]"
        if kind == "message":
            _require(item, label, {"type", "id", "role", "content", "status", "phase",
                                   "internal_chat_message_metadata_passthrough"})
            role = item.get("role")
            if role not in ("system", "developer", "user", "assistant"):
                raise Failure("invalid_request", f"{label}.role must be system, developer, user, or assistant")
            text = parse_content(item.get("content"), label, allow_empty=role == "assistant")
            if role in ("system", "developer"):
                if started or assistant is not None:
                    flush_assistant()
                    out.append(ChatMessage("user", text))
                else:
                    leading_systems.append(text)
            elif role == "user":
                flush_assistant()
                started = True
                if pending:
                    raise Failure("invalid_request", f"{label} must first supply outputs for outstanding "
                                                     f"function calls")
                out.append(ChatMessage("user", text))
            else:
                started = True
                require_open_assistant(i)
                a = open_assistant()
                a.content += text
                assistant = a
        elif kind == "reasoning":
            _require(item, label, {"type", "id", "summary", "content", "encrypted_content", "status",
                                   "internal_chat_message_metadata_passthrough"})
            enc = item.get("encrypted_content")
            if enc is not None and not isinstance(enc, str):
                raise Failure("invalid_request", f"{label}.encrypted_content must be text")
            if enc:
                raise Failure("unsupported_field", f"{label}: encrypted reasoning from another provider cannot be "
                                                   f"replayed here")
            started = True
            require_open_assistant(i)
            text = _reasoning_text(item, label)
            if text:
                a = open_assistant()
                a.reasoning = text if a.reasoning is None else a.reasoning + "\n" + text
                assistant = a
        elif kind == "function_call":
            require_open_assistant(i)
            _require(item, label, {"type", "id", "call_id", "name", "arguments", "status", "namespace",
                                   "internal_chat_message_metadata_passthrough"})
            bare = item.get("name")
            if not isinstance(bare, str) or not bare:
                raise Failure("invalid_request", f"{label} needs a function name")
            ns = item.get("namespace", "")
            if not isinstance(ns, str):
                raise Failure("invalid_request", f"{label}.namespace must be text")
            name = bare if not ns else ns + "." + bare
            cid = item.get("call_id")
            if not isinstance(cid, str) or not cid:
                raise Failure("invalid_request", f"{label} needs a call_id")
            if cid in call_ids:
                raise Failure("invalid_request", f"{label}: duplicate call_id '{cid}'")
            call_ids.add(cid)
            args_text = item.get("arguments", "")
            if not isinstance(args_text, str):
                raise Failure("invalid_request", f"{label}.arguments must be a JSON object encoded as text")
            if not args_text.strip():
                arguments = {}
            else:
                ok, parsed = parse_json(args_text)
                if not (ok and isinstance(parsed, dict)):
                    raise Failure("invalid_request", f"{label}.arguments must be a JSON object encoded as text")
                arguments = parsed
            started = True
            a = open_assistant()
            a.tool_calls.append(ParsedToolCall(cid, name, arguments, tuple(arguments)))
            assistant = a
            pending[cid] = name
            pending_order.append(cid)
        elif kind == "custom_tool_call":
            require_open_assistant(i)
            _require(item, label, {"type", "id", "call_id", "name", "input", "status", "namespace",
                                   "internal_chat_message_metadata_passthrough"})
            name = item.get("name")
            if not isinstance(name, str) or not name:
                raise Failure("invalid_request", f"{label} needs a tool name")
            cid = item.get("call_id")
            if not isinstance(cid, str) or not cid:
                raise Failure("invalid_request", f"{label} needs a call_id")
            if cid in call_ids:
                raise Failure("invalid_request", f"{label}: duplicate call_id '{cid}'")
            call_ids.add(cid)
            if not isinstance(item.get("input"), str):
                raise Failure("invalid_request", f"{label}.input must be text")
            started = True
            a = open_assistant()
            a.tool_calls.append(ParsedToolCall(cid, name, {"input": item["input"]}, ("input",)))
            assistant = a
            pending[cid] = name
            pending_order.append(cid)
        elif kind in ("function_call_output", "custom_tool_call_output"):
            _require(item, label, {"type", "id", "call_id", "output", "status", "name", "namespace",
                                   "internal_chat_message_metadata_passthrough"})
            cid = item.get("call_id")
            if not isinstance(cid, str) or cid not in pending:
                raise Failure("invalid_request", f"{label}: function_call_output must answer an outstanding "
                                                 f"call_id exactly once")
            name = pending.pop(cid)
            supplied = item.get("name")
            if supplied is not None and not isinstance(supplied, str):
                raise Failure("invalid_request", f"{label}.name must be text")
            if supplied and supplied != name:
                raise Failure("invalid_request", f"{label}: output name does not match its call")
            flush_assistant()
            pending_results[cid] = ChatMessage("tool", parse_tool_output(item.get("output"), label),
                                               tool_call_id=cid, tool_name=name)
            if not pending:
                out.extend(pending_results[c] for c in pending_order)
                pending_order.clear()
                pending_results.clear()
        elif kind == "":
            raise Failure("invalid_request", f"{label} needs a type")
        else:
            raise Failure("unsupported_item", f"{label}: item type '{kind}' is not supported")
    flush_assistant()
    if pending:
        raise Failure("invalid_request", f"input is missing outputs for function call(s): "
                                         f"{', '.join(pending_order)}")
    return out


def parse(raw):
    """Validate a request body and read it into a Request. Raises Failure with the code and sentence to send."""
    if not isinstance(raw, dict):
        raise Failure("invalid_request", "request body must be a JSON object")
    body = without_nulls(raw)
    unknown = sorted(set(body) - KNOWN_FIELDS)
    if unknown:
        raise Failure("unsupported_field", f"unsupported request field(s): {', '.join(unknown)}")
    if not _finite_json(body):
        raise Failure("invalid_request", "the request contains a number that is not finite or is nested too deeply")
    # Server-side state. This server stores nothing, so every request must carry its whole conversation, which is
    # what Codex does (`store: false`).
    for key in ("previous_response_id", "conversation"):
        if key in body:
            raise Failure("stored_state_unsupported",
                          f"{key} is not supported: this server stores no responses; send the full input each time")
    if "prompt" in body:
        raise Failure("unsupported_field", "prompt templates are not supported; send instructions and input")
    if "context_management" in body:
        raise Failure("unsupported_field", "context_management is not supported; the client owns compaction")
    if "background" in body:
        b = as_bool(body["background"])
        if b is None:
            raise Failure("invalid_request", "background must be true or false")
        if b:
            raise Failure("unsupported_field", "background responses are not supported; stream or wait for the reply")
    for key in ("max_tool_calls", "top_logprobs"):
        if key in body:
            raise Failure("unsupported_field", f"{key} is not supported")
    if "truncation" in body:
        if not (isinstance(body["truncation"], str) and body["truncation"] in ("disabled", "auto")):
            raise Failure("invalid_request", 'truncation must be "disabled" or "auto"')
        if body["truncation"] == "auto":
            raise Failure("unsupported_field", 'truncation "auto" is not supported: this server never drops '
                                               'context; send a shorter input')
    if "text" in body:
        text = body["text"]
        if not isinstance(text, dict):
            raise Failure("invalid_request", "text must be an object")
        extra = sorted(set(text) - {"format", "verbosity"})
        if extra:
            raise Failure("unsupported_field", f"unsupported text field(s): {', '.join(extra)}")
        fmt = text.get("format")
        if fmt is not None and not (isinstance(fmt, dict) and fmt.get("type") == "text"):
            raise Failure("unsupported_field", 'text.format is not supported for constrained output; only '
                                               '{"type": "text"} is supported')
        verbosity = text.get("verbosity")
        if verbosity is not None and not (isinstance(verbosity, str) and verbosity in ("low", "medium", "high")):
            raise Failure("invalid_request", "text.verbosity must be low, medium, or high")
    # Accepted without effect: `store` (nothing is stored either way, and the API's default is true, so refusing it
    # would refuse every stock client), `include` (encrypted reasoning does not exist here), `metadata`,
    # `prompt_cache_key`, `prompt_cache_retention`, `safety_identifier`, `user`, `service_tier`, `stream_options`,
    # `client_metadata`, `access_programs`. Their types are still checked so a malformed request is named.
    if "store" in body and as_bool(body["store"]) is None:
        raise Failure("invalid_request", "store must be true or false")
    if "include" in body and not isinstance(body["include"], list):
        raise Failure("invalid_request", "include must be an array")
    if "metadata" in body and not isinstance(body["metadata"], dict):
        raise Failure("invalid_request", "metadata must be an object")
    for key in ("prompt_cache_key", "safety_identifier", "user", "service_tier", "prompt_cache_retention"):
        if key in body and not isinstance(body[key], str):
            raise Failure("invalid_request", f"{key} must be text")
    for key in ("stream_options", "client_metadata", "access_programs"):
        if key in body and not isinstance(body[key], dict):
            raise Failure("invalid_request", f"{key} must be an object")
    if "model" in body and not isinstance(body["model"], str):
        raise Failure("invalid_request", "model must be text")

    stream = False
    if "stream" in body:
        stream = as_bool(body["stream"])
        if stream is None:
            raise Failure("invalid_request", "stream must be true or false")
    max_output = temperature = top_p = None
    if "max_output_tokens" in body:
        max_output = as_int(body["max_output_tokens"])
        if max_output is None or max_output <= 0:
            raise Failure("invalid_request", "max_output_tokens must be a positive integer")
    if "temperature" in body:
        temperature = as_num(body["temperature"])
        if temperature is None:
            raise Failure("invalid_request", "temperature must be a number")
    if "top_p" in body:
        top_p = as_num(body["top_p"])
        if top_p is None:
            raise Failure("invalid_request", "top_p must be a number")

    tools, names, namespaces, freeform = parse_tools(body.get("tools"))
    choice = parse_tool_choice(body.get("tool_choice"), names)
    if choice == REQUIRED and not tools:
        raise Failure("invalid_request", "tool_choice required needs at least one tool")
    parallel = True
    if "parallel_tool_calls" in body:
        parallel = as_bool(body["parallel_tool_calls"])
        if parallel is None:
            raise Failure("invalid_request", "parallel_tool_calls must be true or false")
    thinking, effort = parse_reasoning(body.get("reasoning"))

    systems = []
    if "instructions" in body:
        if not isinstance(body["instructions"], str):
            raise Failure("invalid_request", "instructions must be text")
        if body["instructions"]:
            systems.append(body["instructions"])
    value = body.get("input")
    if value is None:
        raise Failure("invalid_request", "input is required: text or an array of items")
    if isinstance(value, str):
        if not value:
            raise Failure("invalid_request", "input must not be empty")
        messages = [ChatMessage("user", value)]
    elif isinstance(value, list):
        if not value:
            raise Failure("invalid_request", "input must not be empty")
        messages = parse_items(value, systems)
    else:
        raise Failure("invalid_request", "input must be text or an array of items")
    if not messages:
        raise Failure("invalid_request", "input needs at least one user, assistant, or tool item")
    if messages[-1].role == "assistant":
        raise Failure("assistant_prefill_unsupported", "input may not end with an assistant item")
    if systems:
        messages.insert(0, ChatMessage("system", "\n\n".join(systems)))
    return Request(messages, tools, choice, parallel, thinking, effort, max_output, temperature, top_p, stream,
                   frozenset(namespaces), frozenset(freeform), echoed(body))


# ---- preparation -------------------------------------------------------------------------------------------

def instructing(messages, line):
    """Append an instruction to the system turn, creating one when there is none."""
    out = list(messages)
    for i, m in enumerate(out):
        if m.role == "system":
            out[i] = ChatMessage("system", m.content + "\n\n" + line)
            return out
    return [ChatMessage("system", line)] + out


def prompt(request):
    """(messages, render_tools): the tool choice is written into the system prompt as the other dialects do."""
    render_tools = [] if request.choice == DISABLED else request.tools
    messages = request.messages
    if request.choice.is_named_tool:
        messages = instructing(messages, f"You must call the {request.choice.name} tool now.")
    elif request.choice == REQUIRED:
        messages = instructing(messages, "You must call one of the available tools now.")
    if not request.parallel and render_tools:
        messages = instructing(messages, "Call at most one tool in this response.")
    return messages, render_tools


def render_prompt(request, tokenizer):
    messages, render_tools = prompt(request)
    return template.render(tokenizer, messages, tools=render_tools, thinking=request.thinking,
                           effort=request.effort)


@dataclass
class Prepared:
    request: Request
    messages: list
    render_tools: list
    params: object
    seed: int

    def check_prompt(self, prompt_tokens, window):
        """The context errors once the prompt is tokenized; clamps the reply budget to the room left. Codex sends no
        output limit, so the default is the advertised budget bounded by the room in the window, never the 512-token
        chat default, which truncates every real edit."""
        if prompt_tokens <= 0:
            raise Failure("empty_prompt", "input must not be empty")
        if prompt_tokens >= window:
            raise Failure("context_length_exceeded", f"prompt is {prompt_tokens} tokens, leaving no reply room in "
                                                     f"the {window}-token window")
        room = max(1, window - prompt_tokens)
        wanted = self.request.max_output_tokens or common.output_budget(window)
        self.params.max_tokens = min(wanted, room)


def prepare(request, window, random_seed=None):
    messages, render_tools = prompt(request)
    params = (common.agent_params() if render_tools else
              (common.thinking_params() if request.thinking else common.instruct_params()))
    if request.temperature is not None:
        params.temperature = request.temperature
    if request.top_p is not None:
        params.top_p = request.top_p
    params.max_tokens = request.max_output_tokens or common.output_budget(window)
    params = params.sanitized()
    seed = random.randint(1, (1 << 64) - 1) if random_seed is None else random_seed
    return Prepared(request, messages, render_tools, params, seed)


# ---- wire --------------------------------------------------------------------------------------------------

def new_id():
    return "resp_" + uuid.uuid4().hex


def call_id():
    return "call_" + uuid.uuid4().hex


def usage_body(input_tokens, cached_tokens, output_tokens, reasoning_tokens):
    return {"input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": cached_tokens, "cache_write_tokens": 0},
            "output_tokens": output_tokens, "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
            "total_tokens": input_tokens + output_tokens}


class ResponseStream:
    """One response as it is produced: the output items in order, the item currently open, and the event frames
    that describe each step. Every method returns the frames to write and writes nothing; the non-streaming reply is
    the same object the `response.completed` frame carries."""

    def __init__(self, model, rid=None, created_at=None, namespaces=(), freeform=(), echo=None, clock=time.time):
        self.model = model
        self.namespaces, self.freeform = frozenset(namespaces), frozenset(freeform)
        self.echo = dict(echoed({}) if echo is None else echo)
        self.id = rid or new_id()
        self.created_at = int(clock()) if created_at is None else created_at
        self.sequence = 0
        self.items = []
        self.open_text = None  # [item_id, index, text]
        self.open_reasoning = None
        self.counter = 0
        self.call_count = 0
        self.text_total = ""
        self.reasoning_total = ""

    def frame(self, type_, fields=None):
        """One SSE frame: the event line, then `data: ` and compact JSON, then a blank line. Codex reads only the
        data; the event line is what the specification shows and what other consumers key on."""
        obj = dict(fields or {}, type=type_, sequence_number=self.sequence)
        self.sequence += 1
        return f"event: {type_}\ndata: " + json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                                        separators=(",", ":")) + "\n\n"

    def _item_id(self, prefix):
        self.counter += 1
        return f"{prefix}_{self.id[len('resp_'):][:12]}_{self.counter}"

    def response_object(self, status, usage=None, error=None, incomplete_reason=None):
        r = dict(self.echo)
        r.update({"id": self.id, "object": "response", "created_at": self.created_at, "status": status,
                  "model": self.model, "output": list(self.items), "error": error,
                  "incomplete_details": None if incomplete_reason is None else {"reason": incomplete_reason}})
        if usage is not None:
            r["usage"] = usage
        return r

    def created(self):
        """The frames that open a stream: `response.created`, then an `in_progress` so the client has two events
        before any token."""
        return [self.frame("response.created", {"response": self.response_object("in_progress")}),
                self.keepalive()]

    def keepalive(self):
        """A real event, not a comment: Codex's idle timer only resets on an event, and a cold prefill is minutes."""
        return self.frame("response.in_progress", {"response": self.response_object("in_progress")})

    def reasoning(self, delta):
        if not delta:
            return []
        out = self._close_text()
        if self.open_reasoning is None:
            item_id, index = self._item_id("rs"), len(self.items)
            self.open_reasoning = [item_id, index, ""]
            out.append(self.frame("response.output_item.added", {"output_index": index, "item": {
                "id": item_id, "type": "reasoning", "summary": [], "encrypted_content": None}}))
            out.append(self.frame("response.reasoning_summary_part.added", {
                "item_id": item_id, "output_index": index, "summary_index": 0,
                "part": {"type": "summary_text", "text": ""}}))
        self.open_reasoning[2] += delta
        self.reasoning_total += delta
        out.append(self.frame("response.reasoning_summary_text.delta", {
            "item_id": self.open_reasoning[0], "output_index": self.open_reasoning[1], "summary_index": 0,
            "delta": delta}))
        return out

    def text(self, delta):
        if not delta:
            return []
        out = self._close_reasoning()
        if self.open_text is None:
            item_id, index = self._item_id("msg"), len(self.items)
            self.open_text = [item_id, index, ""]
            out.append(self.frame("response.output_item.added", {"output_index": index, "item": {
                "id": item_id, "type": "message", "role": "assistant", "status": "in_progress", "content": []}}))
            out.append(self.frame("response.content_part.added", {
                "item_id": item_id, "output_index": index, "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []}}))
        self.open_text[2] += delta
        self.text_total += delta
        out.append(self.frame("response.output_text.delta", {
            "item_id": self.open_text[0], "output_index": self.open_text[1], "content_index": 0, "delta": delta,
            "logprobs": []}))
        return out

    def function_call(self, call):
        """A complete call. The parser publishes a call only after its closing tag, so the arguments arrive whole:
        one delta, then done."""
        out = self._close_reasoning() + self._close_text()
        if call.name in self.freeform:
            return out + self._custom_tool_call(call)
        item_id, index = self._item_id("fc"), len(self.items)
        arguments = call.input_json
        name, namespace = call.name, None
        if "." in call.name and call.name.split(".", 1)[0] in self.namespaces:
            namespace, name = call.name.split(".", 1)
        opened = {"id": item_id, "type": "function_call", "status": "in_progress", "call_id": call.id,
                  "name": name, "arguments": ""}
        item = {"id": item_id, "type": "function_call", "status": "completed", "call_id": call.id, "name": name,
                "arguments": arguments}
        if namespace is not None:
            opened["namespace"] = item["namespace"] = namespace
        out.append(self.frame("response.output_item.added", {"output_index": index, "item": opened}))
        out.append(self.frame("response.function_call_arguments.delta", {
            "item_id": item_id, "output_index": index, "delta": arguments}))
        out.append(self.frame("response.function_call_arguments.done", {
            "item_id": item_id, "output_index": index, "arguments": arguments}))
        self.items.append(item)
        self.call_count += 1
        out.append(self.frame("response.output_item.done", {"output_index": index, "item": item}))
        return out

    def _custom_tool_call(self, call):
        """A freeform call: the model's `input` argument is the tool's raw text, delivered as one item."""
        item_id, index = self._item_id("ctc"), len(self.items)
        value = call.arguments.get("input", "")
        text = value if isinstance(value, str) else json_text(value)
        out = [self.frame("response.output_item.added", {"output_index": index, "item": {
            "id": item_id, "type": "custom_tool_call", "status": "in_progress", "call_id": call.id,
            "name": call.name, "input": ""}}),
               self.frame("response.custom_tool_call_input.delta", {
                   "item_id": item_id, "output_index": index, "delta": text}),
               self.frame("response.custom_tool_call_input.done", {
                   "item_id": item_id, "output_index": index, "input": text})]
        item = {"id": item_id, "type": "custom_tool_call", "status": "completed", "call_id": call.id,
                "name": call.name, "input": text}
        self.items.append(item)
        self.call_count += 1
        out.append(self.frame("response.output_item.done", {"output_index": index, "item": item}))
        return out

    def _close_text(self):
        if self.open_text is None:
            return []
        item_id, index, text = self.open_text
        self.open_text = None
        part = {"type": "output_text", "text": text, "annotations": []}
        item = {"id": item_id, "type": "message", "role": "assistant", "status": "completed", "content": [part]}
        self.items.append(item)
        return [self.frame("response.output_text.done", {"item_id": item_id, "output_index": index,
                                                         "content_index": 0, "text": text, "logprobs": []}),
                self.frame("response.content_part.done", {"item_id": item_id, "output_index": index,
                                                          "content_index": 0, "part": part}),
                self.frame("response.output_item.done", {"output_index": index, "item": item})]

    def _close_reasoning(self):
        if self.open_reasoning is None:
            return []
        item_id, index, text = self.open_reasoning
        self.open_reasoning = None
        part = {"type": "summary_text", "text": text}
        item = {"id": item_id, "type": "reasoning", "summary": [part], "encrypted_content": None}
        self.items.append(item)
        return [self.frame("response.reasoning_summary_text.done", {"item_id": item_id, "output_index": index,
                                                                    "summary_index": 0, "text": text}),
                self.frame("response.reasoning_summary_part.done", {"item_id": item_id, "output_index": index,
                                                                    "summary_index": 0, "part": part}),
                self.frame("response.output_item.done", {"output_index": index, "item": item})]

    def is_incomplete(self, engine_reason):
        """A delivered call is whole by construction, so a turn that made one and then hit the limit still
        completed: the client's next move is to run the call, and `incomplete` would make Codex discard it."""
        return engine_reason == "length" and self.call_count == 0

    def finish(self, engine_reason, usage):
        out = self._close_reasoning() + self._close_text()
        if self.is_incomplete(engine_reason):
            out.append(self.frame("response.incomplete", {"response": self.response_object(
                "incomplete", usage, incomplete_reason="max_output_tokens")}))
        else:
            out.append(self.frame("response.completed", {"response": self.response_object("completed", usage)}))
        return out

    def finished(self, engine_reason, usage):
        """The non-streaming reply for a finished generation."""
        self._close_reasoning()
        self._close_text()
        if self.is_incomplete(engine_reason):
            return self.response_object("incomplete", usage, incomplete_reason="max_output_tokens")
        return self.response_object("completed", usage)

    def fail(self, code, message):
        """An error after the head went out. Nothing open is closed as complete: a partial item must not read as a
        finished one."""
        self.open_text = self.open_reasoning = None
        return [self.frame("response.failed", {"response": self.response_object(
            "failed", error={"code": code, "message": message})})]


class ResponsesReply:
    """One response in flight: engine deltas through the reasoning splitter, the tool-call parser and the
    accumulator, out as Responses frames."""

    def __init__(self, prepared, model, rid=None, created_at=None, id_factory=None):
        self.prepared = prepared
        req = prepared.request
        self.response = ResponseStream(model, rid, created_at, req.namespaces, req.freeform, req.echo)
        self.acc = OutputAccumulator(prepared.render_tools, req.choice, req.parallel)
        self.thinker = ThinkSplitter() if req.thinking else None
        self.parser = (ToolCallSplitter([t.schema for t in prepared.render_tools], id_factory or call_id)
                       if prepared.render_tools else None)
        # One per reasoning delta, as the Swift handler counts: the engine reports no separate reasoning count.
        self.reasoning_tokens = 0
        self.published = 0
        self.incremental = req.stream or (not req.parallel and bool(prepared.render_tools))

    @property
    def should_stop(self):
        return self.acc.error is not None or self.acc.finished_single_call

    def _events(self, body):
        return self.parser.push(body) if self.parser else [Text(body)]

    def _publish(self, deltas):
        frames = []
        for d in deltas:
            if "content" in d:
                frames += self.response.text(d["content"])
            if "tool_calls" in d and self.published < len(self.acc.calls):
                frames += self.response.function_call(self.acc.calls[self.published])
                self.published += 1
        return frames

    def _think(self, thought):
        if not thought:
            return []
        self.reasoning_tokens += 1
        self.acc.reasoning_delta(thought)
        return self.response.reasoning(thought)

    def consume(self, delta):
        frames, body = [], delta
        if self.thinker:
            thought, body = self.thinker.push(delta)
            frames += self._think(thought)
        if body:
            frames += self._publish(self.acc.consume(self._events(body)))
        return frames

    def end(self, engine_reason, text=None):
        """Flush the splitters. `text` is the whole reply when deltas were not read incrementally. Returns the
        frames still owed; `failure()` says whether the turn failed."""
        frames = []
        if text is not None and not self.incremental:
            frames += self.consume(text)
        if self.thinker:
            thought, body = self.thinker.flush()
            frames += self._think(thought)
            if body:
                frames += self._publish(self.acc.consume(self._events(body)))
        if self.parser:
            frames += self._publish(self.acc.consume(self.parser.flush()))
        self.acc.finish_reason(engine_reason)
        return frames

    def failure(self):
        return self.acc.error

    def usage(self, prompt_tokens, cached_tokens, output_tokens):
        return usage_body(prompt_tokens, cached_tokens, output_tokens, self.reasoning_tokens)


__all__ = ["Failure", "Prepared", "Request", "ResponseStream", "ResponsesReply", "echoed", "error_body",
           "model_error", "parse", "prepare", "usage_body"]
