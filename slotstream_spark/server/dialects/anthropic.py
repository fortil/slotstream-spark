# SPDX-License-Identifier: MIT
# Port of AnthropicDialect.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""Anthropic Messages dialect (port of AnthropicDialect.swift and the v1Messages / count_tokens handlers).

Pure over values: parsing, validation, request preparation and the event builders, no socket and no engine.
Differences from the OpenAI dialects, each read from Claude Code's own requests:

* Unknown top-level fields are ignored and reported, not refused: Claude Code adds a field in most releases and a
  refusal would break every update. Fields that ask for something this server cannot do are refused by name.
* The attribution line at the top of the system prompt changes with every conversation; keeping it would give
  every prompt a different first line and defeat prefix reuse, so it is removed.
* A thinking block's signature carries its own text, so a client that asked for `display: "omitted"` and replays
  the empty block does not lose the reasoning.
* Every accepted scalar is checked here (finite, in range, integer where an integer is needed). The values reach
  the sampler, and an unchecked NaN or an out-of-range integer there is a 500, not a 400.
"""
import base64
import binascii
import json
import math
import random
import uuid
from dataclasses import dataclass, field

from slotstream_spark.server import common, template
from slotstream_spark.server.common import (AUTO, DISABLED, REQUIRED, VISION_OFF, ChatMessage, DialectError,
                                            ToolChoice, as_bool, as_int, as_num, map_reasoning, named_tool,
                                            without_nulls)
from slotstream_spark.server.output import OutputAccumulator
from slotstream_spark.server.think import ThinkSplitter
from slotstream_spark.server.tool_calls import (ParsedToolCall, Text, ToolCallSplitter, ToolDefinition)

HANDLED_FIELDS = frozenset({"model", "messages", "max_tokens", "system", "stop_sequences", "stream", "temperature",
                            "top_p", "top_k", "tools", "tool_choice", "thinking", "output_config"})
# Their types are checked; they change nothing here. `context_management` asks the API to clear old thinking or
# tool results, and keeping them costs context, never correctness.
INERT_FIELDS = frozenset({"metadata", "context_management", "service_tier"})
REFUSED_FIELDS = {
    "container": "code execution containers are not available; tools run in the client",
    "mcp_servers": "remote MCP servers are not available; connect MCP tools in the client",
}
# Tool types the API runs on its own servers. Showing the model a tool nothing can run invites a call that can
# only fail, so they are dropped before rendering.
HOSTED_TOOL_PREFIXES = ("web_search_", "web_fetch_", "code_execution_", "tool_search_tool_", "mcp_toolset",
                        "advisor_")
EFFORT_LABELS = frozenset({"low", "medium", "high", "xhigh", "max"})
ATTRIBUTION_PREFIX = "x-anthropic-billing-header:"
SIGNATURE_PREFIX = "slotstream.thinking.v1."
IMAGE_PREFIX = "Could not process image: "
IGNORED_HEADER = "x-slotstream-ignored-fields"
KEEPALIVE_INTERVAL_S = 10
MAX_IGNORED_NAMES = 16
MAX_IGNORED_NAME_CHARS = 64
MAX_DEPTH = 64


class Failure(DialectError):
    """A request that never reaches the model, in Anthropic's error shape. `type` is the Anthropic error type."""

    def __init__(self, message, type_="invalid_request_error", status=400):
        super().__init__(message, code=type_, status=status)
        self.type = type_

    @property
    def body(self):
        return error_body(self.type, self.message)


def error_body(type_, message):
    return {"type": "error", "error": {"type": type_, "message": message}}


def error_type(http_status):
    """The Anthropic error type for an HTTP status. Claude Code retries `overloaded_error`, so a transient memory
    refusal maps there; everything else keeps the meaning of its status."""
    return {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
            404: "not_found_error", 413: "request_too_large", 431: "request_too_large", 429: "rate_limit_error",
            503: "overloaded_error", 529: "overloaded_error"}.get(int(http_status), "api_error")


def model_error(raw, model_name):
    """A 404 not_found_error Failure for a model this server does not serve; None when the name fits (or is
    absent, which parse() then refuses)."""
    if not isinstance(raw, dict):
        return None
    e = common.model_error(raw, model_name)
    return None if e is None else Failure("model: " + e, "not_found_error", 404)


def prompt_too_long(tokens, maximum):
    """Claude Code matches this exact form to start its own compaction, so the wording is the contract."""
    return f"prompt is too long: {tokens} tokens > {maximum} maximum"


@dataclass
class Request:
    messages: list
    tools: list
    choice: ToolChoice
    parallel: bool
    thinking: bool
    effort: str
    show_thinking: bool
    max_tokens: int  # 0 only for a token count, which needs no reply budget
    temperature: float = None
    top_p: float = None
    top_k: int = None
    stop_sequences: list = field(default_factory=list)
    stream: bool = False
    ignored: list = field(default_factory=list)


# ---- scalars -----------------------------------------------------------------------------------------------

def _finite_json(value, depth=0):
    """A JSON value with no NaN/Infinity and a bounded depth: tool inputs are replayed into the template."""
    if depth > MAX_DEPTH:
        return False
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_json(v, depth + 1) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _finite_json(v, depth + 1) for k, v in value.items())
    return False


def header_safe_field_names(names):
    """Names safe to repeat in a header and a log line. They come from the client's JSON, so anything but a plain
    identifier is left out."""
    ok = [n for n in names if n and len(n) <= MAX_IGNORED_NAME_CHARS and n.isascii()
          and all(c.isalnum() or c in "_-." for c in n)]
    return ok[:MAX_IGNORED_NAMES]


# ---- tools -------------------------------------------------------------------------------------------------

def parse_tools(value):
    """The declared tools and the names of the server tools that were dropped."""
    if value is None:
        return [], set()
    if not isinstance(value, list):
        raise Failure("tools: must be an array")
    tools, names, hosted = [], set(), set()
    for i, element in enumerate(value):
        if not isinstance(element, dict):
            raise Failure(f"tools.{i}: must be an object")
        tool = without_nulls(element)
        kind = tool.get("type", "custom")
        if not isinstance(kind, str):
            raise Failure(f"tools.{i}.type: must be text")
        if kind != "custom":
            if kind.startswith(HOSTED_TOOL_PREFIXES):
                if isinstance(tool.get("name"), str):
                    hosted.add(tool["name"])
                continue
            raise Failure(f"tools.{i}: tool type '{kind}' is not supported here; declare the tool with a name, "
                          f"description and input_schema")
        # `cache_control`, `strict`, `defer_loading`, `eager_input_streaming`, `input_examples` and
        # `allowed_callers` change delivery, not the schema the model is shown. `strict` is not enforced.
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise Failure(f"tools.{i}.name: field required")
        if name in names:
            raise Failure(f"tools: tool names must be unique; '{name}' appears twice")
        names.add(name)
        if "description" in tool and not isinstance(tool["description"], str):
            raise Failure(f"tools.{i}.description: must be text")
        schema = tool.get("input_schema")
        if not isinstance(schema, dict) or not _finite_json(schema):
            raise Failure(f"tools.{i}.input_schema: must be a JSON Schema object")
        tools.append(ToolDefinition(name, tool.get("description", ""), schema))
    return tools, hosted


def parse_tool_choice(value, declared, hosted=frozenset()):
    if value is None:
        return AUTO, True
    kind = value.get("type") if isinstance(value, dict) else None
    if not isinstance(kind, str):
        raise Failure("tool_choice: must be an object with a type")
    parallel = True
    if value.get("disable_parallel_tool_use") is not None:
        b = as_bool(value["disable_parallel_tool_use"])
        if b is None:
            raise Failure("tool_choice.disable_parallel_tool_use: must be true or false")
        parallel = not b
    if kind == "auto":
        return AUTO, parallel
    if kind == "any":
        return REQUIRED, parallel
    if kind == "none":
        return DISABLED, parallel
    if kind == "tool":
        name = value.get("name")
        # Claude Code's WebSearch asks for the hosted tool by name; say why it cannot run, not that it is undeclared.
        if isinstance(name, str) and name in hosted:
            raise Failure(f"tool_choice.name: '{name}' is a server tool that runs on Anthropic's servers; "
                          f"this server cannot run it")
        if not isinstance(name, str) or name not in declared:
            raise Failure("tool_choice.name: must name a declared tool")
        return named_tool(name), parallel
    raise Failure("tool_choice.type: must be auto, any, tool, or none")


def parse_thinking(thinking_value, output_config):
    """(thinking, effort, show_thinking). `thinking` and `output_config` together decide whether the model thinks,
    at what effort, and whether the client sees the text."""
    label = None
    if output_config is not None:
        if not isinstance(output_config, dict):
            raise Failure("output_config: must be an object")
        cfg = without_nulls(output_config)
        if "format" in cfg:
            raise Failure("output_config.format: not supported; constrained output is not available on this server")
        if "effort" in cfg:
            if not (isinstance(cfg["effort"], str) and cfg["effort"] in EFFORT_LABELS):
                raise Failure("output_config.effort: must be low, medium, high, xhigh, or max")
            label = cfg["effort"]
    if thinking_value is None:
        return False, None, True
    kind = thinking_value.get("type") if isinstance(thinking_value, dict) else None
    if not isinstance(kind, str):
        raise Failure("thinking: must be an object with a type")
    show = True
    display = thinking_value.get("display")
    if display is not None:
        # `omitted` hides the text; every other mode, including ones added later, shows it.
        if not isinstance(display, str):
            raise Failure("thinking.display: must be text")
        show = display != "omitted"
    if kind == "disabled":
        return False, None, show
    if kind == "enabled":
        budget = as_int(thinking_value.get("budget_tokens"))
        if budget is None or budget <= 0:
            raise Failure("thinking.budget_tokens: must be a positive integer")
        # An explicit effort wins; otherwise the budget picks one.
        mapped = map_reasoning(label or ("low" if budget < 4096 else "medium" if budget < 16384 else "high"))
        return True, mapped.effort, show
    if kind == "adaptive":
        mapped = map_reasoning(label or "high")
        return mapped.thinking, mapped.effort, show
    raise Failure("thinking.type: must be enabled, adaptive, or disabled")


# ---- system and messages ---------------------------------------------------------------------------------

def without_attribution(text):
    """The text without a leading attribution line."""
    if not text.startswith(ATTRIBUTION_PREFIX):
        return text
    at = text.find("\n")
    return "" if at < 0 else text[at + 1:].strip("\n\r")


def parse_system(value):
    if value is None:
        return []
    if isinstance(value, str):
        kept = without_attribution(value)
        return [kept] if kept else []
    if not isinstance(value, list):
        raise Failure("system: must be text or an array of text blocks")
    out = []
    for i, block in enumerate(value):
        if not (isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)):
            raise Failure(f"system.{i}: must be a text block")
        kept = without_attribution(block["text"])
        if kept:
            out.append(kept)
    return out


def content_blocks(value, label):
    if value is None:
        raise Failure(f"{label}: field required")
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    if not isinstance(value, list):
        raise Failure(f"{label}: must be text or an array of content blocks")
    out = []
    for j, block in enumerate(value):
        if not (isinstance(block, dict) and isinstance(block.get("type"), str)):
            raise Failure(f"{label}.{j}: must be a content block with a type")
        out.append(without_nulls(block))
    return out


def _text(block, label):
    if not isinstance(block.get("text"), str):
        raise Failure(f"{label}.text: must be text")
    return block["text"]


def _image(label):
    """This server runs text-only. Claude Code removes the pictures and retries on exactly these words; any other
    wording repeats the failure on every later turn."""
    raise Failure(IMAGE_PREFIX + VISION_OFF)


def _document_text(block, label):
    """Plain-text documents are read inline; PDFs and remote documents become a note, because refusing would end
    the turn and, for Claude Code, repeat on every turn after it."""
    source = block.get("source")
    kind = source.get("type") if isinstance(source, dict) else None
    if not isinstance(kind, str):
        raise Failure(f"{label}.source: field required")
    if kind == "text":
        if not isinstance(source.get("data"), str):
            raise Failure(f"{label}.source.data: must be text")
        body = source["data"]
    elif kind == "content":
        parts = source.get("content")
        if not isinstance(parts, list):
            raise Failure(f"{label}.source.content: must be an array")
        texts = []
        for k, part in enumerate(parts):
            if not (isinstance(part, dict) and part.get("type") == "text"):
                raise Failure(f"{label}.source.content.{k}: only text blocks can be read")
            texts.append(_text(part, f"{label}.source.content.{k}"))
        body = "\n\n".join(texts)
    else:
        title = block.get("title")
        name = f" '{title}'" if isinstance(title, str) else ""
        return (f"[The document{name} is not shown: this model reads plain text and pictures, not PDF or remote "
                f"documents. Extract its text first, for example with `pdftotext file.pdf -`.]")
    if isinstance(block.get("title"), str) and block["title"]:
        body = block["title"] + "\n\n" + body
    if isinstance(block.get("context"), str) and block["context"]:
        body = block["context"] + "\n\n" + body
    return body


def _search_result_text(block, label):
    parts = block.get("content")
    if not isinstance(parts, list):
        raise Failure(f"{label}.content: must be an array")
    lines = []
    if isinstance(block.get("title"), str) and block["title"]:
        lines.append(block["title"])
    if isinstance(block.get("source"), str) and block["source"]:
        lines.append("Source: " + block["source"])
    for k, part in enumerate(parts):
        if not (isinstance(part, dict) and part.get("type") == "text"):
            raise Failure(f"{label}.content.{k}: only text blocks can be read")
        lines.append(_text(part, f"{label}.content.{k}"))
    return "\n".join(lines)


def _tool_result(block, label):
    value = block.get("content")
    texts = []
    if value is None:
        pass
    elif isinstance(value, str):
        texts.append(value)
    elif isinstance(value, list):
        for k, part in enumerate(value):
            plabel = f"{label}.content.{k}"
            if not isinstance(part, dict):
                raise Failure(f"{plabel}: must be a content block")
            kind = part.get("type")
            if kind == "text":
                texts.append(_text(part, plabel))
            elif kind == "image":
                _image(plabel)
            elif kind == "document":
                texts.append(_document_text(part, plabel))
            elif kind == "tool_reference":
                # Tool search results: the named tool is declared already.
                if not isinstance(part.get("tool_name"), str):
                    raise Failure(f"{plabel}.tool_name: must be text")
                texts.append(f"Tool available: {part['tool_name']}")
            elif kind == "search_result":
                texts.append(_search_result_text(part, plabel))
            else:
                raise Failure(f"{plabel}: content type '{kind}' cannot be returned to the model here")
    else:
        raise Failure(f"{label}.content: must be text or an array of content blocks")
    joined = "\n".join(texts)
    if block.get("is_error") is not None:
        flag = as_bool(block["is_error"])
        if flag is None:
            raise Failure(f"{label}.is_error: must be true or false")
        if flag:
            joined = "Error: " + joined if joined else "Error"
    return joined


def signature_for(reasoning):
    """A signature that carries the reasoning it signs. Anthropic's are opaque and verified; this one is plain and
    trusted like the rest of a replayed conversation, which the client could rewrite anyway."""
    return SIGNATURE_PREFIX + base64.b64encode(reasoning.encode("utf-8")).decode("ascii")


def reasoning_from_signature(signature):
    if not signature.startswith(SIGNATURE_PREFIX):
        return None
    try:
        return base64.b64decode(signature[len(SIGNATURE_PREFIX):], validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None


def _thinking_text(block, label):
    thinking = block.get("thinking")
    if not isinstance(thinking, str):
        raise Failure(f"{label}.thinking: must be text")
    if thinking:
        return thinking
    if isinstance(block.get("signature"), str):
        return reasoning_from_signature(block["signature"]) or ""
    return ""


def parse_messages(value, systems, counting=False):
    """Anthropic messages to template messages.

    An assistant turn's tool_use blocks must be answered by tool_result blocks in the next user turn, which may
    span consecutive user messages; the template wants one tool message per call, in call order. Consecutive
    messages of one role are one turn, as the API treats them. A system message before the conversation joins the
    system prompt; after it starts it renders as user text, because the template renders one system turn."""
    if value is None:
        raise Failure("messages: field required")
    if not isinstance(value, list):
        raise Failure("messages: must be an array")
    if not value:
        raise Failure("messages: at least one message is required")
    out, assistant = [], None
    pending = []  # (id, name)
    tool_ids = set()
    results, user_texts = {}, []
    started = False

    def flush_assistant():
        nonlocal assistant
        if assistant is not None:
            out.append(assistant)
            assistant = None

    def unanswered(i):
        missing = ", ".join(c for c, _ in pending if c not in results)
        return Failure(f"messages.{i}: tool_use ids were found without tool_result blocks immediately after: "
                       f"{missing}")

    for i, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise Failure(f"messages.{i}: must be an object")
        message = without_nulls(raw)
        role = message.get("role")
        if not isinstance(role, str):
            raise Failure(f"messages.{i}.role: field required")
        blocks = content_blocks(message.get("content"), f"messages.{i}.content")
        if role == "user":
            flush_assistant()
            started = True
            for j, block in enumerate(blocks):
                label = f"messages.{i}.content.{j}"
                kind = block["type"]
                if kind == "tool_result":
                    tid = block.get("tool_use_id")
                    call = next((c for c in pending if c[0] == tid), None) if isinstance(tid, str) else None
                    if call is None or tid in results:
                        raise Failure(f"{label}: tool_result must answer a tool_use from the previous assistant "
                                      f"message, once")
                    results[tid] = ChatMessage("tool", _tool_result(block, label), tool_call_id=tid,
                                               tool_name=call[1])
                elif kind == "text":
                    user_texts.append(_text(block, label))
                elif kind == "image":
                    _image(label)
                elif kind == "document":
                    user_texts.append(_document_text(block, label))
                elif kind == "search_result":
                    user_texts.append(_search_result_text(block, label))
                else:
                    raise Failure(f"{label}: content type '{kind}' is not supported in a user message")
            # A whole user turn is validated and emitted together, even when its results arrive in several
            # adjacent messages; errors keep their original block index because each message is parsed above.
            nxt = value[i + 1] if i + 1 < len(value) else None
            if isinstance(nxt, dict) and nxt.get("role") == "user":
                continue
            if pending:
                if not all(c in results for c, _ in pending):
                    raise unanswered(i)
                out.extend(results[c] for c, _ in pending)
                pending = []
            if user_texts:
                out.append(ChatMessage("user", "\n\n".join(user_texts)))
            results, user_texts = {}, []
        elif role == "assistant":
            # Calls may span adjacent assistant messages of the same turn.
            if assistant is None and pending:
                raise unanswered(i)
            started = True
            a = assistant or ChatMessage("assistant", "")
            for j, block in enumerate(blocks):
                label = f"messages.{i}.content.{j}"
                kind = block["type"]
                if kind == "text":
                    a.content += _text(block, label)
                elif kind == "thinking":
                    replayed = _thinking_text(block, label)
                    if replayed:
                        a.reasoning = replayed if a.reasoning is None else a.reasoning + "\n" + replayed
                elif kind == "redacted_thinking":
                    continue  # encrypted by another provider; nothing here can read it
                elif kind == "tool_use":
                    tid, name = block.get("id"), block.get("name")
                    if not isinstance(tid, str) or not tid:
                        raise Failure(f"{label}.id: field required")
                    if not isinstance(name, str) or not name:
                        raise Failure(f"{label}.name: field required")
                    if tid in tool_ids:
                        raise Failure(f"{label}: tool_use id '{tid}' is used twice")
                    tool_ids.add(tid)
                    arguments = block.get("input")
                    if not isinstance(arguments, dict) or not _finite_json(arguments):
                        raise Failure(f"{label}.input: must be an object")
                    a.tool_calls.append(ParsedToolCall(tid, name, arguments, tuple(arguments)))
                    pending.append((tid, name))
                else:
                    raise Failure(f"{label}: content type '{kind}' is not supported in an assistant message")
            assistant = a
        elif role == "system":
            if pending:
                raise unanswered(i)
            texts = []
            for j, block in enumerate(blocks):
                kind = block["type"]
                # Claude Code announces tools that became available late this way; every declared tool is
                # already rendered.
                if kind in ("tool_addition", "tool_removal"):
                    continue
                if kind != "text":
                    raise Failure(f"messages.{i}.content.{j}: a system message carries text, not '{kind}'")
                texts.append(without_attribution(_text(block, f"messages.{i}.content.{j}")))
            joined = "\n\n".join(t for t in texts if t)
            if not joined:
                continue
            if started:
                flush_assistant()
                out.append(ChatMessage("user", joined))
            else:
                systems.append(joined)
        else:
            raise Failure(f"messages.{i}.role: must be user or assistant")
    if pending:
        raise Failure("messages: the last assistant message has tool_use blocks with no tool_result after them")
    flush_assistant()
    if not any(m.role in ("user", "tool") for m in out):
        raise Failure("messages: at least one user message is required")
    if out and out[-1].role == "assistant" and not counting:
        raise Failure("messages: the last message must be from the user; continuing a partial assistant reply "
                      "is not supported")
    return out


def parse(raw, counting=False):
    """Parse a Messages request. `counting` is the count_tokens form: no `max_tokens`, no sampling reply."""
    if not isinstance(raw, dict):
        raise Failure("request body must be a JSON object")
    body = without_nulls(raw)
    for key in sorted(REFUSED_FIELDS):
        if key in body:
            raise Failure(f"{key}: not supported; {REFUSED_FIELDS[key]}")
    ignored = sorted(set(body) - HANDLED_FIELDS - INERT_FIELDS - set(REFUSED_FIELDS))
    if "metadata" in body and not isinstance(body["metadata"], dict):
        raise Failure("metadata: must be an object")
    if "context_management" in body and not isinstance(body["context_management"], dict):
        raise Failure("context_management: must be an object")
    if "service_tier" in body and not isinstance(body["service_tier"], str):
        raise Failure("service_tier: must be text")
    if "model" not in body:
        raise Failure("model: field required")
    if not isinstance(body["model"], str):
        raise Failure("model: must be text")

    max_tokens = 0
    if "max_tokens" in body:
        n = as_int(body["max_tokens"])
        if n is None or n <= 0:
            raise Failure("max_tokens: must be a positive integer")
        max_tokens = n
    elif not counting:
        raise Failure("max_tokens: field required")
    stream = False
    if "stream" in body:
        stream = as_bool(body["stream"])
        if stream is None:
            raise Failure("stream: must be true or false")
    temperature = top_p = top_k = None
    if "temperature" in body:
        temperature = as_num(body["temperature"])
        if temperature is None or temperature < 0:
            raise Failure("temperature: must be a number of at least 0")
    if "top_p" in body:
        top_p = as_num(body["top_p"])
        if top_p is None or not 0 <= top_p <= 1:
            raise Failure("top_p: must be a number from 0 to 1")
    if "top_k" in body:
        top_k = as_int(body["top_k"])
        if top_k is None or top_k < 0:
            raise Failure("top_k: must be an integer of at least 0")
    stops = []
    if "stop_sequences" in body:
        s = body["stop_sequences"]
        if not (isinstance(s, list) and all(isinstance(x, str) for x in s)):
            raise Failure("stop_sequences: must be an array of text")
        stops = [x for x in s if x]

    tools, hosted = parse_tools(body.get("tools"))
    choice, parallel = parse_tool_choice(body.get("tool_choice"), {t.name for t in tools}, hosted)
    if choice == REQUIRED and not tools:
        raise Failure('tool_choice: "any" needs at least one tool')
    thinking, effort, show = parse_thinking(body.get("thinking"), body.get("output_config"))

    systems = parse_system(body.get("system"))
    messages = parse_messages(body.get("messages"), systems, counting)
    if systems:
        messages.insert(0, ChatMessage("system", "\n\n".join(systems)))
    return Request(messages, tools, choice, parallel, thinking, effort, show, max_tokens, temperature, top_p,
                   top_k, stops, stream, ignored)


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


def count_tokens(request, tokenizer):
    """Tokens of the rendered prompt; no request is admitted and nothing is generated."""
    return len(tokenizer.encode(render_prompt(request, tokenizer)))


@dataclass
class Prepared:
    request: Request
    messages: list
    render_tools: list
    params: object
    seed: int
    window_limited: bool = False

    def check_prompt(self, prompt_tokens, window):
        """The context errors once the prompt is tokenized; clamps the reply budget to the room left. `max_tokens`
        is a ceiling and Claude Code asks for 32,000 whatever the window, so the reply gets what the window has
        left, and a reply that reaches that point ends with `model_context_window_exceeded`."""
        if prompt_tokens <= 0:
            raise Failure("messages: the prompt is empty")
        if prompt_tokens >= window:
            raise Failure(prompt_too_long(prompt_tokens, window - 1))
        room = max(1, window - prompt_tokens)
        self.window_limited = self.params.max_tokens > room
        self.params.max_tokens = min(self.params.max_tokens, room)


def prepare(request, random_seed=None):
    messages, render_tools = prompt(request)
    params = (common.agent_params() if render_tools else
              (common.thinking_params() if request.thinking else common.instruct_params()))
    if request.temperature is not None:
        params.temperature = request.temperature
    if request.top_p is not None:
        params.top_p = request.top_p
    if request.top_k is not None:
        params.top_k = request.top_k
    params.max_tokens = request.max_tokens
    params.stop = list(request.stop_sequences)
    params = params.sanitized()
    seed = random.randint(1, (1 << 64) - 1) if random_seed is None else random_seed
    return Prepared(request, messages, render_tools, params, seed)


# ---- wire --------------------------------------------------------------------------------------------------

def new_id(prefix):
    return prefix + uuid.uuid4().hex[:24]


def tool_use_id():
    return new_id("toolu_")


def usage_body(prompt_tokens, cached_tokens, output_tokens):
    """Anthropic splits the prompt: `input_tokens` is what was read this time and `cache_read_input_tokens` what
    was reused. Claude Code adds the two to know how full the window is."""
    return {"input_tokens": max(0, prompt_tokens - cached_tokens), "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": max(0, min(cached_tokens, prompt_tokens)), "output_tokens": output_tokens}


def frame(type_, fields=None):
    obj = dict(fields or {}, type=type_)
    return f"event: {type_}\ndata: " + json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                                    separators=(",", ":")) + "\n\n"


class MessageStream:
    """One message as it is produced: the finished content blocks, the block currently open, and the SSE frames
    that describe each step. Every method returns frames and writes nothing; the non-streaming reply is the same
    message object the stream describes."""

    def __init__(self, model, show_thinking=True, mid=None):
        self.model, self.show_thinking = model, show_thinking
        self.id = mid or new_id("msg_")
        self.content = []
        self.open = None  # [index, is_thinking, text]
        self.tool_uses = 0

    def _message(self, stop_reason, stop_sequence, usage):
        return {"id": self.id, "type": "message", "role": "assistant", "model": self.model,
                "content": self.content, "stop_reason": stop_reason, "stop_sequence": stop_sequence,
                "usage": usage}

    def start(self, prompt_tokens, reused=0):
        """`message_start` and a first ping, sent once the request is admitted and the reused part of the prompt
        is known. Claude Code keeps a start's nonzero input count when the final one is zero, so the start must
        already carry the real split."""
        return [frame("message_start", {"message": self._message(None, None, usage_body(prompt_tokens, reused, 0))}),
                self.ping()]

    def ping(self):
        return frame("ping")

    def keepalive(self):
        """A ping, or an empty thinking delta while hidden thinking is open: Claude Code's idle timer counts only
        its first 30 consecutive pings, so a long reasoning it does not display needs events it counts."""
        if self.open is not None and self.open[1] and not self.show_thinking:
            return frame("content_block_delta", {"index": self.open[0],
                                                 "delta": {"type": "thinking_delta", "thinking": ""}})
        return self.ping()

    def thinking(self, delta):
        if not delta:
            return []
        out = []
        if self.open is None or not self.open[1]:
            out += self._close()
            self.open = [len(self.content), True, ""]
            out.append(frame("content_block_start", {"index": len(self.content), "content_block": {
                "type": "thinking", "thinking": "", "signature": ""}}))
        self.open[2] += delta
        if self.show_thinking:
            out.append(frame("content_block_delta", {"index": self.open[0], "delta": {
                "type": "thinking_delta", "thinking": delta}}))
        return out

    def text(self, delta):
        if not delta:
            return []
        out = []
        if self.open is None or self.open[1]:
            out += self._close()
            self.open = [len(self.content), False, ""]
            out.append(frame("content_block_start", {"index": len(self.content),
                                                     "content_block": {"type": "text", "text": ""}}))
        self.open[2] += delta
        out.append(frame("content_block_delta", {"index": self.open[0],
                                                 "delta": {"type": "text_delta", "text": delta}}))
        return out

    def tool_use(self, call):
        """A complete call. The parser publishes a call only after its closing tag, so the input arrives whole:
        one delta, then the stop."""
        out = self._close()
        index = len(self.content)
        text = call.input_json
        out.append(frame("content_block_start", {"index": index, "content_block": {
            "type": "tool_use", "id": call.id, "name": call.name, "input": {}}}))
        out.append(frame("content_block_delta", {"index": index, "delta": {
            "type": "input_json_delta", "partial_json": text}}))
        out.append(frame("content_block_stop", {"index": index}))
        self.content.append({"type": "tool_use", "id": call.id, "name": call.name, "input": json.loads(text)})
        self.tool_uses += 1
        return out

    def _close(self):
        if self.open is None:
            return []
        index, thinking, text = self.open
        self.open = None
        out = []
        if thinking:
            sig = signature_for(text)
            out.append(frame("content_block_delta", {"index": index, "delta": {
                "type": "signature_delta", "signature": sig}}))
            self.content.append({"type": "thinking", "thinking": text if self.show_thinking else "",
                                 "signature": sig})
        else:
            self.content.append({"type": "text", "text": text})
        out.append(frame("content_block_stop", {"index": index}))
        return out

    def stop_reason(self, engine_reason, stop_sequence=None, window_limited=False):
        """Why the reply ended. A delivered call wins: the client's next move is to run it, whatever stopped the
        model afterwards. A reply cut short by the window rather than by `max_tokens` says so, because a larger
        `max_tokens` would not have helped."""
        if self.tool_uses > 0:
            return "tool_use"
        if stop_sequence is not None:
            return "stop_sequence"
        if engine_reason != "length":
            return "end_turn"
        return "model_context_window_exceeded" if window_limited else "max_tokens"

    def finish(self, engine_reason, stop_sequence, usage, window_limited=False):
        out = self._close()
        reason = self.stop_reason(engine_reason, stop_sequence, window_limited)
        out.append(frame("message_delta", {"delta": {
            "stop_reason": reason, "stop_sequence": stop_sequence if reason == "stop_sequence" else None},
            "usage": usage}))
        out.append(frame("message_stop"))
        return out

    def message(self, engine_reason, stop_sequence, usage, window_limited=False):
        """The non-streaming reply."""
        self._close()
        reason = self.stop_reason(engine_reason, stop_sequence, window_limited)
        return self._message(reason, stop_sequence if reason == "stop_sequence" else None, usage)

    def fail(self, type_, message):
        """An error after the head went out. Nothing open is closed as complete: a partial block must not read as
        a finished one."""
        self.open = None
        return [frame("error", {"error": {"type": type_, "message": message}})]


class MessagesReply:
    """One response in flight: engine deltas through the reasoning splitter, the tool-call parser and the
    accumulator, out as Anthropic frames."""

    def __init__(self, prepared, model, id_factory=None, mid=None):
        self.prepared = prepared
        req = prepared.request
        self.message = MessageStream(model, req.show_thinking, mid)
        self.acc = OutputAccumulator(prepared.render_tools, req.choice, req.parallel)
        self.thinker = ThinkSplitter() if req.thinking else None
        self.parser = (ToolCallSplitter([t.schema for t in prepared.render_tools], id_factory or tool_use_id)
                       if prepared.render_tools else None)
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
                frames += self.message.text(d["content"])
            if "tool_calls" in d and self.published < len(self.acc.calls):
                frames += self.message.tool_use(self.acc.calls[self.published])
                self.published += 1
        return frames

    def _think(self, thought):
        if not thought:
            return []
        self.acc.reasoning_delta(thought)
        return self.message.thinking(thought)

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

    def failure(self, engine_reason):
        """The accumulator's error unless the reply simply ran out of room inside a call, or before the call a tool
        choice demanded: then the partial call is withheld and the client sees `max_tokens`, which Claude Code
        knows how to continue from."""
        err = self.acc.error
        if err is None:
            return None
        truncated = engine_reason == "length" and (err.startswith("model produced an incomplete")
                                                    or err.startswith("model did not satisfy tool_choice"))
        return None if truncated else err


__all__ = ["Failure", "MessagesReply", "MessageStream", "Prepared", "Request", "count_tokens", "error_body",
           "error_type", "header_safe_field_names", "model_error", "parse", "prepare", "prompt_too_long", "reasoning_from_signature",
           "signature_for"]
