# SPDX-License-Identifier: MIT
# Port of OpenAIDialect.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""OpenAI Chat Completions dialect (port of OpenAIDialect.swift and the v1Chat handler in Server.swift).

Pure over values: parsing, validation, request preparation and wire builders, no socket and no engine. The
splitters, the accumulator and the template layer come from step 79; nothing of them is repeated here.

Refusals worth remembering: a field this server does not implement is refused by name, except the exact no-op
default a stock SDK sends on every call (`n: 1`, `frequency_penalty: 0`, `store: false`, ...), which names the
behaviour the server already has. Images are refused with `invalid_image` because the server runs text-only.
"""
import json
import random
import time
import uuid
from dataclasses import dataclass, field

from slotstream_spark.server import common, template
from slotstream_spark.server.common import (AUTO, DISABLED, REQUIRED, VISION_OFF, ChatMessage, DialectError,
                                            ToolChoice, as_bool, as_int, as_num, as_stop_list, map_reasoning,
                                            named_tool, without_nulls)
from slotstream_spark.server.output import OutputAccumulator, wire_tool_call
from slotstream_spark.server.think import ThinkSplitter
from slotstream_spark.server.tool_calls import (ParsedToolCall, Text, ToolCallSplitter, ToolDefinition,
                                                parse_json)

KEEPALIVE = ": keepalive\n\n"
KEEPALIVE_INTERVAL_S = 10
DONE = "data: [DONE]\n\n"

# Accepted only at the value this server already implements; see no_op_error. Stock SDKs send these on every call.
FIELDS = frozenset({
    "model", "messages", "stream", "temperature", "top_p", "top_k", "presence_penalty", "max_tokens",
    "max_completion_tokens", "seed", "stop", "stream_options", "min_p",
    "n", "frequency_penalty", "logprobs", "top_logprobs", "logit_bias", "response_format", "tools", "tool_choice",
    "parallel_tool_calls", "user", "reasoning_effort", "think", "options",
    "store", "metadata", "prompt_cache_key", "prompt_cache_retention", "safety_identifier", "service_tier",
})

REASONING_LABELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
tool_call = wire_tool_call


@dataclass
class Conversation:
    messages: list
    tools: list
    choice: ToolChoice
    parallel: bool
    thinking: bool
    effort: str
    context_limit: int


def no_op_error(body):
    """The sentence for a field set to something other than the behaviour this server has; None when fine."""
    if "n" in body and as_int(body["n"]) != 1:
        return "n must be 1; this server returns a single choice"
    if "frequency_penalty" in body and as_num(body["frequency_penalty"]) != 0:
        return "frequency_penalty is not supported (0 only); use presence_penalty"
    if "logprobs" in body and as_bool(body["logprobs"]) is not False:
        return "logprobs are not supported"
    if "top_logprobs" in body:
        return "top_logprobs are not supported"
    if "logit_bias" in body and not (isinstance(body["logit_bias"], dict) and not body["logit_bias"]):
        return "logit_bias is not supported"
    if "response_format" in body:
        rf = body["response_format"]
        if not (isinstance(rf, dict) and rf.get("type") in ("text", "json_object")):
            return ("response_format is not supported for constrained output; only {\"type\": \"text\"} "
                    "and {\"type\": \"json_object\"} are supported")
    if "user" in body and not isinstance(body["user"], str):
        return "user must be text"
    # `store: false` is what the OpenAI SDKs send by default; `true` asks for a completion to fetch later.
    if "store" in body:
        store = as_bool(body["store"])
        if store is None:
            return "store must be true or false"
        if store:
            return "store: true is not supported; this server keeps no completions"
    if "metadata" in body:
        md = body["metadata"]
        if not (isinstance(md, dict) and all(isinstance(v, str) for v in md.values())):
            return "metadata must be an object of text values"
    for key in ("prompt_cache_key", "prompt_cache_retention", "safety_identifier", "service_tier"):
        if key in body and not isinstance(body[key], str):
            return f"{key} must be text"
    return None


def validation_error(body, model_name, context_limit):
    """The first reason a request is refused, or None. Order: model, unknown fields, no-ops, conversation, scalars."""
    if (e := common.model_error(body, model_name)) is not None:
        return e
    if (e := common.unsupported_keys(body, FIELDS)) is not None:
        return e
    if (e := no_op_error(body)) is not None:
        return e
    try:
        conversation(body, context_limit)
    except DialectError as err:
        return err.message
    for key in ("temperature", "top_p", "presence_penalty", "min_p"):
        if key in body and as_num(body[key]) is None:
            return f"{key} must be a number"
    for key in ("top_k", "max_tokens", "max_completion_tokens", "seed"):
        if key in body and as_int(body[key]) is None:
            return f"{key} must be an integer"
    for key in ("max_tokens", "max_completion_tokens"):
        if (v := as_int(body.get(key))) is not None and v <= 0:
            return f"{key} must be greater than zero"
    if "stop" in body and not (isinstance(body["stop"], str) or
                               (isinstance(body["stop"], list) and all(isinstance(s, str) for s in body["stop"]))):
        return "stop must be text or an array of text"
    if "stream" in body and as_bool(body["stream"]) is None:
        return "stream must be true or false"
    if "stream_options" in body:
        so = body["stream_options"]
        if not isinstance(so, dict):
            return "stream_options must be an object"
        extras = sorted(set(so) - {"include_usage"})
        if extras:
            return f"unsupported stream_options field(s): {', '.join(extras)}"
        if "include_usage" in so and as_bool(so["include_usage"]) is None:
            return "stream_options.include_usage must be true or false"
    return None


def _content(value, allow_empty):
    if value is None and allow_empty:
        return ""
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or not all(isinstance(p, dict) for p in value):
        raise DialectError("message content must be text or a content array")
    text = []
    for part in value:
        if not set(part) <= {"type", "text", "image_url", "image"}:
            raise DialectError("unsupported content part field")
        kind = part.get("type", "text")
        # An image is named before anything else: a part with an image key and no type would otherwise read as
        # a broken text part and hide the real reason.
        if kind in ("image_url", "image") or "image_url" in part or "image" in part:
            raise DialectError(VISION_OFF, code="invalid_image")
        if kind in ("text", "input_text"):
            if not isinstance(part.get("text"), str):
                raise DialectError("text content part needs text")
            text.append(part["text"])
        else:
            raise DialectError(f"unsupported content type '{kind}'")
    return "".join(text)


def _tools(value):
    if not isinstance(value, list) or not all(isinstance(t, dict) for t in value):
        raise DialectError("tools must be an array of function tools")
    out, names = [], set()
    for tool in value:
        f = tool.get("function")
        if not (set(tool) <= {"type", "function"} and tool.get("type") == "function" and isinstance(f, dict)
                and set(f) <= {"name", "description", "parameters", "strict"}
                and isinstance(f.get("name"), str) and f["name"] and f["name"] not in names):
            raise DialectError("tools must contain uniquely named function definitions")
        names.add(f["name"])
        if f.get("strict") is not None and f["strict"] is not False:
            raise DialectError("strict tool schemas are not supported; omit strict or use false")
        if "description" in f and not isinstance(f["description"], str):
            raise DialectError("tool description must be text")
        parameters = f["parameters"] if "parameters" in f else {"type": "object", "properties": {}}
        if not isinstance(parameters, dict):
            raise DialectError("tool parameters must be a JSON Schema object")
        out.append(ToolDefinition(f["name"], f.get("description", ""), parameters))
    return out


def _choice(value, tools):
    if value == "auto":
        return AUTO
    if value == "none":
        return DISABLED
    if value == "required":
        return REQUIRED
    if not (isinstance(value, dict) and value.get("type") == "function" and set(value) <= {"type", "function"}
            and isinstance(value.get("function"), dict) and set(value["function"]) == {"name"}
            and isinstance(value["function"]["name"], str)
            and any(t.name == value["function"]["name"] for t in tools)):
        raise DialectError("tool_choice must be auto, none, required, or a declared function")
    return named_tool(value["function"]["name"])


def conversation(body, context_limit):
    """Validate and normalize a request body. Raises DialectError with the sentence to send."""
    body = without_nulls(body)
    tools = _tools(body["tools"]) if "tools" in body else []
    choice = _choice(body["tool_choice"], tools) if "tool_choice" in body else AUTO
    if choice == REQUIRED and not tools:
        raise DialectError("tool_choice required needs at least one tool")
    parallel = True
    if "parallel_tool_calls" in body:
        parallel = as_bool(body["parallel_tool_calls"])
        if parallel is None:
            raise DialectError("parallel_tool_calls must be true or false")

    thinking, effort = False, None
    if "reasoning_effort" in body:
        label = body["reasoning_effort"]
        if not (isinstance(label, str) and label in REASONING_LABELS):
            raise DialectError("reasoning_effort must be none, minimal, low, medium, high, xhigh, or max")
        mapped = map_reasoning(label)
        thinking, effort = mapped.thinking, mapped.effort
    if "think" in body:
        b = as_bool(body["think"])
        if b is None:
            raise DialectError("think must be true or false")
        if "reasoning_effort" in body and b != thinking:
            raise DialectError("think conflicts with reasoning_effort")
        thinking = b

    limit = context_limit
    if "options" in body:
        options = body["options"]
        if not (isinstance(options, dict) and set(options) <= {"num_ctx"}):
            raise DialectError("OpenAI options supports only num_ctx")
        if "num_ctx" in options:
            n = as_int(options["num_ctx"])
            if n is None or n <= 0 or n > context_limit:
                raise DialectError(f"options.num_ctx must be between 1 and the served context limit {context_limit}")
            limit = n

    raw = body.get("messages")
    if not isinstance(raw, list) or not raw:
        raise DialectError("messages must be a nonempty array")
    messages, systems, pending = [], [], {}
    ids = set()
    for i, message in enumerate(raw):
        if not isinstance(message, dict):
            raise DialectError(f"messages[{i}].role must be system, developer, user, assistant, or tool")
        m = without_nulls(message)
        role = m.get("role")
        if role not in ("system", "developer", "user", "assistant", "tool"):
            raise DialectError(f"messages[{i}].role must be system, developer, user, assistant, or tool")
        # Hermes's iteration-limit summary leaves three bookkeeping fields on messages; they carry nothing.
        allowed = {"role", "content", "images", "effect_disposition", "display_kind", "display_metadata"}
        if role == "assistant":
            allowed |= {"tool_calls", "reasoning_content", "reasoning"}
        if role == "tool":
            allowed |= {"tool_call_id", "name"}
        extra = set(m) - allowed
        if extra:
            raise DialectError(f"messages[{i}] has unsupported field(s): {', '.join(sorted(extra))}")
        if role != "tool" and pending:
            raise DialectError(f"messages[{i}] must first supply results for outstanding tool calls")
        text = _content(m.get("content"), allow_empty=role == "assistant")
        if "images" in m:
            if not (isinstance(m["images"], list) and all(isinstance(x, str) for x in m["images"])):
                raise DialectError(f"messages[{i}].images must be an array of base64 strings")
            if m["images"]:
                raise DialectError(VISION_OFF, code="invalid_image")
        if role in ("system", "developer"):
            if messages:
                raise DialectError("system/developer messages must precede the conversation and contain only text")
            systems.append(text)
            continue
        out = ChatMessage(role, text)
        if role == "assistant":
            if "reasoning" in m and "reasoning_content" in m:
                raise DialectError("use only one assistant reasoning field")
            reason = m.get("reasoning_content", m.get("reasoning"))
            if reason is not None:
                if not isinstance(reason, str):
                    raise DialectError("assistant reasoning_content must be text")
                out.reasoning = reason
            if "tool_calls" in m:
                calls = m["tool_calls"]
                if not (isinstance(calls, list) and all(isinstance(c, dict) for c in calls)):
                    raise DialectError("assistant tool_calls must be an array")
                for call in calls:
                    f = call.get("function")
                    ok = (set(call) <= {"id", "type", "function"} and call.get("type") == "function"
                          and isinstance(call.get("id"), str) and call["id"] and call["id"] not in ids
                          and isinstance(f, dict) and set(f) == {"name", "arguments"}
                          and isinstance(f["name"], str) and f["name"] and isinstance(f["arguments"], str))
                    parsed = parse_json(f["arguments"]) if ok else (False, None)
                    if not (ok and parsed[0] and isinstance(parsed[1], dict)):
                        raise DialectError("assistant tool_calls need unique IDs, function names, "
                                           "and JSON object argument strings")
                    ids.add(call["id"])
                    pending[call["id"]] = f["name"]
                    out.tool_calls.append(ParsedToolCall(call["id"], f["name"], parsed[1], tuple(parsed[1])))
        elif role == "tool":
            call_id = m.get("tool_call_id")
            if not isinstance(call_id, str) or call_id not in pending:
                raise DialectError("tool result must reference an outstanding tool_call_id exactly once")
            name = pending.pop(call_id)
            if "name" in m and m["name"] != name:
                raise DialectError("tool result name does not match its call")
            out.tool_call_id, out.tool_name = call_id, name
        messages.append(out)
    if pending:
        raise DialectError("messages are missing tool results")
    # Results may arrive in any order; the native template pairs them by position.
    messages = template.reorder_tool_results(messages)
    if systems:
        messages.insert(0, ChatMessage("system", "\n\n".join(systems)))
    return Conversation(messages, tools, choice, parallel, thinking, effort, limit)


def instructing(messages, line):
    """Append an instruction to the system turn, creating one when there is none."""
    out = list(messages)
    for i, m in enumerate(out):
        if m.role == "system":
            out[i] = ChatMessage("system", m.content + "\n\n" + line)
            return out
    return [ChatMessage("system", line)] + out


def prefixing(messages, line):
    """Put an instruction first in the system turn, creating one when there is none. At the front so the prefix
    stays the same for every call that shares the system text."""
    out = list(messages)
    if out and out[0].role == "system":
        out[0] = ChatMessage("system", line + "\n\n" + out[0].content)
        return out
    return [ChatMessage("system", line)] + out


def wants_json_object(body):
    rf = body.get("response_format")
    return isinstance(rf, dict) and rf.get("type") == "json_object"


def random_seed():
    return random.randint(1, (1 << 64) - 1)


@dataclass
class Prepared:
    """Everything the handler needs after validation: the messages to render, the tools shown to the model, the
    sampling knobs, and the output switches."""
    conversation: Conversation
    messages: list
    render_tools: list
    params: object
    seed: int
    stream: bool
    want_usage: bool
    max_tokens_requested: int

    def check_prompt(self, prompt_tokens):
        """The context errors, once the prompt is tokenized; clamps the reply budget to the room left."""
        limit = self.conversation.context_limit
        if prompt_tokens >= limit:
            raise DialectError(f"prompt is {prompt_tokens} tokens, leaving no reply room in the requested "
                               f"context limit {limit}", code="context_length_exceeded")
        self.params.max_tokens = min(self.params.max_tokens, limit - prompt_tokens)


def prepare(body, context_limit):
    body = without_nulls(body)
    conv = conversation(body, context_limit)
    render_tools = [] if conv.choice == DISABLED else conv.tools
    messages = conv.messages
    if wants_json_object(body):
        # No grammar: thinking off and an instruction. Whatever reasoning_effort or think asked for is overridden.
        conv.thinking, conv.effort = False, None
        messages = prefixing(messages, common.JSON_OBJECT_INSTRUCTION)
    if conv.choice.is_named_tool:
        messages = instructing(messages, f"You must call the {conv.choice.name} tool now.")
    elif conv.choice == REQUIRED:
        messages = instructing(messages, "You must call one of the available tools now.")
    if not conv.parallel and render_tools:
        messages = instructing(messages, "Call at most one tool in this response.")
    params = (common.agent_params() if render_tools else
              (common.thinking_params() if conv.thinking else common.instruct_params()))
    for key, attr in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"),
                      ("presence_penalty", "presence_penalty"), ("min_p", "min_p")):
        v = as_int(body.get(key)) if key == "top_k" else as_num(body.get(key))
        if v is not None:  # a bad value is refused by validation_error; here it can only keep the preset
            setattr(params, attr, v)
    # Without a limit the reply gets the budget /v1/models advertises: agents' summary requests often send none,
    # and the 512-token chat default cut them off.
    params.max_tokens = common.output_budget(context_limit)
    for key in ("max_tokens", "max_completion_tokens"):
        if (v := as_int(body.get(key))) is not None:
            params.max_tokens = v
    seed = as_int(body.get("seed"))
    seed = random_seed() if seed is None else seed & ((1 << 64) - 1)
    if (stop := as_stop_list(body.get("stop"))) is not None:
        params.stop = stop
    params = params.sanitized()
    so = body.get("stream_options")
    return Prepared(conv, messages, render_tools, params, seed, bool(as_bool(body.get("stream"))),
                    bool(as_bool(so.get("include_usage"))) if isinstance(so, dict) else False,
                    params.max_tokens)


def render_prompt(prepared, tokenizer):
    conv = prepared.conversation
    return template.render(tokenizer, prepared.messages, tools=prepared.render_tools, thinking=conv.thinking,
                           effort=conv.effort)


# ---- wire ------------------------------------------------------------------------------------------------

def sse(obj):
    return "data: " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n\n"


def new_id():
    return "chatcmpl-" + str(uuid.uuid4()).upper()


def usage_body(prompt_tokens, completion_tokens, cached_tokens):
    return {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens}}


def error_body(message, code="invalid_request_error", type_=None):
    return {"error": {"message": message, "type": type_ or code, "code": code}}


def server_error_body(message):
    return {"error": {"message": message, "type": "server_error", "code": "inference_error"}}


def refusal_body(err):
    """The body for a DialectError: the type stays invalid_request_error, the code names the cause."""
    return error_body(err.message, err.code, "invalid_request_error")


class ChatReply:
    """One response in flight: feeds engine deltas through the reasoning splitter, the tool-call parser and the
    accumulator, and builds the chunks and the final body."""

    def __init__(self, prepared, model, rid=None, created=None, id_factory=None, clock=time.time):
        self.prepared, self.model = prepared, model
        self.rid = rid or new_id()
        self.created = int(clock()) if created is None else created
        conv = prepared.conversation
        self.acc = OutputAccumulator(prepared.render_tools, conv.choice, conv.parallel,
                                     stream_tool_arguments=True, allow_length_truncation=True)
        self.thinker = ThinkSplitter() if conv.thinking else None
        self.parser = (ToolCallSplitter([t.schema for t in prepared.render_tools], id_factory or
                                        (lambda: "call_" + uuid.uuid4().hex)) if prepared.render_tools else None)
        self.sent_role = False
        # Non-streamed single-call turns still stop at the first call, so they read deltas as they come.
        self.incremental = prepared.stream or (not conv.parallel and bool(prepared.render_tools))

    @property
    def should_stop(self):
        return self.acc.error is not None or self.acc.finished_single_call

    def _events(self, body):
        return self.parser.push(body) if self.parser else [Text(body)]

    def consume(self, delta):
        """Wire deltas (dicts for choices[0].delta) for one engine delta."""
        out = []
        body = delta
        if self.thinker:
            reasoning, body = self.thinker.push(delta)
            if reasoning:
                out.append(self.acc.reasoning_delta(reasoning))
        if body:
            out.extend(self.acc.consume(self._events(body)))
        return out

    def end(self, engine_finish_reason, text=None):
        """Flush the splitters. `text` is the whole reply when deltas were not read incrementally.
        Returns (wire deltas, finish_reason)."""
        out = []
        if text is not None and not self.incremental:
            out.extend(self.consume(text))
        if self.thinker:
            reasoning, body = self.thinker.flush()
            if reasoning:
                out.append(self.acc.reasoning_delta(reasoning))
            if body:
                out.extend(self.acc.consume(self._events(body)))
        if self.parser:
            out.extend(self.acc.consume(self.parser.flush()))
        return out, self.acc.finish_reason(engine_finish_reason)

    def chunk(self, delta, finish_reason=None):
        if delta and not self.sent_role:
            delta = dict(delta, role="assistant")
            self.sent_role = True
        return {"id": self.rid, "object": "chat.completion.chunk", "created": self.created, "model": self.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}

    def final_chunk(self, finish, usage=None):
        c = self.chunk({}, finish)
        if self.prepared.want_usage and usage is not None:
            c["usage"] = usage  # kept in the final choice chunk, the shape current clients read
        return c

    def completion(self, finish, usage):
        return {"id": self.rid, "object": "chat.completion", "created": self.created, "model": self.model,
                "choices": [{"index": 0, "finish_reason": finish, "message": self.acc.message}], "usage": usage}


__all__ = ["ChatReply", "Conversation", "DONE", "FIELDS", "KEEPALIVE", "Prepared", "conversation", "prepare",
           "tool_call", "validation_error"]
