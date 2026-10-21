# SPDX-License-Identifier: MIT
# Port of Server.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""Ollama dialect: chat, generate, show, tags, ps (port of the /api handlers in Server.swift).

Pure over values. Each prepare_* returns what the handler needs or raises DialectError (status 400, or 404 for an
unknown model on show); replies are built by OllamaReply. Behaviour pinned by the Ollama CLI's own wire shapes:
its show request serializes every field, its one-shot mode sends empty suffix/system/template, its interactive
mode opens with an empty-prompt "load" request, and its chat can send `keep_alive` and `options: null`. Unknown
fields and non-empty overrides are still refused.
"""
import datetime
import json
import random
from dataclasses import dataclass, field

from slotstream_spark.server import common, template
from slotstream_spark.server.common import (VISION_OFF, ChatMessage, DialectError, as_bool, as_int, as_num,
                                            as_stop_list, without_nulls)
from slotstream_spark.server.think import ThinkSplitter

OPTIONS = frozenset({"temperature", "top_p", "top_k", "min_p", "presence_penalty", "num_predict", "seed", "stop"})
CHAT_FIELDS = frozenset({"model", "messages", "stream", "think", "options", "keep_alive"})
GENERATE_FIELDS = frozenset({"model", "prompt", "system", "raw", "stream", "think", "options", "keep_alive",
                             "suffix", "template", "images"})
SHOW_FIELDS = frozenset({"model", "name", "verbose", "system", "template", "options"})
CAPABILITIES = ["completion"]  # no vision, no tools: Ollama tools are refused, OpenAI tools live elsewhere


@dataclass
class ModelInfo:
    """What the metadata endpoints report. `resident_bytes` is the process's resident memory: Ollama renders
    (size - size_vram) as a CPU share, and 104 GB of streamed weights would make `ollama ps` claim ~98% CPU."""
    name: str
    digest: str = ""
    weights_bytes: int = 0
    resident_bytes: int = 0
    quantization: str = "affine-4bit"
    max_context: int = 32768
    details_extra: dict = field(default_factory=dict)


def iso(dt):
    return dt.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


# ---- validation -----------------------------------------------------------------------------------------

def options_error(body):
    options = body.get("options")
    if options is None:  # the Ollama CLI sends "options": null when none are set
        return None
    if not isinstance(options, dict):
        return "options must be an object"
    extras = sorted(set(options) - OPTIONS)
    if extras:
        return f"unsupported options field(s): {', '.join(extras)}"
    for key in ("temperature", "top_p", "min_p", "presence_penalty"):
        if key in options and as_num(options[key]) is None:
            return f"options.{key} must be a number"
    for key in ("top_k", "num_predict", "seed"):
        if key in options and as_int(options[key]) is None:
            return f"options.{key} must be an integer"
    stop = options.get("stop")
    if "stop" in options and not (isinstance(stop, str) or (isinstance(stop, list)
                                                            and all(isinstance(s, str) for s in stop))):
        return "options.stop must be text or an array of text"
    return None


def tool_error(body):
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    if "tools" in body or any(isinstance(m, dict) and (m.get("role") == "tool" or "tool_calls" in m
                                                       or "tool_call_id" in m) for m in messages):
        return ("tool calling is supported at /v1/chat/completions in OpenAI format; "
                "/api/chat does not implement Ollama tools")
    return None


def _message_error(body):
    raw = body.get("messages")
    if not isinstance(raw, list):
        return "messages must be an array"
    for i, m in enumerate(raw):
        if not isinstance(m, dict):
            return f"messages[{i}].role must be system, user, or assistant"
        extra = set(m) - {"role", "content", "images", "tool_calls", "tool_call_id"}
        if extra:
            return f"messages[{i}] has unsupported field(s): {', '.join(sorted(extra))}"
        if "images" in m and m["images"] is not None:
            if not (isinstance(m["images"], list) and all(isinstance(x, str) for x in m["images"])):
                return f"messages[{i}].images must be an array of base64 strings"
        if m.get("role") not in ("system", "user", "assistant"):
            return f"messages[{i}].role must be system, user, or assistant"
        content = m.get("content")
        if isinstance(content, list):
            for j, part in enumerate(content):
                if not isinstance(part, dict):
                    return f"messages[{i}].content[{j}] has unsupported field(s)"
                extra = set(part) - {"type", "text", "image_url", "image"}
                if extra:
                    return f"messages[{i}].content[{j}] has unsupported field(s): {', '.join(sorted(extra))}"
                kind = part.get("type", "text")
                if kind in ("text", "input_text"):
                    if not isinstance(part.get("text"), str):
                        return f"messages[{i}] has a text part without text"
                elif kind in ("image_url", "image") or "image_url" in part or "image" in part:
                    continue  # refused as invalid_image when the messages are built
                else:
                    return f"messages[{i}] contains unsupported content type '{kind}'"
        elif not isinstance(content, str):
            return f"messages[{i}].content must be text or a content array"
    return None


def validation_error(body, model_name, allowed, messages=False):
    if (e := common.model_error(body, model_name)) is not None:
        return e
    if (e := common.unsupported_keys(body, allowed)) is not None:
        return e
    if (e := options_error(body)) is not None:
        return e
    if "think" in body and as_bool(body["think"]) is None:
        return "think must be true or false; named reasoning levels are not supported"
    if "stream" in body and as_bool(body["stream"]) is None:
        return "stream must be true or false"
    return _message_error(body) if messages else None


def _content_text(content):
    if isinstance(content, str):
        return content
    out = []
    for part in content:
        if part.get("type", "text") in ("text", "input_text"):
            out.append(part["text"])
        else:
            raise DialectError(VISION_OFF, code="invalid_image")
    return "".join(out)


def sample_params(body):
    """(SampleParams, seed). `seed: -1` and `num_predict: -1` are Ollama's "random" and "as many as allowed"."""
    thinking = as_bool(body.get("think")) or False
    p = common.thinking_params() if thinking else common.instruct_params()
    o = body.get("options") or {}
    for key, attr in (("temperature", "temperature"), ("top_p", "top_p"), ("min_p", "min_p"),
                      ("presence_penalty", "presence_penalty")):
        if (v := as_num(o.get(key))) is not None:
            setattr(p, attr, v)
    if (v := as_int(o.get("top_k"))) is not None:
        p.top_k = v
    if (v := as_int(o.get("num_predict"))) is not None:
        p.max_tokens = v
    seed = as_int(o.get("seed"))
    seed = random.randint(1, (1 << 64) - 1) if seed is None or seed < 0 else seed
    if (stop := as_stop_list(o.get("stop"))) is not None:
        p.stop = stop
    return p.sanitized(), seed


# ---- requests -------------------------------------------------------------------------------------------

@dataclass
class Load:
    """Ollama's documented "load" request (nothing to generate): acknowledged without touching the engine."""
    body: dict


@dataclass
class Prepared:
    kind: str  # chat | generate
    messages: list
    raw_prompt: str
    thinking: bool
    stream: bool
    params: object
    seed: int

    def prompt_text(self, tokenizer):
        if self.raw_prompt is not None:
            return self.raw_prompt
        return template.render(tokenizer, self.messages, thinking=self.thinking)


def prepare_chat(body, info, now=_now):
    body = without_nulls(body)
    if (e := tool_error(body)) is not None:
        raise DialectError(e)
    if (e := validation_error(body, info.name, CHAT_FIELDS, messages=True)) is not None:
        raise DialectError(e)
    raw = body["messages"]
    if not raw:
        return Load({"model": info.name, "created_at": iso(now()), "message": {"role": "assistant", "content": ""},
                     "done": True, "done_reason": "load"})
    messages = []
    for i, m in enumerate(raw):
        if m.get("images"):
            raise DialectError(VISION_OFF, code="invalid_image")
        if m["role"] == "system" and messages:
            raise DialectError(f"messages[{i}]: a system message must come first")
        messages.append(ChatMessage(m["role"], _content_text(m["content"])))
    if messages[-1].role == "assistant":
        raise DialectError("the conversation must end with a user message")
    params, seed = sample_params(body)
    return Prepared("chat", messages, None, bool(as_bool(body.get("think"))), as_bool(body.get("stream")) is not False,
                    params, seed)


def prepare_generate(body, info, now=_now):
    body = without_nulls(body)
    if (e := validation_error(body, info.name, GENERATE_FIELDS)) is not None:
        raise DialectError(e)
    # A non-empty suffix asks for fill-in-the-middle and a non-empty template for a modelfile override; neither exists.
    for key in ("suffix", "template"):
        if key in body:
            if not isinstance(body[key], str):
                raise DialectError(f"{key} must be text")
            if body[key]:
                raise DialectError("fill-in-the-middle (suffix) is not supported" if key == "suffix"
                                   else "template overrides are not supported on this server")
    for key in ("prompt", "system"):
        if key in body and not isinstance(body[key], str):
            raise DialectError(f"{key} must be text")
    if "raw" in body and as_bool(body["raw"]) is None:
        raise DialectError("raw must be true or false")
    images = body.get("images", [])
    if not (isinstance(images, list) and all(isinstance(x, str) for x in images)):
        raise DialectError("images must be an array of base64 strings")
    prompt = body.get("prompt", "")
    thinking = bool(as_bool(body.get("think")))
    if not prompt:
        return Load({"model": info.name, "created_at": iso(now()), "response": "", "done": True,
                     "done_reason": "load"})
    if images:
        raise DialectError(VISION_OFF, code="invalid_image")
    raw = bool(as_bool(body.get("raw")))
    if raw and (thinking or "system" in body):
        raise DialectError("raw generation cannot apply system or think; remove raw or those fields")
    params, seed = sample_params(body)
    stream = as_bool(body.get("stream")) is not False
    if raw:
        return Prepared("generate", [], prompt, thinking, stream, params, seed)
    messages = []
    if body.get("system"):
        messages.append(ChatMessage("system", body["system"]))
    messages.append(ChatMessage("user", prompt))
    return Prepared("generate", messages, None, thinking, stream, params, seed)


# ---- replies --------------------------------------------------------------------------------------------

class OllamaReply:
    """NDJSON frames (or the single JSON body) for one chat or generate response. With think on, the reasoning
    goes in `thinking` and never in the answer; the final frame carries counts, durations and `cached_tokens`."""

    def __init__(self, prepared, model, now=_now):
        self.p, self.model, self.now = prepared, model, now
        self.splitter = ThinkSplitter() if prepared.thinking else None

    def _frame(self, thinking, content, done=False):
        obj = {"model": self.model, "created_at": iso(self.now())}
        if self.p.kind == "chat":
            message = {"role": "assistant"}
            if thinking:
                message["thinking"] = thinking
            message["content"] = content
            obj["message"] = message
        else:
            if thinking:
                obj["thinking"] = thinking
            obj["response"] = content
        obj["done"] = done
        return obj

    def delta(self, text):
        """The frame for one engine delta, or None when the splitter is still holding it back."""
        if not text:
            return None
        if self.splitter is None:
            return self._frame("", text)
        think, content = self.splitter.push(text)
        if not think and not content:
            return None
        return self._frame(think, content)

    def final(self, stats, text, total_seconds):
        """The last frame (streamed) or the whole body (not streamed). `text` is the full reply when not streamed."""
        stream = self.p.stream
        if self.splitter is not None:
            think, content = self.splitter.flush() if stream else ThinkSplitter.split(text)
        else:
            think, content = "", ("" if stream else text)
        obj = self._frame(think, content, done=True)
        obj.update({
            "done_reason": stats.finish_reason,
            "total_duration": int(total_seconds * 1e9),
            "prompt_eval_count": stats.prompt_tokens,
            "prompt_eval_duration": int(stats.prefill_seconds * 1e9),
            "eval_count": stats.decode_tokens,
            "eval_duration": int(stats.decode_seconds * 1e9),
            "cached_tokens": stats.reused_prefix_tokens,
        })
        return obj


def ndjson(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"


# ---- metadata -------------------------------------------------------------------------------------------

def _details(info):
    return {"format": "safetensors", "family": "qwen4_exp", "parameter_size": "176B-A6B",
            "quantization_level": info.quantization, **info.details_extra}


def _card(info, now, loaded=False):
    card = {"name": info.name, "model": info.name, "modified_at": iso(now()), "size": info.weights_bytes,
            "digest": info.digest, "details": _details(info)}
    if loaded:
        card["size"] = info.resident_bytes
        card["size_vram"] = info.resident_bytes
        card["expires_at"] = iso(now() + datetime.timedelta(hours=1))
    return card


def tags(info, now=_now):
    return {"models": [_card(info, now)]}


def ps(info, now=_now):
    return {"models": [_card(info, now, loaded=True)]}


def show(body, info):
    """Raises DialectError (404 for a model this server does not have, 400 for the rest)."""
    body = without_nulls(body)
    # `ollama run` puts the name in `model` and sends an empty `name`; `ollama show` does the reverse, so an
    # empty `model` falls back to the deprecated alias too.
    if not body.get("model") and body.get("name"):
        body = dict(body, model=body["name"])
    if (e := common.model_error(body, info.name)) is not None:
        raise DialectError(e, status=404)
    if (e := common.unsupported_keys(body, SHOW_FIELDS)) is not None:
        raise DialectError(e)
    for key in ("system", "template"):
        if key in body:
            if not isinstance(body[key], str):
                raise DialectError(f"{key} must be text")
            if body[key]:
                raise DialectError(f"{key} overrides are not supported on this server")
    if "options" in body:
        if not isinstance(body["options"], dict):
            raise DialectError("options must be an object")
        if body["options"]:
            raise DialectError("options overrides are not supported on /api/show")
    if "verbose" in body and as_bool(body["verbose"]) is None:
        raise DialectError("verbose must be true or false")
    return {"modelfile": "# slotstream-spark: NVMe-streamed qwen4_exp", "parameters": f"num_ctx {info.max_context}",
            "capabilities": list(CAPABILITIES), "template": "{{ .Prompt }}", "details": _details(info),
            "model_info": {"general.architecture": "qwen4_exp", "general.parameter_count": 176_000_000_000,
                           "qwen4_exp.context_length": info.max_context}}


def version_body(version):
    return {"version": version}


def error_body(err):
    return {"error": err.message}
