# SPDX-License-Identifier: MIT
# Port of Server.swift, GatewayDialect.swift and Generate.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""Pieces every dialect shares: null stripping, strict scalars, reasoning labels, tool choice, sampling presets.

Ported from the slotstream Swift server (Server.swift withoutNulls, GatewayDialect.swift mapReasoning,
Generate.swift presets). The strictness is the point: a JSON `true` is never an integer, and a float with a
fraction is never a context size.
"""
import math
from dataclasses import dataclass, field

from slotstream_spark.engine.sampler import SampleParams

FLOAT32_MAX = 3.4028234663852886e38
JSON_OBJECT_INSTRUCTION = "Answer with a single valid JSON object and nothing else, with no code fence."
VISION_OFF = "this server was started with --vision off; images are not accepted"


class DialectError(Exception):
    """A request refused before it reaches the engine. `code` is the stable identifier the wire carries."""

    def __init__(self, message, code="invalid_request_error", status=400):
        super().__init__(message)
        self.message, self.code, self.status = message, code, status


def without_nulls(d):
    """JSON null means "not set": stock SDKs send null for every unset field and a 400 for that is a bug."""
    return {k: v for k, v in d.items() if v is not None}


def is_bool(v):
    return isinstance(v, bool)


def as_bool(v):
    return v if isinstance(v, bool) else None


def as_int(v):
    """An integer, or None. Bool is never an int; 64000.0 is, 100.5 is not (NSNumber Int(exactly:)).
    Nothing outside 64 bits is either."""
    if isinstance(v, bool):
        return None
    if isinstance(v, float) and math.isfinite(v) and v == int(v):
        v = int(v)
    if isinstance(v, int) and -(1 << 63) <= v < (1 << 63):  # Swift Int is 64-bit: Int(exactly:) fails beyond it
        return v
    return None


def as_num(v):
    """A finite number that survives conversion to float32, or None."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    d = float(v)
    if not math.isfinite(d) or abs(d) > FLOAT32_MAX:
        return None
    return d


def as_stop_list(v):
    if isinstance(v, list) and all(isinstance(s, str) for s in v):
        return [s for s in v if s]
    if isinstance(v, str) and v:
        return [v]
    return None


@dataclass(frozen=True)
class Thinking:
    thinking: bool
    effort: str = None


def map_reasoning(label):
    """A bare effort label, or {"effort": label}, to thinking on/off. Missing means off: faster, and the shape
    the model was measured producing well-formed tool calls in. An unknown label turns thinking on."""
    if isinstance(label, dict):
        label = label.get("effort")
    if not isinstance(label, str):
        return Thinking(False, None)
    if label in ("none", "minimal", "off"):
        return Thinking(False, None)
    if label == "low":
        return Thinking(True, "low")
    if label == "medium":
        return Thinking(True, "medium")
    if label in ("high", "xhigh", "max"):
        return Thinking(True, "xhigh")
    return Thinking(True, None)


@dataclass(frozen=True)
class ToolChoice:
    kind: str  # auto | disabled | required | tool
    name: str = None

    @property
    def is_named_tool(self):
        return self.kind == "tool"

    @property
    def label(self):
        return {"auto": "auto", "disabled": "none", "required": "required"}.get(self.kind, f"tool({self.name})")


AUTO = ToolChoice("auto")
DISABLED = ToolChoice("disabled")
REQUIRED = ToolChoice("required")


def named_tool(name):
    return ToolChoice("tool", name)


@dataclass
class ChatMessage:
    """One conversation turn in the engine's own shape; `template.py` turns it into what the Jinja template reads."""
    role: str
    content: str = ""
    reasoning: str = None
    tool_calls: list = field(default_factory=list)  # ParsedToolCall
    tool_call_id: str = None
    tool_name: str = None
    images: list = field(default_factory=list)


def instruct_params():
    return SampleParams(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5)


def agent_params():
    """Tool turns: presence penalty 0 because the call grammar must repeat its closing tags, and a low
    temperature because a call has one right shape."""
    return SampleParams(temperature=0.2, top_p=0.9, top_k=20, min_p=0.0, presence_penalty=0.0)


def thinking_params():
    return SampleParams(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0)


def output_budget(context_cap):
    """The reply budget advertised for the served model: a quarter of the cap, floored at 256, capped at 8192,
    and always strictly below the window."""
    desired = min(8192, max(256, context_cap // 4))
    return max(1, min(desired, context_cap - 1))


def model_error(json_body, model_name):
    """None when the request names the served model (or no model); the sentence to send otherwise."""
    if "model" not in json_body:
        return None
    requested = json_body["model"]
    if not isinstance(requested, str):
        return "model must be text"
    if not requested:
        return "model must not be empty"
    if requested in (model_name, "qwen3.8-flash-next", "qwen3.8-flash-next:latest"):
        return None
    return f"model '{requested}' is not loaded; this server has only '{model_name}'"


def unsupported_keys(json_body, allowed):
    extras = sorted(set(json_body) - set(allowed))
    return f"unsupported request field(s): {', '.join(extras)}" if extras else None
