# SPDX-License-Identifier: MIT
# Port of ToolCallSplitter.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""Native tool-call parsing, in the model's own grammar (port of ToolCallSplitter.swift).

The model does not emit JSON calls. It writes
    <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>
and the API streams, so the parser is fed deltas and must never emit half of a tag as text: a client that has
seen "<tool" in its transcript has been given a wrong answer no later event removes.

Values arrive untyped. The declared schema decides how a span becomes JSON; a span that does not parse stays a
string, because a tool that reports a bad argument is a better failure than a call that silently lost one.
JSON values are plain Python (str, int, float, bool, list, dict, None); the int/float/bool distinction is kept.
"""
import json
import math
import random
import re
from dataclasses import dataclass, field

INT64_MIN, INT64_MAX = -(1 << 63), (1 << 63) - 1
KINDS = ("string", "integer", "number", "boolean", "array", "object", "unknown")

_INT_RE = re.compile(r"[+-]?[0-9]+")
_FLOAT_RE = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_INDEX_RE = re.compile(r"0|[1-9][0-9]*")
_WS = " \t\n\r\x0b\x0c\x85   "


# ---- JSON text -------------------------------------------------------------------------------------------

def quote(s):
    """JSON string literal. Only the characters JSON requires are escaped, so non-ASCII text stays readable."""
    out = ['"']
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20:
            out.append("\\u%04x" % ord(ch))
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def json_text(v):
    """Compact JSON with sorted object keys, so a delta stream is reproducible. A whole double prints as an int:
    JSON has one numeric type and the tool expects 1, not 1.0."""
    if v is None:
        return "null"
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, str):
        return quote(v)
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            return "null"
        if v == round(v) and abs(v) < 1e15:
            return str(int(v))
        return repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(json_text(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(quote(k) + ":" + json_text(v[k]) for k in sorted(v)) + "}"
    raise TypeError(f"not a JSON value: {type(v).__name__}")


def _reject_constant(name):
    raise ValueError(name)


def _normalize(v):
    """json.loads gives an arbitrary-size int; a value outside int64 is a double, as on the Swift side."""
    if isinstance(v, bool) or v is None or isinstance(v, (str, float)):
        return v
    if isinstance(v, int):
        return v if INT64_MIN <= v <= INT64_MAX else float(v)
    if isinstance(v, list):
        return [_normalize(x) for x in v]
    return {k: _normalize(x) for k, x in v.items()}


def parse_json(text):
    """Parse text as JSON (bare numbers and strings included). Returns (ok, value); NaN and Infinity are refused."""
    try:
        return True, _normalize(json.loads(text, parse_constant=_reject_constant))
    except (ValueError, RecursionError):
        return False, None


# ---- schema ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolSchema:
    name: str
    params: dict = field(default_factory=dict)  # parameter name -> kind


_REF_SIBLINGS = frozenset({"$ref", "$defs", "definitions", "$schema", "$id", "$comment", "$anchor", "title",
                           "description", "default", "examples", "deprecated", "readOnly", "writeOnly"})


def _has_resource_id(field_):
    # In a properties/$defs map a key named "$id" holds a schema, not a URI; only a string value is an identifier.
    return any(isinstance(field_.get(k), str) for k in ("$id", "id"))


def _percent_decode(s):
    """Strict percent-decoding: a bad escape or a result that is not UTF-8 is None."""
    out = bytearray()
    raw = s.encode("utf-8")
    i = 0
    while i < len(raw):
        if raw[i] == 0x25:
            hexa = raw[i + 1:i + 3]
            if len(hexa) != 2 or not re.fullmatch(rb"[0-9A-Fa-f]{2}", hexa):
                return None
            out.append(int(hexa, 16))
            i += 3
        else:
            out.append(raw[i])
            i += 1
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError:
        return None


class SchemaInference:
    """Type inference only: no validation, remote retrieval, anchors or resource rebasing. A nested `$id` can
    change what `#` means, so it is not traversed. The 128-visit budget (schema visits plus pointer steps)
    bounds branching anyOf/ref graphs; running out keeps the unknown-type fallback."""

    def __init__(self, document):
        self.document = document
        self.remaining = 128

    def resolve(self, value, visited=frozenset(), references_allowed=True, is_root=False):
        if self.remaining <= 0 or not isinstance(value, dict):
            return None
        self.remaining -= 1
        can_reference = references_allowed and (is_root or not _has_resource_id(value))
        if "$ref" not in value:
            return value, visited, can_reference
        # Siblings of $ref mean different things across drafts: when constraints are present, keep the
        # inference from the field's own declarations instead of merging with the target.
        if not set(value) <= _REF_SIBLINGS:
            return value, visited, can_reference
        ref = value["$ref"]
        if not can_reference or not isinstance(ref, str) or not ref.startswith("#"):
            return None
        pointer = _percent_decode(ref[1:])
        if pointer is None or (pointer != "" and not pointer.startswith("/")):
            return None
        tokens = []
        for raw in pointer.split("/")[1:]:
            token, i = [], 0
            while i < len(raw):
                ch = raw[i]
                if ch == "~":
                    if i + 1 >= len(raw) or raw[i + 1] not in "01":
                        return None
                    token.append("~" if raw[i + 1] == "0" else "/")
                    i += 2
                else:
                    token.append(ch)
                    i += 1
            tokens.append("".join(token))
        tokens = tuple(tokens)
        if tokens in visited:
            return None
        target = self.document
        for index, token in enumerate(tokens):
            if self.remaining <= 0:
                return None
            self.remaining -= 1
            if isinstance(target, dict):
                # Do not cross an embedded resource on the way to a target.
                if (index != 0 and _has_resource_id(target)) or token not in target:
                    return None
                target = target[token]
            elif isinstance(target, list):
                if not _INDEX_RE.fullmatch(token) or int(token) >= len(target):
                    return None
                target = target[int(token)]
            else:
                return None
        if tokens and isinstance(target, dict) and _has_resource_id(target):
            return None
        return self.resolve(target, visited | {tokens}, True, len(tokens) == 0)

    def types(self, value, visited=frozenset(), references_allowed=True):
        got = self.resolve(value, visited, references_allowed)
        if got is None:
            return None
        field_, path, can_reference = got
        if "type" in field_:
            t = field_["type"]
            if isinstance(t, str):
                return [t]
            if not isinstance(t, list):
                return None
            return list(t) if all(isinstance(x, str) for x in t) else None
        options = field_.get("anyOf")
        if not isinstance(options, list) or not options:
            return None
        names = []
        for option in options:
            branch = self.types(option, path, can_reference)
            if branch is None:
                return None
            names += branch
        return names


@dataclass
class ToolDefinition:
    """A tool as the client declared it: the raw JSON Schema the template renders, plus (through `schema`) the
    reduced type table the parser coerces with. One object holds both so the schema the model is shown and the
    schema its answer is typed against cannot drift."""
    name: str
    description: str
    parameters: object

    @property
    def schema(self):
        params = {}
        root = SchemaInference(self.parameters).resolve(self.parameters, is_root=True)
        if root is not None:
            root_field, _, references_allowed = root
            props = root_field.get("properties")
            if isinstance(props, dict):
                for key, value in props.items():
                    # A shared definition used by another property is not a cycle: fresh budget per field.
                    types = SchemaInference(self.parameters).types(value, references_allowed=references_allowed) or []
                    concrete = [t for t in types if t != "null"]
                    params[key] = concrete[0] if len(concrete) == 1 and concrete[0] in KINDS else "unknown"
        return ToolSchema(self.name, params)

    @property
    def template_value(self):
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


# ---- events ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedToolCall:
    id: str
    name: str
    arguments: dict
    order: tuple = ()  # parameter names in the order the model emitted them

    def __post_init__(self):
        # Streamed input deltas arrive as each parameter closes; if the final input were key-sorted the two would
        # disagree whenever the model wrote parameters out of alphabetical order.
        object.__setattr__(self, "order", tuple(self.order) or tuple(sorted(self.arguments)))

    @property
    def input_json(self):
        return "{" + ",".join(quote(k) + ":" + json_text(self.arguments[k]) for k in self.order
                              if k in self.arguments) + "}"


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class ToolInputStart:
    id: str
    name: str


@dataclass(frozen=True)
class ToolInputDelta:
    id: str
    delta: str


@dataclass(frozen=True)
class ToolInputEnd:
    id: str


@dataclass(frozen=True)
class ToolCall:
    call: ParsedToolCall


@dataclass(frozen=True)
class Malformed:
    """A <tool_call> block that never closed. The text comes back verbatim so nothing the model wrote is lost."""
    text: str


OPEN, CLOSE = "<tool_call>", "</tool_call>"
FUNC_OPEN, FUNC_CLOSE = "<function=", "</function>"
PARAM_OPEN, PARAM_CLOSE = "<parameter=", "</parameter>"

_TEXT, _CALL, _PARAMS, _VALUE, _CLOSING = "text", "call", "params", "value", "closing"


def random_id():
    return "call_%08x" % random.getrandbits(32)


def holdback(s, tag):
    """How many trailing characters of `s` must be withheld because they could still grow into `tag`. Zero for
    ordinary prose, so text streams at once."""
    n = len(tag) - 1
    tail = s[-n:] if n > 0 else ""
    for length in range(len(tail), 0, -1):
        if tag.startswith(tail[-length:]):
            return length
    return 0


def _strip_one_newline_each(raw):
    v = raw
    if v.startswith("\n"):
        v = v[1:]
    if v.endswith("\n"):
        v = v[:-1]
    return v


def coerce(raw, kind):
    """The declared type decides how a raw span becomes JSON. The template writes a newline after
    `<parameter=NAME>` and before `</parameter>`; both are framing and exactly one of each is removed, so a
    value that itself ends in a blank line keeps it."""
    v = _strip_one_newline_each(raw)
    trimmed = v.strip(_WS)
    if kind == "string":
        return v
    if kind == "integer":
        # A model can write 1e20 or inf; those stay numbers, or text, and never reach an int conversion that traps.
        i = _int64(trimmed)
        if i is not None:
            return i
        d = _finite(trimmed)
        if d is not None:
            return int(d) if d == round(d) and INT64_MIN <= d <= INT64_MAX else d
        return v
    if kind == "number":
        i = _int64(trimmed)
        if i is not None:
            return i
        d = _finite(trimmed)
        return d if d is not None else v
    if kind == "boolean":
        if trimmed == "true":
            return True
        if trimmed == "false":
            return False
        return v
    if kind in ("array", "object"):
        ok, parsed = parse_json(trimmed)
        return parsed if ok else v
    # unknown: anyOf and undeclared names. JSON when unambiguous; a bare word stays a string.
    if trimmed.startswith(("{", "[")):
        ok, parsed = parse_json(trimmed)
        if ok:
            return parsed
    if trimmed == "true":
        return True
    if trimmed == "false":
        return False
    i = _int64(trimmed)
    return i if i is not None else v


def _int64(text):
    if _INT_RE.fullmatch(text):
        n = int(text)
        if INT64_MIN <= n <= INT64_MAX:
            return n
    return None


def _finite(text):
    # JSON has no infinity or NaN: "inf", "nan" and an overflowing exponent are not numbers here.
    if not _FLOAT_RE.fullmatch(text):
        return None
    d = float(text)
    return d if math.isfinite(d) else None


class ToolCallSplitter:
    """Splits a generated stream into text and tool calls. Feed it the deltas that survive ThinkSplitter
    (reasoning is never scanned for calls); `flush()` closes the stream."""

    def __init__(self, tools, id_factory=None):
        # Accepts ToolSchema or ToolDefinition; the type table is derived once, not per parameter.
        self.schemas = {t.name: (t.schema if isinstance(t, ToolDefinition) else t) for t in tools}
        self.next_id = id_factory or random_id
        self.buf = ""
        self.state = _TEXT
        self.param = None
        self._reset_call()

    def _reset_call(self):
        self.current_id = ""
        self.current_name = ""
        self.current_args = {}
        self.current_order = []
        self.emitted_any_arg = False
        self.raw_call = ""
        self.value_raw = ""
        self.streams_string = False
        self.value_at_start = True

    def push(self, s):
        self.buf += s
        out = []
        while True:
            if self.state == _TEXT:
                at = self.buf.find(OPEN)
                if at >= 0:
                    before = self.buf[:at]
                    if before:
                        out.append(Text(before))
                    self.raw_call = OPEN
                    self.buf = self.buf[at + len(OPEN):]
                    self.state = _CALL
                    continue
                keep = holdback(self.buf, OPEN)
                emit = self.buf[:len(self.buf) - keep]
                if emit:
                    out.append(Text(emit))
                self.buf = self.buf[len(self.buf) - keep:]
                break

            if self.state == _CALL:
                # Both tags are located before either is acted on: taking whichever was in the buffer swallowed
                # a whole call when the complete text arrived in one push.
                f_open, c_tag = self.buf.find(FUNC_OPEN), self.buf.find(CLOSE)
                if c_tag >= 0 and (f_open < 0 or c_tag < f_open):
                    end = c_tag + len(CLOSE)
                    self.raw_call += self.buf[:end]
                    self.buf = self.buf[end:]
                    self._reset_call()
                    self.state = _TEXT
                    continue
                if f_open < 0:
                    break
                after = f_open + len(FUNC_OPEN)
                gt = self.buf.find(">", after)
                if gt < 0:
                    break
                self.current_name = self.buf[after:gt]
                self.current_id = self.next_id()
                self.current_args, self.current_order, self.emitted_any_arg = {}, [], False
                self.raw_call += self.buf[:gt + 1]
                self.buf = self.buf[gt + 1:]
                out.append(ToolInputStart(self.current_id, self.current_name))
                self.state = _PARAMS
                continue

            if self.state == _PARAMS:
                hits = [(self.buf.find(t), label, t) for label, t in
                        (("param", PARAM_OPEN), ("func", FUNC_CLOSE), ("call", CLOSE))]
                hits = [h for h in hits if h[0] >= 0]
                if not hits:
                    break
                at, label, tag = min(hits)
                if label == "param":
                    after = at + len(PARAM_OPEN)
                    gt = self.buf.find(">", after)
                    if gt < 0:
                        break
                    name = self.buf[after:gt]
                    if name in self.current_args:  # a duplicate parameter makes the block malformed
                        out.append(Malformed(self.raw_call + self.buf))
                        self.buf = ""
                        self._reset_call()
                        self.state = _TEXT
                        return out
                    self.raw_call += self.buf[:gt + 1]
                    self.buf = self.buf[gt + 1:]
                    self.value_raw = ""
                    self.value_at_start = True
                    self.streams_string = self._kind(name) == "string"
                    if self.streams_string:
                        out.append(ToolInputDelta(self.current_id, ("," if self.emitted_any_arg else "{")
                                                  + quote(name) + ':"'))
                    self.param = name
                    self.state = _VALUE
                    continue
                # </function> ends the parameters but not the block: the call is published only once
                # </tool_call> is consumed, so a truncated block is reported malformed, not dispatched.
                end = at + len(tag)
                self.raw_call += self.buf[:end]
                self.buf = self.buf[end:]
                if label == "call":
                    out.extend(self._finish_call())
                    self.state = _TEXT
                else:
                    self.state = _CLOSING
                continue

            if self.state == _VALUE:
                at = self.buf.find(PARAM_CLOSE)
                if at < 0:
                    # Only a bounded tail is searched on the next push; keeping a closing tag's length also
                    # withholds its framing newline. Rescanning the whole growing argument never streamed a
                    # long file's contents.
                    keep = min(len(self.buf), len(PARAM_CLOSE))
                    self._consume_value(self.buf[:len(self.buf) - keep], False, out)
                    self.buf = self.buf[len(self.buf) - keep:]
                    break
                self._consume_value(self.buf[:at], True, out)
                self.raw_call += PARAM_CLOSE
                self.buf = self.buf[at + len(PARAM_CLOSE):]
                value = coerce(self.value_raw, self._kind(self.param))
                if self.param not in self.current_args:
                    self.current_order.append(self.param)
                self.current_args[self.param] = value
                out.append(ToolInputDelta(self.current_id, '"' if self.streams_string else
                                          ("," if self.emitted_any_arg else "{") + quote(self.param)
                                          + ":" + json_text(value)))
                self.emitted_any_arg = True
                self.state = _PARAMS
                continue

            if self.state == _CLOSING:
                at = self.buf.find(CLOSE)
                if at < 0:
                    break
                end = at + len(CLOSE)
                self.raw_call += self.buf[:end]
                self.buf = self.buf[end:]
                out.extend(self._finish_call())
                self.state = _TEXT
                continue
        return out

    def flush(self):
        """Close the stream. Anything still buffered is text; an unterminated call comes back verbatim as
        `Malformed` and the turn ends as text."""
        out = []
        if self.state == _TEXT:
            if self.buf:
                out.append(Text(self.buf))
        elif self.state == _VALUE:
            # A length-limited reply still owns the stable argument text it generated.
            keep = holdback(self.buf, PARAM_CLOSE)
            tail = self.buf[len(self.buf) - keep:]
            self._consume_value(self.buf[:len(self.buf) - keep], False, out)
            out.append(Malformed(self.raw_call + tail))
        else:
            out.append(Malformed(self.raw_call + self.buf))
        self.buf = ""
        self.raw_call = ""
        self.state = _TEXT
        return out

    def _kind(self, param):
        schema = self.schemas.get(self.current_name)
        return schema.params.get(param, "unknown") if schema else "unknown"

    def _finish_call(self):
        out = [ToolInputDelta(self.current_id, "}" if self.emitted_any_arg else "{}"),
               ToolInputEnd(self.current_id),
               ToolCall(ParsedToolCall(self.current_id, self.current_name, self.current_args,
                                       tuple(self.current_order)))]
        self._reset_call()
        return out

    def _consume_value(self, raw, final, out):
        self.value_raw += raw
        self.raw_call += raw
        if not self.streams_string:
            return
        value = raw
        if self.value_at_start and value:
            if value.startswith("\n"):
                value = value[1:]
            self.value_at_start = False
        if final and value.endswith("\n"):
            value = value[:-1]
        if value:
            out.append(ToolInputDelta(self.current_id, quote(value)[1:-1]))

    @staticmethod
    def parse_all(text, tools, id_factory=None):
        """The whole-text parse, for non-streamed replies and for the gate that a split stream and a single push agree."""
        s = ToolCallSplitter(tools, id_factory)
        return s.push(text) + s.flush()
