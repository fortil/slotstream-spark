"""Chat template layer: engine messages to the Jinja template, over `engine/tokenizer.py::render_chat`.

The template file is sha-checked against model.lock.json by the tokenizer, so a silently different prompt format
is a hard error there. This layer owns the rules the template does not enforce: tool results sit in the order of
the calls they answer, and a conversation to be continued ends on a user or tool turn.
"""
import os
from pathlib import Path

from slotstream_spark.engine.tokenizer import ROOT, TEMPLATE, Tokenizer
from slotstream_spark.server.common import VISION_OFF

CHECKOUT_TEMPLATE = ROOT / "third_party" / "qwen38-flash-next-metadata" / TEMPLATE


class TemplateError(ValueError):
    pass


def template_path():
    """`$SLOTSTREAM_MODEL` on the Spark, the checkout copy on the Mac (T0)."""
    model = os.environ.get("SLOTSTREAM_MODEL")
    if model and (Path(model) / TEMPLATE).exists():
        return Path(model) / TEMPLATE
    return CHECKOUT_TEMPLATE if CHECKOUT_TEMPLATE.exists() else None


def load_tokenizer():
    """A Tokenizer (template only, plus tokenizer.json when the model directory has it), or None when neither
    location has a template."""
    model = os.environ.get("SLOTSTREAM_MODEL")
    if model and (Path(model) / TEMPLATE).exists():
        return Tokenizer.from_checkpoint(model)
    return Tokenizer(CHECKOUT_TEMPLATE) if CHECKOUT_TEMPLATE.exists() else None


def reorder_tool_results(messages):
    """Put each assistant turn's tool results in the order of its calls.

    OpenAI-style clients may send results in any order, and the template pairs them by position, not by id. A
    result that answers no call of the preceding assistant turn is left where it is for the dialect to refuse."""
    out = []
    i = 0
    while i < len(messages):
        m = messages[i]
        out.append(m)
        i += 1
        if m.role != "assistant" or not m.tool_calls:
            continue
        j = i
        while j < len(messages) and messages[j].role == "tool":
            j += 1
        results = messages[i:j]
        rank = {c.id: n for n, c in enumerate(m.tool_calls)}
        known = [r for r in results if r.tool_call_id in rank]
        if len(known) == len(results):
            results = sorted(results, key=lambda r: rank[r.tool_call_id])
        out.extend(results)
        i = j
    return out


def require_open_turn(messages):
    if not messages:
        raise TemplateError("the conversation has no messages")
    if messages[-1].role not in ("user", "tool"):
        raise TemplateError("the conversation must end with a user or tool message")


def template_messages(messages):
    out = []
    for m in messages:
        if m.images:
            raise TemplateError(VISION_OFF)
        d = {"role": m.role, "content": m.content or ""}
        if m.role == "assistant":
            if m.reasoning is not None:
                d["reasoning_content"] = m.reasoning
            if m.tool_calls:
                d["tool_calls"] = [{"type": "function", "function": {
                    "name": c.name, "arguments": {k: c.arguments[k] for k in c.order if k in c.arguments}}}
                    for c in m.tool_calls]
        out.append(d)
    return out


def template_tools(tools):
    return [t.template_value for t in tools] or None


def render(tokenizer, messages, *, tools=(), thinking=False, effort=None):
    """The prompt text with the generation prompt appended."""
    messages = reorder_tool_results(list(messages))
    require_open_turn(messages)
    rendered, schemas = template_messages(messages), template_tools(tools)
    if effort is None:
        return tokenizer.render_chat(rendered, enable_thinking=thinking, tools=schemas)
    # render_chat has no effort argument; the template reads `reasoning_effort`, so the compiled template is
    # called directly with the same arguments render_chat passes.
    return tokenizer._template.render(messages=rendered, tools=schemas, add_generation_prompt=True,
                                      enable_thinking=thinking, add_vision_id=False, reasoning_effort=effort)

