# SPDX-License-Identifier: MIT
# Port of OpenAIOutput.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""Output accumulator for Chat Completions (port of OpenAIOutput.swift).

Completed calls and provisional wire deltas are separate: consumers execute only completed calls, while Chat
Completions can show arguments as they stream.
"""
from slotstream_spark.server.tool_calls import (Malformed, Text, ToolCall, ToolInputDelta, ToolInputEnd,
                                                ToolInputStart)


def wire_tool_call(call):
    return {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.input_json}}


class OutputAccumulator:
    def __init__(self, tools, choice, parallel, stream_tool_arguments=False, allow_length_truncation=False):
        self.names = {t.name for t in tools}
        self.choice = choice
        self.parallel = parallel
        self.stream_tool_arguments = stream_tool_arguments
        self.allow_length_truncation = allow_length_truncation
        self.text = ""
        self.reasoning = ""
        self.calls = []
        self.error = None
        self.pending = None  # [id, name, arguments]
        self.incomplete = False
        self.truncated = False

    @property
    def finished_single_call(self):
        return not self.parallel and bool(self.calls)

    def reasoning_delta(self, value):
        self.reasoning += value
        return {"reasoning_content": value}

    def consume(self, events):
        deltas = []
        for event in events:
            if self.error is not None or self.finished_single_call:
                break
            if isinstance(event, Text):
                if event.text:
                    self.text += event.text
                    deltas.append({"content": event.text})
            elif isinstance(event, ToolCall):
                call = event.call
                if not self._validate(call.name):
                    continue
                wire = wire_tool_call(call)
                wire["index"] = len(self.calls)
                if not self.stream_tool_arguments or self.pending is None or self.pending[0] != call.id:
                    deltas.append({"tool_calls": [wire]})
                self.calls.append(call)
                self.pending = None
            elif isinstance(event, Malformed):
                self.incomplete = True
                self.error = "model produced an incomplete or malformed tool call"
            elif isinstance(event, ToolInputStart):
                if not self._validate(event.name):
                    continue
                self.pending = [event.id, event.name, ""]
                if self.stream_tool_arguments:
                    deltas.append({"tool_calls": [{"index": len(self.calls), "id": event.id, "type": "function",
                                                   "function": {"name": event.name, "arguments": ""}}]})
            elif isinstance(event, ToolInputDelta):
                if self.pending is None or self.pending[0] != event.id:
                    continue
                self.pending[2] += event.delta
                if self.stream_tool_arguments and event.delta:
                    deltas.append({"tool_calls": [{"index": len(self.calls),
                                                   "function": {"arguments": event.delta}}]})
            elif isinstance(event, ToolInputEnd):
                pass
        return deltas

    def finish_reason(self, reason):
        if reason == "length" and self.allow_length_truncation and not self.finished_single_call:
            self.truncated = True
            if self.incomplete:
                self.error = None
            return "length"
        if not self.calls and (self.choice.kind == "required" or self.choice.is_named_tool):
            self.error = self.error or f"model did not satisfy tool_choice: {self.choice.label}"
        return reason if not self.calls else "tool_calls"

    def _validate(self, name):
        if name not in self.names:
            self.error = f"model called an undeclared tool: {name}"
            return False
        if self.choice.is_named_tool and name != self.choice.name:
            self.error = "model did not satisfy the named tool_choice"
            return False
        return True

    @property
    def message(self):
        message = {"role": "assistant", "content": self.text}
        wire_calls = [wire_tool_call(c) for c in self.calls]
        if self.truncated and self.pending is not None:
            wire_calls.append({"id": self.pending[0], "type": "function",
                               "function": {"name": self.pending[1], "arguments": self.pending[2]}})
        if wire_calls:
            message["tool_calls"] = wire_calls
            if not self.text:
                message["content"] = None
        if self.reasoning:
            message["reasoning_content"] = self.reasoning
        return message

