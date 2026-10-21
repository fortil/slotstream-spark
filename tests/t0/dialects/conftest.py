"""Shared fixtures for the dialect tests: the fx tool schemas, the call fixtures, a deterministic id factory,
the event normalizer and a fake engine that replays scripted text.

The tool-call fixtures live inline in the Swift checks, so they are inline here. `fixture_dir()` is the loader
for files under Tools/fixtures/ (checkout first, then the pinned third_party copy), for the dialect tests that
read them.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from slotstream_spark.server import tool_calls as tc
from slotstream_spark.server.tool_calls import ToolSchema

ROOT = Path(__file__).resolve().parents[3]

ONE_CALL = """<tool_call>
<function=read_file>
<parameter=path>
hello.txt
</parameter>
</function>
</tool_call>"""

PROSE_THEN_CALL = """Let me read it.
<tool_call>
<function=read_file>
<parameter=path>
hello.txt
</parameter>
<parameter=start_line>
1
</parameter>
</function>
</tool_call>"""

TWO_CALLS = """<tool_call>
<function=read_file>
<parameter=path>
a.txt
</parameter>
</function>
</tool_call>
<tool_call>
<function=list_dir>
<parameter=path>
.
</parameter>
<parameter=depth>
2
</parameter>
</function>
</tool_call>"""


def fx_tool_schemas():
    """fx's real schemas reduced to what the parser needs, including the anyOf fields on `terminal`."""
    return [
        ToolSchema("read_file", {"path": "string", "start_line": "integer", "end_line": "integer"}),
        ToolSchema("write_file", {"path": "string", "content": "string"}),
        ToolSchema("edit_file", {"path": "string", "old_string": "string", "new_string": "string",
                                 "replace_all": "boolean"}),
        ToolSchema("terminal", {"command": "string", "timeout": "unknown", "background": "unknown"}),
        ToolSchema("list_dir", {"path": "string", "depth": "integer"}),
        ToolSchema("grep", {"pattern": "string", "path": "string", "glob": "string", "case_sensitive": "boolean"}),
        ToolSchema("todo_write", {"todos": "array", "merge": "boolean"}),
        ToolSchema("task", {"prompt": "string", "context": "object"}),
    ]


def counting_ids():
    """A deterministic id factory (call_00000001, ...), so events compare by value."""
    box = {"n": 0}

    def next_id():
        box["n"] += 1
        return "call_%08x" % box["n"]
    return next_id


def _raw_normalize(events):
    out = []
    for e in events:
        if isinstance(e, tc.Text):
            out.append(f"text:{e.text}")
        elif isinstance(e, tc.ToolInputStart):
            out.append(f"start:{e.name}")
        elif isinstance(e, tc.ToolInputDelta):
            out.append(f"delta:{e.delta}")
        elif isinstance(e, tc.ToolInputEnd):
            out.append("end")
        elif isinstance(e, tc.ToolCall):
            out.append(f"call:{e.call.name}:{e.call.input_json}")
        elif isinstance(e, tc.Malformed):
            out.append(f"malformed:{e.text}")
    return out


def normalize(events):
    """Events with ids dropped. Adjacent text and delta events are coalesced: how prose is chunked depends on the
    delta boundaries and is not a semantic difference. What must not vary is the concatenated text, the calls
    and their order."""
    out = []
    for s in _raw_normalize(events):
        prefix = next((p for p in ("text:", "delta:") if s.startswith(p)), None)
        if prefix and out and out[-1].startswith(prefix):
            out[-1] += s[len(prefix):]
        else:
            out.append(s)
    return out


def fixture_dir():
    """Tools/fixtures of the pinned third_party/slotstream copy; None if absent."""
    for base in (ROOT / "third_party" / "slotstream",):
        d = base / "Tools" / "fixtures"
        if d.is_dir():
            return d
    return None


@dataclass
class FakeStats:
    prompt_tokens: int = 0
    decode_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    finish_reason: str = "stop"
    reused_prefix_tokens: int = 0
    runtime_error: str = None


@dataclass
class FakeEngine:
    """Replays scripted text through the same callback shape the real engine uses: `on_token(index, delta)`
    returns False to stop, and the loop also stops when `should_continue()` says so. `chunk` is the number of
    characters per delta; the last delta carries the remainder."""
    text: str = ""
    chunk: int = 1
    finish_reason: str = "stop"
    reused_prefix_tokens: int = 0
    prompts: list = field(default_factory=list)

    def generate(self, prompt, params=None, on_token=None, should_continue=None):
        self.prompts.append(prompt)
        produced, n = "", 0
        for start in range(0, len(self.text), self.chunk):
            if should_continue is not None and not should_continue():
                break
            delta = self.text[start:start + self.chunk]
            produced += delta
            n += 1
            if on_token is not None and on_token(n, delta) is False:
                break
        stats = FakeStats(prompt_tokens=len(str(prompt)), decode_tokens=n, finish_reason=self.finish_reason,
                          reused_prefix_tokens=self.reused_prefix_tokens)
        return produced, [], stats


@pytest.fixture
def fx():
    return SimpleNamespace(
        tools=fx_tool_schemas(), one_call=ONE_CALL, prose_then_call=PROSE_THEN_CALL, two_calls=TWO_CALLS,
        ids=counting_ids, normalize=normalize, fixture_dir=fixture_dir, FakeEngine=FakeEngine)


@pytest.fixture(scope="session")
def tokenizer():
    from slotstream_spark.server import template
    tok = template.load_tokenizer()
    if tok is None:
        pytest.skip("no chat_template.jinja (third_party/ on a checkout, $SLOTSTREAM_MODEL on the Spark)")
    return tok


@pytest.fixture
def load_fixture():
    def load(name):
        d = fixture_dir()
        if d is None or not (d / name).exists():
            pytest.skip(f"no fixture {name}")
        return json.loads((d / name).read_text())
    return load
