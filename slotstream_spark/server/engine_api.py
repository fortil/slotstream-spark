"""What the HTTP layer needs from an engine, and nothing else.

The server never imports the real engine: it sees this protocol, so the same app runs on the fake engine in the
tests and on the adapter that wraps the model (step 84).
"""
from dataclasses import dataclass, field
from typing import Callable, Protocol

from slotstream_spark.engine.sampler import SampleParams


@dataclass
class PromptRequest:
    """A prompt waiting to be tokenized. `render(tokenizer) -> str` is the dialect's own template call, so the
    engine owns the tokenizer and the dialect owns the text."""
    render: Callable


@dataclass
class GenParams:
    sample: SampleParams
    seed: int


@dataclass
class GenerationResult:
    finish_reason: str  # stop | length
    prompt_tokens: int = 0
    decode_tokens: int = 0
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    reused_prefix_tokens: int = 0
    stop_sequence: str = None
    text: str = ""


@dataclass
class EngineSnapshot:
    """Lock-free view for the metadata endpoints and /metrics. Taking one never waits for a generation."""
    footprint_bytes: int = 0
    resident_bytes: int = 0
    weights_bytes: int = 0
    digest: str = ""
    quantization: str = "affine-4bit"
    pool_hits: int = 0
    governor_mode: str = "normal"
    details: dict = field(default_factory=dict)


class EngineFailure(Exception):
    """A failure the engine names with one of the request-control codes (request_control.FAILURE_STATUS)."""

    def __init__(self, code, message=""):
        super().__init__(message or code)
        self.code, self.message = code, message or code


class EngineProtocol(Protocol):
    model_name: str
    context_window: int

    def render_and_tokenize(self, req: PromptRequest) -> list: ...

    def count_tokens(self, req: PromptRequest) -> int: ...

    def generate(self, ids, params: GenParams, *, on_admitted, on_token, should_continue) -> GenerationResult:
        """Blocking. `on_admitted(reused_tokens)` once the prompt is admitted and before the first token;
        `on_token(text)` per decoded text delta; stop soon after `should_continue()` returns False."""

    def snapshot(self) -> EngineSnapshot: ...
