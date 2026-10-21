"""A fake engine and an ASGI driver for the server tests. No model, no socket (except the uvicorn test)."""
import asyncio
import json
import threading
import time
from types import SimpleNamespace

from slotstream_spark.server.engine_api import EngineFailure, EngineSnapshot, GenerationResult


class StubTokenizer:
    """Enough of the Tokenizer surface for template.render: a deterministic text and a word-count encode."""

    def render_chat(self, messages, enable_thinking=False, tools=None):
        parts = []
        for m in messages:
            c = m.get("content")
            parts.append(f"<{m['role']}>{c if isinstance(c, str) else json.dumps(c)}")
        return "\n".join(parts) + "\n<assistant>" + ("<think>" if enable_thinking else "")

    def encode(self, text):
        return [hash(w) & 0xFFFF for w in text.split()] or [0]


class FakeEngine:
    """Scripted engine. `script` is the list of text deltas. `block_after_admit` holds the generation until
    `release()`; the engine still watches `should_continue` while it waits."""

    def __init__(self, script=("Hello", " world"), model_name="slotstream-test", context_window=4096,
                 block_after_admit=False, token_delay=0.0, reused=0, fail=None, fail_after=None,
                 footprint=123456, tokens_per_word=1):
        self.script, self.model_name, self.context_window = list(script), model_name, context_window
        self.block_after_admit, self.token_delay, self.reused = block_after_admit, token_delay, reused
        self.fail, self.fail_after = fail, fail_after
        self.footprint = footprint
        self.tokenizer = StubTokenizer()
        self.generate_calls = []
        self.render_calls = 0
        self.started = threading.Event()
        self.released = threading.Event()
        self.finished = threading.Event()
        self.saw_stop = False
        self.in_generate = 0
        self.max_in_generate = 0
        self.snapshots = 0
        self._lock = threading.Lock()

    def release(self):
        self.released.set()

    def render_and_tokenize(self, req):
        self.render_calls += 1
        return self.tokenizer.encode(req.render(self.tokenizer))

    def count_tokens(self, req):
        return len(self.tokenizer.encode(req.render(self.tokenizer)))

    def snapshot(self):
        self.snapshots += 1
        return EngineSnapshot(footprint_bytes=self.footprint, resident_bytes=self.footprint, weights_bytes=1000,
                              digest="sha256:abc", pool_hits=7, governor_mode="normal")

    def generate(self, ids, params, *, on_admitted, on_token, should_continue):
        with self._lock:
            self.in_generate += 1
            self.max_in_generate = max(self.max_in_generate, self.in_generate)
        self.generate_calls.append((list(ids), params))
        self.started.set()
        try:
            if self.fail == "before_admit":
                raise EngineFailure("insufficient_memory", "no room")
            if self.fail == "boom":
                raise RuntimeError("secret internal detail")
            on_admitted(self.reused)
            if self.block_after_admit:
                while not self.released.wait(0.01):
                    if not should_continue():
                        self.saw_stop = True
                        return GenerationResult("stop", len(ids), 0, reused_prefix_tokens=self.reused)
            text, n = "", 0
            for delta in self.script:
                if not should_continue():
                    self.saw_stop = True
                    break
                if self.fail_after is not None and n >= self.fail_after:
                    raise EngineFailure(self.fail or "inference_error", "failed mid-stream")
                if self.token_delay:
                    time.sleep(self.token_delay)
                text += delta
                n += 1
                on_token(delta)
            return GenerationResult("stop", len(ids), n, 0.01, 0.02, self.reused, text=text)
        finally:
            with self._lock:
                self.in_generate -= 1
            self.finished.set()


class Reply(SimpleNamespace):
    @property
    def text(self):
        return self.body.decode()

    def json(self):
        return json.loads(self.body)

    def header(self, name):
        for k, v in self.headers:
            if k.decode().lower() == name:
                return v.decode()
        return None

    def frames(self):
        return [f for f in self.text.split("\n\n") if f]


def make_scope(method, path, headers=()):
    return {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "path": path,
            "raw_path": path.encode(), "query_string": b"", "root_path": "", "scheme": "http",
            "server": ("127.0.0.1", 18435), "client": ("127.0.0.1", 50000),
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers]}


async def call(app, method, path, body=None, *, headers=(), raw=None, send_hook=None, disconnect=None):
    """Run one request through the ASGI app. `body` is JSON; `raw` replaces it. `disconnect` is an asyncio.Event:
    when set the client goes away. `send_hook(message)` may be a coroutine function that delays the send."""
    payload = raw if raw is not None else (b"" if body is None else json.dumps(body).encode())
    hdrs = list(headers)
    if not any(k.lower() == "content-length" for k, _ in hdrs) and raw is None:
        hdrs.append(("content-length", str(len(payload))))
    sent = False
    gone = disconnect or asyncio.Event()

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": payload, "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    messages = []

    async def send(message):
        if send_hook is not None:
            await send_hook(message)
        messages.append(message)

    try:
        await app(make_scope(method, path, hdrs), receive, send)
    finally:
        gone.set()
    start = next((m for m in messages if m["type"] == "http.response.start"), None)
    return Reply(status=start["status"] if start else None, headers=start["headers"] if start else [],
                 body=b"" if method == "HEAD" else  # a server never writes a HEAD body
                 b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body"),
                 messages=messages)


def chat_body(**extra):
    return {"model": "slotstream-test", "messages": [{"role": "user", "content": "hi there"}], **extra}


def anthropic_body(**extra):
    return {"model": "slotstream-test", "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi there"}], **extra}


def responses_body(**extra):
    return {"model": "slotstream-test", "input": "hi there", **extra}


def ollama_chat_body(**extra):
    return {"model": "slotstream-test", "messages": [{"role": "user", "content": "hi there"}], **extra}


def ollama_generate_body(**extra):
    return {"model": "slotstream-test", "prompt": "hi there", **extra}


ROUTES = {
    "chat": ("/v1/chat/completions", chat_body),
    "anthropic": ("/v1/messages", anthropic_body),
    "responses": ("/v1/responses", responses_body),
    "ollama_chat": ("/api/chat", ollama_chat_body),
    "ollama_generate": ("/api/generate", ollama_generate_body),
}
