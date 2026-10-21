# SPDX-License-Identifier: MIT
# Port of Server.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""The HTTP server: four wire dialects over one engine (port of Server.swift's routing and request handling).

Engine access goes through `engine_api.EngineProtocol` only. One generation runs at a time (`GenerationGate`);
metadata endpoints and count_tokens never touch the gate. Every request body is validated by its dialect before
anything is tokenized, and an image is refused before tokenizing and without taking the gate.
"""
import asyncio
import contextlib
import json
import secrets
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from slotstream_spark import config as platform_config
from slotstream_spark.server import common, http_limits, template
from slotstream_spark.server.bounded_output import BoundedStream
from slotstream_spark.server.common import VISION_OFF, DialectError, without_nulls
from slotstream_spark.server.dialects import anthropic, ollama, openai_chat, responses
from slotstream_spark.server.engine_api import EngineFailure, GenParams, PromptRequest
from slotstream_spark.server.metrics import Metrics
from slotstream_spark.server.request_control import (BODY_RESERVATION_FACTOR, FIRST_TOKEN_DEADLINE_S,
                                                     BodyBudget, GenerationGate, QueueFull, RequestControl,
                                                     status_for)

FINAL_DRAIN_S = 5.0


def fresh_seed():
    return secrets.randbelow(2 ** 64 - 1) + 1


def ignore_sigpipe():
    """A write to a closed socket must raise, not kill the process."""
    if hasattr(signal, "SIGPIPE"):
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGPIPE, signal.SIG_IGN)
        if signal.getsignal(signal.SIGPIPE) != signal.SIG_IGN:
            raise RuntimeError("SIGPIPE is not ignored")


@dataclass
class ServerConfig:
    queue_max: int = None
    first_token_deadline_s: float = FIRST_TOKEN_DEADLINE_S
    keepalive_s: float = 10.0
    stream_max_bytes: int = 1 << 20
    stream_max_frames: int = 64
    write_timeout_s: float = 5.0
    max_body: int = http_limits.MAX_BODY_BYTES
    body_budget_bytes: int = 1 << 30
    version: str = "0.1.0"
    workers: int = 8

    def __post_init__(self):
        if self.queue_max is None:
            self.queue_max = platform_config.get("server.queue_max")


@dataclass
class Done:
    frames: list = field(default_factory=list)
    body: dict = None
    error: tuple = None  # (code, message)


@dataclass
class Immediate:
    """A request answered without generating (Ollama's empty "load" request)."""
    body: dict


# ---- errors in each dialect's shape ---------------------------------------------------------------------

def error_body(dialect, status, code, message):
    if dialect == "anthropic":
        return anthropic.error_body(anthropic.error_type(status), message)
    if dialect == "responses":
        return responses.error_body(code, message)
    if dialect == "ollama":
        return {"error": message}
    return openai_chat.error_body(message, code, "server_error" if status >= 500 else "invalid_request_error")


def dialect_error(dialect, err):
    """(status, body) for a DialectError raised by a dialect or by the template layer."""
    if isinstance(err, template.TemplateError):
        return 400, error_body(dialect, 400, "invalid_request_error", str(err))
    if isinstance(err, (anthropic.Failure, responses.Failure)):
        return err.status, err.body
    return err.status, error_body(dialect, err.status, err.code, err.message)


# ---- flows: one per request, owning the dialect's reply object --------------------------------------------

class Flow:
    stream = False
    media_type = "text/event-stream"
    headers = ()

    def __init__(self):
        self.text = ""
        self.reply = None
        self.lock = threading.Lock()  # the engine thread and the keepalive timer both touch the reply

    @property
    def should_stop(self):
        return bool(self.reply is not None and getattr(self.reply, "should_stop", False))

    def start(self, prompt_tokens, reused):
        return []

    def keepalive(self):
        return None

    def stream_error(self, status, code, message):
        return []


class ChatFlow(Flow):
    def __init__(self, ctx, body):
        super().__init__()
        engine = ctx.engine
        body = without_nulls(body)
        if (e := openai_chat.validation_error(body, engine.model_name, engine.context_window)) is not None:
            raise DialectError(e, code="invalid_image" if VISION_OFF in e else "invalid_request_error")
        self.prepared = openai_chat.prepare(body, engine.context_window)
        if "seed" not in body:
            self.prepared.seed = fresh_seed()
        self.stream = self.prepared.stream
        self.reply = openai_chat.ChatReply(self.prepared, engine.model_name)
        p = self.prepared
        self.prompt = PromptRequest(lambda tok: openai_chat.render_prompt(p, tok))
        self.params = GenParams(p.params, p.seed)

    def check_prompt(self, n):
        self.prepared.check_prompt(n)

    def keepalive(self):
        return openai_chat.KEEPALIVE

    def token(self, text):
        self.text += text
        if not self.reply.incremental:
            return []
        frames = [openai_chat.sse(self.reply.chunk(d)) for d in self.reply.consume(text)]
        return frames if self.stream else []

    def finish(self, result):
        deltas, finish = self.reply.end(result.finish_reason, text=self.text)
        if self.reply.acc.error:
            return Done(error=("inference_error", self.reply.acc.error))
        usage = openai_chat.usage_body(result.prompt_tokens, result.decode_tokens, result.reused_prefix_tokens)
        if not self.stream:
            return Done(body=self.reply.completion(finish, usage))
        frames = [openai_chat.sse(self.reply.chunk(d)) for d in deltas]
        frames += [openai_chat.sse(self.reply.final_chunk(finish, usage)), openai_chat.DONE]
        return Done(frames)

    def stream_error(self, status, code, message):
        return [openai_chat.sse(error_body("chat", status, code, message))]


class AnthropicFlow(Flow):
    def __init__(self, ctx, raw):
        super().__init__()
        engine = ctx.engine
        if (e := anthropic.model_error(raw, engine.model_name)) is not None:
            raise e
        self.request = anthropic.parse(raw)
        self.prepared = anthropic.prepare(self.request, random_seed=fresh_seed())
        self.window = engine.context_window
        self.stream = self.request.stream
        self.reply = anthropic.MessagesReply(self.prepared, engine.model_name)
        req = self.request
        self.prompt = PromptRequest(lambda tok: anthropic.render_prompt(req, tok))
        self.params = GenParams(self.prepared.params, self.prepared.seed)
        names = anthropic.header_safe_field_names(self.request.ignored)
        self.headers = ((b"x-slotstream-ignored-fields", ",".join(names).encode()),) if names else ()

    def check_prompt(self, n):
        self.prepared.check_prompt(n, self.window)

    def start(self, prompt_tokens, reused):
        return self.reply.message.start(prompt_tokens, reused)

    def keepalive(self):
        return self.reply.message.keepalive()

    def token(self, text):
        self.text += text
        if not self.reply.incremental:
            return []
        frames = self.reply.consume(text)
        return frames if self.stream else []

    def finish(self, result):
        frames = self.reply.end(result.finish_reason, text=self.text)
        if (err := self.reply.failure(result.finish_reason)) is not None:
            return Done(error=("inference_error", err))
        usage = anthropic.usage_body(result.prompt_tokens, result.reused_prefix_tokens, result.decode_tokens)
        m, limited = self.reply.message, self.prepared.window_limited
        if not self.stream:
            return Done(body=m.message(result.finish_reason, result.stop_sequence, usage, limited))
        return Done(frames + m.finish(result.finish_reason, result.stop_sequence, usage, limited))

    def stream_error(self, status, code, message):
        return self.reply.message.fail(anthropic.error_type(status), message)


class ResponsesFlow(Flow):
    def __init__(self, ctx, raw):
        super().__init__()
        engine = ctx.engine
        if isinstance(raw, dict) and (e := responses.model_error(without_nulls(raw), engine.model_name)):
            raise e
        self.request = responses.parse(raw)
        self.window = engine.context_window
        self.prepared = responses.prepare(self.request, self.window, random_seed=fresh_seed())
        self.stream = self.request.stream
        self.reply = responses.ResponsesReply(self.prepared, engine.model_name)
        req = self.request
        self.prompt = PromptRequest(lambda tok: responses.render_prompt(req, tok))
        self.params = GenParams(self.prepared.params, self.prepared.seed)

    def check_prompt(self, n):
        self.prepared.check_prompt(n, self.window)

    def start(self, prompt_tokens, reused):
        return self.reply.response.created()

    def keepalive(self):
        return self.reply.response.keepalive()

    def token(self, text):
        self.text += text
        if not self.reply.incremental:
            return []
        frames = self.reply.consume(text)
        return frames if self.stream else []

    def finish(self, result):
        frames = self.reply.end(result.finish_reason, text=self.text)
        if (err := self.reply.failure()) is not None:
            return Done(error=("inference_error", err))
        usage = self.reply.usage(result.prompt_tokens, result.reused_prefix_tokens, result.decode_tokens)
        if not self.stream:
            return Done(body=self.reply.response.finished(result.finish_reason, usage))
        return Done(frames + self.reply.response.finish(result.finish_reason, usage))

    def stream_error(self, status, code, message):
        return self.reply.response.fail(code, message)


class OllamaFlow(Flow):
    media_type = "application/x-ndjson"

    def __init__(self, ctx, body, kind):
        super().__init__()
        info = ctx.model_info()
        prep = (ollama.prepare_chat if kind == "chat" else ollama.prepare_generate)(body, info)
        if isinstance(prep, ollama.Load):
            raise _Immediate(prep.body)
        self.prepared, self.info = prep, info
        o = without_nulls(body).get("options") or {}
        seed = common.as_int(o.get("seed"))
        if seed is None or seed < 0:
            seed = fresh_seed()
        else:
            seed = prep.seed
        prep.seed = seed
        self.stream = prep.stream
        self.reply = ollama.OllamaReply(prep, info.name)
        self.window = ctx.engine.context_window
        self.prompt = PromptRequest(lambda tok: prep.prompt_text(tok))
        self.params = GenParams(prep.params, prep.seed)

    def check_prompt(self, n):
        if n <= 0:
            raise DialectError("the prompt is empty")
        if n >= self.window:
            raise DialectError(f"prompt is {n} tokens, leaving no reply room in the {self.window}-token window",
                               code="context_length_exceeded")
        p = self.params.sample
        p.max_tokens = min(p.max_tokens if p.max_tokens > 0 else self.window, self.window - n)

    def token(self, text):
        self.text += text
        if not self.stream:
            return []
        frame = self.reply.delta(text)
        return [ollama.ndjson(frame)] if frame else []

    def finish(self, result):
        total = result.prefill_seconds + result.decode_seconds
        final = self.reply.final(result, self.text, total)
        return Done([ollama.ndjson(final)] if self.stream else [], None if self.stream else final)

    def stream_error(self, status, code, message):
        return [ollama.ndjson({"error": message})]


class _Immediate(Exception):
    def __init__(self, body):
        self.body = body


# ---- context ----------------------------------------------------------------------------------------------

class Context:
    def __init__(self, engine, cfg):
        self.engine, self.cfg = engine, cfg
        self.gate = GenerationGate(cfg.queue_max)
        self.metrics = Metrics()
        self.budget = BodyBudget(cfg.body_budget_bytes)
        self.executor = ThreadPoolExecutor(max_workers=cfg.workers, thread_name_prefix="slotstream-http")

    def model_info(self):
        s = self.engine.snapshot()
        return ollama.ModelInfo(name=self.engine.model_name, digest=s.digest, weights_bytes=s.weights_bytes,
                                resident_bytes=s.footprint_bytes, quantization=s.quantization,
                                max_context=self.engine.context_window, details_extra=dict(s.details))

    async def offload(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, fn, *args)


# ---- generation -------------------------------------------------------------------------------------------

class Generation:
    """The ASGI response of an admitted-to-queue request: waits for the gate, runs the engine in a worker thread,
    and writes the reply. Owns the connection from here on, so it reads `receive` for the disconnect itself."""

    def __init__(self, ctx, route, dialect, flow, ids, cost):
        self.ctx, self.route, self.dialect, self.flow, self.ids, self.cost = ctx, route, dialect, flow, ids, cost

    async def __call__(self, scope, receive, send):
        control = RequestControl()
        control.reply_stop = lambda: self.flow.should_stop
        self.gone = asyncio.Event()
        watcher = asyncio.create_task(self._watch(receive, control))
        try:
            await self._run(control, send)
        finally:
            watcher.cancel()
            self.ctx.budget.release(self.cost)

    async def _watch(self, receive, control):
        while True:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                control.cancel()
                self.gone.set()
                return

    async def _plain(self, send, status, body, headers=()):
        self.ctx.metrics.request(self.route, status)
        if status == 499 or body is None:
            raw, hdrs = b"", [(b"content-length", b"0")]
        else:
            raw = json.dumps(body, ensure_ascii=False).encode()
            hdrs = [(b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode())]
        try:
            await send({"type": "http.response.start", "status": status, "headers": hdrs + list(headers)})
            await send({"type": "http.response.body", "body": raw})
        except Exception:
            pass

    async def _failure(self, send, control):
        code, message = control.failure
        status = status_for(code)
        headers = [(b"retry-after", b"1")] if code == "queue_full" else []
        if code == "client_cancelled":
            self.ctx.metrics.inc("client_cancelled_total")
            return await self._plain(send, 499, None)
        await self._plain(send, status, error_body(self.dialect, status, code, message), headers)

    async def _run(self, control, send):
        ctx, flow, cfg = self.ctx, self.flow, self.ctx.cfg
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        acquire = asyncio.create_task(ctx.gate.acquire())

        def deadline():
            if not control.first_token and control.fail("prefill_deadline_exceeded",
                                                        "no token within the wait limit"):
                acquire.cancel()

        timer = loop.call_later(cfg.first_token_deadline_s, deadline)
        gone = asyncio.create_task(self.gone.wait())
        try:
            await asyncio.wait({acquire, gone}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            gone.cancel()
        acquired = acquire.done() and not acquire.cancelled() and acquire.exception() is None
        if acquired and control.failure is not None:
            ctx.gate.release()  # the client left (or the wait ran out) in the instant the slot came
            acquired = False
        if not acquired:
            timer.cancel()
            if not acquire.done():
                acquire.cancel()
            await asyncio.gather(acquire, return_exceptions=True)
            if not acquire.cancelled() and isinstance(acquire.exception(), QueueFull):
                control.fail("queue_full", "the server is busy; try again")
            control.fail("inference_error", "the generation did not start")
            return await self._failure(send, control)
        queue_wait = time.monotonic() - t0
        stream = (BoundedStream(cfg.stream_max_bytes, cfg.stream_max_frames, cfg.write_timeout_s)
                  if flow.stream else None)
        admitted = asyncio.Event()
        state = {"admitted": False}

        def put(frames):
            if stream is not None:
                for f in frames:
                    if not stream.put(f):
                        control.overflowed = True
                        control.fail("client_cancelled", "output buffer overflowed")
                        return

        def on_admitted(reused):
            with flow.lock:
                frames = flow.start(len(self.ids), reused)
            put(frames)
            state["admitted"] = True
            loop.call_soon_threadsafe(admitted.set)

        def on_token(text):
            control.first_token = True
            with flow.lock:
                frames = flow.token(text)
            put(frames)

        def work():
            try:
                result = ctx.engine.generate(self.ids, flow.params, on_admitted=on_admitted, on_token=on_token,
                                             should_continue=control.should_continue)
                if control.failure is None:
                    with flow.lock:
                        done = flow.finish(result)
                    if done.error is None:
                        put(done.frames)
                        return result, done
                    control.fail(*done.error)
            except EngineFailure as e:
                control.fail(e.code, e.message)
            except Exception:  # noqa: BLE001 - the engine's internals must not reach the wire
                control.fail("inference_error", "inference failed")
                ctx.metrics.inc("engine_exceptions_total")
            if state["admitted"]:
                code, message = control.failure
                put(flow.stream_error(status_for(code), code, message))
            return None, Done(error=control.failure)

        gen = loop.run_in_executor(ctx.executor, work)
        gen.add_done_callback(lambda _f: ctx.gate.release())
        closer = asyncio.ensure_future(_close_when_done(gen, stream))
        try:
            if stream is None:
                result, done = await asyncio.shield(gen)
                if control.failure is not None:
                    return await self._failure(send, control)
                self._record(result, queue_wait)
                return await self._plain(send, 200, done.body, flow.headers)
            await asyncio.wait({asyncio.ensure_future(admitted.wait()), gen}, return_when=asyncio.FIRST_COMPLETED)
            if not state["admitted"] and gen.done() and control.failure is not None:
                return await self._failure(send, control)
            await self._drain(send, control, stream, gen)
            result, _done = await asyncio.shield(gen)
            if result is not None:
                self._record(result, queue_wait)
        finally:
            timer.cancel()
            if not gen.done():
                control.cancel()
            await asyncio.gather(closer, return_exceptions=True)

    def _record(self, result, queue_wait):
        if result is not None:
            self.ctx.metrics.generation(result, queue_wait)

    async def _drain(self, send, control, stream, gen):
        ctx = self.ctx
        ctx.metrics.request(self.route, 200)
        hdrs = [(b"content-type", self.flow.media_type.encode()), (b"cache-control", b"no-cache"),
                (b"x-accel-buffering", b"no"), *self.flow.headers]
        final_deadline = None
        try:
            await stream.send_with_timeout(send, {"type": "http.response.start", "status": 200, "headers": hdrs})
            async for frame in stream.frames(idle_timeout=ctx.cfg.keepalive_s):
                if frame is None:
                    with self.flow.lock:
                        ka = self.flow.keepalive()
                    if not ka:
                        continue
                    frame = ka.encode()
                if final_deadline is None and gen.done():
                    final_deadline = time.monotonic() + FINAL_DRAIN_S
                timeout = stream.write_timeout if final_deadline is None else max(
                    0.0, min(stream.write_timeout, final_deadline - time.monotonic()))
                await asyncio.wait_for(send({"type": "http.response.body", "body": frame, "more_body": True}),
                                       timeout)
            if stream.failed:
                ctx.metrics.inc("stream_overflow_total")
                control.overflowed = True
            await asyncio.wait_for(send({"type": "http.response.body", "body": b"", "more_body": False}),
                                   FINAL_DRAIN_S)
        except asyncio.TimeoutError:
            ctx.metrics.inc("stream_write_timeouts_total")
            control.overflowed = True
            control.fail("client_cancelled", "write timed out")
        except Exception:
            control.cancel()


async def _close_when_done(gen, stream):
    try:
        await asyncio.shield(gen)
    except BaseException:
        pass
    if stream is not None:
        stream.close()


# ---- routes -----------------------------------------------------------------------------------------------

def create_app(engine, cfg=None):
    cfg = cfg or ServerConfig()
    ctx = Context(engine, cfg)
    metrics = ctx.metrics

    def respond(route, status, body, headers=None):
        metrics.request(route, status)
        return JSONResponse(body, status_code=status, headers=headers)

    async def read_json(request, route, dialect, cost_box):
        raw = await request.body()
        cost = ctx.budget.reserve(len(raw))
        if not cost:
            return None, respond(route, 503, error_body(dialect, 503, "insufficient_memory",
                                                        "not enough memory to hold this request"))
        cost_box.append(cost)
        try:
            return json.loads(raw), None
        except (ValueError, RecursionError):
            return None, respond(route, 400, error_body(dialect, 400, "invalid_request_error",
                                                        "request body is not valid JSON"))

    def generation_endpoint(route, dialect, factory):
        async def endpoint(request):
            cost_box = []
            handed_over = False
            try:
                body, bad = await read_json(request, route, dialect, cost_box)
                if bad is not None:
                    return bad
                try:
                    flow = await ctx.offload(factory, body)
                except _Immediate as imm:
                    return respond(route, 200, imm.body)
                except (DialectError, template.TemplateError) as err:
                    if VISION_OFF in str(err):
                        metrics.inc("images_rejected_total")
                    status, payload = dialect_error(dialect, err)
                    return respond(route, status, payload)
                try:
                    ids = await ctx.offload(engine.render_and_tokenize, flow.prompt)
                    flow.check_prompt(len(ids))
                except (DialectError, template.TemplateError) as err:
                    status, payload = dialect_error(dialect, err)
                    return respond(route, status, payload)
                except EngineFailure as e:
                    return respond(route, status_for(e.code), error_body(dialect, status_for(e.code), e.code,
                                                                         e.message))
                handed_over = True
                return Generation(ctx, route, dialect, flow, ids, cost_box[0])
            finally:
                if cost_box and not handed_over:
                    ctx.budget.release(cost_box[0])
        return endpoint

    async def count_tokens(request):
        cost_box = []
        try:
            body, bad = await read_json(request, "/v1/messages/count_tokens", "anthropic", cost_box)
            if bad is not None:
                return bad
            try:
                def count():
                    if (e := anthropic.model_error(body, engine.model_name)) is not None:
                        raise e
                    req = anthropic.parse(body, counting=True)
                    return engine.count_tokens(PromptRequest(lambda tok: anthropic.render_prompt(req, tok)))
                n = await ctx.offload(count)
            except (DialectError, template.TemplateError) as err:
                if VISION_OFF in str(err):
                    metrics.inc("images_rejected_total")
                status, payload = dialect_error("anthropic", err)
                return respond("/v1/messages/count_tokens", status, payload)
            return respond("/v1/messages/count_tokens", 200, {"input_tokens": n})
        finally:
            if cost_box:
                ctx.budget.release(cost_box[0])

    async def models(request):
        window = engine.context_window
        return respond("/v1/models", 200, {"object": "list", "data": [{
            "id": engine.model_name, "object": "model", "created": 0, "owned_by": "slotstream-spark",
            "context_window": window, "max_output_tokens": common.output_budget(window)}]})

    async def stored_state(request):
        return respond("/v1/responses/{id}", 404, responses.error_body(
            "stored_state_unsupported", "this server stores no responses"))

    async def tags(request):
        return respond("/api/tags", 200, ollama.tags(ctx.model_info()))

    async def ps(request):
        return respond("/api/ps", 200, ollama.ps(ctx.model_info()))

    async def version(request):
        return respond("/api/version", 200, ollama.version_body(cfg.version))

    async def show(request):
        cost_box = []
        try:
            body, bad = await read_json(request, "/api/show", "ollama", cost_box)
            if bad is not None:
                return bad
            if not isinstance(body, dict):
                return respond("/api/show", 400, {"error": "request body must be a JSON object"})
            try:
                return respond("/api/show", 200, ollama.show(body, ctx.model_info()))
            except DialectError as err:
                return respond("/api/show", err.status, ollama.error_body(err))
        finally:
            if cost_box:
                ctx.budget.release(cost_box[0])

    async def metrics_endpoint(request):
        text = metrics.render(engine.snapshot(), ctx.gate.queued, ctx.gate.busy)
        return PlainTextResponse(text, media_type="text/plain; version=0.0.4")

    async def root(request):
        return PlainTextResponse("slotstream-spark is running")

    async def not_found(request):
        path = request.url.path
        return respond("other", 404, http_limits.error_payload(path, 404, "not found", "not_found"))

    def chat(ctx_, body):
        if not isinstance(body, dict):
            raise DialectError("request body must be a JSON object")
        return ChatFlow(ctx_, body)

    def ollama_flow(kind):
        def make(ctx_, body):
            if not isinstance(body, dict):
                raise DialectError("request body must be a JSON object")
            return OllamaFlow(ctx_, body, kind)
        return make

    def bound(factory):
        return lambda body: factory(ctx, body)

    routes = [
        Route("/v1/messages", generation_endpoint("/v1/messages", "anthropic", bound(AnthropicFlow)),
              methods=["POST"]),
        Route("/v1/messages/count_tokens", count_tokens, methods=["POST"]),
        Route("/v1/chat/completions", generation_endpoint("/v1/chat/completions", "chat", bound(chat)),
              methods=["POST"]),
        Route("/v1/responses", generation_endpoint("/v1/responses", "responses", bound(ResponsesFlow)),
              methods=["POST"]),
        Route("/v1/responses/{rid}", stored_state, methods=["GET", "DELETE"]),
        Route("/v1/models", models, methods=["GET"]),
        Route("/api/chat", generation_endpoint("/api/chat", "ollama", bound(ollama_flow("chat"))),
              methods=["POST"]),
        Route("/api/generate", generation_endpoint("/api/generate", "ollama", bound(ollama_flow("generate"))),
              methods=["POST"]),
        Route("/api/tags", tags, methods=["GET"]),
        Route("/api/show", show, methods=["POST"]),
        Route("/api/ps", ps, methods=["GET"]),
        Route("/api/version", version, methods=["GET"]),
        Route("/metrics", metrics_endpoint, methods=["GET"]),
        Route("/", root, methods=["GET"]),
        Route("/{path:path}", not_found, methods=["GET", "POST", "PUT", "PATCH", "DELETE"]),
    ]

    @contextlib.asynccontextmanager
    async def lifespan(app):
        ignore_sigpipe()
        yield
        ctx.executor.shutdown(wait=False, cancel_futures=True)

    app = Starlette(routes=routes, lifespan=lifespan,
                    middleware=[Middleware(http_limits.HttpLimits, max_body=cfg.max_body)])
    app.state.ctx = ctx
    return app


def serve(engine, cfg=None, port=18435, host=None):
    """Listen on 127.0.0.1 only; a request for another address is refused."""
    import uvicorn
    ignore_sigpipe()
    options = http_limits.uvicorn_options()
    options["host"] = http_limits.bind_host(host)
    uvicorn.run(create_app(engine, cfg), port=port, log_level="warning", **options)
