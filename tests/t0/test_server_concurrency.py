"""Gate, queue, keepalives, bounded output and cancellation, on the fake engine; plus the uvicorn connection cap."""
import asyncio
import socket
import threading
import time

import pytest
from fakes import ROUTES, FakeEngine, call, chat_body

from slotstream_spark.server import http_limits
from slotstream_spark.server.app import ServerConfig, create_app
from slotstream_spark.server.bounded_output import BoundedStream
from slotstream_spark.server.request_control import GenerationGate, QueueFull


async def wait_started(engine, timeout=5):
    await asyncio.get_running_loop().run_in_executor(None, engine.started.wait, timeout)
    assert engine.started.is_set()


def make(engine, **cfg):
    return create_app(engine, ServerConfig(queue_max=cfg.pop("queue_max", 4), **cfg))


# ---- gate -------------------------------------------------------------------------------------------------

def test_gate_queue_max_zero_refuses_second_at_once():
    async def go():
        g = GenerationGate(0)
        await g.acquire()
        with pytest.raises(QueueFull):
            await g.acquire()
        g.release()
        await g.acquire()
    asyncio.run(go())


def test_gate_orders_waiters_and_cancel_leaves_the_queue():
    async def go():
        g = GenerationGate(2)
        await g.acquire()
        order = []

        async def waiter(n):
            await g.acquire()
            order.append(n)
            g.release()
        a, b = asyncio.create_task(waiter("a")), asyncio.create_task(waiter("b"))
        await asyncio.sleep(0.01)
        assert g.queued == 2
        with pytest.raises(QueueFull):
            await g.acquire()
        a.cancel()
        await asyncio.sleep(0.01)
        assert g.queued == 1
        g.release()
        await b
        assert order == ["b"] and not g.busy and g.queued == 0
    asyncio.run(go())


def test_gate_wait_timeout():
    async def go():
        g = GenerationGate(1)
        await g.acquire()
        with pytest.raises(asyncio.TimeoutError):
            await g.acquire(timeout=0.05)
        assert g.queued == 0
    asyncio.run(go())


def test_http_queue_max_zero_second_request_is_503_with_retry_after():
    async def go():
        engine = FakeEngine(block_after_admit=True)
        app = make(engine, queue_max=0)
        first = asyncio.create_task(call(app, "POST", "/v1/chat/completions", chat_body()))
        await wait_started(engine)
        for name in ("chat", "anthropic", "responses", "ollama_chat"):
            path, mk = ROUTES[name]
            t = time.monotonic()
            r = await call(app, "POST", path, mk())
            assert r.status == 503 and r.header("retry-after") == "1", name
            assert time.monotonic() - t < 0.5
        engine.release()
        assert (await first).status == 200
        assert len(engine.generate_calls) == 1 and engine.max_in_generate == 1
    asyncio.run(go())


def test_http_queue_max_one_second_waits_third_is_503():
    async def go():
        engine = FakeEngine(block_after_admit=True)
        app = make(engine, queue_max=1)
        first = asyncio.create_task(call(app, "POST", "/v1/chat/completions", chat_body()))
        await wait_started(engine)
        second = asyncio.create_task(call(app, "POST", "/v1/chat/completions", chat_body()))
        await asyncio.sleep(0.2)
        assert not second.done() and app.state.ctx.gate.queued == 1
        r3 = await call(app, "POST", "/v1/chat/completions", chat_body())
        assert r3.status == 503 and r3.header("retry-after") == "1"
        assert r3.json()["error"]["code"] == "queue_full"
        engine.release()
        assert (await first).status == 200 and (await second).status == 200
        assert len(engine.generate_calls) == 2 and engine.max_in_generate == 1
        assert not app.state.ctx.gate.busy
    asyncio.run(go())


def test_queue_full_503_shape_per_dialect():
    async def go():
        engine = FakeEngine(block_after_admit=True)
        app = make(engine, queue_max=0)
        first = asyncio.create_task(call(app, "POST", "/v1/chat/completions", chat_body()))
        await wait_started(engine)
        a = (await call(app, "POST", "/v1/messages", ROUTES["anthropic"][1]())).json()
        assert a["type"] == "error" and a["error"]["type"] == "overloaded_error"
        o = (await call(app, "POST", "/api/chat", ROUTES["ollama_chat"][1]())).json()
        assert isinstance(o["error"], str)
        r = (await call(app, "POST", "/v1/responses", ROUTES["responses"][1]())).json()
        assert r["error"]["code"] == "queue_full"
        engine.release()
        await first
    asyncio.run(go())


# ---- metadata stays responsive ----------------------------------------------------------------------------

def test_metadata_and_count_tokens_answer_fast_during_a_generation():
    async def go():
        engine = FakeEngine(block_after_admit=True)
        app = make(engine, queue_max=0)
        first = asyncio.create_task(call(app, "POST", "/v1/chat/completions", chat_body()))
        await wait_started(engine)
        before = engine.snapshots
        count_body = {"model": "slotstream-test", "messages": [{"role": "user", "content": "a b c"}]}
        for method, path, body in [("GET", "/v1/models", None), ("GET", "/api/tags", None), ("GET", "/api/ps", None),
                                   ("GET", "/api/version", None), ("GET", "/metrics", None),
                                   ("POST", "/api/show", {"model": "slotstream-test"}),
                                   ("POST", "/v1/messages/count_tokens", count_body)]:
            t = time.monotonic()
            r = await call(app, method, path, body)
            assert r.status == 200, path
            assert time.monotonic() - t < 0.2, path
        assert engine.snapshots > before
        assert app.state.ctx.gate.busy and len(engine.generate_calls) == 1
        engine.release()
        await first
    asyncio.run(go())


# ---- keepalive ---------------------------------------------------------------------------------------------

async def stream_until(app, path, body, predicate, engine, seconds=3.0):
    """Start a streaming request against a blocked engine, wait for `predicate(frames)`, then release."""
    seen = []

    async def hook(message):
        if message["type"] == "http.response.body" and message.get("body"):
            seen.append(message["body"].decode())
    task = asyncio.create_task(call(app, "POST", path, body, send_hook=hook))
    end = time.monotonic() + seconds
    while time.monotonic() < end and not predicate(seen):
        await asyncio.sleep(0.02)
    engine.release()
    reply = await task
    return seen, reply


def test_keepalive_for_the_three_sse_dialects():
    expectations = {
        "chat": lambda seen: sum(1 for s in seen if s == ": keepalive\n\n") >= 2,
        "anthropic": lambda seen: sum(1 for s in seen if s.startswith("event: ping")) >= 3,  # one at the start
        "responses": lambda seen: sum(1 for s in seen if s.startswith("event: response.in_progress")) >= 3,
    }
    for name, predicate in expectations.items():
        engine = FakeEngine(block_after_admit=True)
        app = make(engine, keepalive_s=0.05)
        path, mk = ROUTES[name]
        seen, reply = asyncio.run(stream_until(app, path, mk(stream=True), predicate, engine))
        assert predicate(seen), (name, seen)
        assert reply.status == 200


def test_ollama_never_sends_a_keepalive():
    async def go():
        engine = FakeEngine(block_after_admit=True)
        app = make(engine, keepalive_s=0.03)
        seen = []

        async def hook(message):
            if message["type"] == "http.response.body" and message.get("body"):
                seen.append(message["body"])
        task = asyncio.create_task(call(app, "POST", "/api/chat", ROUTES["ollama_chat"][1](), send_hook=hook))
        await wait_started(engine)
        await asyncio.sleep(0.4)
        assert seen == []  # many keepalive intervals passed and nothing was written
        engine.release()
        reply = await task
        assert all(line.startswith(b"{") for line in reply.body.strip().split(b"\n"))
    asyncio.run(go())


def test_headers_wait_for_admission():
    async def go():
        engine = FakeEngine(block_after_admit=True)
        app = make(engine, queue_max=1)
        first = asyncio.create_task(call(app, "POST", "/v1/chat/completions", chat_body()))
        await wait_started(engine)
        starts = []

        async def hook(message):
            if message["type"] == "http.response.start":
                starts.append(message["status"])
        second = asyncio.create_task(call(app, "POST", "/v1/chat/completions", chat_body(stream=True),
                                          send_hook=hook))
        await asyncio.sleep(0.2)
        assert starts == []  # queued: no head yet
        engine.release()
        await first
        assert (await second).status == 200 and starts == [200]
    asyncio.run(go())


# ---- bounded output ----------------------------------------------------------------------------------------

def test_bounded_stream_overflow_by_frames_and_bytes():
    async def go():
        s = BoundedStream(max_bytes=100, max_frames=3)
        assert s.put("a") and s.put("b") and s.put("c")
        assert not s.put("d") and s.failed
        s = BoundedStream(max_bytes=10, max_frames=100)
        assert s.put("12345") and not s.put("1234567")
        from slotstream_spark.server import bounded_output as bo
        assert (bo.MAX_BYTES, bo.MAX_FRAMES, bo.WRITE_TIMEOUT_S) == (1 << 20, 64, 5.0)
    asyncio.run(go())


def test_bounded_stream_drains_in_order_and_ends_on_close():
    async def go():
        s = BoundedStream()
        threading.Thread(target=lambda: [s.put(f"f{i}") for i in range(10)] and s.close()).start()
        got = [f async for f in s.frames(idle_timeout=1)]
        assert got == [f"f{i}".encode() for i in range(10)]
    asyncio.run(go())


def test_slow_reader_overflows_and_gate_is_released_in_under_6_seconds():
    async def go():
        engine = FakeEngine(script=["tok "] * 5000)
        app = make(engine)

        async def stuck(message):
            await asyncio.sleep(3600)  # a reader that never reads
        t = time.monotonic()
        reply = await call(app, "POST", "/v1/chat/completions", chat_body(stream=True), send_hook=stuck)
        elapsed = time.monotonic() - t
        assert elapsed < 6.0, elapsed
        assert engine.saw_stop  # generation was cancelled, not run to the end
        assert not app.state.ctx.gate.busy
        # and the next request is served
        ok = await call(app, "POST", "/v1/chat/completions", chat_body())
        assert ok.status == 200
        assert reply.status is None  # the head never went out
    asyncio.run(go())


def test_slow_reader_mid_stream_gate_released_in_under_6_seconds():
    async def go():
        engine = FakeEngine(script=["tok "] * 5000)
        app = make(engine)
        count = 0

        async def stuck_after_head(message):
            nonlocal count
            if message["type"] == "http.response.body":
                count += 1
                if count > 1:
                    await asyncio.sleep(3600)
        t = time.monotonic()
        await call(app, "POST", "/v1/chat/completions", chat_body(stream=True), send_hook=stuck_after_head)
        assert time.monotonic() - t < 6.0
        assert engine.saw_stop and not app.state.ctx.gate.busy
    asyncio.run(go())


# ---- cancellation ------------------------------------------------------------------------------------------

def test_disconnect_during_generation_stops_engine_and_frees_gate():
    async def go():
        engine = FakeEngine(block_after_admit=True)
        app = make(engine)
        gone = asyncio.Event()
        task = asyncio.create_task(call(app, "POST", "/v1/chat/completions", chat_body(), disconnect=gone))
        await wait_started(engine)
        gone.set()
        r = await asyncio.wait_for(task, 3)
        assert r.status == 499 and r.body == b""
        await asyncio.get_running_loop().run_in_executor(None, engine.finished.wait, 3)
        assert engine.saw_stop
        for _ in range(100):
            if not app.state.ctx.gate.busy:
                break
            await asyncio.sleep(0.02)
        assert not app.state.ctx.gate.busy
    asyncio.run(go())


def test_disconnect_while_queued_leaves_the_queue():
    async def go():
        engine = FakeEngine(block_after_admit=True)
        app = make(engine, queue_max=2)
        first = asyncio.create_task(call(app, "POST", "/v1/chat/completions", chat_body()))
        await wait_started(engine)
        gone = asyncio.Event()
        second = asyncio.create_task(call(app, "POST", "/v1/chat/completions", chat_body(), disconnect=gone))
        await asyncio.sleep(0.2)
        assert app.state.ctx.gate.queued == 1
        gone.set()
        r = await asyncio.wait_for(second, 3)
        assert r.status == 499
        assert app.state.ctx.gate.queued == 0
        engine.release()
        assert (await first).status == 200
        assert len(engine.generate_calls) == 1  # the cancelled one never ran
    asyncio.run(go())


# ---- uvicorn connection cap ---------------------------------------------------------------------------------

def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_33rd_connection_gets_503_in_under_a_second():
    import uvicorn
    engine = FakeEngine()
    app = create_app(engine, ServerConfig(queue_max=1))
    port = _free_port()
    opts = http_limits.uvicorn_options()
    server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="error", **opts))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started
    held = []
    try:
        ok = socket.create_connection(("127.0.0.1", port), timeout=2)
        ok.sendall(b"GET /api/version HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        assert b" 200 " in ok.recv(4096).split(b"\r\n")[0]
        ok.close()
        time.sleep(0.2)
        for _ in range(32):
            held.append(socket.create_connection(("127.0.0.1", port), timeout=2))
        time.sleep(0.3)
        t = time.monotonic()
        extra = socket.create_connection(("127.0.0.1", port), timeout=2)
        extra.sendall(b"GET /api/version HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        first_line = extra.recv(4096).split(b"\r\n")[0]
        assert b" 503 " in first_line, first_line
        assert time.monotonic() - t < 1.0
        extra.close()
    finally:
        for c in held:
            c.close()
        server.should_exit = True
        thread.join(5)
