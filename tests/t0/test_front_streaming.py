"""The front relays bytes as they arrive, answers in the dialect's shape while the engine loads, and never starts
an engine during a development run. Sockets are loopback and ephemeral; nothing touches systemd."""
import asyncio
import fcntl
import json
import time

import pytest

from slotstream_spark import front


async def upstream(handler):
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[:2]


async def front_server(fr):
    server = await asyncio.start_server(fr.handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[:2]


def proxying(addr):
    fr = front.Front(engine_addr=addr, dev_run_fn=lambda: False)
    fr.decide = lambda: (front.PROXY, "engine ready")
    return fr


def test_pings_arrive_at_the_engine_pace_not_in_a_burst():
    async def scenario():
        async def engine(r, w):
            await r.readuntil(b"\r\n\r\n")
            w.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
            await w.drain()
            for _ in range(5):
                w.write(b"event: ping\ndata: {}\n\n")
                await w.drain()
                await asyncio.sleep(0.2)
            w.close()

        up, up_addr = await upstream(engine)
        fr, fr_addr = await front_server(proxying(up_addr))
        r, w = await asyncio.open_connection(*fr_addr)
        w.write(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n")
        await w.drain()
        stamps, t0 = [], time.monotonic()
        while True:
            chunk = await r.read(4096)
            if not chunk:
                break
            stamps += [time.monotonic() - t0] * chunk.count(b"event: ping")
        fr.close(), up.close()
        return stamps

    stamps = asyncio.run(scenario())
    assert len(stamps) == 5
    assert stamps[0] < 0.15                          # the first ping did not wait for the stream to end
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert all(0.12 < g < 0.4 for g in gaps), gaps   # spaced like the engine's own pings


def test_request_body_reaches_the_engine_and_the_answer_comes_back_whole():
    async def scenario():
        async def engine(r, w):
            head = await r.readuntil(b"\r\n\r\n")
            n = int(head.lower().split(b"content-length: ")[1].split(b"\r\n")[0])
            body = await r.readexactly(n)
            w.write(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: close\r\n\r\n" % len(body) + body)
            await w.drain()
            w.close()

        up, up_addr = await upstream(engine)
        fr, fr_addr = await front_server(proxying(up_addr))
        r, w = await asyncio.open_connection(*fr_addr)
        payload = b'{"a": 1}'
        w.write(b"POST /x HTTP/1.1\r\nContent-Length: %d\r\n\r\n" % len(payload) + payload)
        await w.drain()
        data = await r.read()
        fr.close(), up.close()
        return data

    assert asyncio.run(scenario()).endswith(b'{"a": 1}')


def loading_front():
    fr = front.Front(engine_addr=("127.0.0.1", 1), dev_run_fn=lambda: False, wait_s=0)
    fr.decide = lambda: (front.LOADING, "engine loading")
    return fr


def ask(fr, path):
    async def scenario():
        srv, addr = await front_server(fr)
        r, w = await asyncio.open_connection(*addr)
        w.write(f"POST {path} HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n".encode())
        await w.drain()
        data = await r.read()
        srv.close()
        return data

    raw = asyncio.run(scenario())
    head, body = raw.split(b"\r\n\r\n", 1)
    return head.decode(), json.loads(body)


def test_loading_answers_503_with_retry_after_in_the_anthropic_shape():
    head, body = ask(loading_front(), "/v1/messages")
    assert head.startswith("HTTP/1.1 503") and "Retry-After: 10" in head
    assert body["type"] == "error" and body["error"]["type"] == "overloaded_error"


@pytest.mark.parametrize("path,key", [("/v1/chat/completions", "error"), ("/v1/responses", "error"), ("/api/chat", "error")])
def test_loading_in_the_other_dialects(path, key):
    head, body = ask(loading_front(), path)
    assert head.startswith("HTTP/1.1 503") and "Retry-After: 10" in head and key in body
    if path == "/api/chat":
        assert isinstance(body["error"], str)
    elif path == "/v1/responses":
        assert body["error"]["code"] == "loading"
    else:
        assert body["error"]["type"] == "server_error"


# ---- development runs ---------------------------------------------------------------------------------------
def test_policy_refuses_during_a_development_run_and_proxies_a_ready_engine():
    from slotstream_spark.memory import governor as G
    cfg = G.load_config()
    assert front.policy(False, False, 720, (), 60000, 20000, cfg, dev_run=True) == (front.REFUSE, front.DEV_RUN)
    assert front.policy(True, False, 720, (), 60000, 20000, cfg, dev_run=True)[0] == front.PROXY
    assert front.policy(False, False, 720, (), 60000, 20000, cfg, dev_run=False)[0] == front.START


def test_dev_run_active_sees_a_held_lock_and_a_running_job(tmp_path):
    assert front.dev_run_active(tmp_path) is False
    assert not (tmp_path / ".model.lock").exists()          # looking creates nothing
    lock = tmp_path / ".model.lock"
    lock.write_text("")
    fd = open(lock)
    fcntl.flock(fd, fcntl.LOCK_EX)
    assert front.dev_run_active(tmp_path) is True
    fcntl.flock(fd, fcntl.LOCK_UN)
    fd.close()
    assert front.dev_run_active(tmp_path) is False          # and it did not keep the lock
    jobs = tmp_path / "run" / "jobs"
    jobs.mkdir(parents=True)
    (jobs / "a.status").write_text("done 0 2026-10-14T10:00:00\n")
    assert front.dev_run_active(tmp_path) is False
    (jobs / "b.status").write_text("running - 2026-10-14T10:00:00\n")
    assert front.dev_run_active(tmp_path) is True


def test_front_never_starts_the_engine_while_a_guard_job_is_active(monkeypatch, tmp_path):
    from slotstream_spark.memory import signals
    started = []
    monkeypatch.setattr(front, "_start_engine", lambda: started.append(1))
    monkeypatch.setattr(front, "_engine_ready", lambda: False)
    monkeypatch.setattr(front, "_engine_active", lambda: False)
    monkeypatch.setattr(front, "_now_min", lambda: 720)
    monkeypatch.setattr(front.modes, "dynamic_cap_mib", lambda m, avail: 60000)
    monkeypatch.setattr(front.modes, "meminfo_mib", lambda name: 90000)
    monkeypatch.setattr(front.G, "floor_mib", lambda g, cfg: 20000)
    monkeypatch.setattr(signals, "fetch_running", lambda url, timeout: ())
    monkeypatch.setenv("SLOTSTREAM_R", str(tmp_path))
    (tmp_path / "run" / "jobs").mkdir(parents=True)
    (tmp_path / "run" / "jobs" / "f7-x.status").write_text("running - 2026-10-14T10:00:00\n")
    fr = front.Front(engine_addr=("127.0.0.1", 1), wait_s=0)

    async def scenario():
        srv, addr = await front_server(fr)
        r, w = await asyncio.open_connection(*addr)
        w.write(b"POST /v1/messages HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n")
        await w.drain()
        data = await r.read()
        srv.close()
        return data

    raw = asyncio.run(scenario())
    assert raw.startswith(b"HTTP/1.1 503") and b"development run in progress" in raw and started == []
    (tmp_path / "run" / "jobs" / "f7-x.status").write_text("done 0 2026-10-14T10:05:00\n")
    assert fr.decide()[0] == front.START


# ---- front wait ---------------------------------------------------------------------------------------------
def test_front_wait_holds_until_engine_ready(monkeypatch):
    """The engine comes up 5 s after the request arrives; the held request returns the engine's 200, not a 503."""
    monkeypatch.setattr(front, "_start_engine", lambda: None)

    async def scenario():
        async def engine(r, w):
            head = await r.readuntil(b"\r\n\r\n")
            assert head.startswith(b"POST /v1/chat/completions")
            w.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            await w.drain()
            w.close()

        up, up_addr = await upstream(engine)
        t0 = time.monotonic()
        fr = front.Front(engine_addr=up_addr, dev_run_fn=lambda: False, wait_s=30,
                         ready_fn=lambda: time.monotonic() - t0 >= 5)
        fr.decide = lambda: (front.START, "ok")
        srv, addr = await front_server(fr)
        r, w = await asyncio.open_connection(*addr)
        w.write(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 0\r\n\r\n")
        await w.drain()
        data = await r.read()
        srv.close(), up.close()
        return data, time.monotonic() - t0

    data, took = asyncio.run(scenario())
    assert data.startswith(b"HTTP/1.1 200") and data.endswith(b"ok")
    assert 5 <= took < 10


def test_front_wait_times_out_with_503(monkeypatch):
    monkeypatch.setattr(front, "WAIT_POLL_S", 0.05)
    fr = front.Front(engine_addr=("127.0.0.1", 1), dev_run_fn=lambda: False, wait_s=0.4, ready_fn=lambda: False)
    fr.decide = lambda: (front.LOADING, "engine loading")
    t0 = time.monotonic()
    head, body = ask(fr, "/v1/chat/completions")
    assert head.startswith("HTTP/1.1 503") and "Retry-After: 10" in head
    assert 0.35 <= time.monotonic() - t0 < 3


def test_front_wait_refusal_is_immediate():
    """`/running` non-empty (policy REFUSE): no hold, even with a long wait configured."""
    fr = front.Front(engine_addr=("127.0.0.1", 1), dev_run_fn=lambda: False, wait_s=240, ready_fn=lambda: False)
    fr.decide = lambda: (front.REFUSE, "running not empty")
    t0 = time.monotonic()
    head, body = ask(fr, "/v1/chat/completions")
    assert head.startswith("HTTP/1.1 503") and "Retry-After" not in head
    assert time.monotonic() - t0 < 1


# ---- probes -------------------------------------------------------------------------------------------------
def probe(fr, method="GET", path="/v1/models"):
    async def scenario():
        srv, addr = await front_server(fr)
        r, w = await asyncio.open_connection(*addr)
        w.write(f"{method} {path} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
        await w.drain()
        data = await asyncio.wait_for(r.read(), 5)
        srv.close()
        return data

    head, body = asyncio.run(scenario()).split(b"\r\n\r\n", 1)
    return head.decode(), json.loads(body)


def test_a_get_never_starts_the_engine_and_is_not_held(monkeypatch):
    started = []
    monkeypatch.setattr(front, "_start_engine", lambda: started.append(1))
    fr = front.Front(engine_addr=("127.0.0.1", 1), dev_run_fn=lambda: False, wait_s=240, ready_fn=lambda: False)
    fr.decide = lambda: (front.START, "ok")
    t0 = time.monotonic()
    head, body = probe(fr)
    assert head.startswith("HTTP/1.1 503") and "Retry-After" not in head
    assert body["error"]["code"] == "engine_stopped"
    assert started == [] and time.monotonic() - t0 < 1


def test_a_get_while_loading_answers_loading_at_once():
    fr = front.Front(engine_addr=("127.0.0.1", 1), dev_run_fn=lambda: False, wait_s=240, ready_fn=lambda: False)
    fr.decide = lambda: (front.LOADING, "engine loading")
    t0 = time.monotonic()
    head, body = probe(fr)
    assert "Retry-After: 10" in head and body["error"]["code"] == "loading"
    assert time.monotonic() - t0 < 1


def test_a_get_the_rule_refuses_says_not_allowed_and_is_not_logged(capsys):
    fr = front.Front(engine_addr=("127.0.0.1", 1), dev_run_fn=lambda: False, wait_s=240, ready_fn=lambda: False)
    fr.decide = lambda: (front.REFUSE, "running not empty")
    head, body = probe(fr)
    assert head.startswith("HTTP/1.1 503") and body["error"]["code"] == "not_allowed"
    assert "refused" not in capsys.readouterr().err
    ask(fr, "/v1/chat/completions")                                  # a real request is still logged
    assert "refused: running not empty" in capsys.readouterr().err


def test_a_post_still_starts_the_engine_and_waits(monkeypatch):
    started = []
    monkeypatch.setattr(front, "_start_engine", lambda: started.append(1))
    fr = front.Front(engine_addr=("127.0.0.1", 1), dev_run_fn=lambda: False, wait_s=0.3, ready_fn=lambda: False)
    monkeypatch.setattr(front, "WAIT_POLL_S", 0.05)
    fr.decide = lambda: (front.START, "ok")
    head, body = ask(fr, "/v1/chat/completions")
    assert started == [1] and body["error"]["code"] == "loading"
