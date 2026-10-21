"""HTTP surface of the server on the fake engine: routes, limits, shapes, seeds, errors, bind."""
import asyncio
import json
import signal

import pytest
from fakes import (ROUTES, FakeEngine, anthropic_body, call, chat_body, ollama_chat_body, responses_body)

from slotstream_spark.server import app as app_mod
from slotstream_spark.server import common, http_limits
from slotstream_spark.server.app import ServerConfig, create_app
from slotstream_spark.server.request_control import FAILURE_STATUS, RequestControl


def run(coro):
    return asyncio.run(coro)


def make(engine=None, **cfg):
    engine = engine or FakeEngine()
    return engine, create_app(engine, ServerConfig(queue_max=cfg.pop("queue_max", 4), **cfg))


# ---- metadata -------------------------------------------------------------------------------------------

@pytest.mark.parametrize("window", [2, 100, 1000, 4096, 262144])
def test_models_max_output_formula(window):
    engine, app = make(FakeEngine(context_window=window))
    body = run(call(app, "GET", "/v1/models")).json()
    expected = max(1, min(min(8192, max(256, window // 4)), window - 1))
    assert body["data"][0]["max_output_tokens"] == expected == common.output_budget(window)
    assert body["data"][0]["id"] == "slotstream-test"
    assert body["data"][0]["context_window"] == window


def test_ollama_metadata_and_ps_size_is_footprint():
    engine, app = make(FakeEngine(footprint=987654321))
    assert run(call(app, "GET", "/api/version")).json() == {"version": "0.1.0"}
    assert run(call(app, "GET", "/api/tags")).json()["models"][0]["name"] == "slotstream-test"
    ps = run(call(app, "GET", "/api/ps")).json()["models"][0]
    assert ps["size"] == 987654321
    show = run(call(app, "POST", "/api/show", {"model": "slotstream-test"}))
    assert show.status == 200 and show.json()["capabilities"] == ["completion"]
    assert run(call(app, "POST", "/api/show", {"model": "other"})).status == 404


def test_gateway_and_unknown_routes_are_404():
    engine, app = make()
    for path in ("/gateway/status", "/v1/gateway/models", "/nope", "/v1/embeddings"):
        assert run(call(app, "GET", path)).status == 404
    assert run(call(app, "POST", "/v1/embeddings", {})).status == 404
    assert engine.generate_calls == []


def test_responses_stored_state_is_404():
    engine, app = make()
    for method in ("GET", "DELETE"):
        r = run(call(app, method, "/v1/responses/resp_abc"))
        assert r.status == 404 and r.json()["error"]["code"] == "stored_state_unsupported"


def test_head_and_options():
    engine, app = make()
    r = run(call(app, "HEAD", "/api/tags"))
    assert r.status == 200 and r.body == b""
    r = run(call(app, "OPTIONS", "/v1/messages", headers=[("origin", "http://localhost:3000")]))
    assert r.status == 204 and r.header("access-control-allow-origin") == "http://localhost:3000"


def test_metrics_text():
    engine, app = make()
    run(call(app, "POST", "/v1/chat/completions", chat_body()))
    text = run(call(app, "GET", "/metrics")).text
    assert 'slotstream_requests_total{route="/v1/chat/completions",status="200"} 1' in text
    assert "slotstream_footprint_bytes 123456" in text
    assert "slotstream_pool_hits_total 7" in text
    assert 'slotstream_governor_mode{mode="normal"} 1' in text
    assert "slotstream_decode_tokens_per_second" in text
    assert "slotstream_reused_tokens_total" in text
    assert "slotstream_queue_wait_seconds_sum" in text and "slotstream_prefill_seconds_sum" in text


def test_count_tokens_without_generating():
    engine, app = make()
    r = run(call(app, "POST", "/v1/messages/count_tokens",
                 {"model": "slotstream-test", "messages": [{"role": "user", "content": "one two three"}]}))
    assert r.status == 200 and r.json()["input_tokens"] >= 3
    assert engine.generate_calls == []
    bad = run(call(app, "POST", "/v1/messages/count_tokens", {"model": "other", "messages": []}))
    assert bad.status == 404 and bad.json()["type"] == "error"


# ---- limits -----------------------------------------------------------------------------------------------

def test_body_over_limit_is_413_after_drain_in_dialect_shape():
    engine, app = make(max_body=1024)
    big = ("content-length", str(2048))
    r = run(call(app, "POST", "/v1/messages", raw=b"x" * 2048, headers=[big]))
    assert r.status == 413 and r.json()["error"]["type"] == "request_too_large"
    r = run(call(app, "POST", "/api/chat", raw=b"x" * 2048, headers=[big]))
    assert r.status == 413 and "error" in r.json() and isinstance(r.json()["error"], str)
    r = run(call(app, "POST", "/v1/chat/completions", raw=b"x" * 2048, headers=[big]))
    assert r.status == 413 and r.json()["error"]["code"] == "request_too_large"
    assert engine.generate_calls == []


def test_real_limit_is_32_mib():
    assert http_limits.MAX_BODY_BYTES == 32 * 1024 * 1024
    engine, app = make()
    r = run(call(app, "POST", "/v1/chat/completions", raw=b"{}", headers=[("content-length", str(33 << 20))]))
    assert r.status == 413


def test_drain_is_bounded():
    async def go():
        chunks = [{"type": "http.request", "body": b"x" * 100, "more_body": True}] * 50

        async def receive():
            return chunks.pop() if chunks else {"type": "http.request", "body": b"", "more_body": False}
        await http_limits.drain(receive, max_bytes=1000)
        assert len(chunks) > 30  # stopped after ~1000 bytes, not at the end of the body

        async def never():
            await asyncio.sleep(10)
        t = asyncio.get_running_loop().time()
        await http_limits.drain(never, chunk_timeout=0.05, total=0.2)
        assert asyncio.get_running_loop().time() - t < 1.0
    run(go())


def test_chunked_body_is_411():
    engine, app = make()
    for path in ("/v1/messages", "/v1/chat/completions", "/api/chat", "/v1/responses"):
        r = run(call(app, "POST", path, raw=b"{}", headers=[("transfer-encoding", "chunked")]))
        assert r.status == 411
    r = run(call(app, "POST", "/v1/messages", raw=b"{}", headers=[("transfer-encoding", "chunked")]))
    assert r.json()["type"] == "error"  # read failures on /v1/messages use the Anthropic shape
    assert engine.generate_calls == []


def test_cors_loopback_echoed_other_403():
    engine, app = make()
    for origin in ("http://localhost:5173", "http://127.0.0.1:8080", "https://localhost", "http://[::1]:3000"):
        r = run(call(app, "GET", "/api/tags", headers=[("origin", origin)]))
        assert r.status == 200 and r.header("access-control-allow-origin") == origin
    for origin in ("http://evil.example", "http://localhost.evil.example", "https://203.0.113.5", "null"):
        r = run(call(app, "GET", "/api/tags", headers=[("origin", origin)]))
        assert r.status == 403 and r.header("access-control-allow-origin") is None
    r = run(call(app, "POST", "/v1/messages", anthropic_body(), headers=[("origin", "http://evil.example")]))
    assert r.status == 403 and r.json()["type"] == "error"
    assert engine.generate_calls == []


# ---- generation shapes ------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", list(ROUTES))
def test_bad_json_is_400_in_each_shape(name):
    engine, app = make()
    path, _ = ROUTES[name]
    r = run(call(app, "POST", path, raw=b"{not json", headers=[("content-length", "9")]))
    assert r.status == 400
    assert engine.generate_calls == []


@pytest.mark.parametrize("name", list(ROUTES))
def test_non_object_body_is_400(name):
    engine, app = make()
    path, _ = ROUTES[name]
    assert run(call(app, "POST", path, [1, 2, 3])).status == 400
    assert engine.generate_calls == []


def test_chat_stream_and_not():
    engine, app = make(FakeEngine(reused=2))
    r = run(call(app, "POST", "/v1/chat/completions", chat_body()))
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Hello world"
    assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == 2
    r = run(call(app, "POST", "/v1/chat/completions", chat_body(stream=True, stream_options={"include_usage": True})))
    assert r.status == 200 and r.header("content-type") == "text/event-stream"
    assert r.text.rstrip().endswith("data: [DONE]")
    chunks = [json.loads(f[6:]) for f in r.frames() if f.startswith("data: {")]
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == "Hello world"
    assert chunks[-1]["usage"]["completion_tokens"] == 2


def test_json_object_usage_present():
    engine, app = make(FakeEngine(reused=2))
    r = run(call(app, "POST", "/v1/chat/completions",
                 chat_body(response_format={"type": "json_object"}, reasoning_effort="high")))
    body = r.json()
    assert r.status == 200 and body["choices"][0]["finish_reason"]
    assert body["usage"]["prompt_tokens"] > 0 and body["usage"]["completion_tokens"] == 2
    schema = run(call(app, "POST", "/v1/chat/completions", chat_body(response_format={"type": "json_schema"})))
    assert schema.status == 400


def test_anthropic_stream_and_not():
    engine, app = make(FakeEngine(reused=1))
    body = run(call(app, "POST", "/v1/messages", anthropic_body())).json()
    assert body["content"][0]["text"] == "Hello world" and body["stop_reason"] == "end_turn"
    r = run(call(app, "POST", "/v1/messages", anthropic_body(stream=True, bogus_field=1)))
    kinds = [f.split("\n")[0] for f in r.frames()]
    assert kinds[0] == "event: message_start" and kinds[-1] == "event: message_stop"
    assert r.header("x-slotstream-ignored-fields") == "bogus_field"
    assert any("text_delta" in f for f in r.frames())


def test_responses_stream_and_not():
    engine, app = make()
    body = run(call(app, "POST", "/v1/responses", responses_body())).json()
    assert body["status"] == "completed" and body["output"][0]["content"][0]["text"] == "Hello world"
    r = run(call(app, "POST", "/v1/responses", responses_body(stream=True)))
    kinds = [f.split("\n")[0] for f in r.frames()]
    assert kinds[0] == "event: response.created" and kinds[-1] == "event: response.completed"


def test_ollama_default_streams_and_not():
    engine, app = make()
    r = run(call(app, "POST", "/api/chat", ollama_chat_body()))
    lines = [json.loads(x) for x in r.text.strip().split("\n")]
    assert r.header("content-type") == "application/x-ndjson" and lines[-1]["done"] is True
    assert "".join(x["message"]["content"] for x in lines) == "Hello world"
    r = run(call(app, "POST", "/api/chat", ollama_chat_body(stream=False)))
    assert r.json()["message"]["content"] == "Hello world" and r.json()["done_reason"] == "stop"
    load = run(call(app, "POST", "/api/chat", {"model": "slotstream-test", "messages": []}))
    assert load.json()["done_reason"] == "load"
    assert len(engine.generate_calls) == 2


def test_null_max_tokens_and_options_mean_unset():
    engine, app = make()
    run(call(app, "POST", "/v1/chat/completions", chat_body(max_tokens=None, temperature=None)))
    sample = engine.generate_calls[-1][1].sample
    assert sample.max_tokens == common.output_budget(4096)
    r = run(call(app, "POST", "/api/chat", ollama_chat_body(options=None, stream=False)))
    assert r.status == 200
    r = run(call(app, "POST", "/api/generate", {"model": "slotstream-test", "prompt": "hi", "options": None,
                                                 "stream": False}))
    assert r.status == 200


def test_unseeded_requests_get_different_seeds_and_given_seed_is_kept():
    engine, app = make()
    for name, (path, mk) in ROUTES.items():
        seeds = set()
        for _ in range(3):
            run(call(app, "POST", path, mk(stream=False)))
            seeds.add(engine.generate_calls[-1][1].seed)
        assert len(seeds) == 3, name
        assert all(1 <= s < 2 ** 64 for s in seeds)
    run(call(app, "POST", "/v1/chat/completions", chat_body(seed=42)))
    assert engine.generate_calls[-1][1].seed == 42
    run(call(app, "POST", "/api/chat", ollama_chat_body(options={"seed": 7}, stream=False)))
    assert engine.generate_calls[-1][1].seed == 7
    run(call(app, "POST", "/api/chat", ollama_chat_body(options={"seed": -1}, stream=False)))
    assert engine.generate_calls[-1][1].seed != -1


def test_seed_generator_range():
    seen = {app_mod.fresh_seed() for _ in range(200)}
    assert len(seen) == 200 and all(1 <= s <= 2 ** 64 - 1 for s in seen)


def test_validation_errors_never_reach_the_engine():
    engine, app = make()
    cases = [
        ("/v1/chat/completions", chat_body(temperature="hot")),
        ("/v1/chat/completions", chat_body(max_tokens=-5)),
        ("/v1/chat/completions", chat_body(top_p=float("nan"))),
        ("/v1/messages", anthropic_body(temperature=-1)),
        ("/v1/messages", anthropic_body(max_tokens=True)),
        ("/v1/responses", responses_body(temperature="x")),
        ("/api/chat", ollama_chat_body(options={"temperature": "x"})),
        ("/api/generate", {"model": "slotstream-test", "prompt": "x", "options": {"top_k": 1.5}}),
        ("/v1/chat/completions", {**chat_body(), "model": "other"}),
    ]
    for path, body in cases:
        raw = json.dumps(body).encode()  # NaN serialises as NaN, which json.loads accepts: still refused
        r = run(call(app, "POST", path, raw=raw, headers=[("content-length", str(len(raw)))]))
        assert r.status in (400, 404), (path, body, r.status)
    assert engine.generate_calls == []


def test_prompt_too_long_is_400_before_the_gate():
    engine, app = make(FakeEngine(context_window=8))
    long = {"model": "slotstream-test", "messages": [{"role": "user", "content": "w " * 50}]}
    r = run(call(app, "POST", "/v1/chat/completions", long))
    assert r.status == 400 and r.json()["error"]["code"] == "context_length_exceeded"
    r = run(call(app, "POST", "/v1/messages", anthropic_body(messages=long["messages"])))
    assert r.status == 400 and "prompt is too long" in r.json()["error"]["message"]
    r = run(call(app, "POST", "/api/chat", ollama_chat_body(messages=long["messages"])))
    assert r.status == 400
    assert engine.generate_calls == [] and not app.state.ctx.gate.busy


# ---- engine failures --------------------------------------------------------------------------------------

def test_failure_codes_map_to_statuses():
    assert FAILURE_STATUS == {
        "context_length_exceeded": 400, "invalid_configuration": 400, "prefill_wait_exceeded": 400,
        "insufficient_memory": 503, "prefill_deadline_exceeded": 503, "queue_full": 503, "client_cancelled": 499,
        "inference_error": 500}


@pytest.mark.parametrize("code", ["context_length_exceeded", "invalid_configuration", "prefill_wait_exceeded",
                                  "insufficient_memory", "prefill_deadline_exceeded", "inference_error"])
def test_engine_failure_before_admission(code):
    from slotstream_spark.server.engine_api import EngineFailure

    class Failing(FakeEngine):
        def generate(self, ids, params, **kw):
            raise EngineFailure(code, "nope")
    engine, app = make(Failing())
    for name in ("chat", "anthropic", "responses", "ollama_chat"):
        path, mk = ROUTES[name]
        for stream in (False, True):
            r = run(call(app, "POST", path, mk(stream=stream)))
            assert r.status == FAILURE_STATUS[code], (name, stream, r.status)
            assert "nope" in r.text
    assert not app.state.ctx.gate.busy


def test_unknown_engine_exception_is_500_without_leaking():
    engine, app = make(FakeEngine(fail="boom"))
    for name, (path, mk) in ROUTES.items():
        r = run(call(app, "POST", path, mk(stream=False)))
        assert r.status == 500 and "secret" not in r.text, name
    assert not app.state.ctx.gate.busy


def test_failure_after_headers_stays_in_the_stream():
    for name in ("chat", "anthropic", "responses", "ollama_chat"):
        engine, app = make(FakeEngine(fail="inference_error", fail_after=1))
        path, mk = ROUTES[name]
        r = run(call(app, "POST", path, mk(stream=True)))
        assert r.status == 200, name
        assert "failed mid-stream" in r.text, name
        assert not app.state.ctx.gate.busy


def test_request_control_first_failure_is_sticky():
    c = RequestControl()
    assert c.fail("insufficient_memory", "first") is True
    assert c.fail("inference_error", "second") is False
    assert c.failure == ("insufficient_memory", "first")
    assert c.should_continue() is False


def test_first_token_deadline():
    engine, app = make(FakeEngine(block_after_admit=True), first_token_deadline_s=0.2)
    r = run(call(app, "POST", "/v1/chat/completions", chat_body()))
    assert r.status == 503 and r.json()["error"]["code"] == "prefill_deadline_exceeded"
    assert not app.state.ctx.gate.busy


def test_body_reservation_refuses_when_budget_is_short():
    engine, app = make(body_budget_bytes=100)
    r = run(call(app, "POST", "/v1/chat/completions", chat_body()))
    assert r.status == 503 and r.json()["error"]["code"] == "insufficient_memory"
    assert engine.generate_calls == []
    ok = create_app(FakeEngine(), ServerConfig(queue_max=1, body_budget_bytes=1 << 20))
    assert run(call(ok, "POST", "/v1/chat/completions", chat_body())).status == 200
    assert ok.state.ctx.budget.used == 0  # the reservation is returned


# ---- bind and process --------------------------------------------------------------------------------------

def test_bind_is_loopback_only():
    assert http_limits.bind_host() == "127.0.0.1"
    assert http_limits.bind_host("localhost") == "127.0.0.1"
    for bad in ("0.0.0.0", "::", "203.0.113.5", "198.51.100.2", ""):
        with pytest.raises(ValueError):
            http_limits.bind_host(bad)
    opts = http_limits.uvicorn_options()
    assert opts == {"host": "127.0.0.1", "http": "h11", "limit_concurrency": 32, "backlog": 16,
                    "h11_max_incomplete_event_size": 65536}
    with pytest.raises(ValueError):
        app_mod.serve(FakeEngine(), host="0.0.0.0")


def test_sigpipe_is_ignored():
    app_mod.ignore_sigpipe()
    assert signal.getsignal(signal.SIGPIPE) == signal.SIG_IGN


def test_queue_max_comes_from_platform_toml():
    assert ServerConfig().queue_max == 4


def test_deadline_does_not_fire_once_a_token_has_been_produced():
    engine, app = make(FakeEngine(script=["a", "b", "c"], token_delay=0.15), first_token_deadline_s=0.25)
    r = run(call(app, "POST", "/v1/chat/completions", chat_body()))
    assert r.status == 200 and r.json()["choices"][0]["message"]["content"] == "abc"
