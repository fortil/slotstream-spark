"""Images are refused in all four dialects before tokenizing and without touching the gate."""
import asyncio

import pytest
from fakes import FakeEngine, call

from slotstream_spark.server.app import ServerConfig, create_app
from slotstream_spark.server.common import VISION_OFF

B64 = "iVBORw0KGgo="

CASES = {
    "chat_image_url": ("/v1/chat/completions", {"model": "slotstream-test", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "what"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64," + B64}}]}]},
        "invalid_image"),
    "anthropic_image": ("/v1/messages", {"model": "slotstream-test", "max_tokens": 10, "messages": [
        {"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                                    "data": B64}}]}]}, None),
    "responses_input_image": ("/v1/responses", {"model": "slotstream-test", "input": [
        {"type": "message", "role": "user", "content": [{"type": "input_image",
                                                          "image_url": "data:image/png;base64," + B64}]}]},
        "invalid_image"),
    "ollama_chat_images": ("/api/chat", {"model": "slotstream-test", "messages": [
        {"role": "user", "content": "x", "images": [B64]}]}, None),
    "ollama_generate_images": ("/api/generate", {"model": "slotstream-test", "prompt": "x", "images": [B64]}, None),
}


@pytest.mark.parametrize("name", list(CASES))
def test_image_is_400_and_never_reaches_engine_or_gate(name):
    path, body, code = CASES[name]
    engine = FakeEngine()
    app = create_app(engine, ServerConfig(queue_max=1))
    r = asyncio.run(call(app, "POST", path, body))
    assert r.status == 400, r.text
    assert VISION_OFF in r.text
    if code:
        assert code in r.text
    assert engine.generate_calls == []
    assert engine.render_calls == 0  # refused before tokenizing
    ctx = app.state.ctx
    assert not ctx.gate.busy and ctx.gate.queued == 0
    assert ctx.metrics.get("images_rejected_total") == 1
    assert "slotstream_images_rejected_total 1" in asyncio.run(call(app, "GET", "/metrics")).text


def test_image_is_refused_even_while_the_gate_is_busy():
    async def go():
        engine = FakeEngine(block_after_admit=True)
        app = create_app(engine, ServerConfig(queue_max=0))
        first = asyncio.create_task(call(app, "POST", "/v1/chat/completions",
                                         {"model": "slotstream-test", "messages": [{"role": "user", "content": "hi"}]}))
        await asyncio.get_running_loop().run_in_executor(None, engine.started.wait, 5)
        path, body, _ = CASES["chat_image_url"]
        r = await call(app, "POST", path, body)
        assert r.status == 400  # not 503: the gate was never consulted
        engine.release()
        assert (await first).status == 200
    asyncio.run(go())
