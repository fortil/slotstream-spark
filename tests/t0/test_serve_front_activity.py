"""The idle exit counts generation requests only: status paths a poller calls every second do not keep the engine up."""
import asyncio

from slotstream_spark.cli import ServeFront


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def call(front, path):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    front.app = app
    sent = []

    async def send(msg):
        sent.append(msg)

    asyncio.run(front(dict(type="http", path=path), None, send))
    return sent


def make(clock):
    return ServeFront(None, None, None, None, 1800, clock=clock)


def test_a_model_list_poll_does_not_move_the_idle_clock():
    clock = Clock()
    front = make(clock)
    for _ in range(3):
        clock.t += 1000
        call(front, "/v1/models")
    assert front.last_activity == 0.0
    assert front.idle_expired()


def test_a_generation_request_moves_the_idle_clock():
    clock = Clock()
    front = make(clock)
    clock.t = 1000
    call(front, "/v1/chat/completions")
    assert front.last_activity == 1000
    clock.t = 2000
    assert not front.idle_expired()
    clock.t = 2800
    assert front.idle_expired()
