"""Minimal engine server on 127.0.0.1:18435 (replaced in F7): /health, /internal/memory, /internal/generate.

One generation at a time (503 when busy), admission before allocation, idle exit after SLOTSTREAM_IDLE_EXIT_S
(default 30 min). The engine object is injected: `chunks`, `plan`, `generate(ids, max_tokens)`.
"""
import asyncio
import os
import time

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from slotstream_spark.engine.pressure import Cancelled
from slotstream_spark.memory import admission

HOST, PORT = "127.0.0.1", 18435


def idle_exit_s():
    return float(os.environ.get("SLOTSTREAM_IDLE_EXIT_S", "1800"))


class State:
    def __init__(self, engine, ledger, controller, cap_fn, clock=time.monotonic, idle_s=None):
        self.engine, self.ledger, self.controller, self.cap_fn, self.clock = engine, ledger, controller, cap_fn, clock
        self.idle_s = idle_exit_s() if idle_s is None else idle_s
        self.last_activity = clock()
        self.ready = False
        self.busy = False

    def idle_expired(self):
        return not self.busy and self.clock() - self.last_activity >= self.idle_s


def make_app(state):
    async def health(request):
        return JSONResponse({"ok": state.ready, "busy": state.busy, "chunks": state.engine.chunks},
                            status_code=200 if state.ready else 503)

    async def memory(request):
        plan = state.engine.plan
        return JSONResponse({"chunks": state.engine.chunks, "plan_chunks": plan.chunks, "window": plan.window,
                             "expected_peak": state.ledger.expected_peak_bytes(state.engine.chunks),
                             "cap_bytes": state.cap_fn(), "terms": state.ledger.terms()})

    async def generate(request):
        if not state.ready:
            return JSONResponse({"error": {"type": "loading"}}, status_code=503, headers={"Retry-After": "10"})
        body = await request.json()
        ids = [int(t) for t in body["ids"]]
        adm = admission.admit(len(ids), state.engine.plan, state.ledger, state.cap_fn())
        if not adm.ok:
            return JSONResponse(adm.body, status_code=adm.status)
        if state.busy:
            return JSONResponse({"error": {"type": "busy"}}, status_code=503, headers={"Retry-After": "1"})
        if not state.controller.gate.acquire(0):
            return JSONResponse({"error": {"type": "busy"}}, status_code=503, headers={"Retry-After": "1"})
        state.busy = True
        try:
            res = await asyncio.get_running_loop().run_in_executor(
                None, lambda: state.engine.generate(ids, int(body.get("max_tokens", 16)), state.controller.boundary))
            return JSONResponse({"tokens": res["tokens"]})
        except Cancelled as e:
            return JSONResponse({"error": {"type": "cancelled", "reason": str(e)}}, status_code=503)
        finally:
            state.busy = False
            state.last_activity = state.clock()
            state.controller.gate.release()

    return Starlette(routes=[Route("/health", health), Route("/internal/memory", memory),
                             Route("/internal/generate", generate, methods=["POST"])])
