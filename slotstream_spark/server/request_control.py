# SPDX-License-Identifier: MIT
# Port of RequestControl.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""Request control (port of RequestControl.swift): the one-at-a-time gate with a bounded queue, the failure codes,
the sticky first failure, and the body reservation."""
import asyncio
import threading
from collections import deque

FIRST_TOKEN_DEADLINE_S = 30 * 60
BODY_RESERVATION_FACTOR = 16

FAILURE_STATUS = {
    "context_length_exceeded": 400,
    "invalid_configuration": 400,
    "prefill_wait_exceeded": 400,
    "insufficient_memory": 503,
    "prefill_deadline_exceeded": 503,
    "queue_full": 503,
    "client_cancelled": 499,
    "inference_error": 500,
}


def status_for(code):
    return FAILURE_STATUS.get(code, 500)


class QueueFull(Exception):
    pass


class GenerationGate:
    """One generation runs; up to `queue_max` more wait in order. `queue_max=0` refuses the second request at once.
    Waits are cancelable: a cancelled waiter leaves the queue without taking the slot."""

    def __init__(self, queue_max):
        self.queue_max = max(0, int(queue_max))
        self.busy = False
        self.waiters = deque()

    @property
    def queued(self):
        return len(self.waiters)

    async def acquire(self, timeout=None):
        """Returns once the caller owns the slot. Raises QueueFull, asyncio.TimeoutError or CancelledError."""
        if not self.busy and not self.waiters:
            self.busy = True
            return
        if len(self.waiters) >= self.queue_max:
            raise QueueFull()
        fut = asyncio.get_running_loop().create_future()
        self.waiters.append(fut)
        try:
            await asyncio.wait_for(fut, timeout)
        except BaseException:
            if fut in self.waiters:
                self.waiters.remove(fut)
            elif fut.done() and not fut.cancelled():
                self.release()  # the slot was handed over just as the wait ended: pass it on
            raise

    def release(self):
        while self.waiters:
            fut = self.waiters.popleft()
            if not fut.done():
                fut.set_result(None)  # the slot stays busy and passes to the waiter
                return
        self.busy = False


class BodyBudget:
    """Memory reserved for request bodies: each body costs `factor` times its size while it is being handled."""

    def __init__(self, limit_bytes, factor=BODY_RESERVATION_FACTOR):
        self.limit, self.factor, self.used = limit_bytes, factor, 0
        self._lock = threading.Lock()

    def reserve(self, nbytes):
        cost = nbytes * self.factor
        with self._lock:
            if self.used + cost > self.limit:
                return 0
            self.used += cost
            return cost

    def release(self, cost):
        with self._lock:
            self.used = max(0, self.used - cost)


class RequestControl:
    """Per-request state shared by the event loop and the engine thread. The first failure sticks; later ones are
    ignored, so the client sees the cause and not its consequences."""

    def __init__(self):
        self._lock = threading.Lock()
        self.failure = None  # (code, message)
        self.cancelled = False
        self.overflowed = False
        self.reply_stop = lambda: False
        self.first_token = False

    def fail(self, code, message=""):
        with self._lock:
            if self.failure is None:
                self.failure = (code, message or code)
                return True
            return False

    def cancel(self):
        self.cancelled = True
        self.fail("client_cancelled", "the client went away")

    def should_continue(self):
        return self.failure is None and not self.overflowed and not self.reply_stop()
