# SPDX-License-Identifier: MIT
# Port of PressureBoundary.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""Cancellation of an in-flight generation on memory pressure (port of PressureBoundary.swift:5-30).

A ticket is an object: identity, not a counter, so a late acknowledgement cannot clear a newer request. The
generation loop checks `boundary.check()` between steps and raises `Cancelled`; `GenerationGate` is the lock a
generation holds while it owns the stream, and a resize takes it exclusively.
"""
import threading
import time


class Cancelled(Exception):
    pass


class PressureTicket:
    def __init__(self, reason=""):
        self.requested_at = time.monotonic()
        self.reason = reason


class PressureBoundary:
    def __init__(self):
        self._lock = threading.Lock()
        self._pending = None

    def request(self, reason=""):
        t = PressureTicket(reason)
        with self._lock:
            self._pending = t
        return t

    def snapshot(self):
        with self._lock:
            return self._pending

    def acknowledge(self, ticket):
        with self._lock:
            if self._pending is ticket:
                self._pending = None

    def check(self):
        """Raise Cancelled when a ticket is pending; called by the generation between steps."""
        t = self.snapshot()
        if t is not None:
            raise Cancelled(t.reason or "memory pressure")


class GenerationGate:
    """Held by a generation for its whole run (stream owned); a resize or shrink takes it exclusively."""

    def __init__(self):
        self._lock = threading.Lock()

    def acquire(self, timeout=None):
        return self._lock.acquire(timeout=-1 if timeout is None else timeout)

    def release(self):
        self._lock.release()

    def held(self):
        return self._lock.locked()

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, *exc):
        self._lock.release()
