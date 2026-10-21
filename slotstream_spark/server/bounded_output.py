# SPDX-License-Identifier: MIT
# Port of BoundedOutput.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""Bounded output (port of BoundedOutput.swift): the engine thread only enqueues; if the reader falls behind by
more than the caps, the stream fails and generation is cancelled instead of buffering without limit."""
import asyncio
import threading
from collections import deque

MAX_BYTES = 1 << 20
MAX_FRAMES = 64
WRITE_TIMEOUT_S = 5.0


class BoundedStream:
    def __init__(self, max_bytes=MAX_BYTES, max_frames=MAX_FRAMES, write_timeout=WRITE_TIMEOUT_S):
        self.max_bytes, self.max_frames, self.write_timeout = max_bytes, max_frames, write_timeout
        self._q = deque()
        self._bytes = 0
        self._lock = threading.Lock()
        self._closed = False
        self.failed = False
        self._wake = asyncio.Event()
        self._loop = asyncio.get_running_loop()

    def _notify(self):
        try:
            self._loop.call_soon_threadsafe(self._wake.set)
        except RuntimeError:
            pass  # loop already closed

    def put(self, frame):
        """Thread-safe, never blocks. False once the stream has failed or closed."""
        data = frame.encode() if isinstance(frame, str) else frame
        with self._lock:
            if self.failed or self._closed:
                return False
            if len(self._q) + 1 > self.max_frames or self._bytes + len(data) > self.max_bytes:
                self.failed = True
                self._q.clear()
                self._bytes = 0
            else:
                self._q.append(data)
                self._bytes += len(data)
        self._notify()
        return not self.failed

    def close(self):
        with self._lock:
            self._closed = True
        self._notify()

    def _take(self):
        with self._lock:
            if self._q:
                data = self._q.popleft()
                self._bytes -= len(data)
                return data
            return None

    async def frames(self, idle_timeout=None):
        """Yields frames as bytes; yields None after `idle_timeout` seconds with nothing to send (the keepalive
        moment). Ends when the stream is closed and drained, or has failed."""
        while True:
            data = self._take()
            if data is not None:
                yield data
                continue
            if self.failed or self._closed:
                with self._lock:
                    if self.failed or not self._q:
                        return
                continue
            self._wake.clear()
            if self._q or self._closed or self.failed:
                continue
            try:
                await asyncio.wait_for(self._wake.wait(), idle_timeout)
            except asyncio.TimeoutError:
                yield None

    async def send_with_timeout(self, send, message):
        await asyncio.wait_for(send(message), self.write_timeout)
