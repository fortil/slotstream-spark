# SPDX-License-Identifier: MIT
# Port of Server.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""HTTP limits as an ASGI layer (port of the checks in Server.swift): body size, chunked bodies, CORS, bind host.

Everything here answers before a handler runs, so a hostile request never reaches the engine or the gate.
"""
import asyncio
import json
import time
from urllib.parse import urlsplit

from slotstream_spark.server.dialects import anthropic

LOOPBACK_BIND = "127.0.0.1"
LOOPBACK_NAMES = frozenset({"127.0.0.1", "localhost", "::1"})
MAX_BODY_BYTES = 32 << 20
MAX_DRAIN_BYTES = 256 << 20
DRAIN_CHUNK_TIMEOUT_S = 2.0
DRAIN_TOTAL_S = 10.0
LIMIT_CONCURRENCY = 32
BACKLOG = 16
H11_MAX_INCOMPLETE_EVENT_SIZE = 65536


def bind_host(host=None):
    """serve() listens on loopback only: no argument means 127.0.0.1, any other address is refused."""
    if host is None or host in LOOPBACK_NAMES:
        return LOOPBACK_BIND
    raise ValueError(f"refusing to bind {host!r}: this server listens on 127.0.0.1 only")


def uvicorn_options():
    return {"host": LOOPBACK_BIND, "http": "h11", "limit_concurrency": LIMIT_CONCURRENCY, "backlog": BACKLOG,
            "h11_max_incomplete_event_size": H11_MAX_INCOMPLETE_EVENT_SIZE}


def is_loopback_origin(origin):
    try:
        host = urlsplit(origin).hostname
    except ValueError:
        return False
    return host in LOOPBACK_NAMES


def error_payload(path, status, message, code):
    """The refusal in the shape of the route's dialect, so a client's SDK can parse it."""
    if path.startswith("/v1/messages"):
        return anthropic.error_body(anthropic.error_type(status), message)
    if path.startswith("/api/"):
        return {"error": message}
    return {"error": {"message": message, "type": "invalid_request_error", "code": code}}


async def _json_response(send, status, payload, extra=()):
    body = json.dumps(payload).encode()
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), *extra]
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


async def drain(receive, *, max_bytes=MAX_DRAIN_BYTES, chunk_timeout=DRAIN_CHUNK_TIMEOUT_S, total=DRAIN_TOTAL_S):
    """Read and discard a refused body so the client can read the 413: bounded in bytes, per chunk and overall."""
    deadline = time.monotonic() + total
    seen = 0
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            return
        try:
            msg = await asyncio.wait_for(receive(), min(chunk_timeout, left))
        except asyncio.TimeoutError:
            return
        if msg["type"] != "http.request":
            return
        seen += len(msg.get("body", b""))
        if seen > max_bytes or not msg.get("more_body", False):
            return


class HttpLimits:
    def __init__(self, app, max_body=MAX_BODY_BYTES, **drain_options):
        self.app, self.max_body, self.drain_options = app, max_body, drain_options

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        path = scope["path"]
        origin = headers.get("origin")
        cors = []
        if origin is not None:
            if not is_loopback_origin(origin):
                return await _json_response(send, 403, error_payload(path, 403, "origin not allowed", "forbidden"))
            cors = [(b"access-control-allow-origin", origin.encode("latin-1")), (b"vary", b"Origin")]
        if scope["method"] == "OPTIONS":
            allow = [(b"allow", b"GET, POST, DELETE, HEAD, OPTIONS"),
                     (b"access-control-allow-methods", b"GET, POST, DELETE, HEAD, OPTIONS"),
                     (b"access-control-allow-headers",
                      headers.get("access-control-request-headers", "content-type").encode("latin-1"))]
            await send({"type": "http.response.start", "status": 204, "headers": cors + allow})
            return await send({"type": "http.response.body", "body": b""})
        if "chunked" in headers.get("transfer-encoding", "").lower():
            return await _json_response(send, 411, error_payload(
                path, 411, "chunked request bodies are not accepted; send Content-Length", "length_required"), cors)
        if "content-length" in headers:
            try:
                length = int(headers["content-length"])
                if length < 0:
                    raise ValueError
            except ValueError:
                return await _json_response(send, 400, error_payload(
                    path, 400, "Content-Length is not a valid size", "invalid_request_error"), cors)
            if length > self.max_body:
                await drain(receive, **self.drain_options)
                return await _json_response(send, 413, error_payload(
                    path, 413, f"request body is larger than {self.max_body >> 20} MiB", "request_too_large"), cors)

        async def send_cors(message):
            if cors and message["type"] == "http.response.start":
                message = dict(message, headers=list(message.get("headers", [])) + cors)
            await send(message)

        await self.app(scope, receive, send_cors)
