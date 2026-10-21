"""Front on 127.0.0.1:18434 (socket activation): starts the engine on demand and proxies to 127.0.0.1:18435.

Policy (plan rule 1, `governor.may_start`): a request starts the engine only inside 04:00-23:45 (no window while `$R/state/PAUSED` has NO_WINDOWS=1), with /running
readable and empty and a cap above the floor, and never while a development run owns the model (`$R/.model.lock`
held, or any `$R/run/jobs/*.status` saying `running`). While the engine loads it answers 503 + Retry-After: 10 in the
shape of the dialect the request spoke; when the rule refuses it answers 503 and starts nothing. Bytes are relayed as
they arrive (no buffering, TCP_NODELAY), so SSE pings reach the client at the engine's own pace. Exits after 5
minutes with no engine and no connections.

Optional extras, all off unless configured:
- `SLOTSTREAM_EXTRA_BIND`: a second listening address (a socket-unit drop-in adds it; run by hand the front binds it
  itself). Meant for containers on the same host. The engine port never leaves loopback.
- `SLOTSTREAM_API_KEY`: when set, a request that arrives on a non-loopback socket must carry `Authorization: Bearer
  <key>`. Loopback stays open.
- `SLOTSTREAM_FRONT_WAIT_S` (default 240, 0 = old behaviour): a request that finds the engine loading, or starts it, is
  held open until the engine answers, then relayed; on timeout it gets 503 + Retry-After. Refusals stay immediate.
"""
import asyncio
import fcntl
import functools
import hmac
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from slotstream_spark.config import modes
from slotstream_spark.guard import paused
from slotstream_spark.memory import governor as G
from slotstream_spark.memory import signals

ENGINE = ("127.0.0.1", 18435)
IDLE_EXIT_S = 300
RETRY_AFTER = 10
RUNNING_URL = os.environ.get("SLOTSTREAM_RUNNING_URL", "")      # empty: no model router to watch
FRONT_PORT = 18434
WAIT_POLL_S = 0.5
LOOPBACK = "127.0.0.1"


def front_wait_s():
    """Seconds a request is held while the engine loads (`SLOTSTREAM_FRONT_WAIT_S`, default 240; bad value: default)."""
    try:
        return max(0.0, float(os.environ.get("SLOTSTREAM_FRONT_WAIT_S", "240")))
    except ValueError:
        return 240.0


def api_key():
    return os.environ.get("SLOTSTREAM_API_KEY", "")


def bearer_ok(head, key):
    """True when the request head carries `Authorization: Bearer <key>` (constant-time compare)."""
    for line in head.split(b"\r\n")[1:]:
        name, _, value = line.partition(b":")
        if name.strip().lower() == b"authorization":
            parts = value.strip().split(None, 1)
            return (len(parts) == 2 and parts[0].lower() == b"bearer"
                    and hmac.compare_digest(parts[1].strip(), key.encode()))
    return False

PROXY, LOADING, START, REFUSE = "proxy", "loading", "start", "refuse"
DEV_RUN = "development run in progress"


def policy(engine_ready, engine_active, now_min, running, cap_mib, floor_mib, cfg, dev_run=False, no_windows=False):
    """-> (action, reason). `engine_ready`: engine answers /health 200; `engine_active`: its unit is active or
    activating (loading); `dev_run`: a development run owns the model. Pure."""
    if engine_ready:
        return PROXY, "engine ready"
    if engine_active:
        return LOADING, "engine loading"
    if dev_run:
        return REFUSE, DEV_RUN
    ok, why = G.may_start(now_min, running, cap_mib, floor_mib, cfg, no_windows=no_windows)
    return (START, why) if ok else (REFUSE, why)


def state_root():
    return Path(os.environ.get("SLOTSTREAM_R") or Path.home() / "slotstream-spark")


def dev_run_active(root=None):
    """True while `<root>/.model.lock` is held by someone or any `<root>/run/jobs/*.status` starts with `running`.
    Looking never creates the lock and never keeps it."""
    root = Path(root) if root is not None else state_root()
    lock = root / ".model.lock"
    if lock.exists():
        try:
            fd = os.open(str(lock), os.O_RDONLY)
        except OSError:
            return True
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            return True
        finally:
            os.close(fd)
    for st in sorted((root / "run" / "jobs").glob("*.status")):
        try:
            if st.read_text().split(None, 1)[:1] == ["running"]:
                return True
        except OSError:
            continue
    return False


def _no_windows():
    """Pause mode ($R/state/PAUSED, NO_WINDOWS=1). A malformed file keeps the windows on (the front never starts less safely)."""
    try:
        return paused.no_windows(state_root())
    except paused.PausedError:
        return False


def _now_min():
    t = time.localtime()
    return t.tm_hour * 60 + t.tm_min


def _engine_ready():
    try:
        with socket.create_connection(ENGINE, timeout=0.5) as s:
            s.sendall(b"GET /health HTTP/1.0\r\n\r\n")
            return b" 200 " in s.recv(64)
    except OSError:
        return False


def _engine_active():
    r = subprocess.run(["systemctl", "--user", "is-active", "slotstream-spark.service"],
                       capture_output=True, text=True)
    return r.stdout.strip() in ("active", "activating")


def _start_engine():
    subprocess.run(["systemctl", "--user", "start", "--no-block", "slotstream-spark.service"], check=False)


MESSAGES = {"loading": "the engine is loading; retry shortly", "not_allowed": "the engine may not start now",
            "unauthorized": "missing or wrong bearer key"}


def error_body(path, code, message):
    """The 503 body in the shape of the dialect behind `path` (Anthropic: overloaded_error)."""
    if path.startswith("/v1/messages"):
        return {"type": "error", "error": {"type": "overloaded_error", "message": message}}
    if path.startswith("/v1/responses"):
        return {"error": {"code": code, "message": message}}
    if path.startswith("/api/"):
        return {"error": message}
    return {"error": {"message": message, "type": "server_error", "code": code}}


def request_path(head):
    try:
        return head.split(b"\r\n", 1)[0].split(b" ")[1].decode("latin-1").split("?", 1)[0]
    except IndexError:
        return "/"


def _answer(status, text, retry=False, path="/", message=None):
    body = json.dumps(error_body(path, text, message or MESSAGES.get(text, text))).encode()
    head = f"HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\nConnection: close\r\n"
    if retry:
        head += f"Retry-After: {RETRY_AFTER}\r\n"
    return (head + "\r\n").encode() + body


def _nodelay(writer):
    sock = writer.get_extra_info("socket")
    if sock is not None:
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass


class Front:
    def __init__(self, cfg=None, engine_addr=ENGINE, dev_run_fn=None, wait_s=None, ready_fn=None):
        self.cfg = cfg or G.load_config()
        self.engine_addr = engine_addr
        self.dev_run_fn = dev_run_fn or dev_run_active
        self.wait_s = front_wait_s() if wait_s is None else wait_s
        self.ready_fn = ready_fn
        self.conns = 0
        self.idle_since = time.monotonic()

    def decide(self):
        m = modes.get(os.environ.get("SLOTSTREAM_MODE", "D5"))
        cap = modes.dynamic_cap_mib(m, modes.meminfo_mib("MemAvailable"))
        floor = G.floor_mib(G.GovInputs(m.name, 0, (), 0, 0, 0, 0), self.cfg)
        running = signals.fetch_running(RUNNING_URL, 1.0) if RUNNING_URL else ()
        return policy(_engine_ready(), _engine_active(), _now_min(), running, cap, floor, self.cfg,
                      dev_run=self.dev_run_fn(), no_windows=_no_windows())

    async def _wait_ready(self, seconds):
        """Poll the engine until it answers /health or `seconds` pass."""
        loop = asyncio.get_running_loop()
        ready = self.ready_fn or (lambda: _engine_ready())
        deadline = time.monotonic() + seconds
        while True:
            if await loop.run_in_executor(None, ready):
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(WAIT_POLL_S)

    async def handle(self, reader, writer, extra=False):
        """`extra`: the connection came in on a non-loopback socket (bearer key required when one is set)."""
        self.conns += 1
        try:
            prefix = b""
            key = api_key()
            if extra and key:
                try:
                    prefix = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 5)
                except Exception:
                    prefix = b""
                if not bearer_ok(prefix, key):
                    writer.write(_answer("401 Unauthorized", "unauthorized", path=request_path(prefix)))
                    await writer.drain()
                    return
            action, reason = await asyncio.get_running_loop().run_in_executor(None, self.decide)
            if action == PROXY:
                await self._proxy(reader, writer, prefix)
                return
            if action in (START, LOADING) and self.wait_s > 0:
                if action == START:
                    await asyncio.get_running_loop().run_in_executor(None, _start_engine)
                if await self._wait_ready(self.wait_s):
                    await self._proxy(reader, writer, prefix)
                    return
            head = prefix
            if not head:
                try:
                    head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 2)
                except Exception:
                    pass
            path = request_path(head)
            if action == START:
                if self.wait_s <= 0:
                    await asyncio.get_running_loop().run_in_executor(None, _start_engine)
                writer.write(_answer("503 Service Unavailable", "loading", retry=True, path=path))
            elif action == LOADING:
                writer.write(_answer("503 Service Unavailable", "loading", retry=True, path=path))
            else:
                print(f"refused: {reason}", file=sys.stderr, flush=True)
                writer.write(_answer("503 Service Unavailable", "not_allowed", path=path,
                                     message=DEV_RUN if reason == DEV_RUN else None))
            await writer.drain()
        finally:
            writer.close()
            self.conns -= 1
            self.idle_since = time.monotonic()

    async def _proxy(self, reader, writer, prefix=b""):
        ur, uw = await asyncio.open_connection(*self.engine_addr)
        _nodelay(writer)
        _nodelay(uw)
        if prefix:
            uw.write(prefix)                         # the head already read for the bearer check

        async def pipe(r, w, half_close):
            """Relay whatever arrives, as it arrives: one write and one drain per read, no accumulation."""
            try:
                while True:
                    b = await r.read(16384)
                    if not b:
                        break
                    w.write(b)
                    await w.drain()
            except (ConnectionError, asyncio.CancelledError):
                pass
            finally:
                if half_close and w.can_write_eof():
                    try:
                        w.write_eof()
                    except OSError:
                        pass
                else:
                    w.close()

        up = asyncio.ensure_future(pipe(reader, uw, True))
        down = asyncio.ensure_future(pipe(ur, writer, False))
        await down                                   # the engine's answer ended: the exchange is over
        up.cancel()
        uw.close()

    async def serve(self, socks):
        socks = list(socks) if isinstance(socks, (list, tuple)) else [socks]
        servers = [await asyncio.start_server(functools.partial(self.handle, extra=is_extra(s)), sock=s)
                   for s in socks]
        try:
            while True:
                await asyncio.sleep(5)
                if self.conns == 0 and time.monotonic() - self.idle_since >= IDLE_EXIT_S and not _engine_active():
                    return
        finally:
            for srv in servers:
                srv.close()


def is_extra(sock):
    """True for a listening socket that is not loopback (the optional extra bind)."""
    try:
        return sock.getsockname()[0] not in (LOOPBACK, "::1")
    except OSError:
        return False


def listen_sockets():
    """The activated sockets (fds from 3, LISTEN_PID checked: one per `ListenStream`); when run by hand, the loopback
    listener plus `SLOTSTREAM_EXTRA_BIND` if set. The engine port is never bound here."""
    if os.environ.get("LISTEN_PID") == str(os.getpid()) and int(os.environ.get("LISTEN_FDS", "0")) >= 1:
        return [socket.socket(fileno=3 + i) for i in range(int(os.environ["LISTEN_FDS"]))]
    socks = []
    hosts = [LOOPBACK] + ([os.environ["SLOTSTREAM_EXTRA_BIND"]] if os.environ.get("SLOTSTREAM_EXTRA_BIND") else [])
    for host in hosts:
        s = socket.socket()
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, FRONT_PORT))
        s.listen(16)
        socks.append(s)
    return socks


def main():
    signals.write_oom_score_adj()
    asyncio.run(Front().serve(listen_sockets()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
