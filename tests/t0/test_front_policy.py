"""Front policy (rule 1) and the unit files: no clock, no network, no systemd."""
import re
from pathlib import Path

from slotstream_spark import front
from slotstream_spark.memory import governor as G

CFG = G.load_config()
DEPLOY = Path(__file__).resolve().parents[2] / "deploy"
NOON = 12 * 60
FLOOR = 20000


def pol(ready=False, active=False, now=NOON, running=(), cap=60000):
    return front.policy(ready, active, now, running, cap, FLOOR, CFG)


def test_ready_engine_is_proxied_whatever_the_clock():
    assert pol(ready=True, now=100)[0] == front.PROXY


def test_loading_engine_gets_503_not_a_second_start():
    assert pol(active=True)[0] == front.LOADING


def test_start_inside_the_window_with_nothing_running():
    assert pol() == (front.START, "ok")


def test_refused_before_04_and_from_23_45():
    assert pol(now=3 * 60 + 59) == (front.REFUSE, "before 04:00")
    assert pol(now=CFG.may_start_end_min)[0] == front.REFUSE
    assert pol(now=CFG.may_start_end_min - 1)[0] == front.START


def test_refused_when_running_is_not_empty_or_unreadable():
    assert pol(running=(G.Resident("large-x", "loading"),)) == (front.REFUSE, "running not empty")
    assert pol(running=None) == (front.REFUSE, "running unreadable")


def test_no_url_starts_when_memory_allows(monkeypatch):
    """Empty SLOTSTREAM_RUNNING_URL: there is no model router to watch, so decide() passes () and never calls the network."""
    monkeypatch.setattr(front, "RUNNING_URL", "")
    monkeypatch.setattr(front.signals, "fetch_running", lambda *a, **k: (_ for _ in ()).throw(AssertionError("fetched")))
    monkeypatch.setattr(front, "_engine_ready", lambda: False)
    monkeypatch.setattr(front, "_engine_active", lambda: False)
    monkeypatch.setattr(front, "_now_min", lambda: NOON)
    monkeypatch.setattr(front, "_no_windows", lambda: False)
    monkeypatch.setattr(front.modes, "dynamic_cap_mib", lambda m, avail: 60000)
    monkeypatch.setattr(front.modes, "meminfo_mib", lambda k, *a: 70000)
    action, _ = front.Front(cfg=CFG, dev_run_fn=lambda: False).decide()
    assert action == front.START


def test_refused_when_the_cap_is_below_the_floor():
    assert pol(cap=FLOOR - 1) == (front.REFUSE, "cap below the floor")


def test_503_answer_carries_retry_after_only_while_loading():
    assert b"Retry-After: 10" in front._answer("503 Service Unavailable", "loading", retry=True)
    assert b"Retry-After" not in front._answer("503 Service Unavailable", "not_allowed")


def test_units():
    names = ["slotstream-spark.socket", "slotstream-spark-front.service", "slotstream-spark.service",
             "slotstream-spark-start.timer", "slotstream-spark-start.service", "slotstream-spark-stop.timer",
             "slotstream-spark-stop.service"]
    text = {n: (DEPLOY / n).read_text() for n in names}
    for n, t in text.items():
        assert not re.search(r"^\s*Restart=", t, re.M), n
        assert "/home/" not in t, n
    sock = text["slotstream-spark.socket"]
    assert "ListenStream=127.0.0.1:18434" in sock and "[Install]" not in sock
    assert "Service=slotstream-spark-front.service" in sock
    front_u, eng = text["slotstream-spark-front.service"], text["slotstream-spark.service"]
    assert "-m slotstream_spark.front" in front_u and "OOMScoreAdjust=1000" in front_u
    assert "-m slotstream_spark.cli serve" in eng and "OOMScoreAdjust=1000" in eng
    assert "SLOTSTREAM_MODE=D5" in eng and "SLOTSTREAM_TAKE_LOCK=1" in eng
    assert "EnvironmentFile=-%h/slotstream-spark/serve.env" in eng       # the context window, shared with the front
    assert "EnvironmentFile=-%h/slotstream-spark/serve.env" in front_u
    assert eng.index("Environment=SLOTSTREAM_MODE=D5") < eng.index("EnvironmentFile=")
    assert "KillMode=control-group" in eng and "TimeoutStopSec=15" in eng
    assert "OnCalendar=*-*-* 04:00:00" in text["slotstream-spark-start.timer"]
    assert "OnCalendar=*-*-* 23:59:00" in text["slotstream-spark-stop.timer"]
    assert "start slotstream-spark.socket" in text["slotstream-spark-start.service"]
    stop = text["slotstream-spark-stop.service"]
    assert all(u in stop for u in ("slotstream-spark.socket", "slotstream-spark-front.service", "slotstream-spark.service"))


def test_crontab_lines_are_tagged_and_keep_the_kill_all():
    lines = (DEPLOY / "crontab.backup").read_text().splitlines()
    assert len(lines) == 2 and all(l.rstrip().endswith("#slotstream-spark") for l in lines)
    assert any(l.startswith("59 23 * * * XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user stop slotstream-spark.socket "
                            "slotstream-spark-front.service slotstream-spark.service") for l in lines)


# ---- extra bind and bearer key -------------------------------------------------------------------------------
class FakeSock:
    def __init__(self, fileno=None, host="127.0.0.1"):
        self.fd, self.host = fileno, host

    def getsockname(self):
        return (self.host, front.FRONT_PORT)


def test_extra_bind_accepts_second_fd(monkeypatch):
    """LISTEN_FDS=2: both activated fds (3 and 4) become listening sockets; only the non-loopback one is `extra`."""
    monkeypatch.setenv("LISTEN_PID", str(front.os.getpid()))
    monkeypatch.setenv("LISTEN_FDS", "2")
    monkeypatch.setattr(front.socket, "socket", lambda fileno=None: FakeSock(fileno, "127.0.0.1" if fileno == 3 else "203.0.113.9"))
    socks = front.listen_sockets()
    assert [s.fd for s in socks] == [3, 4]
    assert [front.is_extra(s) for s in socks] == [False, True]
    monkeypatch.setenv("LISTEN_FDS", "1")
    assert [s.fd for s in front.listen_sockets()] == [3]


def test_extra_bind_never_exposes_engine_port():
    """The engine stays on loopback and no unit or example lists its port."""
    assert front.ENGINE[0] == "127.0.0.1"
    example = (DEPLOY / "slotstream-spark.socket.d" / "extra-bind.conf.example").read_text()
    listens = re.findall(r"^ListenStream=(.+)$", example, re.M)
    assert listens and all(l.endswith(f":{front.FRONT_PORT}") for l in listens)
    assert str(front.ENGINE[1]) not in "".join(listens)
    assert "SLOTSTREAM_EXTRA_BIND" in (Path(front.__file__)).read_text()
    assert re.search(r"\d+\.\d+\.\d+\.\d+", example) is None      # a placeholder, not a host address


def test_bearer_required_on_extra_bind(monkeypatch):
    import asyncio
    monkeypatch.setenv("SLOTSTREAM_API_KEY", "k-test")

    async def scenario(auth):
        async def engine(r, w):
            await r.readuntil(b"\r\n\r\n")
            w.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            await w.drain()
            w.close()

        up = await asyncio.start_server(engine, "127.0.0.1", 0)
        fr = front.Front(engine_addr=up.sockets[0].getsockname()[:2], dev_run_fn=lambda: False, wait_s=0)
        fr.decide = lambda: (front.PROXY, "engine ready")
        srv = await asyncio.start_server(lambda r, w: fr.handle(r, w, extra=True), "127.0.0.1", 0)
        r, w = await asyncio.open_connection(*srv.sockets[0].getsockname()[:2])
        w.write(f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n{auth}Content-Length: 0\r\n\r\n".encode())
        await w.drain()
        data = await r.read()
        srv.close(), up.close()
        return data

    assert asyncio.run(scenario("")).startswith(b"HTTP/1.1 401")
    assert asyncio.run(scenario("Authorization: Bearer nope\r\n")).startswith(b"HTTP/1.1 401")
    assert asyncio.run(scenario("authorization: bearer k-test\r\n")).endswith(b"ok")


def test_bearer_not_required_on_loopback(monkeypatch):
    import asyncio
    monkeypatch.setenv("SLOTSTREAM_API_KEY", "k-test")

    async def scenario():
        async def engine(r, w):
            await r.readuntil(b"\r\n\r\n")
            w.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
            await w.drain()
            w.close()

        up = await asyncio.start_server(engine, "127.0.0.1", 0)
        fr = front.Front(engine_addr=up.sockets[0].getsockname()[:2], dev_run_fn=lambda: False, wait_s=0)
        fr.decide = lambda: (front.PROXY, "engine ready")
        srv = await asyncio.start_server(fr.handle, "127.0.0.1", 0)       # extra=False: the loopback socket
        r, w = await asyncio.open_connection(*srv.sockets[0].getsockname()[:2])
        w.write(b"GET /v1/models HTTP/1.1\r\nHost: x\r\n\r\n")
        await w.drain()
        data = await r.read()
        srv.close(), up.close()
        return data

    assert asyncio.run(scenario()).endswith(b"ok")


def test_decide_floor_follows_the_context_window(monkeypatch):
    seen = []
    monkeypatch.setattr(front, "RUNNING_URL", "")
    monkeypatch.setattr(front, "_engine_ready", lambda: False)
    monkeypatch.setattr(front, "_engine_active", lambda: False)
    monkeypatch.setattr(front, "_now_min", lambda: NOON)
    monkeypatch.setattr(front, "_no_windows", lambda: False)
    monkeypatch.setattr(front.modes, "dynamic_cap_mib", lambda m, avail: 60000)
    monkeypatch.setattr(front.modes, "meminfo_mib", lambda k, *a: 70000)
    real = front.policy
    monkeypatch.setattr(front, "policy", lambda *a, **k: (seen.append(a[5]), real(*a, **k))[1])
    f = front.Front(cfg=CFG, dev_run_fn=lambda: False)
    monkeypatch.delenv("SLOTSTREAM_CONTEXT_WINDOW", raising=False)
    assert f.decide()[0] == front.START
    monkeypatch.setenv("SLOTSTREAM_CONTEXT_WINDOW", "131072")
    assert f.decide()[0] == front.START
    assert seen == [17_166, 26_148]                                     # the floor the engine will enforce
    monkeypatch.setenv("SLOTSTREAM_CONTEXT_WINDOW", "131000")
    assert f.decide() == (front.REFUSE, "context window: context window 131000 is not a multiple of 64")
    monkeypatch.setattr(front, "_engine_ready", lambda: True)
    assert f.decide()[0] == front.PROXY                                 # a running engine is proxied whatever the environment says
