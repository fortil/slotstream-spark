"""Fixtures of the F4 drills (step 66): a /running stub, the 20 ms series, the engine process, the drill records.

Heavy part: CAP is the guard's capture dir of label `drills`; light part: the capture.sh dir of `drills-light`.
Every drill writes `$CAP/h4-<name>.json`; `tools/gates/records.py h4 <dir>...` aggregates them.
The engine under test is `cli serve` in its own process, with its own sampler (20 ms, `footprint-serve.jsonl`) and
Arbiter wired by cli. This conftest also builds the test process's own `slotio.Sampler` (the MemAvailable series at
20 ms, no allocation registered) and a `slotio.Arbiter` with the throttle point of the device, as the plan asks.
Missing inputs FAIL a test, never skip it.
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, ROOT / "tools" / "drills"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

STUB_PORT = 18080
STUB_URL = f"http://127.0.0.1:{STUB_PORT}/running"
ENGINE_URL = "http://127.0.0.1:18435"
FLOOR_CHUNKS = 2
GIB = 1 << 30


def cap_dir():
    v = os.environ.get("CAP")
    if not v:
        pytest.fail("CAP is not set: run under deploy/guard.sh (heavy) or tools/capture.sh (light)", pytrace=False)
    p = Path(v)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_record(name, data):
    """The drill's verdict inputs: `$CAP/h4-<name>.json`."""
    p = cap_dir() / f"h4-{name}.json"
    p.write_text(json.dumps(data, indent=1, sort_keys=True, default=str) + "\n")
    return p


def oom_lines(since):
    """Kernel log lines with an oom kill or NV_ERR_NO_MEMORY since `since` (a journalctl time string)."""
    r = subprocess.run(["journalctl", "-k", "--since", since, "--no-pager", "-q"], capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail(f"journalctl -k failed (rc {r.returncode}): {r.stderr.strip()}", pytrace=False)
    return [ln for ln in r.stdout.splitlines() if "oom-kill" in ln.lower() or "out of memory" in ln.lower()
            or "NV_ERR_NO_MEMORY" in ln]


def stamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Stub:
    """tools/drills/running_stub.py on 127.0.0.1:18080; `set` writes the state file it serves."""

    def __init__(self, workdir):
        self.state = Path(workdir) / "stub-state.json"
        self.log = Path(workdir) / "stub-flips.jsonl"
        self.state.write_text("")
        self.proc = subprocess.Popen([sys.executable, str(ROOT / "tools" / "drills" / "running_stub.py"),
                                      "--port", str(STUB_PORT), "--state-file", str(self.state), "--log", str(self.log)],
                                     stderr=subprocess.DEVNULL)
        for _ in range(50):
            try:
                urllib.request.urlopen(STUB_URL, timeout=1).read()
                return
            except OSError:
                time.sleep(0.1)
        self.close()
        pytest.fail("running stub did not answer", pytrace=False)

    def set(self, *models):
        """models: (name, state) pairs; no argument = empty. Returns the monotonic time of the write."""
        body = json.dumps({"running": [{"model": n, "state": s} for n, s in models]}) if models else ""
        tmp = self.state.with_suffix(".tmp")
        tmp.write_text(body)
        t = time.monotonic()
        os.replace(tmp, self.state)
        return t

    def close(self):
        self.proc.terminate()
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class Series:
    """The test process's `slotio.Sampler` (20 ms) drained to a list of (t_ns, MemAvailable bytes)."""

    def __init__(self, dest):
        import slotio
        self.sampler = slotio.Sampler(period_ms=20, use_nvml=False)
        self.rows, self._lock = [], threading.Lock()
        self._out = open(dest, "w")
        self._stop = threading.Event()
        self.sampler.start()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        time.sleep(0.2)
        self._drain()
        if not self.rows or abs(self.rows[-1][0] - time.monotonic_ns()) > 2e9:
            self.close()
            pytest.fail("Sampler t_ns is not CLOCK_MONOTONIC: cannot align the series with the engine's", pytrace=False)

    def _drain(self):
        d = self.sampler.drain()
        with self._lock:
            for r in d:
                self.rows.append((r["t_ns"], r["avail"]))
                self._out.write(json.dumps({"t_ns": r["t_ns"], "avail": r["avail"]}) + "\n")
            self._out.flush()

    def _run(self):
        while not self._stop.wait(0.5):
            self._drain()

    def snapshot(self):
        self._drain()
        with self._lock:
            return list(self.rows)

    def close(self):
        self._stop.set()
        self._th.join(2)
        self.sampler.stop()
        self._drain()
        self._out.close()


def arbiter_throttle():
    """slotio.Arbiter built the way cli serve builds it (throttle from the device, never 80.0); -> throttle_c."""
    import slotio
    from slotstream_spark.host import eligibility
    lim, src = eligibility.nvme_limit_mc()
    if lim is None:
        pytest.fail("nvme hwmon limit unreadable: the Arbiter has no throttle point", pytrace=False)
    reader = slotio.Reader()
    arb = slotio.Arbiter(reader, device=os.environ.get("SLOTSTREAM_NVME_DEVICE") or "nvme0n1", hwmon=eligibility.hwmon_input(), throttle_c=lim / 1000.0)
    caps = arb.caps()
    del arb, reader
    return lim / 1000.0, src, caps


class Engine:
    """`cli serve` in a child process, in its own capture subdir (governor.jsonl, footprint-serve.jsonl, serve.log)."""

    def __init__(self, label, mode, stub_url=STUB_URL, env=None):
        self.label, self.mode = label, mode
        self.dir = cap_dir() / label
        self.dir.mkdir(parents=True, exist_ok=True)
        if subprocess.run(["systemctl", "--user", "is-active", "--quiet", "slotstream-spark.service"]).returncode == 0:
            pytest.fail("slotstream-spark.service is active: the drill needs the engine port", pytrace=False)
        e = dict(os.environ)
        e.update({"SLOTSTREAM_MODE": mode, "SLOTSTREAM_RUNNING_URL": stub_url, "CAP": str(self.dir)})
        e.pop("SLOTSTREAM_TAKE_LOCK", None)                       # the guard holds the model lock for this run
        e.update(env or {})
        self.log_path = self.dir / "serve.log"
        self.log = open(self.log_path, "w")
        self.start_ts = stamp()
        self.t_start = time.monotonic()
        self.proc = subprocess.Popen([sys.executable, "-m", "slotstream_spark.cli", "serve"], env=e, cwd=str(ROOT),
                                     stdout=self.log, stderr=subprocess.STDOUT)
        self.t_exit = None

    def get(self, path, timeout=2):
        with urllib.request.urlopen(ENGINE_URL + path, timeout=timeout) as r:
            return json.loads(r.read())

    def wait_ready(self, timeout_s=900):
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            if self.proc.poll() is not None:
                pytest.fail(f"{self.label}: engine exited {self.proc.returncode} while loading; "
                            f"tail: {self.log_path.read_text()[-600:]}", pytrace=False)
            try:
                if self.get("/health")["ok"]:
                    return time.monotonic() - self.t_start
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(1.0)
        self.kill()
        pytest.fail(f"{self.label}: engine not ready after {timeout_s} s", pytrace=False)

    def wait_exit(self, timeout_s):
        """-> monotonic time of the exit, or None when still alive after timeout_s (50 ms resolution)."""
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            if self.proc.poll() is not None:
                self.t_exit = time.monotonic()
                return self.t_exit
            time.sleep(0.05)
        return None

    def governor(self):
        p = self.dir / "governor.jsonl"
        if not p.exists():
            return []
        return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]

    def footprint(self):
        """-> (list of raw sample dicts, flags). Raw dicts keep `t_ns` and `avail`."""
        p = self.dir / "footprint-serve.jsonl"
        rows, flags = [], None
        for ln in p.read_text().splitlines():
            if not ln.strip():
                continue
            d = json.loads(ln)
            if d.get("header"):
                flags = d["flags"]
            else:
                rows.append(d)
        return rows, flags

    def kill(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(5)
        self.log.close()


@pytest.fixture
def stub(tmp_path):
    s = Stub(cap_dir())
    yield s
    s.close()


@pytest.fixture
def series():
    s = Series(cap_dir() / f"avail-{os.getpid()}-{time.time_ns()}.jsonl")
    yield s
    s.close()


@pytest.fixture(scope="session", autouse=True)
def _throttle_record():
    if not os.environ.get("CAP"):
        pytest.fail("CAP is not set", pytrace=False)
    t, src, caps = arbiter_throttle()
    write_record(f"arbiter-{os.getpid()}", {"throttle_c": t, "source": src, "caps_at_construction": list(caps)})


@pytest.fixture
def engine_factory():
    made = []

    def make(label, mode, **kw):
        e = Engine(label, mode, **kw)
        made.append(e)
        return e
    yield make
    for e in made:
        e.kill()
