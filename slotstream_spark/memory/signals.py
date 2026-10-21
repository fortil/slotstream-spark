"""Signal thread of the engine: reads /running (250 ms, GET only) and meminfo (1 s), asks the governor, acts.

Everything that touches the world is injected (`get_running`, `read_mem`, `read_footprint`, `engine`, `clock`), so
the loop runs against fakes. The footprint handed to the governor is rule 6: max(ledger.expected_peak_bytes(chunks),
footprint.once_bytes(sample, flags)); no path sums VmRSS + NVML + pinned without the per-name subtraction.
"""
import json
import threading
import time
import urllib.request

from slotstream_spark.engine.pressure import GenerationGate, PressureBoundary
from slotstream_spark.guard import paused
from slotstream_spark.memory import footprint as fp
from slotstream_spark.memory import governor as G


def _paused_no_windows():
    """Pause mode (`$R/state/PAUSED` with NO_WINDOWS=1). A malformed file keeps the windows on."""
    try:
        return paused.no_windows()
    except paused.PausedError:
        return False


MIB = 1 << 20


def write_oom_score_adj(path="/proc/self/oom_score_adj", value=1000):
    """Rule 3: every process we start is the first the kernel kills. Returns True when written."""
    try:
        with open(path, "w") as f:
            f.write(str(int(value)))
        return True
    except OSError:
        return False


def parse_running(body):
    """`{"running":[{"model":..,"state":..}]}` -> tuple of Resident; None when the body is not that shape."""
    try:
        d = json.loads(body) if isinstance(body, (str, bytes)) else body
        return tuple(G.Resident(str(r["model"]), r.get("state")) for r in d["running"])
    except (KeyError, TypeError, ValueError):
        return None


def fetch_running(url, timeout=1.0):
    """GET only, 1 s timeout. None means the GET failed (the governor treats it as non-empty, unknown kind).
    An empty `url` means there is no model router to watch: nothing is running, so `()` (never None)."""
    if not url:
        return ()
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return parse_running(r.read().decode())
    except Exception:
        return None


def governor_footprint_bytes(ledger, chunks, sample, flags):
    """Rule 6 in bytes. `sample` None (no reading yet) falls back to the ledger's expected peak."""
    expected = ledger.expected_peak_bytes(chunks)
    if sample is None:
        return expected
    return max(expected, fp.once_bytes(sample, flags))


class Controller:
    """Executes decisions. A shrink never runs while a generation holds the gate: with `cancel` a ticket is raised
    and the gate is awaited; without it the shrink waits for the request in flight to end."""

    def __init__(self, engine, boundary=None, gate=None, gate_timeout_s=5.0):
        self.engine = engine
        self.boundary = boundary or PressureBoundary()
        self.gate = gate or GenerationGate()
        self.gate_timeout_s = gate_timeout_s

    def shrink(self, chunks, reason, cancel=False, drop_prefix=False):
        ticket = self.boundary.request(reason) if cancel else None
        if not self.gate.acquire(self.gate_timeout_s):
            if ticket is not None:
                self.boundary.acknowledge(ticket)
            return False
        try:
            if drop_prefix and hasattr(self.engine, "drop_prefix"):
                self.engine.drop_prefix()
            self.engine.resize_chunks(chunks)
        finally:
            self.gate.release()
            if ticket is not None:
                self.boundary.acknowledge(ticket)
        return True

    def grow(self, chunks):
        if not self.gate.acquire(self.gate_timeout_s):
            return False
        try:
            self.engine.resize_chunks(chunks)
        finally:
            self.gate.release()
        return True

    def exit(self, reason):
        t = self.boundary.request(reason)
        got = self.gate.acquire(self.gate_timeout_s)    # exit proceeds even if the cancelled run is slow to yield
        try:
            self.engine.request_exit(reason)
        finally:
            if got:
                self.gate.release()
            self.boundary.acknowledge(t)


class Signals:
    RUNNING_PERIOD_S = 0.25
    MEM_PERIOD_S = 1.0

    def __init__(self, *, mode, cfg, ledger, engine, get_running, read_mem, read_footprint=None, log=None,
                 minutes_now=None, no_windows=None, controller=None, decide=G.decide, oom_path="/proc/self/oom_score_adj",
                 write_oom=True):
        self.mode, self.cfg, self.ledger, self.engine = mode, cfg, ledger, engine
        self.get_running, self.read_mem, self.read_footprint = get_running, read_mem, read_footprint
        self.log = log
        self.no_windows = no_windows or _paused_no_windows
        self.minutes_now = minutes_now or (lambda: time.localtime().tm_hour * 60 + time.localtime().tm_min)
        self.controller = controller or Controller(engine)
        self.decide = decide
        self.oom_written = write_oom_score_adj(oom_path) if write_oom else None
        self.running = ()
        self.avail = self.free = 0
        self.sample, self.flags = None, {}
        self._t_run = self._t_mem = None
        self._last_pressure_t = None
        self._busy_floor_since = None
        self._a_since = None
        self._stable_ref, self._stable_since = None, None
        self._prev_chunks = None
        self.exited = None

    # --- inputs -------------------------------------------------------------------------------------------
    def _refresh_mem(self, now):
        self.avail, self.free = self.read_mem()
        if self.read_footprint is not None:
            self.sample, self.flags = self.read_footprint()
        band = self.cfg.stable_band_mib
        if self._stable_ref is None or abs(self.avail - self._stable_ref) > band:
            self._stable_ref, self._stable_since = self.avail, now

    def inputs(self, now):
        chunks = self.engine.chunks
        floor = self.cfg.floor_chunks
        busy = bool(self.running) or self.running is None
        if busy and chunks <= floor:
            if self._busy_floor_since is None:
                self._busy_floor_since = now
        else:
            self._busy_floor_since = None
        a_present = bool(self.running) and any(G.classify(r.name, self.cfg) == "A" for r in self.running)
        if a_present:
            if self._a_since is None:
                self._a_since = now
        else:
            self._a_since = None
        calm = now - (self._last_pressure_t if self._last_pressure_t is not None else now - 10 ** 6)
        fbytes = governor_footprint_bytes(self.ledger, chunks, self.sample, self.flags)
        return G.GovInputs(
            mode=self.mode, now_min=self.minutes_now(), running=self.running, chunks=chunks,
            footprint_mib=-(-fbytes // MIB), mem_available_mib=self.avail, mem_free_mib=self.free,
            calm_s=int(calm), busy_at_floor_s=int(now - self._busy_floor_since) if self._busy_floor_since is not None else 0,
            a_stable_s=int(now - self._stable_since) if self._stable_since is not None else 0,
            a_since_s=int(now - self._a_since) if self._a_since is not None else 0,
            prev_chunks=self._prev_chunks, no_windows=self.no_windows())

    # --- one tick -----------------------------------------------------------------------------------------
    def poll(self, now):
        """Call as often as you like; polls /running every 250 ms and meminfo every 1 s, decides after each
        /running poll. Returns the decision taken at this call or None when nothing was due."""
        if self._t_run is not None and now - self._t_run < self.RUNNING_PERIOD_S - 1e-9:
            return None
        self._t_run = now
        self.running = self.get_running()
        if self._t_mem is None or now - self._t_mem >= self.MEM_PERIOD_S - 1e-9:
            self._t_mem = now
            self._refresh_mem(now)
        g = self.inputs(now)
        d = self.decide(g, self.cfg)
        return self._act(now, g, d)

    def _act(self, now, g, d):
        before = g.chunks
        t0 = time.monotonic()
        freed_t = None
        if isinstance(d, G.ShrinkTo):
            if self._prev_chunks is None and "A starting" in d.reason:
                self._prev_chunks = before
            self.controller.shrink(d.chunks, d.reason, cancel=d.cancel, drop_prefix=d.drop_prefix)
            freed_t = round(time.monotonic() - t0, 4)
            self._last_pressure_t = now
        elif isinstance(d, G.GrowTo):
            self.controller.grow(d.chunks)
            self._last_pressure_t = now
        elif isinstance(d, G.Exit):
            self.controller.exit(d.reason)
            self.exited = d.reason
        elif isinstance(d, G.Hold) and not self.running and self._prev_chunks is not None:
            self._prev_chunks = None
        self._write_log(now, g, d, before, freed_t)
        return d

    def _write_log(self, now, g, d, before, freed_t):
        if self.log is None:
            return
        rec = {"t": round(now, 3), "avail": g.mem_available_mib, "free": g.mem_free_mib,
               "running": None if self.running is None else [r.name for r in self.running],
               "cap": G.cap_mib(g, self.cfg, empty=not self.running), "footprint": g.footprint_mib,
               "decision": type(d).__name__ + (f"({getattr(d, 'chunks', getattr(d, 'reason', ''))})"
                                                if not isinstance(d, G.Hold) else ""),
               "chunks_before": before, "chunks_after": self.engine.chunks, "freed_t": freed_t}
        self.log.write(json.dumps(rec, sort_keys=True) + "\n")
        self.log.flush()

    # --- thread -------------------------------------------------------------------------------------------
    def run(self, stop_event, clock=time.monotonic):
        while not stop_event.is_set() and self.exited is None:
            self.poll(clock())
            stop_event.wait(self.RUNNING_PERIOD_S)

    def start(self, clock=time.monotonic):
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, args=(self.stop_event, clock), daemon=True, name="signals")
        self.thread.start()
        return self.thread

    def stop(self):
        self.stop_event.set()
        self.thread.join(2)
