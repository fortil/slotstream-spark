"""Wall-time partition of one decode token: NVTX ranges plus perf_counter_ns accumulators.

A token is the whole loop iteration: sample, n-gram hash and prefetch, embed read, n-gram wait, graph launches,
the spin on the GPU progress word, the expert ensure and the logits readback. The nine phases are disjoint by
construction (opening a phase inside another raises), and `other = wall - sum(phases)` is what is left, so the
partition adds up to 100% of the wall time and a double count shows as a negative `other`.

The pool timing is split without overlap: `read_wall_ns` of an ensure is submit -> join and contains the copy of
the last record, so expert-miss-read = read_wall - copy_tail, expert-copy-tail = copy_tail (`t_copy`) and
ensure-hits = (ensure wall) - read_wall (bookkeeping, pins, publish, the Python call).
"""
import time

PHASES = ("sample", "ngram_hash", "embed_read", "ngram_wait", "launch", "gpu_wait", "ensure_hits",
          "expert_miss_read", "expert_copy_tail")
_now = time.perf_counter_ns


class _Range:
    __slots__ = ("clock", "name", "t0")

    def __init__(self, clock, name):
        self.clock, self.name = clock, name

    def __enter__(self):
        c = self.clock
        if c._cur is None:
            return self
        if c._open is not None:
            raise RuntimeError(f"phase {self.name!r} opened inside phase {c._open!r}")
        c._open = self.name
        c.push(self.name)
        self.t0 = _now()
        return self

    def __exit__(self, *exc):
        c = self.clock
        if c._cur is None:
            return False
        c._cur[self.name] += _now() - self.t0
        c.pop()
        c._open = None
        return False


class PhaseClock:
    def __init__(self, nvtx=True):
        self.nvtx = False
        if nvtx:
            try:
                import torch
                self._push, self._pop = torch.cuda.nvtx.range_push, torch.cuda.nvtx.range_pop
                self.nvtx = True
            except Exception:
                self.nvtx = False
        self.tokens = []
        self.tau_ns = []                      # ensure + launch of the all-hit layer steps (M0' definition)
        self._cur = None
        self._open = None
        self._t0 = 0
        self.hits = self.misses = 0
        self.read_wall_ns = 0

    def push(self, name):
        if self.nvtx:
            self._push(name)

    def pop(self):
        if self.nvtx:
            self._pop()

    def phase(self, name):
        if name not in PHASES:
            raise KeyError(name)
        return _Range(self, name)

    def add(self, name, ns):
        if self._cur is not None:
            self._cur[name] += int(ns)

    def begin(self):
        self._cur = dict.fromkeys(PHASES, 0)
        self._open = None
        self.push("token")
        self._t0 = _now()

    def end(self):
        wall = _now() - self._t0
        self.pop()
        cur, self._cur = self._cur, None
        self.tokens.append({"wall_ns": wall, "phases": cur, "other_ns": wall - sum(cur.values())})
        return self.tokens[-1]

    def cancel(self):
        """Drop the open token (the loop ended before its step ran)."""
        if self._cur is not None:
            self.pop()
        self._cur = None
        self._open = None

    def layer(self, wait_ns, ensure_ns, launch_ns, timing):
        """One layer step of the decode loop: the spin, the ensure (split by the pool timing) and the launch."""
        if self._cur is None:
            return
        read = min(int(timing["read_wall_ns"]), ensure_ns)
        tail = min(int(timing["copy_tail_ns"]), read)
        cur = self._cur
        cur["gpu_wait"] += wait_ns
        cur["launch"] += launch_ns
        cur["expert_miss_read"] += read - tail
        cur["expert_copy_tail"] += tail
        cur["ensure_hits"] += ensure_ns - read
        self.read_wall_ns += read
        self.hits += int(timing["hits"])
        self.misses += int(timing["misses"])
        if int(timing["misses"]) == 0:
            self.tau_ns.append(ensure_ns + launch_ns)
