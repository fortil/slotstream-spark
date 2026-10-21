"""One CUDA graph per decode token, with host-memory waits (plan step 91, F8).

The F3 path (runtime.graphs.GraphSet) launches 49 segment graphs per token: G(0) ... G(48), and between two of them the
host waits for the router ids, runs ensure(l) and launches the next one. Here the same 49 bodies are captured one after
the other into ONE graph, and the launch gap between them is replaced by a stream memory operation node:

    G(0) -> [wait ready[0]] -> G(1) -> [wait ready[1]] -> ... -> [wait ready[47]] -> G(48)

The wait is cuStreamWaitValue32 on a pinned host word (no spinning kernel: the stream stops in the front end), re-armed
to 0 by a cuStreamWriteValue32 node right behind it, because the value of a memory-operation node is fixed at capture
(so `ready[l]` is a 0/1 flag the host raises and the graph lowers again, not a growing sequence number). The host loop is
the F3 one with the launch removed: wait for the layer's ids (slotio.wait_seq on the same sequence word), ensure(l), then
publish `ready[l]` (slotio.ControlRing, a release store). After MoE(l), the segment l + 1 copies the layer's sequence
number `base + l` into `done[l]` (a device-to-host copy node), which the tests read back.

Pins: unchanged. A pin of generation base + l is released when the sequence word passes it, and the word is published by
the same `signal` kernels at the end of each body, so the graph releases pins exactly as the segmented path does.

Launch thread: cudaGraphLaunch of a graph this long (about 2,500 kernel nodes) blocks the caller when the stream's launch
queue is full, and the queue cannot drain while the first memory operation waits for the host. The launch therefore runs
on a worker thread (measured: the same launch from the host loop's own thread never returns), and the host loop starts
at once on the main thread.

Bounded wait and fallback: the host waits at most `wait_bound_s` (50 ms) for the sequence word of a layer. A graph that is
already launched cannot be cancelled, so a miss is counted (`fallbacks`), the host keeps waiting up to the Readback bound
(2 s, the F3 limit) and the NEXT token runs on the F3 segmented path; the graph is used again after that token. Errors
(a timeout past 2 s, a pool that cannot ensure) publish every remaining `ready` word so the stream drains, and re-raise.

The token output is bit-identical to the F3 path: same bodies, same order, same kernels.
"""
import queue
import threading

import numpy as np
import torch

from slotstream_spark.model.forward import ngram_finish, ngram_start
from slotstream_spark.runtime.readback import LAYERS

WAIT_BOUND_S = 0.050      # plan: bounded host wait before the F3 segmented path takes the next token

_CU_STREAM_WAIT_VALUE_EQ = 1


def _driver():
    from cuda.bindings import driver  # cuda-python, installed with the torch build on the Spark
    return driver


def _check(res, what):
    err = res[0] if isinstance(res, tuple) else res
    if int(err) != 0:
        raise RuntimeError(f"{what} failed: CUresult {int(err)}")
    return res[1:] if isinstance(res, tuple) and len(res) > 1 else None


def _slotio():
    import slotio
    return slotio


def eligible(gs):
    """A GraphSet the token graph can replace: one row, no forecast tap, no lookahead scheduler."""
    return gs.rows == 1 and gs.forecast is None and gs.lookahead is None and gs.record is None


class TokenGraph:
    """The single per-token graph over the bodies of a 1-row GraphSet. `step` is GraphSet.step with one launch."""

    def __init__(self, gs, wait_bound_s=WAIT_BOUND_S):
        if not eligible(gs):
            raise ValueError("the token graph needs a 1-row GraphSet without forecast tap, lookahead or step record")
        self.gs = gs
        self.wait_bound_s = float(wait_bound_s)
        self.ready = torch.zeros(LAYERS, dtype=torch.int32).pin_memory()
        self.done = torch.zeros(LAYERS, dtype=torch.int32).pin_memory()
        self.ring = _slotio().ControlRing(self.ready.data_ptr(), self.done.data_ptr(), LAYERS)
        self.ring.clear()
        self.fallbacks = 0              # tokens whose host wait passed the bound
        self.graph_tokens = 0           # tokens run on the single graph
        self.segmented_tokens = 0       # tokens run on the F3 path (after a fallback, or while a hook needs segments)
        self._segmented_next = False
        self.ensure_hook = None         # tests: called with the layer before ensure(l) (an injected delay)
        self.memop_nodes = 0
        self.graph = None
        self._capture()
        self._jobs = queue.SimpleQueue()
        self._worker = threading.Thread(target=self._launch_loop, name="token-graph-launch", daemon=True)
        self._worker.start()

    def _launch_loop(self):
        while True:
            job = self._jobs.get()
            if job is None:
                return
            stream, done, errors = job
            try:
                with torch.cuda.stream(stream):
                    self.graph.replay()
            except BaseException as e:      # noqa: BLE001 - handed to the host loop
                errors.append(e)
            finally:
                done.set()

    def close(self):
        self._jobs.put(None)
        self._worker.join(timeout=5)

    # ------------------------------------------------------------------ capture
    def _capture(self):
        gs, drv = self.gs, _driver()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph(keep_graph=True)
        seq = torch.zeros(1, dtype=torch.int32, device=gs.device)
        with torch.cuda.graph(g, pool=gs.pool_handle, capture_error_mode="thread_local"):
            stream = drv.CUstream(torch.cuda.current_stream().cuda_stream)
            for k in range(LAYERS):
                if k > 0:
                    addr = drv.CUdeviceptr(self.ready.data_ptr() + 4 * (k - 1))
                    _check(drv.cuStreamWaitValue32(stream, addr, 1, _CU_STREAM_WAIT_VALUE_EQ), "cuStreamWaitValue32")
                    _check(drv.cuStreamWriteValue32(stream, addr, 0, 0), "cuStreamWriteValue32")
                gs.bodies[k]()
                if k > 0:                                    # segment k ran MoE(k - 1): done[k - 1] = base + k - 1
                    seq.copy_((gs.ctrl_dev[1:2] + (k - 1)).to(torch.int32))
                    self.done[k - 1:k].copy_(seq, non_blocking=True)
        torch.cuda.synchronize()
        self.graph = g
        self.memop_nodes = self.count_memop_nodes()

    def count_memop_nodes(self):
        """Memory-operation nodes of the captured graph (expected 2 per wait: the wait and its re-arm = 96)."""
        drv = _driver()
        raw = self.graph.raw_cuda_graph()
        handle = drv.CUgraph(int(raw))
        count = _check(drv.cuGraphGetNodes(handle), "cuGraphGetNodes")[1]       # (nodes, count): the first call sizes it
        nodes = _check(drv.cuGraphGetNodes(handle, count), "cuGraphGetNodes")[0]
        n = 0
        for node in nodes:
            t = _check(drv.cuGraphNodeGetType(node), "cuGraphNodeGetType")[0]
            if t == drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_BATCH_MEM_OP:
                n += 1
        return n

    # ------------------------------------------------------------------ one token
    def _wait_layer(self, rb, target):
        """Wait for the layer's sequence word: bounded to wait_bound_s, then (counted) up to the Readback bound."""
        from slotstream_spark.runtime.pool import wait_seq
        try:
            wait_seq(rb.seq_addr, int(target), int(self.wait_bound_s * 1e9))
        except TimeoutError:
            if not self._segmented_next:             # one fallback per token, however many layers missed the bound
                self.fallbacks += 1
            self._segmented_next = True
            rb.wait(target)

    def step(self, toks):
        """Decode one token: logits fp32 [1, V] (CPU), the GraphSet.step contract."""
        gs = self.gs
        if self._segmented_next or gs.clock is not None or gs.probe is not None or not eligible(gs):
            self._segmented_next = False
            self.segmented_tokens += 1
            return gs.step(toks)
        self.graph_tokens += 1
        return self._run(toks)

    def _run(self, toks):
        gs = self.gs
        seg, st, rb, R = gs.seg, gs.model.st, gs.rb, gs.rows
        ids = [int(t) for t in toks]
        if len(ids) != R:
            raise ValueError("this graph set takes 1 token per step")
        if st.pos + R > gs.model.max_ctx:
            raise ValueError(f"context {st.pos + R} exceeds max_ctx {gs.model.max_ctx}")
        if not rb.attached:
            gs.attach()
        if self.ring.armed():
            raise RuntimeError("token graph: a ready word is still raised from the previous token")
        with torch.no_grad():
            if gs.ple_emb is not None:
                if gs.ple_late:       # the rows must land before the graph starts: G(1) is no longer behind a host launch
                    started = ngram_start(seg.ngram, st.tokens, ids)
                    gs.ple_emb.copy_(ngram_finish(seg.ngram, started).to(gs.ple_emb.dtype))
                else:
                    gs.ple_emb.copy_(seg.ngram(st.tokens, ids).to(gs.ple_emb.dtype))
            gs.hyp_in.copy_(seg.embed(ids).repeat(1, seg.hc))
        base = rb.base
        rb.base += LAYERS
        rb.begin(st.pos, base)
        pool = seg.pool
        launched, launch_errors = threading.Event(), []
        self._jobs.put((torch.cuda.current_stream(), launched, launch_errors))
        L = 0
        try:
            for L in range(gs.NL):
                self._wait_layer(rb, base + L)
                uniq = np.unique(rb.layer_ids(L, R * seg.topk)).astype(np.int32)
                if self.ensure_hook is not None:
                    self.ensure_hook(L)
                pool.ensure(L, uniq, base + L)
                self.ring.publish(L)
            rb.wait(base + gs.NL)
        except BaseException:
            self.ring.publish_from(L)        # let the stream drain; the caller gets the error
            launched.wait(timeout=10)
            raise
        if not launched.wait(timeout=10):
            raise RuntimeError("token graph: the launch thread did not return after the last layer")
        if launch_errors:
            raise launch_errors[0]
        st.pos += R
        st.tokens += ids
        for c in st.attn.values():
            c.pos = st.pos
        return rb.logits[:R].clone()

    def done_ok(self, base):
        """True when done[l] == base + l for every layer 0..47 (the last token's segments all ran)."""
        return all(int(self.done[l]) == base + l for l in range(gs_layers(self.gs)))

    def stats(self):
        return {"graph_tokens": self.graph_tokens, "segmented_tokens": self.segmented_tokens, "fallbacks": self.fallbacks,
                "memop_nodes": self.memop_nodes}


def gs_layers(gs):
    return gs.NL


def enabled(platform_flag=None):
    """SLOTSTREAM_TOKEN_GRAPH=on|off overrides the platform flag ([f8] token_graph); unset = the flag."""
    import os
    v = os.environ.get("SLOTSTREAM_TOKEN_GRAPH")
    if v is not None:
        if v not in ("on", "off"):
            raise ValueError(f"SLOTSTREAM_TOKEN_GRAPH must be on or off, got {v!r}")
        return v == "on"
    if platform_flag is None:
        from slotstream_spark import config
        platform_flag = bool(config.load().get("f8", {}).get("token_graph", False))
    return bool(platform_flag)


def for_graphs(gs):
    """The TokenGraph of a GraphSet (built once, kept on `gs.token_graph`), or None when it is off or not eligible."""
    tg = getattr(gs, "token_graph", None)
    if tg is not None:
        return tg
    if not enabled() or not eligible(gs):
        return None
    gs.token_graph = TokenGraph(gs)
    return gs.token_graph
