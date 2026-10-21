"""Eager forward of Qwen3.8-Flash-Next: prefill, decode and teacher forcing, one layer at a time.

Per layer: seg_attn -> seg_router -> router ids to the host (sync) -> ExpertPool.ensure per row group ->
moe_decode -> HC residual. `teacher_force(ids, dump_dir)` writes the step 31 golden layout
(tools/make_goldens.py), which every engine-vs-oracle gate reads:
  h_in.f32, layer_{LL}_out.f32 (hyper width [T,10240]), router_{LL}_ids.i32, router_{LL}_w.f32,
  ngram_ids.i64, logits.f32 [T,V], meta.json
"""
import json
from pathlib import Path

import numpy as np
import torch

from .layers import SeqState, Segments, lin


def ngram_start(src, history, ids):
    """Steps (2)-(3) of the decode loop: hash the 16 gids of every token and start the row prefetch; returns at
    once. `src` is a weights.NgramSource. The pinned buffer is reused between calls: a decode step ends with
    the logits readback, so the previous step's copy out of it is complete before the next prefetch is waited on."""
    gids = np.ascontiguousarray(src.ids(history, ids), dtype=np.int64)          # [T, 16]
    src.last_ids = gids
    buf = getattr(src, "_pinned", {}).get(gids.size)
    if buf is None:
        buf = torch.empty((gids.size, src.row_bytes), dtype=torch.uint8).pin_memory()
        if not hasattr(src, "_pinned"):
            src._pinned = {}
        src._pinned[gids.size] = buf
    return gids, src.rows.prefetch(gids.reshape(-1)), buf


def ngram_finish(src, started):
    """Step (6): wait for the rows into the pinned buffer, dequantize on the device: bf16 [T, width]."""
    from slotstream_spark.kernels.rows import ple_gather_gs32
    gids, ticket, buf = started
    src.rows.wait(ticket, buf.numpy())
    dev = buf.to(src.device, non_blocking=True).view(len(gids), 16, src.row_bytes)
    return torch.stack([ple_gather_gs32(dev[t].contiguous(), src.row_bytes) for t in range(len(gids))])


class Model:
    def __init__(self, cfg, W, embed, pool, ngram=None, device="cuda", gdn="fused", max_ctx=4096, chunk=1024,
                 max_rows=32, cuda=None):
        self.cfg, self.W, self.device = cfg, W, device
        self.cuda = str(device).startswith("cuda") if cuda is None else cuda
        self.max_ctx, self.chunk = max_ctx, chunk
        self.seg = Segments(cfg, W, embed, pool, ngram, device, self.cuda, gdn, max_rows)
        self.NL = cfg["num_hidden_layers"]
        self.vocab = cfg["vocab_size"]
        self.nvtx = False                 # NVTX ranges per chunk and layer on the eager path (profiling)
        self.multi_sink = None            # callable(ids, hyp) with the pre-mixer stream of each eager chunk (MTP head, step 76)
        self.st = self.new_state()

    # ------------------------------------------------------------ state
    def new_state(self):
        return SeqState(self.cfg, self.device, self.max_ctx, self.cuda)

    def reset(self):
        self.st = self.new_state()
        self.seg.gen = 0

    def set_gdn(self, mode):
        if mode not in ("fused", "ref"):
            raise ValueError(f"gdn mode must be 'fused' or 'ref', got {mode!r}")
        self.seg.gdn = mode

    @property
    def gdn(self):
        return self.seg.gdn

    # ------------------------------------------------------------ one chunk through the 48 layers
    def _chunk(self, ids, sink=None):
        st, seg = self.st, self.seg
        T = len(ids)
        if st.pos + T > self.max_ctx:
            raise ValueError(f"context {st.pos + T} exceeds max_ctx {self.max_ctx}")
        nv = torch.cuda.nvtx if self.nvtx and self.cuda else None
        with torch.no_grad():
            if nv:
                nv.range_push(f"chunk{T}")
            hyp = seg.seg_begin(ids)
            if sink is not None:
                sink.h_in(hyp)
            for L in range(self.NL):
                if nv:
                    nv.range_push(f"layer{L}")
                hyp = seg.seg_attn(L, st, hyp, ids)
                mixed, inj, rids, rw = seg.seg_router(L, hyp)
                h = seg.seg_moe(L, mixed, rids, rw)
                hyp = hyp + (h.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)
                if sink is not None:
                    sink.layer(L, hyp, *seg.last_routing[L])
                if nv:
                    nv.range_pop()
            if self.multi_sink is not None:
                self.multi_sink(ids, hyp)
            final = seg.seg_final(hyp)
            if nv:
                nv.range_pop()
        if sink is not None:
            sink.ngram(seg.last_ngram_ids)
        st.pos += T
        st.tokens += [int(t) for t in ids]
        return final

    def _logits(self, final):
        return lin(final, self.W("lm_head.weight")).float()

    # ------------------------------------------------------------ public passes
    def prefill(self, ids):
        """Process `ids` (chunks of at most `chunk` tokens); logits of the last token, fp32 [V]."""
        ids = [int(t) for t in ids]
        if not ids:
            raise ValueError("prefill needs at least one token")
        final = None
        for a in range(0, len(ids), self.chunk):
            final = self._chunk(ids[a:a + self.chunk])
        return self._logits(final[-1:])[0]

    def decode(self, tok):
        """One more token; logits fp32 [V]."""
        return self._logits(self._chunk([int(tok)])[-1:])[0]

    def teacher_force(self, ids, dump_dir=None):
        """Fresh sequence over `ids`; logits fp32 [T, V] (CPU). With `dump_dir` the golden layout is written."""
        ids = [int(t) for t in ids]
        self.reset()
        sink = _Dump(dump_dir) if dump_dir else None
        out = []
        try:
            for a in range(0, len(ids), self.chunk):
                final = self._chunk(ids[a:a + self.chunk], sink)
                lg = self._logits(final).cpu()
                out.append(lg)
                if sink:
                    sink.logits(lg)
            if sink:
                sink.finish(self.seg.gdn)
        finally:
            if sink:
                sink.close()
        return torch.cat(out, 0)


class _Dump:
    """Appends the chunks of a teacher-forced pass to the raw golden files."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.files = {}

    def _f(self, name):
        if name not in self.files:
            self.files[name] = open(self.dir / name, "wb")
        return self.files[name]

    def _put(self, name, arr, dtype):
        np.ascontiguousarray(arr, dtype=dtype).tofile(self._f(name))

    def h_in(self, hyp):
        self._put("h_in.f32", hyp.float().cpu().numpy(), np.float32)

    def layer(self, L, hyp, ids, w):
        self._put(f"layer_{L:02d}_out.f32", hyp.float().cpu().numpy(), np.float32)
        self._put(f"router_{L:02d}_ids.i32", ids, np.int32)
        self._put(f"router_{L:02d}_w.f32", w, np.float32)

    def ngram(self, gids):
        if gids is not None:
            self._put("ngram_ids.i64", gids, np.int64)

    def logits(self, lg):
        self._put("logits.f32", lg.numpy(), np.float32)

    def finish(self, gdn):
        (self.dir / "meta.json").write_text(json.dumps(
            {"dtype": "bfloat16", "engine": "slotstream-spark", "gdn_impl": gdn, "torch": torch.__version__},
            sort_keys=True))

    def close(self):
        for f in self.files.values():
            f.close()
