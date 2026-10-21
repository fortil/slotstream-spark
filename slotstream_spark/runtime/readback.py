"""Pinned readback of the CUDA-graph decode path (plan section 3.3).

One sequence word, one control word pair and one slot per layer, all in pinned host memory the GPU can write:
  seq     int64 [1]      published by kernels/signal.py as t * 49 + layer after the layer's ids are in its slot
  ctrl    int64 [2]      (position of the first row, t * 49) written by the host before the first graph launches
  ids/w   [49 * SLOT]    top-10 expert ids (int32) and weights (fp32) of each layer, rows back to back
  logits  fp32 [R, V]    written by the last graph
  f_ids/f_logits  [49, R, K]  forecast tap (step 73): the top-K = 24 ids and logits of layer l + 1 computed after layer l's
                         attention, written before layer l's signal, so the same sequence value publishes them
The host waits with slotio.wait_seq (acquire load, bounded spin, 2 s by default): no cudaStreamSynchronize.
The same word is the progress pointer of the Ensurer: a pin of generation g is released once seq > g, and seq(l+1)
implies MoE(l) finished because the graph that publishes it runs MoE(l) first.
"""
import numpy as np
import torch

from slotstream_spark.kernels.signal import SLOT
from slotstream_spark.runtime.pool import wait_seq

LAYERS = 49           # 48 layers plus the final segment
K_FORECAST = 24       # forecast tap width


class Readback:
    def __init__(self, vocab, max_rows=3, timeout_s=2.0, forecast=False):
        self.timeout_ns = int(timeout_s * 1e9)
        self.base = 0              # generation base of the next step (shared by the 1- and 3-row graph sets)
        self.attached = False      # True while seq is the Ensurer's progress pointer
        self.seq = torch.zeros(1, dtype=torch.int64).pin_memory()
        self.ctrl = torch.zeros(2, dtype=torch.int64).pin_memory()
        self.ids = torch.zeros(LAYERS * SLOT, dtype=torch.int32).pin_memory()
        self.w = torch.zeros(LAYERS * SLOT, dtype=torch.float32).pin_memory()
        self.logits = torch.zeros(max_rows, vocab, dtype=torch.float32).pin_memory()
        self.f_ids = self.f_logits = None
        if forecast:
            self.f_ids = torch.full((LAYERS, max_rows, K_FORECAST), -1, dtype=torch.int32).pin_memory()
            self.f_logits = torch.zeros(LAYERS, max_rows, K_FORECAST, dtype=torch.float32).pin_memory()
        self._ids_np = self.ids.numpy()
        self._w_np = self.w.numpy()

    @property
    def seq_addr(self):
        return self.seq.data_ptr()

    def set_seq(self, value):
        self.seq[0] = int(value)

    def begin(self, pos, base):
        self.ctrl[0] = int(pos)
        self.ctrl[1] = int(base)

    def wait(self, target):
        """Spin until seq >= target; raises slotio.Timeout after the bound."""
        wait_seq(self.seq_addr, int(target), self.timeout_ns)

    def layer_ids(self, layer, n):
        """The first n ids of a layer slot (a copy)."""
        o = layer * SLOT
        return self._ids_np[o:o + n].copy()

    def layer_weights(self, layer, n):
        o = layer * SLOT
        return self._w_np[o:o + n].copy()

    def forecast(self, layer, rows):
        """(ids int32 [rows, K], logits fp32 [rows, K]) the tap of layer `layer` wrote for layer + 1 (copies, in index
        order as the stable top-k emits them); valid once seq >= base + layer."""
        if self.f_ids is None:
            raise RuntimeError("this Readback has no forecast buffers")
        return self.f_ids[layer, :rows].clone().numpy(), self.f_logits[layer, :rows].clone().numpy()
