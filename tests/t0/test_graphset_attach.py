"""GraphSet.attach() keeps the step generations above every pin the eager path made, attached or not.

The pool retires a pin once the progress word passes its generation. While the graphs are attached the progress word is
the readback's seq, which only the steps move; a sweep prefill in between advances `seg.gen` by two per layer. Measured
on the GB10 (2026-10-21, tools/probes/pins_after_prefill.py, 131,072-token window, graphs left attached by the capture):
seg.gen 1,502 against a base of 784 after a 32,768-token prompt, every one of the 14,356 slots pinned, and the first
decode step failed with `slotio.Exhausted`; 8,192 tokens only reached 719 against 637 and found victims by luck.
"""
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from slotstream_spark.runtime.graphs import GraphSet  # noqa: E402
from slotstream_spark.runtime.readback import LAYERS  # noqa: E402


def _fake(gen, base=0, attached=False, seq=-1):
    ptrs = []
    seg = SimpleNamespace(gen=gen, progress=np.zeros(1, dtype=np.uint64),
                          pool=SimpleNamespace(set_progress_ptr=ptrs.append))
    rb = SimpleNamespace(base=base, attached=attached, seq=[seq], seq_addr=0xA11CE)
    rb.set_seq = lambda v: rb.seq.__setitem__(0, int(v))
    gs = SimpleNamespace(seg=seg, rb=rb)
    return gs, ptrs


def _roundup(gen):
    return ((gen + 1 + LAYERS - 1) // LAYERS) * LAYERS


def test_first_attach_bases_the_step_above_the_eager_generation():
    gs, ptrs = _fake(gen=527)
    GraphSet.attach(gs)
    assert gs.rb.base == _roundup(527) == 539
    assert gs.rb.seq == [538] and gs.rb.attached and ptrs == [0xA11CE]


def test_attach_while_attached_and_quiet_changes_nothing():
    gs, ptrs = _fake(gen=600, base=784, attached=True, seq=784)
    GraphSet.attach(gs)
    assert gs.rb.base == 784 and gs.rb.seq == [784] and ptrs == []


def test_attach_rebases_after_a_sweep_ran_while_attached():
    """The measured first request: capture left base 784, the 32K sweep took seg.gen to 1,502."""
    gs, ptrs = _fake(gen=1502, base=784, attached=True, seq=783)
    GraphSet.attach(gs)
    assert gs.rb.base == _roundup(1502) == 1519 > 1502
    assert gs.rb.seq == [1518] and gs.rb.attached and ptrs == [0xA11CE]


def test_attach_rebases_when_gen_reaches_the_base_exactly():
    gs, _ = _fake(gen=784, base=784, attached=True, seq=784)
    GraphSet.attach(gs)
    assert gs.rb.base == _roundup(784) == 833 and gs.rb.seq == [832]


def test_build_detaches_and_run_attaches_unconditionally():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "slotstream_spark" / "runtime" / "graphs.py").read_text()
    build = src[src.index("def _build"):src.index("def _probe_tokens")]
    assert build.rstrip().endswith("self.detach()                       # the eager probes attached the graphs; a fresh set starts like any request ends")
    run = src[src.index("def _run"):src.index("def step")]
    assert "        self.attach()\n" in run and "if not self.rb.attached" not in run
    tg = (Path(__file__).resolve().parents[2] / "slotstream_spark" / "runtime" / "token_graph.py").read_text()
    assert "if not rb.attached" not in tg and "gs.attach()" in tg
