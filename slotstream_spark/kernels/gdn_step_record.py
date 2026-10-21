"""Recorded verify step for the 36 GDN layers (plan step 76).

A verify round runs `rows` (3) tokens through the decode graphs in one pass. If a draft is rejected the recurrent state must go
back to the one after the last kept token, so the pass writes the state after every position into a preallocated ring:

    S     fp32 [rows, 36, 48, 128, 128]   113,246,208 B per entry (the 36 GDN layers)
    conv  act  [rows, 36, ck, C]          GDN conv state
    ple   act  [rows, hist, hc*H]         PLE conv state

`gdn_step_record` is `rows` calls of the same one-row recurrent kernel the plain decode step runs, each starting from the state
the previous one left, so after the round the state and the outputs equal `rows` one-row steps bit for bit. It only adds the
copies into the ring; the buffers are static, so the calls capture into the 3-row graphs.
"""
import torch

from slotstream_spark.kernels import vendored


class StepRecord:
    """The ring, bound to the tensors of one SeqState. `outstanding` counts rounds whose record has not been consumed
    (restored from or released): 0 at the end of a request, or a buffer leaked."""

    def __init__(self, st, rows=3):
        if st.conv is None:
            raise ValueError("StepRecord needs st.ensure_act() first")
        self.rows = rows
        self.S = torch.zeros(rows, *st.S.shape, dtype=torch.float32, device=st.S.device)
        self.conv = torch.zeros(rows, *st.conv.shape, dtype=st.conv.dtype, device=st.conv.device)
        self.ple = torch.zeros(rows, *st.ple_conv.shape, dtype=st.ple_conv.dtype, device=st.ple_conv.device)
        self.outstanding = 0
        self.rounds = 0

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.S, self.conv, self.ple))

    def begin(self):
        """A verify round is about to run (the graphs write the ring)."""
        self.outstanding += 1
        self.rounds += 1

    def release(self):
        if self.outstanding <= 0:
            raise RuntimeError("StepRecord.release without a matching begin")
        self.outstanding -= 1

    def restore(self, st, keep):
        """State after the first `keep` rows of the round (1 <= keep <= rows) -> st, then release the round."""
        if not 1 <= keep <= self.rows:
            raise ValueError(f"keep must be in 1..{self.rows}, got {keep}")
        i = keep - 1
        st.S.copy_(self.S[i])
        st.conv.copy_(self.conv[i])
        st.ple_conv.copy_(self.ple[i])
        self.release()


def gdn_step_record(q, k, v, g, beta, state, ring_layer):
    """`rows` recurrent steps with the state after each one copied into `ring_layer` [rows, 48, 128, 128] (a view of the ring
    for this layer). q, k [1, T, 16, 128], v [1, T, 48, 128], g, beta [1, T, 48], state fp32 [1, 48, 128, 128]
    -> (o [1, T, 48, 128], final state [1, 48, 128, 128]); same arguments as vendored.gdn_decode."""
    T = q.shape[1]
    outs = []
    for i in range(T):
        o, state = vendored.gdn_decode(q[:, i:i + 1], k[:, i:i + 1], v[:, i:i + 1], g[:, i:i + 1], beta[:, i:i + 1], state)
        ring_layer[i].copy_(state[0])
        outs.append(o)
    return torch.cat(outs, 1), state
