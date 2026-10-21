"""Window scoring for the perplexity gate (plan step 93).

`nll_windows(model, driver, ids, window=2048)` cuts `ids` into non-overlapping windows, runs each one through the sweep
prefill driver and scores the tokens 1..window-1 of the window (token 0 has no context to be predicted from). The scorer
hangs on the driver's `on_rows(pos, fin)` hook: it receives the final hidden rows of a tile and runs the lm_head in
`TILE` = 256-row tiles, each with a full-vocabulary (248,320) log-softmax on the GPU, so a [2048, V] logits matrix never
exists (one tile is 256 x V x 4 B = 254 MB). The lm_head weight is read as it is stored (4-bit, through the same dense
call the engine uses); nothing here makes an fp32 copy of it.

Pure helpers (`window_slices`, `ppl`) have no torch dependency.
"""
import math
import time

TILE = 256
WINDOW = 2048


def window_slices(n_tokens, window=WINDOW, max_windows=None):
    """[(start, stop)] of the non-overlapping full windows of an `n_tokens` text, at most `max_windows`."""
    k = n_tokens // window
    if max_windows is not None:
        k = min(k, int(max_windows))
    return [(i * window, (i + 1) * window) for i in range(k)]


def scored_tokens(n_windows, window=WINDOW):
    """Tokens that carry a loss: 1..window-1 of every window."""
    return n_windows * (window - 1)


def ppl(nll_sum, tokens):
    return math.exp(nll_sum / tokens)


class TileScorer:
    """`on_rows` hook: accumulates the negative log-likelihood of ids[pos + i + 1] under row i, row by row in tiles."""

    def __init__(self, logits_fn, tile=TILE):
        import torch
        self.torch = torch
        self.logits_fn = logits_fn
        self.tile = int(tile)
        self.ids = None
        self.total = None
        self.rows = 0
        self.max_tile_elems = 0

    def begin(self, ids, device):
        self.ids = self.torch.as_tensor([int(t) for t in ids], dtype=self.torch.int64, device=device)
        self.total = self.torch.zeros((), dtype=self.torch.float64, device=device)
        self.rows = 0

    def __call__(self, pos, fin):
        torch = self.torch
        n = int(fin.shape[0])
        for a in range(0, n, self.tile):
            b = min(n, a + self.tile)
            p0, p1 = pos + a, pos + b                       # rows predict the ids at p0+1 .. p1 (the last row of the window has none)
            last = min(p1, self.ids.numel() - 1)            # rows whose target exists
            if last <= p0:
                continue
            lg = self.logits_fn(fin[a:a + (last - p0)])     # fp32 [t, V]
            self.max_tile_elems = max(self.max_tile_elems, int(lg.numel()))
            tgt = self.ids[p0 + 1:last + 1]
            lse = torch.logsumexp(lg, dim=-1)
            nll = lse - lg.gather(1, tgt.unsqueeze(1)).squeeze(1)
            self.total += nll.double().sum()
            self.rows += int(nll.numel())
            del lg, lse, nll

    def result(self):
        return float(self.total.item()), self.rows


def nll_windows(model, driver, ids, *, window=WINDOW, max_windows=None, plan=None, control=None, on_window=None,
                tile=TILE):
    """-> list of {"window", "start", "nll", "tokens", "seconds"}; `nll` is the sum over the tokens 1..window-1.

    Every window starts from a reset sequence state. `on_window(row)` is called after each one (progress, partial files)."""
    import torch
    from slotstream_spark.engine.prefill import PrefillPlan, d5_budget
    plan = plan or PrefillPlan.from_envelope(driver.env, prefill_pass=4096, read_scope=8192)
    control = control or d5_budget()
    scorer = TileScorer(model._logits, tile)
    out = []
    for k, (a, b) in enumerate(window_slices(len(ids), window, max_windows)):
        w = [int(t) for t in ids[a:b]]
        model.st.reset_inplace()
        scorer.begin(w, model.device)
        t0 = time.perf_counter()
        with torch.no_grad():
            driver.run(w, model.st, plan=plan, control=control, should_continue=lambda: True, on_rows=scorer)
        torch.cuda.synchronize()
        nll, rows = scorer.result()
        if rows != window - 1:
            raise RuntimeError(f"window {k}: scored {rows} tokens, expected {window - 1}")
        row = {"window": k, "start": a, "nll": nll, "tokens": rows, "seconds": round(time.perf_counter() - t0, 3)}
        out.append(row)
        if on_window:
            on_window(row)
    model.st.reset_inplace()
    return out
