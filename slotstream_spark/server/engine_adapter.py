"""The real engine behind `engine_api.EngineProtocol`.

`EngineAdapter(generator, tokenizer, template, ledger, governor)` owns the order of a request on the engine side:
the governor's gate, then admission (window and ledger against the cap), then the first allocation check, and only
then `on_admitted`. Cancellation is checked per prefill pass and layer (the driver calls `should_continue`), and per
decoded token (the adapter stops the loop from `on_token`). The metadata snapshot has its own lock and never touches
the gate.

`generator` is what runs the model (`ModelGenerator` below, or a stub in the tests):
`run(ids, sample, seed, max_tokens, on_token, should_continue) -> dict` and `reusable(ids) -> int`.
`governor` is a `memory.signals.Controller` (its `gate` and `boundary`), optionally with `cap_fn()` and `mode`.
"""
import threading
from dataclasses import dataclass

from slotstream_spark.engine.pressure import Cancelled
from slotstream_spark.memory import admission as memory_admission
from slotstream_spark.server import detok
from slotstream_spark.server.engine_api import EngineFailure, EngineSnapshot, GenerationResult

GATE_WAIT_S = 30.0


@dataclass(frozen=True)
class Admission:
    reused_tokens: int
    prompt_tokens: int


class _Stop(Exception):
    """Raised from `on_token` to leave the decode loop (client gone, stop sequence, memory pressure)."""


def stop_ids_for(eos, model_dir=None):
    """The ids that end a reply: the config's eos plus the ones generation_config.json names (for this checkpoint
    <|im_end|> 248046 and <|endoftext|> 248044; config.json alone lists only the second)."""
    import json
    import os
    from pathlib import Path
    ids = set() if eos is None else ({int(eos)} if not isinstance(eos, (set, frozenset, list, tuple)) else set(eos))
    d = model_dir or os.environ.get("SLOTSTREAM_MODEL")
    if d:
        try:
            g = json.loads((Path(d) / "generation_config.json").read_text()).get("eos_token_id", [])
            ids |= {int(x) for x in (g if isinstance(g, list) else [g])}
        except (OSError, ValueError, TypeError):
            pass
    return frozenset(ids)


class ModelGenerator:
    """Prefill driver + prefix cache + decode graphs over one loaded `Model`."""

    def __init__(self, model, driver, graphs, cache, key, plan, eos, max_ctx, im_start_id, persistent="auto"):
        self.model, self.driver, self.graphs, self.cache, self.key = model, driver, graphs, cache, key
        self.plan, self.eos, self.max_ctx, self.im_start_id = plan, eos, max_ctx, im_start_id
        self.stop_ids = stop_ids_for(eos)
        self.persistent = self._open_persistent() if persistent == "auto" else persistent

    def _open_persistent(self):
        """The disk tier from the environment (`SLOTSTREAM_PACKED`, `SLOTSTREAM_MODEL`, `[prefix]` in platform.toml,
        `SLOTSTREAM_PREFIX_DIR`). Any failure leaves the engine running on the memory tier alone."""
        import os
        import sys
        from pathlib import Path
        from slotstream_spark.engine import prefix_cache as pc
        try:
            import slotio
            extra = [slotio.__file__] if getattr(slotio, "__file__", None) else []
            return pc.open_tier(
                self.model.st, self.key, self.plan.prefill_pass, model_dir=Path(os.environ["SLOTSTREAM_MODEL"]),
                packed_dir=Path(os.environ["SLOTSTREAM_PACKED"]), envelope=self.driver.env, extra_files=extra,
                on_event=lambda line: print(f"prefix: {line}", file=sys.stderr, flush=True))
        except Exception as e:  # noqa: BLE001
            print(f"prefix: persistent tier off ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
            return None

    def after_gate(self):
        """Called once the gate is released: staged states may be written now."""
        if self.persistent is not None:
            self.persistent.release()

    def reusable(self, ids):
        """Tokens the prefix cache will reuse for `ids` (read-only)."""
        from slotstream_spark.engine import prefill_schedule as ps
        boundaries = ps.resume_boundaries(len(ids), self.plan.prefill_pass, True, self.driver.P)
        n = self.cache.retained_match_length(ids, self.key, boundaries)
        if self.persistent is not None:
            n = max(n, self.persistent.peek(ids, boundaries))
        return n

    def drop_prefix(self):
        self.cache.drop()

    def run(self, ids, sample, seed, max_tokens, on_token, should_continue):
        from slotstream_spark.engine import prefix_cache as pc
        from slotstream_spark.engine.generate import generate
        try:
            return generate(self.model, ids, max_tokens=max_tokens, greedy=False, params=sample, seed=seed, eos=self.stop_ids,
                            on_token=on_token, driver=self.driver, plan=self.plan, should_continue=should_continue,
                            prefix_cache=self.cache, key=self.key,
                            snapshot_target=pc.snapshot_target(ids, self.im_start_id), graphs=self.graphs,
                            persistent=self.persistent)
        finally:
            self.graphs.detach()         # an early exit leaves the graphs attached; the next prefill is eager


class EngineAdapter:
    def __init__(self, generator, tokenizer, template, ledger, governor, *, model_name="qwen38-flash-next",
                 plan=None, window=None, max_ctx=None, cap_fn=None, chunks_fn=None, digest="", eos=None,
                 pool_stats_fn=None):
        self.generator, self.tokenizer, self.template, self.ledger, self.governor = generator, tokenizer, template, ledger, governor
        self.model_name = model_name
        self.plan = plan
        self.context_window = int(window if window is not None else plan.window)
        self.max_ctx = int(max_ctx if max_ctx is not None else self.context_window)
        self.cap_fn = cap_fn
        self.chunks_fn = chunks_fn or (lambda: plan.chunks)
        self.digest, self.eos = digest, eos
        self.stop_ids = getattr(generator, "stop_ids", None) or stop_ids_for(eos)
        self.pool_stats_fn = pool_stats_fn
        self._snap_lock = threading.Lock()
        self._snap = None
        self._publish()

    # ---- prompt side -------------------------------------------------------------------------------------
    def render_and_tokenize(self, req):
        return list(self.tokenizer.encode(req.render(self.tokenizer)))

    def count_tokens(self, req):
        return len(self.render_and_tokenize(req))

    # ---- snapshot (own lock, never the gate) -------------------------------------------------------------
    def _publish(self):
        chunks = self.chunks_fn()
        hits = 0
        if self.pool_stats_fn is not None:
            try:
                hits = int(self.pool_stats_fn().get("hits", 0))
            except Exception:  # noqa: BLE001 - a stats failure must not break a request
                hits = 0
        snap = EngineSnapshot(footprint_bytes=self.ledger.expected_peak_bytes(chunks),
                              resident_bytes=self.ledger.expected_peak_bytes(chunks),
                              weights_bytes=self.ledger.trunk, digest=self.digest, pool_hits=hits,
                              governor_mode=str(getattr(self.governor, "mode", "normal")),
                              details={"chunks": chunks})
        with self._snap_lock:
            self._snap = snap

    def snapshot(self):
        with self._snap_lock:
            return self._snap

    # ---- generation --------------------------------------------------------------------------------------
    def _admit(self, ids):
        n = len(ids)
        if n >= self.context_window:
            raise EngineFailure("context_length_exceeded", f"prompt of {n} tokens does not fit the {self.context_window}-token window")
        cap = self.cap_fn() if self.cap_fn else None
        if cap is not None:
            adm = memory_admission.admit(n, self.plan, self.ledger, cap)
            if not adm.ok:
                raise EngineFailure("insufficient_memory", adm.reason)
        reused = int(self.generator.reusable(ids))
        return Admission(reused, n)

    def generate(self, ids, params, *, on_admitted, on_token, should_continue):
        gate, boundary = self.governor.gate, self.governor.boundary
        if not gate.acquire(GATE_WAIT_S):
            raise EngineFailure("insufficient_memory", "the engine is being resized")
        try:
            return self._generate_locked(ids, params, on_admitted, on_token, should_continue, boundary)
        finally:
            gate.release()
            after = getattr(self.generator, "after_gate", None)
            if after is not None:
                after()
            self._publish()

    def _generate_locked(self, ids, params, on_admitted, on_token, should_continue, boundary):
        ids = [int(t) for t in ids]
        adm = self._admit(ids)
        sample = params.sample.sanitized()
        max_tokens = min(int(sample.max_tokens), self.max_ctx - len(ids))
        if max_tokens <= 0:
            raise EngineFailure("context_length_exceeded", "no room left in the context for a completion")
        sample.max_tokens = max_tokens
        on_admitted(adm.reused_tokens)

        def keep_going():
            return should_continue() and boundary.snapshot() is None

        dec = detok.IncrementalDecoder(self.tokenizer.decode, sample.stop)
        out, state = [], {"eos": False}

        def token(tok):
            if not keep_going():
                raise _Stop
            if tok in self.stop_ids:
                state["eos"] = True
                out.append(tok)
                return
            out.append(tok)
            text = dec.push(tok)
            if text:
                on_token(text)
            if dec.stopped is not None:
                raise _Stop

        info = None
        try:
            info = self.generator.run(ids, sample, params.seed, max_tokens, token, keep_going)
        except _Stop:
            pass
        except Cancelled as e:
            if boundary.snapshot() is not None:
                raise EngineFailure("insufficient_memory", "memory pressure: " + (str(e) or "request cancelled")) from e
        if boundary.snapshot() is not None:
            raise EngineFailure("insufficient_memory", "memory pressure")
        if info is not None and info.get("nan"):
            raise EngineFailure("inference_error", "non-finite logits")
        tail = dec.finish()
        if tail:
            on_token(tail)
        gen_ids = [t for t in out if t not in self.stop_ids]
        text, _matched = detok.decode_once(self.tokenizer.decode, gen_ids, sample.stop)
        reason = "stop" if (state["eos"] or dec.stopped is not None) else "length"
        timings = (info or {}).get("timings", {})
        prefix = (info or {}).get("prefix") or {}
        return GenerationResult(reason, len(ids), len(out), float(timings.get("prefill_s", 0.0)),
                                float(sum(timings.get("decode_s", ()))), int(prefix.get("reused", adm.reused_tokens)),
                                stop_sequence=dec.stopped, text=text)
