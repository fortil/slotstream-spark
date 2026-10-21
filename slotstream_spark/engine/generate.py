"""Generation over token ids: prefill the prompt, then one decode step per token.

It takes ids and returns ids; text prompts, the tokenizer and the chat template arrive in step 57.
Greedy is temperature 0 through the same Sampler as sampled generation, so there is one code path.
"""
import json
import os
import time

import numpy as np

from .sampler import SampleParams, Sampler


SWEEP_MIN_TOKENS = 256      # prompts with at least this many tokens left go through the sweep prefill driver


HEAD_CHUNK = 256            # rows per MTP head step while the head consumes a prompt


def is_eos(tok, eos):
    """`eos` is None, one token id, or a set of ids (generation_config.json lists both <|im_end|> and <|endoftext|>)."""
    if eos is None:
        return False
    if isinstance(eos, (set, frozenset, list, tuple)):
        return tok in eos
    return tok == eos


def _consume_in_pieces(head, ids, hyp):
    for a in range(0, len(ids), HEAD_CHUNK):
        head.consume(ids[a:a + HEAD_CHUNK], hyp[a:a + HEAD_CHUNK])


def _prefill(model, ids, driver, plan, control, should_continue, head=None):
    """Logits of the last prompt position. With a `driver` (engine.prefill.PrefillDriver) a prompt of >= 256 tokens left
    is swept; shorter ones, and runs without a driver, take the eager path. With a `head` (MTP) the head consumes the
    pre-mixer stream of every prompt position as the passes produce it."""
    if head is not None:
        head.reset()
    if driver is not None and len(ids) >= SWEEP_MIN_TOKENS:
        from .prefill import PrefillPlan, d5_budget
        res = driver.run(ids, model.st, plan=plan or PrefillPlan.from_envelope(driver.env),
                         control=control or d5_budget(), should_continue=should_continue or (lambda: True),
                         on_multi=None if head is None else (lambda idt, hyp: _consume_in_pieces(head, idt, hyp)))
        return res.last_logits, res
    if head is None:
        return model.prefill(ids), None
    model.multi_sink = lambda idt, hyp: _consume_in_pieces(head, idt, hyp)
    try:
        return model.prefill(ids), None
    finally:
        model.multi_sink = None


def _prefill_cached(model, ids, driver, plan, control, should_continue, cache, key, snap_target, info, persistent=None):
    """Prefill through the prefix cache (step 71). A hit restores the retained state in place and reads only the new
    tokens (always through the driver, so the passes are the ones a fresh read runs from that boundary); a miss reads
    everything. A snapshot is taken at the last pass end at or before `snap_target` and the whole prompt is stored with
    its raw logits. With a `persistent` tier (engine.prefix_cache.PersistentTier) the disk is consulted only for a state
    longer than the memory hit's (after the miss eviction `take` already did); a restored state is also retained in
    memory as fresh-equivalent, and the snapshot at the pass end is staged for a background write.
    -> (logits tensor, PrefillResult | None)."""
    from . import prefill_schedule as ps
    from . import prefix_cache as pc
    from .prefill import PrefillPlan, d5_budget
    plan = plan or PrefillPlan.from_envelope(driver.env)
    control = control or d5_budget()
    should_continue = should_continue or (lambda: True)
    n = len(ids)
    boundaries = ps.resume_boundaries(n, plan.prefill_pass, True, driver.P)
    hit = cache.take(ids, key, boundaries, reserve_tokens=info.get("reserve_tokens"))
    info["decision"] = cache.decision
    st = model.st
    base, logits = 0, None
    restored = False
    if persistent is not None:
        mem_len = hit.reused if isinstance(hit, pc.Hit) else 0
        entry = persistent.match(ids, mem_len, boundaries)
        if entry is not None:
            try:
                base = persistent.restore_into(st, entry)
                restored = True
                info["decision"] = f"restored {base} tokens from disk"
                info["disk_reused"] = base
            except Exception as e:  # noqa: BLE001 - a failed restore falls back to memory or a cold read
                info["persist_error"] = f"{type(e).__name__}: {e}"
                base = 0
    if restored:
        # Each snapshot below is a device copy of the used pages (about 2 GB at 78K tokens); the cache is asked first and
        # the copy is skipped when it would be refused anyway.
        if cache.would_store_checkpoint(ids[:base], fresh_equivalent=True, produced_key=key,
                                        reserve_tokens=info.get("reserve_tokens")):
            snap0 = pc.snapshot_state(st)
            info["disk_retained"] = cache.store_checkpoint(
                snap0, ids[:base], snap0.nbytes, prompt_key=None, fresh_equivalent=True, produced_key=key,
                reserve_tokens=info.get("reserve_tokens"))
        else:
            info["disk_retained"] = False
    elif isinstance(hit, pc.Hit):
        pc.restore_state(st, hit.state)
        base = hit.reused
        if hit.logits is not None:
            logits = hit.logits.clone()
    else:
        st.reset_inplace()
    info["reused"] = base
    results = []
    if logits is None:
        end = pc.snapshot_end(ids, snap_target, base, plan.prefill_pass, boundaries, driver.P)
        if end is not None:
            r = driver.run(ids[base:end], st, plan=plan, control=control, should_continue=should_continue, admit=False)
            results.append(r)
            info["snapshot_admitted"] = r.admitted
            info["snapshot_end"] = end
            want = cache.would_store_checkpoint(ids[:end], fresh_equivalent=True, produced_key=key,
                                               reserve_tokens=info.get("reserve_tokens"))
            if want or persistent is not None:                   # the disk tier stages the copy even when memory refuses it
                snap = pc.snapshot_state(st)
                info["snapshot_stored"] = cache.store_checkpoint(
                    snap, ids[:end], snap.nbytes, prompt_key=None, fresh_equivalent=True, produced_key=key,
                    reserve_tokens=info.get("reserve_tokens")) if want else False
                if persistent is not None:
                    info["persist_staged"] = persistent.stage(snap, ids[:end], shared=True)
            else:
                info["snapshot_stored"] = False
            base = end
        r = driver.run(ids[base:], st, plan=plan, control=control, should_continue=should_continue, admit=True)
        results.append(r)
        info["admitted"] = r.admitted
        logits = r.last_logits
        lg_bytes = logits.numel() * logits.element_size()
        if cache.would_store_checkpoint(ids, logits=True, logits_bytes=lg_bytes, prompt_key=key, fresh_equivalent=True,
                                        produced_key=key, reserve_tokens=info.get("reserve_tokens")):
            snap = pc.snapshot_state(st)
            lg = logits.clone()
            info["prompt_stored"] = cache.store_checkpoint(
                snap, ids, snap.nbytes, logits=lg, logits_bytes=lg_bytes, prompt_key=key,
                fresh_equivalent=True, produced_key=key, reserve_tokens=info.get("reserve_tokens"))
        else:
            info["prompt_stored"] = False
    pre = results[-1] if results else None
    if pre is not None:
        pre.passes = [x for r in results for x in r.passes]
        pre.seconds = sum(r.seconds for r in results)
        pre.bytes_requested = sum(r.bytes_requested for r in results)
        pre.records_read = sum(r.records_read for r in results)
        pre.admitted = sum(r.admitted for r in results)
    return logits, pre


def _mtp_decode(model, g1, g3, head, sampler, p, seen, logits, eos, on_token, max_tokens, draft_fn):
    """Self-speculative greedy/sampled decode (Generate.swift speculativeDecode, depth fixed at mtp.DEPTH = 2).

    The first token comes off the prefill logits like the plain loop's first iteration. A round drafts 2 tokens with the head
    (on a fork of its cache), verifies pending + drafts in one pass of the 3-row graphs with recording on, samples the rows in
    order with the plain loop's semantics (same RNG order and presence-penalty evolution, a draw only for a token the plain loop
    would have sampled), keeps the first 1 + good rows (rollback restores the recorded state), and feeds the kept tokens and
    their pre-mixer rows to the head. Near the context end (fewer than 2 drafts fit) it switches for good to one-row steps.
    -> (out, nan, round_seconds, stats)."""
    from . import mtp as mtp_mod
    from . import rollback as rb
    out, step_s, nan = [], [], False
    st = {"rounds": 0, "plain_steps": 0, "drafted": 0, "accepted": 0, "attempts_by_depth": [0, 0],
          "accepted_by_depth": [0, 0], "emitted_by_round": [], "verify_rows": 0}
    if not np.isfinite(logits).all():
        return out, True, step_s, st
    tok = sampler.next(logits, p, seen)
    out.append(tok)
    seen.add(tok)
    if on_token:
        on_token(tok)
    if is_eos(tok, eos) or len(out) == max_tokens:
        return out, nan, step_s, st
    pending, use_mtp = tok, True
    while len(out) < max_tokens:
        if use_mtp:
            depth = mtp_mod.draft_depth(model.max_ctx, model.st.pos)
            if head.state.pos + 1 + mtp_mod.DEPTH > head.state.max_ctx:
                depth = 0
            use_mtp = depth >= mtp_mod.DEPTH
        t1 = time.perf_counter()
        if not use_mtp:
            logits = g1.step([pending])[0].numpy()
            step_s.append(time.perf_counter() - t1)
            st["plain_steps"] += 1
            if not np.isfinite(logits).all():
                nan = True
                break
            tok = sampler.next(logits, p, seen)
            out.append(tok)
            seen.add(tok)
            if on_token:
                on_token(tok)
            if is_eos(tok, eos) or len(out) == max_tokens:
                break
            pending = tok
            continue
        drafts = [int(d) for d in (draft_fn(pending, mtp_mod.DEPTH, len(out)) if draft_fn else
                                   head.draft(pending, mtp_mod.DEPTH))]
        ck = rb.checkpoint(model)
        vlog = g3.step([pending] + drafts)
        multi = g3.last_multi
        vlog = vlog.numpy()
        st["rounds"] += 1
        st["drafted"] += len(drafts)
        st["verify_rows"] += 1 + len(drafts)
        good, nextp, done, before = 0, None, False, len(out)
        for i in range(1 + len(drafts)):
            if not np.isfinite(vlog[i]).all():
                nan, done = True, True
                break
            tok = sampler.next(vlog[i], p, seen)
            out.append(tok)
            seen.add(tok)
            if on_token:
                on_token(tok)
            if is_eos(tok, eos) or len(out) == max_tokens:
                done = True
                break
            if i < len(drafts):
                st["attempts_by_depth"][i] += 1
                if tok == drafts[i]:
                    st["accepted_by_depth"][i] += 1
                    good += 1
                    continue
            nextp = tok
            break
        st["accepted"] += good
        st["emitted_by_round"].append(len(out) - before)
        keep = 1 + good
        rb.rollback(model, g3.record, ck, keep)
        head.consume([pending] + drafts[:good], multi[:keep])
        step_s.append(time.perf_counter() - t1)
        if done:
            break
        pending = nextp
    return out, nan, step_s, st


def generate(model, ids, max_tokens=16, greedy=True, params=None, seed=0, eos=None, on_token=None, driver=None,
             plan=None, control=None, should_continue=None, prefix_cache=None, key=None, snapshot_target=None,
             keep_decode_state=True, graphs=None, mtp=None, graphs3=None, mtp_draft=None, persistent=None):
    """-> dict(tokens, first_logits (np.float32 [V], last prompt position), nan (bool), timings, prefill (the
    PrefillResult of a swept prompt, else None), prefix (cache info: decision, reused, snapshot_end, ... or None)).

    Stops after `max_tokens` new tokens or at `eos` (an id or a set of ids; the eos token is kept in the output).

    With `prefix_cache` (engine.prefix_cache.PrefixCache, needs `driver` and a `CheckpointKey`) the prompt resumes from
    the longest retained state it extends; `snapshot_target` is the index of the prompt's last `<|im_start|>`. The
    decode-end state is stored with fresh_equivalent=False (it can never be continued) unless `keep_decode_state` is
    False.

    With `graphs` (runtime.graphs.GraphSet of 1 row) the decode steps replay the captured graphs instead of the eager
    `model.decode`; if its `lookahead` is a Scheduler (step 74) the speculative tickets of the request are all returned when
    the request ends and the result carries `lookahead` (the scheduler's counters, the pool's prefetch counters and the
    number of tickets that were still outstanding at the end, expected 0).

    With `mtp` (engine.mtp.MTPHead, resident) and `graphs3` (GraphSet of 3 rows built with record=True, multi=True; `graphs`
    is the 1-row set used near the context end, or None for the eager step) the decode is the self-speculative loop of
    `_mtp_decode`: the output equals the plain output bit for bit and the result carries `mtp` (rounds, drafted and accepted
    tokens, attempts and acceptances per depth, leaked record rounds). `mtp_draft(pending, k, n_out) -> [k ids]` replaces the
    head's drafts (tests force wrong drafts). It needs the sweep-free path of the prefix cache: no `prefix_cache`.

    With `graphs` and `[f8] token_graph` on (or SLOTSTREAM_TOKEN_GRAPH=on), the plain decode loop replays the single per-token
    graph (runtime.token_graph.TokenGraph) instead of the 49 segment graphs; the tokens are bit-identical.

    With `persistent` (engine.prefix_cache.PersistentTier, needs `prefix_cache`) the disk tier is consulted on a miss or a
    shorter memory hit and the pass-end snapshot is staged; the caller calls `persistent.release()` once the gate is
    released so the write runs after it."""
    if mtp is not None:
        if graphs is None or graphs3 is None or graphs3.record is None or not graphs3.multi_on:
            raise ValueError("mtp needs graphs (1 row) and graphs3: a 3-row GraphSet built with record=True and multi=True")
        if prefix_cache is not None:
            raise ValueError("mtp does not run with the prefix cache (the head's cache is not retained)")
    p = SampleParams(temperature=0.0) if greedy else (params or SampleParams())
    p.max_tokens = max_tokens
    sampler = Sampler(seed)
    t0 = time.perf_counter()
    info = None
    if prefix_cache is not None:
        if driver is None or key is None:
            raise ValueError("a prefix cache needs the sweep driver and a CheckpointKey")
        info = {"reserve_tokens": len(ids) + max_tokens}
        logits, pre = _prefill_cached(model, ids, driver, plan, control, should_continue, prefix_cache, key,
                                      snapshot_target, info, persistent)
    else:
        logits, pre = _prefill(model, ids, driver, plan, control, should_continue, head=mtp)
    logits = logits.cpu().numpy()
    t_prefill = time.perf_counter() - t0
    first = logits.copy()
    out, seen, nan = [], set(), False
    step_s = []
    mtp_info = None
    tg = None
    if graphs is not None and mtp is None:      # step 91: the single per-token graph, when [f8] token_graph / the env turn it on
        from slotstream_spark.runtime import token_graph as tgm
        tg = tgm.for_graphs(graphs)
    if mtp is not None:
        if graphs is not None and graphs.lookahead is not None:
            graphs3.lookahead = graphs.lookahead
        if len(ids) != mtp.consumed:
            raise RuntimeError(f"the head consumed {mtp.consumed} prompt tokens, expected {len(ids)}")
        out, nan, step_s, mtp_info = _mtp_decode(model, graphs, graphs3, mtp, sampler, p, seen, logits, eos, on_token,
                                                 max_tokens, mtp_draft)
        mtp_info["leaked_records"] = graphs3.record.outstanding
        graphs3.lookahead = None
        graphs3.detach()
    for _ in range(0 if mtp is not None else max_tokens):
        if not np.isfinite(logits).all():
            nan = True
            break
        tok = sampler.next(logits, p, seen)
        out.append(tok)
        seen.add(tok)
        if on_token:
            on_token(tok)
        if is_eos(tok, eos) or len(out) == max_tokens:
            break
        t1 = time.perf_counter()
        logits = ((tg or graphs).step([tok])[0].numpy() if graphs is not None else model.decode(tok).cpu().numpy())
        step_s.append(time.perf_counter() - t1)
    la = None
    tg_stats = None
    if tg is not None:
        tg_stats = tg.stats()
        stats_path = os.environ.get("SLOTSTREAM_TOKEN_GRAPH_STATS")      # the F8 A/B and soak arms append their counters here
        if stats_path:
            with open(stats_path, "a") as f:             # one line per request that used the graph
                f.write(json.dumps(dict(tg_stats, pid=os.getpid(), tokens=len(out)), sort_keys=True) + "\n")
    if graphs is not None:
        sched = graphs.lookahead
        if sched is not None:
            la = {"scheduler": sched.stats(), "pool": sched.pool.prefetch_stats(), "leaked_at_end": sched.end_request()}
        graphs.detach()                      # hand the eager path (next prefill) a consistent generation counter
    if prefix_cache is not None and keep_decode_state and out and not nan:
        from . import prefix_cache as pc
        if prefix_cache.would_store(model.st.tokens):             # skip the copy of a state the cache would evict at once
            snap = pc.snapshot_state(model.st)
            info["decode_end"] = "stored" if prefix_cache.store(snap, snap.tokens, snap.nbytes, fresh_equivalent=False,
                                                                 key=key, input_tokens=len(ids)) else "refused"
        else:
            info["decode_end"] = "skipped"
    return {"tokens": out, "first_logits": first, "nan": nan,
            "timings": {"prefill_s": t_prefill, "decode_s": step_s}, "prefill": pre, "prefix": info, "lookahead": la, "mtp": mtp_info,
            "token_graph": tg_stats}
