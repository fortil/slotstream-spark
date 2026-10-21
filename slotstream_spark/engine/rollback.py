# SPDX-License-Identifier: MIT
# Port of Generate.swift and Model.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""Rollback of a verify round (plan step 76; Generate.swift:1430-1651, Model.swift:1384-1404).

A round runs `rows` tokens (the pending one and the drafts) through the 3-row graphs with recording on. The tokens kept are the
first `keep`: the recurrent state (GDN, conv, PLE conv) comes back from the record, the position and the token list are cut
to the checkpoint plus `keep` (the n-gram context is the last two entries of the token list, so cutting it rebuilds it), and
the attention caches are trimmed by position: the QSA ring holds a group (4) plus up to 3 speculative rows in whole groups,
and a compressed row completed by a rejected token is rewritten when the real token at that position is processed, so
nothing else has to be copied back.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Checkpoint:
    pos: int
    ntokens: int


def checkpoint(model):
    st = model.st
    return Checkpoint(int(st.pos), len(st.tokens))


def rollback(model, record, ck, keep, rows=3):
    """Keep the first `keep` (1..rows) rows of the round that started at `ck`; releases the round's record."""
    if not 1 <= keep <= rows:
        raise ValueError(f"keep must be in 1..{rows}, got {keep}")
    st = model.st
    if st.pos != ck.pos + rows:
        raise RuntimeError(f"rollback: state at {st.pos}, expected the checkpoint {ck.pos} + {rows} rows")
    if keep == rows:
        record.release()
    else:
        record.restore(st, keep)
    st.pos = ck.pos + keep
    del st.tokens[ck.ntokens + keep:]
    for c in st.attn.values():
        c.pos = st.pos
        c.last = None
