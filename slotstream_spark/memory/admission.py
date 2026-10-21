"""Admission before allocation: a request that does not fit is refused with 503 insufficient_memory."""
from dataclasses import dataclass

INSUFFICIENT = {"error": {"type": "insufficient_memory"}}


@dataclass(frozen=True)
class Admission:
    ok: bool
    status: int = 200
    body: dict = None
    reason: str = ""


def admit(prompt_tokens, plan, ledger, cap):
    """`plan`: planner.Plan (chunks, window); `ledger`: memory.ledger.Ledger; `cap`: bytes the engine may hold.

    Refused when the prompt does not fit the planned window or when the planned pool plus the fixed bytes exceed
    the cap (the cap shrank since the plan was made)."""
    if int(prompt_tokens) < 0:
        raise ValueError("prompt_tokens must be >= 0")
    if int(prompt_tokens) >= int(plan.window):
        return Admission(False, 503, INSUFFICIENT, f"prompt {prompt_tokens} tokens does not fit window {plan.window}")
    need = ledger.expected_peak_bytes(plan.chunks)
    if need > int(cap):
        return Admission(False, 503, INSUFFICIENT, f"expected peak {need} B over the cap {int(cap)} B")
    return Admission(True)
