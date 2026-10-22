"""Control paths of the prefill driver (step 70/71): the pass budget, the cancel hook, the D5 budget. Fakes only."""
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cuda
torch = pytest.importorskip("torch")

from slotstream_spark.engine import pressure                                                   # noqa: E402
from slotstream_spark.engine.prefill import (PassBudget, PassBudgetExceeded, PrefillCancelled, PrefillDriver,  # noqa: E402
                                             PrefillPlan, d5_budget)
from slotstream_spark.memory.ledger import Ledger                                              # noqa: E402


def _driver():
    model = SimpleNamespace(max_ctx=9216, NL=48, seg=SimpleNamespace())
    moe = SimpleNamespace(stats=SimpleNamespace(as_dict=lambda: {"bytes_requested": 0, "records_read": 0,
                                                                  "admitted": 0, "max_launch_rows": 0}))
    env = {"qk_product": 32833536, "max_rows_per_launch": 81920, "bytes_per_pass_token": 122892.0,
           "bytes_per_pair": 0.0}
    return PrefillDriver(model, moe, env)


def test_pass_budget_check():
    b = PassBudget(10)
    with pytest.raises(PassBudgetExceeded):
        b.check(11)
    assert b.max_seen == 11 and b.calls == 1
    b.check(10)
    assert b.calls == 2


def test_budget_exceeded_before_any_layer():
    d = _driver()
    state = SimpleNamespace(pos=0, tokens=[])
    plan = PrefillPlan(prefill_pass=256, read_scope=0)
    with pytest.raises(PassBudgetExceeded):          # an AttributeError would mean the check ran after seg_begin
        d.run(list(range(300)), state, plan=plan, control=PassBudget(1), should_continue=lambda: True)


def test_cancel_before_the_first_pass():
    d = _driver()
    state = SimpleNamespace(pos=0, tokens=[])
    plan = PrefillPlan(prefill_pass=256, read_scope=0)
    with pytest.raises(PrefillCancelled) as ei:
        d.run(list(range(300)), state, plan=plan, control=PassBudget(10 ** 12), should_continue=lambda: False)
    assert isinstance(ei.value, pressure.Cancelled)


def test_d5_budget_is_pass_plus_read_scope():
    led = Ledger("D5")
    assert d5_budget().limit == led.pass_bytes + led.read_scope


def test_cached_prefill_admits_only_on_the_last_run(monkeypatch):
    from slotstream_spark.engine import generate as gen
    from slotstream_spark.engine import prefix_cache as pc

    admits = []

    class Fake:
        env = {"qk_product": 32833536}
        P = 32833536

        def run(self, ids, st, *, plan, control, should_continue, admit):
            admits.append(admit)
            lg = SimpleNamespace(numel=lambda: 1, element_size=lambda: 4)
            lg.clone = lambda: lg
            return SimpleNamespace(last_logits=lg, passes=[], seconds=0.0, bytes_requested=0, records_read=0,
                                   admitted=5 if admit else 0, scopes=[])

    cache = SimpleNamespace(take=lambda *a, **k: pc.Miss("miss"), decision="miss",
                            would_store_checkpoint=lambda *a, **k: True, store_checkpoint=lambda *a, **k: True)
    monkeypatch.setattr(pc, "snapshot_state", lambda st: SimpleNamespace(nbytes=1))
    monkeypatch.setattr(pc, "restore_state", lambda st, snap: None)
    monkeypatch.setattr(pc, "snapshot_end", lambda *a, **k: 4096)
    model = SimpleNamespace(st=SimpleNamespace(pos=0, tokens=[], reset_inplace=lambda: None))
    info = {}
    gen._prefill_cached(model, list(range(5000)), Fake(), PrefillPlan(prefill_pass=4096, read_scope=0),
                        PassBudget(10 ** 12), lambda: True, cache, "key", 4096, info)
    assert admits == [False, True]
    assert info["snapshot_admitted"] == 0 and info["admitted"] == 5
