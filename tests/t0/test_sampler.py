import itertools
import random
import subprocess
import sys
from pathlib import Path

import pytest

from slotstream_spark.engine.sampler import SampleParams, Sampler, make_logits

REF = Path(__file__).resolve().parents[2] / "third_party/slotstream/Tools/sampler_ref.py"

_GRID = list(itertools.product((0.0, 0.6, 0.8, 1.2), (1.0, 0.95, 0.5), (0, 1, 40), (0.0, 0.05),
                               (0.0, 1.5), (True, False), (256, 4096), (7, 0, 12345)))
CASES = random.Random(1729).sample(_GRID, 40)


def _mine(t, tp, tk, mp, pp, acc, vocab, seed, draws=24):
    p = SampleParams(temperature=t, top_p=tp, top_k=tk, min_p=mp, presence_penalty=pp)
    logits = make_logits(vocab, 99)
    s, gen, out = Sampler(seed), set(), []
    for _ in range(draws):
        tok = s.next(logits, p, gen)
        out.append(tok)
        if acc:
            gen.add(tok)
    return ",".join(map(str, out))


def _ref(t, tp, tk, mp, pp, acc, vocab, seed, draws=24):
    cmd = [sys.executable, str(REF), "--vocab", str(vocab), "--draws", str(draws), "--seed", str(seed),
           "--temperature", str(t), "--top-p", str(tp), "--top-k", str(tk), "--min-p", str(mp),
           "--presence-penalty", str(pp)]
    if acc:
        cmd.append("--accumulate")
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(map(str, c)))
def test_matches_reference(case):
    assert _mine(*case) == _ref(*case)


def test_case_count():
    assert len(CASES) == 40 and len(set(CASES)) == 40


def test_sanitized():
    p = SampleParams(temperature=float("nan"), top_p=0.0, top_k=-3, min_p=2.0,
                     presence_penalty=float("inf"), max_tokens=-1, stop=["a", ""]).sanitized()
    assert (p.temperature, p.top_p, p.top_k, p.min_p, p.presence_penalty) == (0.0, 1.0, 0, 1.0, 0.0)
    assert p.max_tokens == 32_768 and p.stop == ["a"]
    assert SampleParams(max_tokens=10**9).sanitized().max_tokens == 32_768


def test_seed_zero_is_deadbeef():
    assert Sampler(0).state == 0xDEADBEEF and Sampler(5).state == 5
