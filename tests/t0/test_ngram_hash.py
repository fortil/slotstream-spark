import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _ngram_pyref as ref  # noqa: E402

from slotstream_spark.model.ngram_hash import NgramHasher  # noqa: E402

EOS = 248044
PARITY = Path(__file__).resolve().parents[2] / "third_party/slotstream/bench/parity31"


@pytest.fixture(scope="module")
def hasher():
    return NgramHasher.from_seed()


@pytest.fixture(scope="module")
def pyref_params():
    return ref.build()


def test_geometry(hasher):
    assert hasher.rows_per_shard == 2_500_012
    assert 128 * hasher.rows_per_shard == 320_001_536
    assert hasher.padded_rows == 320_001_536
    sh, row = hasher.shard_row(2_500_012 * 3 + 5)
    assert (sh, row) == (3, 5)


def test_params_match_pyref(hasher, pyref_params):
    m, s, o = pyref_params
    assert hasher.mult.tolist() == m and hasher.sizes.tolist() == s and hasher.offs.tolist() == o


def test_parity31_ids(hasher):
    toks = [int(x) for x in (PARITY / "tokens.txt").read_text().strip().split(",")]
    assert toks == [9707, 11, 1246, 525, 498, 30]
    want = np.array([[int(x) for x in line.split(",")]
                     for line in (PARITY / "ngram_ids.txt").read_text().strip().splitlines()], dtype=np.int64)
    got = hasher.ids([EOS, EOS], toks)
    assert got.shape == want.shape == (6, 16)
    assert np.array_equal(got, want)


def _incremental(h, seq):
    hist, rows = [], []
    for t in seq:
        prev = hist[-2:]
        prev = [EOS] * (2 - len(prev)) + prev
        rows.append(h.ids(prev, [t])[0])
        hist.append(t)
    return np.array(rows)


def test_random_sequences_with_eos(hasher, pyref_params):
    rng = np.random.default_rng(0)
    m, s, o = pyref_params
    for _ in range(20):
        T = 40
        seq = rng.integers(0, 248320, size=T)
        seq[rng.integers(0, T, size=3)] = EOS
        seq = seq.tolist()
        full = hasher.ids([EOS, EOS], seq)
        assert np.array_equal(full, _incremental(hasher, seq))
        assert full.tolist() == ref.ngram_ids([EOS, EOS], seq, m, s, o, EOS)


def test_long_sequence_matches_pyref(hasher, pyref_params):
    m, s, o = pyref_params
    rng = random.Random(0)
    seq = [rng.randrange(0, 248320) for _ in range(10_000)]
    for i in rng.sample(range(10_000), 20):
        seq[i] = EOS
    full = hasher.ids([EOS, EOS], seq)
    want = ref.ngram_ids([EOS, EOS], seq, m, s, o, EOS)
    assert full.tolist() == want
    assert np.array_equal(full, _incremental(hasher, seq))


def test_bad_context_length(hasher):
    with pytest.raises(ValueError):
        hasher.ids([EOS], [1])


def test_from_buffers_equivalent(hasher):
    h2 = NgramHasher.from_buffers(hasher.mult, hasher.sizes, hasher.offs, EOS)
    assert np.array_equal(h2.ids([EOS, EOS], [5, 6, 7]), hasher.ids([EOS, EOS], [5, 6, 7]))
