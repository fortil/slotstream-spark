import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slotstream_spark.memory import footprint as fp  # noqa: E402

G = 1 << 30


def S(tag, rss, hwm, nv, pool=0):
    return fp.Sample(tag, rss, hwm, nv, 0, {"pool": pool})


def test_both_counters():
    s = S("x", 5 * G, 5 * G, 4 * G, 2 * G)
    f = {"pool": (True, True)}
    assert fp.nonpool_bytes(s, f) == 9 * G - 4 * G
    assert fp.once_bytes(s, f) == 9 * G - 2 * G


def test_one_counter():
    s = S("x", 5 * G, 5 * G, 4 * G, 2 * G)
    f = {"pool": (True, False)}
    assert fp.nonpool_bytes(s, f) == 9 * G - 2 * G
    assert fp.once_bytes(s, f) == 9 * G


def test_shows_in():
    a = S("a", 0, 0, 0, 0)
    assert fp.shows_in(a, S("b", int(0.95 * G), 0, int(0.5 * G), G), "pool") == (True, False)
    with pytest.raises(ValueError):
        fp.shows_in(a, a, "pool")


def test_peak_hwm_bound_wins():
    f = {"pool": (True, True)}
    ss = [S("base", G, G, G, 0), S("c", 4 * G, 6 * G, 4 * G, 3 * G)]
    p = fp.peak(ss, f)
    assert p.argmax_tag == "vmhwm" and p.nonpool_bytes == p.hwm_bound_bytes == 6 * G - 3 * G + 1 * G
    assert p.unified_hwm == {"pool": 3 * G} and p.once_bytes == p.nonpool_bytes + 3 * G


def test_peak_sample_wins():
    f = {"pool": (True, True)}
    p = fp.peak([S("big", 5 * G, 5 * G, 2 * G, 0), S("small", 3 * G, 5 * G, 3 * G, G)], f)
    assert p.argmax_tag == "big" and p.nonpool_bytes == 7 * G
    assert all(isinstance(v, int) for v in (p.nonpool_bytes, p.once_bytes, p.hwm_bound_bytes))


def test_missing_flag():
    with pytest.raises(KeyError):
        fp.peak([S("a", G, G, G, G)], {})
    with pytest.raises(KeyError):
        fp.nonpool_bytes(S("a", G, G, G, G), {})
    with pytest.raises(ValueError):
        fp.peak([], {})


def test_roundtrip():
    f = {"pool": (True, True)}
    ss = [S("a", G, G, G, 0), S("b", 3 * G, 3 * G, 3 * G, G)]
    d = json.loads(json.dumps([s.__dict__ for s in ss]))
    assert [fp.sample_from_dict(x) for x in d] == ss
    assert fp.flags_from_dict(json.loads(json.dumps(f))) == f
    p = fp.peak(ss, f)
    assert fp.peak_to_dict(p) == json.loads(json.dumps(fp.peak_to_dict(p)))
