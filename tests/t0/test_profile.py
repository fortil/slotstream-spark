"""PhaseClock arithmetic, bench/partition.py and `records.py partition` on synthetic data (Mac, no GPU)."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for d in (ROOT / "tools" / "gates", ROOT / "bench"):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

import partition as pt  # noqa: E402
import records  # noqa: E402

from slotstream_spark.runtime.profile import PHASES, PhaseClock  # noqa: E402


def timing(read=1000, tail=200, hits=9, misses=1):
    return {"read_wall_ns": read, "copy_tail_ns": tail, "hits": hits, "misses": misses, "total_ns": read + 50}


def test_phases_match_the_partition_module():
    assert tuple(PHASES) == tuple(pt.PHASES)


def test_token_sums_to_wall_and_other_is_the_rest():
    c = PhaseClock(nvtx=False)
    c.begin()
    with c.phase("sample"):
        sum(range(1000))
    c.layer(wait_ns=500, ensure_ns=3000, launch_ns=700, timing=timing())
    t = c.end()
    assert sum(t["phases"].values()) + t["other_ns"] == t["wall_ns"]
    assert t["phases"]["gpu_wait"] == 500 and t["phases"]["launch"] == 700
    # the ensure wall 3000 = hits 2000 + (read 1000 = miss-read 800 + copy tail 200)
    assert t["phases"]["ensure_hits"] == 2000
    assert t["phases"]["expert_miss_read"] == 800 and t["phases"]["expert_copy_tail"] == 200
    assert c.read_wall_ns == 1000 and (c.hits, c.misses) == (9, 1)


def test_nested_phase_raises_instead_of_double_counting():
    c = PhaseClock(nvtx=False)
    c.begin()
    with c.phase("embed_read"):
        with pytest.raises(RuntimeError, match="opened inside"):
            with c.phase("launch"):
                pass


def test_unknown_phase_and_outside_a_token():
    c = PhaseClock(nvtx=False)
    with pytest.raises(KeyError):
        c.phase("nope")
    with c.phase("sample"):          # outside a token: ignored
        pass
    c.layer(1, 2, 3, timing())
    assert c.tokens == [] and c.tau_ns == []


def test_all_hit_layers_feed_tau_and_cancel_drops_the_token():
    c = PhaseClock(nvtx=False)
    c.begin()
    c.layer(10, 400, 300, timing(read=0, tail=0, hits=10, misses=0))
    c.layer(10, 9000, 300, timing(misses=2))
    assert c.tau_ns == [700]
    c.cancel()
    assert c.tokens == []


def test_read_longer_than_the_ensure_wall_is_clamped_not_negative():
    c = PhaseClock(nvtx=False)
    c.begin()
    c.layer(0, 100, 0, timing(read=500, tail=900))
    t = c.end()
    assert t["phases"]["ensure_hits"] == 0 and t["phases"]["expert_miss_read"] == 0
    assert t["phases"]["expert_copy_tail"] == 100


def entry(wall=1_000_000, other=50_000, tokens=32):
    ph = dict.fromkeys(PHASES, 0)
    ph.update(sample=20_000, ngram_hash=30_000, embed_read=10_000, ngram_wait=5_000, launch=200_000, gpu_wait=400_000,
              ensure_hits=100_000, expert_miss_read=120_000, expert_copy_tail=25_000)
    ph["gpu_wait"] += wall - other - sum(ph.values())
    return {"prompt_tokens": 100, "new_tokens": tokens + 1, "prefill_s": 1.5, "tok_s": 20.0, "tokens": [1],
            "tokens_profiled": tokens, "wall_ns": wall, "phases_ns": ph, "other_ns": other, "read_wall_ns": 145_000,
            "hits": 90, "misses": 10, "hit_rate": 0.9, "per_token_wall_ns": [wall // tokens] * tokens, "nvtx": True}


def result(**kw):
    e = entry(**kw)
    return {"mode": "D5", "slots": 19000, "gdn": "fused", "profile": True, "ok": True, "tokens_requested": 256,
            "prompts": {"short": e, "2k": copy.deepcopy(e)}, "versions": {"torch": "2.13"},
            "tau": {"p50_us": 14.0, "p99_us": 20.0, "steps": 2000}}


def run(tmp_path, res, *extra):
    p = tmp_path / "result.json"
    p.write_text(json.dumps(res))
    return records.main(["partition", str(p), *extra])


def test_partition_table_sums_to_100(tmp_path):
    rows = pt.table(entry())
    assert abs(rows[-1][2] - 1.0) < 1e-12 and rows[-1][1] == 1_000_000
    assert abs(sum(r[2] for r in rows[:-1]) - 1.0) < 1e-12


def test_records_pass_and_write(tmp_path, capsys):
    dest = tmp_path / "partition-d5.md"
    assert run(tmp_path, result(), "--write", str(dest), "--note", "provenance line") == 0
    text = dest.read_text()
    assert text.count("\nVerdict: PASS") == 1 and "provenance line" in text
    assert "| other |" in text and "| expert_copy_tail |" in text


def test_other_over_ten_percent_fails(tmp_path, capsys):
    assert run(tmp_path, result(other=150_000)) == 1
    assert "FAIL" in capsys.readouterr().out


def test_overlap_negative_other_fails(tmp_path, capsys):
    res = result()
    res["prompts"]["short"]["phases_ns"]["launch"] += 200_000          # Σ phases > wall
    assert run(tmp_path, res) == 1
    assert "do not overlap" in capsys.readouterr().out


def test_copy_folded_into_io_fails(tmp_path, capsys):
    res = result()
    res["prompts"]["2k"]["phases_ns"]["expert_miss_read"] += 5_000     # no longer equals the read wall
    res["prompts"]["2k"]["phases_ns"]["gpu_wait"] -= 5_000
    assert run(tmp_path, res) == 1
    assert "t_copy separate" in capsys.readouterr().out


def test_tau_over_150_prints_decision(tmp_path, capsys):
    res = result()
    res["tau"]["p50_us"] = 151.0
    assert run(tmp_path, res) == 1
    out = capsys.readouterr().out
    assert "DECISION: tau=151.0 us > 150 us" in out


def test_too_few_tokens_and_unprofiled_fail(tmp_path):
    assert run(tmp_path, result(tokens=8)) == 1
    res = result()
    res["profile"] = False
    assert run(tmp_path, res) == 1


def test_unreadable_input_is_exit_3(tmp_path):
    assert run(tmp_path, {"profile": True}) == 3                       # no "prompts": cannot evaluate
