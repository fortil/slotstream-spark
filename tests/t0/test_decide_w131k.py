"""The context-window campaign judge (tools/decide.py w131k-campaign, w131k-envelope) on synthetic cells."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import db_check  # noqa: E402
import decide as dm  # noqa: E402
from slotstream_spark.memory import footprint as fp  # noqa: E402
from slotstream_spark.memory.ledger import Ledger  # noqa: E402

GIB = 1 << 30
WINDOW = 131072
CHUNKS = 40
POOL = CHUNKS * GIB


def with_peak(cell, nonpool=20_000_000_000, window=WINDOW, chunks=CHUNKS):
    pool = chunks * GIB
    cell["flags"] = {"pool": [True, True]}
    cell["footprint_samples"] = [{"tag": "0", "vmrss": nonpool + pool, "vmhwm": nonpool + pool, "nvml": nonpool,
                                  "pinned": 0, "unified": {"pool": pool}}]
    pk = fp.peak([fp.sample_from_dict(d) for d in cell["footprint_samples"]], fp.flags_from_dict(cell["flags"]))
    cell["peak"] = fp.peak_to_dict(pk)
    cell["chunks"] = chunks
    cell["memavailable_drop_bytes"] = nonpool + pool
    cell["mode"] = "D5"
    cell["window"] = window
    return cell


def decode_cell(vals=(28.0, 27.5, 29.0, 26.9, 28.3), **kw):
    return with_peak({"cell": "decode-8k", "kind": "decode", "ok": True,
                      "samples": [{"decode_tps": v, "mtp_policy": "resident", "pool_prefill": {"exhausted": 0}} for v in vals]}, **kw)


def ttft_cell(name, tokens, ttft, tps, reused=0, follow=1.0, follow_reused=None, prompt_tokens=None, **kw):
    pt = prompt_tokens or tokens
    fr = pt - 1200 if follow_reused is None else follow_reused
    samples = [{"sample": i, "cold": {"ttft_s": ttft, "prefill_s": pt / tps, "prefill_tps": tps, "reused_tokens": reused,
                                      "prompt_tokens": pt},
                "followup": {"ttft_s": follow, "reused_tokens": fr, "new_tokens": pt + 60 - fr}} for i in range(5)]
    return with_peak({"cell": name, "kind": "ttft-cold", "tokens": tokens, "prompt_tokens": pt, "ok": True, "samples": samples,
                      "medians": {"ttft_s": ttft, "prefill_s": pt / tps, "prefill_tps": tps, "followup_ttft_s": follow}}, **kw)


def cells(decode=(28.0, 27.5, 29.0, 26.9, 28.3), ttft8=23.0, follow8=1.1, ttft78=290.0, follow78=4.0, reused78=76_800, **kw):
    return {"decode-8k": decode_cell(decode, **kw),
            "ttft-cold-8k": ttft_cell("ttft-cold-8k", 8192, ttft8, 360.0, follow=follow8, follow_reused=8192, **kw),
            "ttft-cold-78k": ttft_cell("ttft-cold-78k", 78000, ttft78, 270.0, follow=follow78, follow_reused=reused78,
                                       prompt_tokens=77_990, **kw)}


def test_pass_at_the_window_and_the_record_is_accepted(tmp_path):
    rows, info, decisions, code = dm.w131k_d5(cells())
    assert code == 0 and decisions == [] and all(r[3] for r in rows)
    names = [r[0] for r in rows]
    assert names[0] == "window of every cell" and rows[0][1] == "131072"
    assert any(n.startswith("follow-up reused tokens, 78K") for n in names)
    assert any("ttft-cold-78k: 77990 tokens, TTFT median 290.00 s" in i and "published, not gated" in i for i in info)
    assert any("ledger" in i and "131072-token window" in i for i in info)
    cap = "db/sources/runs/2026/10/20261022T100000-w131k-engine/"
    text = dm.write_w131k_campaign(cap, rows, info, decisions, code)
    assert text.startswith("# w131k-campaign-d5:") and "Verdict: PASS" in text and text.count("Verdict:") == 1
    (tmp_path / cap).mkdir(parents=True)
    (tmp_path / cap / "verdict.json").write_text("{}")
    (tmp_path / "db" / "sources" / "ARCHIVED.txt").write_text("")
    rec = tmp_path / "w131k-campaign-d5.md"
    rec.write_text(text)
    assert db_check.check_measurement(rec, root=tmp_path) == []


def test_peak_is_judged_against_the_ledger_at_the_window():
    led = Ledger("D5", window=WINDOW).expected_peak_bytes(CHUNKS)
    default = Ledger("D5").expected_peak_bytes(CHUNKS)
    nonpool = led - POOL - 1                                      # between the default ledger and the scaled one
    assert nonpool + POOL > default
    rows, info, decisions, code = dm.w131k_d5(cells(nonpool=nonpool))
    assert code == 0, decisions
    rows, info, decisions, code = dm.w131k_d5(cells(nonpool=led - POOL + 1))
    assert code == 1 and any("peak" in d for d in decisions)


def test_cell_peak_reads_the_window_of_the_cell():
    c = decode_cell()
    once, nonpool, drop, led, chunks = dm.cell_peak(c, "D5")
    assert led == Ledger("D5", window=WINDOW).expected_peak_bytes(CHUNKS) and chunks == CHUNKS
    del c["window"]
    assert dm.cell_peak(c, "D5")[3] == Ledger("D5").expected_peak_bytes(CHUNKS)


def test_every_cell_must_run_at_the_window():
    cs = cells()
    cs["decode-8k"]["window"] = 32768
    rows, info, decisions, code = dm.w131k_d5(cs)
    assert code == 1 and rows[0][3] is False and "decode-8k 32768" in rows[0][1]


@pytest.mark.parametrize("kw,needle", [({"reused78": 60_000}, "reused tokens"), ({"follow78": 13.0}, "follow-up TTFT median (s), 78K"),
                                       ({"ttft8": 31.0}, "cold 8K TTFT"), ({"decode": (20.0,) * 5}, "decode median")])
def test_bars(kw, needle):
    rows, info, decisions, code = dm.w131k_d5(cells(**kw))
    assert code == 1 and any(needle in d for d in decisions)


def test_cold_78k_is_published_not_gated():
    rows, info, decisions, code = dm.w131k_d5(cells(ttft78=900.0))
    assert code == 0 and not any("78K" in r[0] and "cold" in r[0].lower() and "TTFT" in r[0] for r in rows)


def test_reused_cold_sample_invalidates_the_run():
    cs = cells()
    cs["ttft-cold-78k"]["samples"][2]["cold"]["reused_tokens"] = 5
    rows, info, decisions, code = dm.w131k_d5(cs)
    assert code == 1 and "sample 2" in decisions[0]


def test_missing_cell_or_failed_cell_is_unreadable():
    cs = cells()
    del cs["ttft-cold-78k"]
    with pytest.raises(dm.BadInput):
        dm.w131k_d5(cs)
    cs = cells()
    cs["decode-8k"]["ok"] = False
    with pytest.raises(dm.BadInput):
        dm.w131k_d5(cs)


def test_cli_writes_the_record(tmp_path, capsys, monkeypatch):
    cap = tmp_path / "db" / "sources" / "runs" / "2026" / "10" / "20261022T100000-w131k-engine"
    for name, c in cells().items():
        d = cap / f"d5-{name}"
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps(c))
    monkeypatch.setattr(dm, "ROOT", tmp_path)
    (tmp_path / "db" / "records" / "measurements").mkdir(parents=True)
    assert dm.main(["w131k-campaign", str(cap), "--write"]) == 0
    out = capsys.readouterr().out
    assert "w131k-campaign: PASS" in out
    rec = tmp_path / dm.W131K_D5
    assert rec.exists() and "Verdict: PASS" in rec.read_text() and "Command: tools/spark/jobs.sh start w131k-engine" in rec.read_text()


def envelope_result():
    cells_ = []
    for q in (1024, 2048, 4096):
        for ctx in (65536, 131072):
            k = ctx + q
            fits = q * k <= 150_994_944 * 1.2
            cells_.append({"pass": q, "context": ctx, "k": k, "product": q * k, "attn_alloc_bytes": 1_000_000 * q, "attn_ms": 50.0 + q / 100,
                           "moe_alloc_bytes": 500_000 * q, "transient_bytes": 1_500_000 * q, "fits": fits})
    cells_.append({"pass": 4096, "context": 131072, "k": 135168, "product": 4096 * 135168, "error": "oom", "attn_alloc_bytes": None,
                   "attn_ms": None, "moe_alloc_bytes": 1, "transient_bytes": None, "fits": False})
    return {"cells": cells_, "budget_bytes": 5_324_800_000, "watchdog_ms": 200.0, "qk_product": 134_217_728 + 1024 * 1024,
            "launch_ms_max": 61.3, "all_launches_within_watchdog": True, "contexts": [65536, 131072], "passes": [1024, 2048, 4096]}


def test_envelope_record_is_info_and_lists_every_cell(tmp_path):
    rows, info = dm.w131k_envelope(envelope_result(), {"qk_product": 150_994_944})
    assert len(rows) == 7 and any("out of device memory" in r[1] for r in rows)
    assert any("not rewritten" in i for i in info)
    text = dm.write_w131k_envelope("cap/", rows, info)
    assert text.count("Verdict:") == 1 and "Verdict: INFO" in text
    with pytest.raises(dm.BadInput):
        dm.w131k_envelope({"cells": []})


def test_envelope_cli(tmp_path, capsys, monkeypatch):
    cap = tmp_path / "cap"
    (cap / "envelope").mkdir(parents=True)
    (cap / "envelope" / "result.json").write_text(json.dumps(envelope_result()))
    monkeypatch.setattr(dm, "ROOT", tmp_path)
    (tmp_path / "db" / "records" / "measurements").mkdir(parents=True)
    assert dm.main(["w131k-envelope", str(cap), "--write"]) == 0
    assert (tmp_path / dm.W131K_ENVELOPE).read_text().startswith("# w131k-envelope:")
    assert dm.main(["w131k-envelope", str(tmp_path / "nothing")]) == 3
