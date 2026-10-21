"""Step 93 tools: the perplexity windows, the tiled lm_head scorer, the campaign helpers. No GPU, no network."""
import hashlib
import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


score = load("score_mod", "slotstream_spark/engine/score.py")
ppl_wikitext = load("ppl_wikitext_mod", "tools/quality/ppl_wikitext.py")
campaign = load("run_campaign_mod", "bench/campaign/run_campaign.py")


# ------------------------------------------------------------------------------------------------ window math
def test_window_math_is_145_by_2047():
    assert score.scored_tokens(145, 2048) == 145 * 2047 == 296_815
    sl = score.window_slices(145 * 2048 + 500, 2048, 145)
    assert len(sl) == 145 and sl[0] == (0, 2048) and sl[-1] == (144 * 2048, 145 * 2048)
    assert all(b - a == 2048 for a, b in sl) and all(sl[i][1] == sl[i + 1][0] for i in range(144))


def test_fewer_tokens_than_windows_gives_fewer_windows():
    assert len(score.window_slices(3 * 2048 + 5, 2048, 145)) == 3
    assert score.window_slices(100, 2048) == []


def test_ppl_is_exp_of_mean_nll():
    assert score.ppl(0.0, 10) == 1.0
    assert score.ppl(10 * 1.5, 10) == pytest.approx(2.718281828 ** 1.5)


def test_band_is_one_percent_of_the_reference():
    ref, tol, windows, wt = ppl_wikitext.contract_ppl()
    assert (ref, tol, windows, wt) == (5.3914, 0.01, 145, 2048)
    inside = ppl_wikitext.summarize([{"nll": 5.3914 and 1.0 * 0, "tokens": 1}], ref, tol, 145)
    assert inside["band"][0] == pytest.approx(5.3375, abs=1e-4) and inside["band"][1] == pytest.approx(5.4453, abs=1e-4)
    rows = [{"nll": 2047 * 1.6, "tokens": 2047}] * 145             # ppl exp(1.6) = 4.95: below the band
    r = ppl_wikitext.summarize(rows, ref, tol, 145)
    assert r["full"] and r["tokens_scored"] == 296_815 and not r["in_band"]
    import math
    rows = [{"nll": 2047 * math.log(5.40), "tokens": 2047}] * 145
    assert ppl_wikitext.summarize(rows, ref, tol, 145)["in_band"]
    assert not ppl_wikitext.summarize(rows[:10], ref, tol, 145)["full"]
    assert "else 1" not in Path(ppl_wikitext.__file__).read_text().split("def main")[1]      # the band no longer decides the exit
    assert ppl_wikitext.contract_reference_informative() is True


def test_pinned_text_is_refused_when_its_sha_changes(tmp_path):
    rows = ["a\n", "", "b\n"]
    f = tmp_path / "test_rows.json"
    f.write_text(json.dumps(rows))
    (tmp_path / "test_rows.sha256").write_text(hashlib.sha256(f.read_bytes()).hexdigest())
    text, sha = ppl_wikitext.load_text(tmp_path)
    assert text == "a\n" + "\n\n" + "" + "\n\n" + "b\n" and sha == hashlib.sha256(f.read_bytes()).hexdigest()
    f.write_text(json.dumps(rows + ["c"]))
    with pytest.raises(SystemExit):
        ppl_wikitext.load_text(tmp_path)
    with pytest.raises(SystemExit):
        ppl_wikitext.load_text(tmp_path / "nowhere")


def test_the_ppl_tool_has_no_literal_window_loss_of_token_zero():
    src = (ROOT / "slotstream_spark/engine/score.py").read_text()
    assert "self.ids[p0 + 1:last + 1]" in src                      # row i is scored against token i + 1: token 0 never is a target


# ------------------------------------------------------------------------------------------------ tiled scorer (torch)
def test_tiled_scorer_matches_a_full_cross_entropy_and_never_builds_the_full_matrix():
    torch = pytest.importorskip("torch")
    V, H, T, TILE = 97, 16, 300, 64
    g = torch.Generator().manual_seed(7)
    W = torch.randn(V, H, generator=g)
    hid = torch.randn(T, H, generator=g)
    ids = torch.randint(0, V, (T,), generator=g).tolist()
    seen = []

    def logits_fn(x):
        seen.append(tuple(x.shape))
        return x @ W.T

    sc = score.TileScorer(logits_fn, tile=TILE)
    sc.begin(ids, "cpu")
    for pos, size in ((0, 200), (200, 100)):                       # two driver tiles, the second one ends the window
        sc(pos, hid[pos:pos + size])
    nll, rows = sc.result()
    full = torch.nn.functional.cross_entropy(hid[:-1] @ W.T, torch.tensor(ids[1:]), reduction="sum")
    assert rows == T - 1 and nll == pytest.approx(float(full), rel=1e-5)
    assert max(s[0] for s in seen) <= TILE and sc.max_tile_elems <= TILE * V
    assert sum(s[0] for s in seen) == T - 1                         # the last row of the window has no target and is not run


# ------------------------------------------------------------------------------------------------ footprint helpers
def sample(i, rss, hwm, nvml, pool=0):
    return {"tag": str(i), "vmrss": rss, "vmhwm": hwm, "nvml": nvml, "pinned": 0, "unified": {"pool": pool}}


def test_thin_samples_keeps_the_peaks_and_the_ends():
    s = [sample(i, 100 + i, 100 + i, 50) for i in range(1000)]
    s[417]["vmrss"] = 10_000
    s[612]["nvml"] = 9_000
    t = campaign.thin_samples(s, 40)
    tags = {x["tag"] for x in t}
    assert len(t) <= 40 and {"0", "999", "417", "612"} <= tags
    assert campaign.thin_samples(s[:10], 40) == s[:10]


def test_read_footprint_parses_the_serve_jsonl(tmp_path):
    p = tmp_path / "footprint-serve.jsonl"
    p.write_text(json.dumps({"header": True, "flags": {"pool": [True, True]}}) + "\n" +
                 json.dumps({"t_ns": 5, "vmrss": 10, "vmhwm": 12, "nvml": 7, "unified": {"pool": 4}}) + "\n")
    samples, flags = campaign.read_footprint(p)
    assert flags == {"pool": [True, True]} and samples[0]["tag"] == "5" and samples[0]["unified"] == {"pool": 4}
    pk = campaign.peak_of(samples, flags)
    # nonpool = max(VmHWM 12 - pool in RSS 4 + NVML 7 - pool in NVML 4 = 11, sample 10 + 7 - 8 = 9) = 11; once = 11 + pool 4
    assert pk["nonpool_bytes"] == 11 and pk["once_bytes"] == 15


# ------------------------------------------------------------------------------------------------ HTTP client
class Fake(BaseHTTPRequestHandler):
    mode = "ok"

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["content-length"]))
        if Fake.mode == "500":
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"boom")
            return
        self.send_response(200)
        self.end_headers()
        frames = [{"message": {"content": ""}, "done": False}, {"message": {"content": "Hi"}, "done": False},
                  {"message": {"content": ""}, "done": True, "prompt_eval_count": 8192, "prompt_eval_duration": 20_000_000_000,
                   "cached_tokens": 0, "eval_count": 2, "eval_duration": 100_000_000}]
        if Fake.mode == "no-final":
            frames = frames[:2]
        for f in frames:
            self.wfile.write((json.dumps(f) + "\n").encode())
            self.wfile.flush()


@pytest.fixture()
def fake_server():
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield srv.server_address[1]
    srv.shutdown()


def test_chat_ttft_reads_the_first_content_frame_and_the_final_counters(fake_server):
    Fake.mode = "ok"
    r = campaign.chat_ttft(fake_server, "m", [{"role": "user", "content": "x"}])
    assert r["prompt_tokens"] == 8192 and r["prefill_s"] == 20.0 and r["reused_tokens"] == 0 and r["eval_count"] == 2
    assert 0 <= r["ttft_s"] <= r["total_s"]
    assert campaign.prefill_tps(8192, 20.0) == pytest.approx(409.6) and campaign.prefill_tps(1, 0) is None


def test_chat_ttft_errors(fake_server):
    Fake.mode = "500"
    with pytest.raises(RuntimeError, match="500"):
        campaign.chat_ttft(fake_server, "m", [])
    Fake.mode = "no-final"
    with pytest.raises(RuntimeError, match="final frame"):
        campaign.chat_ttft(fake_server, "m", [])
    Fake.mode = "ok"


# ------------------------------------------------------------------------------------------------ cold-sample rule
def fake_sample(k, reused, tokens=8192):
    cold = {"ttft_s": 20.0, "prefill_s": 19.0, "prefill_tps": tokens / 19.0, "reused_tokens": reused, "prompt_tokens": tokens}
    return {"sample": k, "chunks": 48, "memavailable_drop_bytes": 1, "cold": cold, "prompt_tokens_expected": 8192,
            "followup": {"ttft_s": 1.0, "prompt_tokens": tokens + 50, "reused_tokens": tokens - 1, "new_tokens": 51}}


def run_cell(monkeypatch, tmp_path, samples):
    monkeypatch.setattr(campaign, "snapshot", lambda est, mode: ({"running": []}, True, []))
    it = iter(samples)
    monkeypatch.setattr(campaign, "ttft_sample", lambda mode, cdir, tokens, wf, k: next(it))
    ok = campaign.run_ttft_cell(tmp_path, "D5", "ttft-cold-8k", len(samples), 100)
    return ok, json.loads((tmp_path / "d5-ttft-cold-8k" / "result.json").read_text())


def test_a_cold_sample_with_reused_tokens_fails_the_cell(monkeypatch, tmp_path):
    ok, doc = run_cell(monkeypatch, tmp_path, [fake_sample(0, 0), fake_sample(1, 4096)])
    assert not ok and any("reused 4096" in f for f in doc["failures"])


def test_a_clean_cell_carries_medians_and_the_cell_file(monkeypatch, tmp_path):
    ok, doc = run_cell(monkeypatch, tmp_path, [fake_sample(0, 0), fake_sample(1, 0), fake_sample(2, 0)])
    assert ok and doc["medians"]["ttft_s"] == 20.0 and doc["medians"]["followup_ttft_s"] == 1.0
    assert doc["snapshot"] == {"running": []} and doc["chunks"] == 48 and doc["tokens"] == 8192


def test_a_wrong_prompt_length_fails_the_cell(monkeypatch, tmp_path):
    ok, doc = run_cell(monkeypatch, tmp_path, [fake_sample(0, 0, tokens=8000)])
    assert not ok and any("not 8192" in f for f in doc["failures"])


def test_32k_cell_sends_the_admissible_size():
    assert campaign.COLD_TOKENS == {"2k": 2048, "8k": 8192, "32k": 32000, "78k": 78000}
    assert campaign.EXACT_CELLS == ("2k", "8k") and campaign.FOLLOWUP_CELLS == ("8k", "78k")
    assert campaign.DEFAULT_CELLS == ("decode-8k", "ttft-cold-2k", "ttft-cold-8k", "ttft-cold-32k")   # the f9 campaign is unchanged


def test_cut_at_most_falls_back_to_the_longest_prompt_that_cuts(monkeypatch):
    from bench import ttft
    ok = {31990, 31989}

    def fake(tok, ids, n):
        if n in ok or n in (2048, 8192):
            return list(range(n)), "body"
        raise RuntimeError(f"cannot cut a {n}-token prompt")
    monkeypatch.setattr(ttft, "cut_prompt", fake)
    assert len(campaign.cut_at_most(None, None, 8192)[0]) == 8192
    assert len(campaign.cut_at_most(None, None, 32000)[0]) == 31990
    with pytest.raises(RuntimeError, match="no prompt"):
        campaign.cut_at_most(None, None, 32000, floor=5)


def test_the_quality_tools_carry_no_literal_logits_limit():
    src = (ROOT / "tools/quality/teacher_forced.py").read_text()
    for lit in ("0.98", "0.01", "0.05", "0.5,", "1.5"):
        assert lit not in src
    assert "ct.logits_check_seq" in src and "oracle_pair" in src
