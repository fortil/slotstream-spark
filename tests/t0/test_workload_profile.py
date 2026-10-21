import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
import workload_profile as wp  # noqa: E402

TEXT = "abcd " * 40000  # 200,000 chars, 5 chars per fake token


class Fake:
    """Fake front: 503 + Retry-After for the first `cold_503` calls, then 200 with usage from the prompt length."""

    def __init__(self, cold_503=2, tok_per_s=30.0):
        self.cold_503 = cold_503
        self.calls = []
        self.now = 0.0
        self.tok_per_s = tok_per_s

    def clock(self):
        return self.now

    def sleep(self, s):
        self.now += s

    def post(self, base_url, body, timeout, key=None):
        if self.cold_503 > 0:
            self.cold_503 -= 1
            return 503, 10.0, None
        content = body["messages"][0]["content"]
        pt = len(content) // 5
        ct = min(body["max_tokens"], 6000) if body["max_tokens"] > 1 else 1
        self.calls.append(body)
        self.now += ct / self.tok_per_s
        return 200, None, {"usage": {"prompt_tokens": pt, "completion_tokens": ct},
                           "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}


def test_cold_start_retries_on_503_until_200():
    f = Fake(cold_503=3)
    r = wp.cold_start("http://x", _post=f.post, _sleep=f.sleep, _clock=f.clock)
    assert r["attempts"] == 4 and r["seconds"] >= 30


def test_cold_start_times_out():
    f = Fake(cold_503=10 ** 6)
    try:
        wp.cold_start("http://x", timeout_s=50, _post=f.post, _sleep=f.sleep, _clock=f.clock)
    except RuntimeError as e:
        assert "not ready" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_profile_shape_and_json_mode():
    f = Fake(cold_503=1, tok_per_s=40.0)
    res = wp.profile("http://x", TEXT, _post=f.post, _sleep=f.sleep, _clock=f.clock)
    names = [c["name"] for c in res["calls"]]
    assert names == ["json16k", "json25k", "text8k"]
    assert len({b["messages"][0]["content"].split("\n", 1)[0] for b in f.calls}) == len(f.calls)  # unique nonces
    json_calls = [b for b in f.calls if "response_format" in b]
    assert len(json_calls) == 2 and all(b["response_format"] == {"type": "json_object"} for b in json_calls)
    assert all("temperature" not in b for b in f.calls)
    assert [c["prompt_tokens"] for c in res["calls"]][0] >= 15500
    assert res["verdict"] == "PASS"  # 6000 / 40 = 150 s


def test_prompts_are_unique_cold_prompts():
    a = wp.make_prompt(TEXT, 1000)
    b = wp.make_prompt(TEXT, 1000)
    assert a != b and a.split("\n", 1)[0] != b.split("\n", 1)[0]


def test_slow_decode_fails_the_pass_rule():
    f = Fake(cold_503=0, tok_per_s=20.0)  # 6000 / 20 = 300 s
    res = wp.profile("http://x", TEXT, _post=f.post, _sleep=f.sleep, _clock=f.clock)
    assert res["verdict"] == "FAIL"
    assert "over 270 s" in res["verdict_line"]
    assert res["calls_over_300s"] == 0


def test_failed_call_is_a_fail():
    calls = [{"name": "json16k", "ok": False, "status": 500, "total_s": 1.0}]
    v, line = wp.verdict({"seconds": 1.0}, calls)
    assert v == "FAIL" and "json16k" in line


def test_frozen_prompt_text_is_long_enough():
    assert len(wp.load_text()) >= 25500 * 3
