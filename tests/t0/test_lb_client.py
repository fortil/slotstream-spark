import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench/lb"))
import client  # noqa: E402


BODIES = []


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        BODIES.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        start = time.perf_counter()
        for i in range(10):
            # absolute deadlines: a late wake-up does not push every later chunk
            time.sleep(max(0.0, start + i * 0.02 - time.perf_counter()))
            chunk = {"choices": [{"delta": {"content": f"t{i}"}}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()
        last = {"choices": [], "usage": {"prompt_tokens": 42}, "timings": {"prompt_n": 2775, "cache_n": 0, "prompt_ms": 100000}}
        self.wfile.write(b"data: " + json.dumps(last).encode() + b"\n\ndata: [DONE]\n\n")
        self.wfile.flush()


def _serve():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_decode_tps_and_schema(tmp_path):
    srv = _serve()
    try:
        out = tmp_path / "o.jsonl"
        rc = client.main(["--port", str(srv.server_address[1]), "--ids", "short", "--reps", "2",
                          "--warmup", "0", "--out", str(out)])
        assert rc == 0
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        assert len(rows) == 2
        for r in rows:
            assert set(r) == {"prompt_id", "rep", "ttft_s", "n", "decode_tps", "prompt_tokens", "timings",
                              "cache_n", "prompt_n", "prefill_tps", "cold"}
            assert r["prompt_id"] == "short" and r["n"] == 10 and r["prompt_tokens"] == 42
            expect = 9 / 0.18
            assert abs(r["decode_tps"] - expect) / expect < 0.15, r["decode_tps"]
            assert 0 <= r["ttft_s"] < 1
    finally:
        srv.shutdown()


def _run(tmp_path, *extra):
    srv = _serve()
    BODIES.clear()
    try:
        out = tmp_path / "o.jsonl"
        assert client.main(["--port", str(srv.server_address[1]), "--ids", "short", "--reps", "1",
                            "--warmup", "0", "--out", str(out), *extra]) == 0
        return json.loads(out.read_text().splitlines()[0])
    finally:
        srv.shutdown()


def test_no_cache_prompt_flag_and_cold_fields(tmp_path):
    r = _run(tmp_path, "--no-cache-prompt")
    assert BODIES[-1]["cache_prompt"] is False
    assert r["cache_n"] == 0 and r["prompt_n"] == 2775 and r["cold"] is True
    assert abs(r["prefill_tps"] - 27.75) < 1e-9


def test_default_body_has_no_cache_prompt_key(tmp_path):
    _run(tmp_path)
    assert "cache_prompt" not in BODIES[-1]


def test_prompts_are_deterministic_and_sized():
    p = json.loads((Path(client.HERE) / "prompts.json").read_text())
    a, b = client.build_prompt(p["2k"]), client.build_prompt(p["2k"])
    assert a == b and 8000 <= len(a) < 8300
    c = client.build_prompt(p["8k"])
    assert 32000 <= len(c) < 32300
    assert len(client.build_prompt(p["short"])) < 400
