#!/usr/bin/env python3
"""Replay the request order Open WebUI sends to an Ollama server, against a running loopback server (stdlib only).

  webui_sequence.py --url http://127.0.0.1:18434 [--json OUT]

Order: GET /api/tags, POST /api/show, POST /api/chat (stream, turn 1), POST /api/chat (title generation, not
streamed, its own short prompt), POST /api/chat (stream, turn 2: the history of turn 1 plus a new question).
Checks: every status is 200; the second chat turn reports `cached_tokens > 0` (the long first message is reused);
a chat message carrying `images` is a 400 and the rejection counter grows by one.
Exit 0 = every check passed, 1 = a check failed. The report is printed as JSON (and written to --json).

`tree_sha256()` is the `server_tree_sha256` of the h5-agents record: sha256 over `<relative path>\\0<file sha256>\\n`
for slotstream_spark/server/**/*.py and slotstream_spark/front.py, sorted by path.
"""
import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PNG_1X1 = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


def tree_sha256(root=ROOT):
    root = Path(root)
    files = sorted(set((root / "slotstream_spark" / "server").rglob("*.py")) | {root / "slotstream_spark" / "front.py"})
    h = hashlib.sha256()
    for p in files:
        if "__pycache__" in p.parts:
            continue
        h.update(p.relative_to(root).as_posix().encode() + b"\0" + hashlib.sha256(p.read_bytes()).hexdigest().encode() + b"\n")
    return h.hexdigest()


class Client:
    def __init__(self, base):
        assert base.startswith(("http://127.0.0.1:", "http://localhost:")), "loopback only"
        self.base = base.rstrip("/")
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(self, path, body=None, timeout=900):
        req = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(req, timeout=timeout) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    def metric(self, name):
        status, text = self.call("/metrics", timeout=30)
        m = re.search(r"^%s (\S+)$" % re.escape(name), text, re.M) if status == 200 else None
        return float(m.group(1)) if m else 0.0


def frames(raw):
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def run(base):
    c = Client(base)
    checks, detail = [], {}

    def check(name, ok, info=None):
        checks.append({"check": name, "ok": bool(ok), "info": info})
        print(("PASS " if ok else "FAIL ") + name + ("" if info is None else "  " + json.dumps(info)), flush=True)

    status, raw = c.call("/api/tags", timeout=60)
    check("tags answers 200", status == 200, status)
    model = json.loads(raw)["models"][0]["name"] if status == 200 else ""
    status, raw = c.call("/api/show", {"model": model}, timeout=60)
    check("show answers 200", status == 200, status)

    # One long first message: about 5,000 tokens so the prefix cache has a 4,096-token pass boundary to reuse.
    document = " ".join("Section %d: the quick brown fox number %d jumps over lazy dog %d near river %d." % (i, i, i * 7, i * 13)
                        for i in range(1, 160))
    system = {"role": "system", "content": "You answer in one short sentence. Reference document follows. " + document}
    q1 = {"role": "user", "content": "In one word, what animal jumps over the dog in Section 3?"}
    opts = {"temperature": 0, "num_predict": 24, "seed": 7}

    status, raw = c.call("/api/chat", {"model": model, "messages": [system, q1], "stream": True, "options": opts})
    f1 = frames(raw) if status == 200 else []
    answer1 = "".join(f.get("message", {}).get("content", "") for f in f1)
    check("chat turn 1 streams and ends with done", status == 200 and f1 and f1[-1].get("done") is True,
          {"status": status, "frames": len(f1), "prompt_eval_count": f1[-1].get("prompt_eval_count") if f1 else None})
    detail["turn1_prompt_tokens"] = f1[-1].get("prompt_eval_count") if f1 else None

    title = {"role": "user", "content": "Generate a concise title of at most 5 words for this chat: ask about a fox. "
                                         "Answer with the title only."}
    status, raw = c.call("/api/chat", {"model": model, "messages": [title], "stream": False, "options": opts})
    body = json.loads(raw) if status == 200 else {}
    check("title generation (not streamed) answers one body", status == 200 and body.get("done") is True
          and isinstance(body.get("message", {}).get("content"), str), {"status": status})

    q2 = {"role": "user", "content": "And in Section 5, which river is mentioned? Answer with the number only."}
    history = [system, q1, {"role": "assistant", "content": answer1}, q2]
    status, raw = c.call("/api/chat", {"model": model, "messages": history, "stream": True, "options": opts})
    f2 = frames(raw) if status == 200 else []
    cached = f2[-1].get("cached_tokens") if f2 else None
    check("chat turn 2 streams and ends with done", status == 200 and f2 and f2[-1].get("done") is True, {"status": status})
    check("chat turn 2 cached_tokens > 0", isinstance(cached, int) and cached > 0,
          {"cached_tokens": cached, "prompt_eval_count": f2[-1].get("prompt_eval_count") if f2 else None})
    detail["turn2_cached_tokens"] = cached

    before = c.metric("slotstream_images_rejected_total")
    status, raw = c.call("/api/chat", {"model": model, "stream": False,
                                       "messages": [{"role": "user", "content": "What is in this picture?", "images": [PNG_1X1]}]},
                         timeout=120)
    after = c.metric("slotstream_images_rejected_total")
    check("a request with images is a 400", status == 400, {"status": status, "body": raw[:160]})
    check("images_rejected_total grew by 1", after - before == 1, {"before": before, "after": after})
    ok = all(x["ok"] for x in checks)
    return ok, {"url": base, "model": model, "checks": checks, "detail": detail, "server_tree_sha256": tree_sha256()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", required=True)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    ok, report = run(a.url)
    text = json.dumps(report, indent=1, sort_keys=True)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(text + "\n")
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
