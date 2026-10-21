#!/usr/bin/env python3
"""Stub of llama-swap GET /running for guard drills.

  running_stub.py --port 18080 --state-file F [--log FILE]

Serves the content of F as the body of GET /running (`{"running":[{"model":...,"state":...}]}`); a missing or empty file
means `{"running":[]}`. Every change of the served state is logged with its time (stderr, and --log when given).
Listens on 127.0.0.1 only.
"""
import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

EMPTY = '{"running":[]}'


def read_state(path):
    try:
        text = Path(path).read_text().strip()
    except OSError:
        return EMPTY
    if not text:
        return EMPTY
    json.loads(text)                                           # a malformed file is an error, not a silent empty
    return text


class Flips:
    """Remembers the last served state and reports each flip with the time."""

    def __init__(self, log=None, clock=time.time):
        self.last, self.log, self.clock, self.events = None, log, clock, []
        self.lock = threading.Lock()

    def note(self, state):
        with self.lock:
            if state != self.last:
                ev = {"t": round(self.clock(), 3), "running": json.loads(state)["running"]}
                if self.last is not None or state != EMPTY:
                    self.events.append(ev)
                    line = json.dumps(ev)
                    print(line, file=sys.stderr, flush=True)
                    if self.log:
                        with open(self.log, "a") as f:
                            f.write(line + "\n")
                self.last = state


def make_server(port, state_file, log=None):
    flips = Flips(log)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.split("?")[0] != "/running":
                self.send_error(404)
                return
            try:
                state = read_state(state_file)
            except ValueError:
                self.send_error(500)
                return
            flips.note(state)
            body = state.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    srv.flips = flips
    return srv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--state-file", required=True)
    ap.add_argument("--log")
    a = ap.parse_args(argv)
    srv = make_server(a.port, a.state_file, a.log)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
