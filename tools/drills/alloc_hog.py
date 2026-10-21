#!/usr/bin/env python3
"""Drill helper: allocate and touch N MiB, optionally ignore SIGTERM, then sleep."""
import argparse
import signal
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=int, required=True)
    ap.add_argument("--ignore-term", action="store_true")
    ap.add_argument("--sleep", type=float, default=600)
    a = ap.parse_args()

    buf = bytearray(a.mib << 20)
    for i in range(0, len(buf), 4096):
        buf[i] = 1

    if a.ignore_term:
        def on_term(signum, frame):
            print(f"TERM {time.time():.3f}", flush=True)
        signal.signal(signal.SIGTERM, on_term)
    print(f"READY {a.mib}", flush=True)
    end = time.time() + a.sleep
    while time.time() < end:
        time.sleep(min(1.0, max(0.0, end - time.time())))


if __name__ == "__main__":
    main()
