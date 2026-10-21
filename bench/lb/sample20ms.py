#!/usr/bin/env python3
"""20-ms sampler for the model-load phase (m-lb2). Linux only, stdlib only.

  sample20ms.py --scope slotstream-spark-lb --out sample.csv [--interval 0.02] [--count N]
                [--disk nvme0n1] [--prod NAME,NAME,...]

Columns: t (s since start), rd_ios, rd_sectors, rd_ms, inflight, io_ms (all from /proc/diskstats),
scope_file and scope_anon (bytes, memory.stat of the scope; empty until the scope exists),
psi0..psi3 (io.pressure some avg10 of up to four production inference containers).
Runs until SIGTERM/SIGINT or --count samples.
"""
import argparse
import glob
import os
import re
import signal
import subprocess
import sys
import time

_STOP = False


def _stop(*_):
    global _STOP
    _STOP = True


def disk_row(name, path="/proc/diskstats"):
    try:
        for line in open(path):
            f = line.split()
            if len(f) > 12 and f[2] == name:
                return [int(f[3]), int(f[5]), int(f[6]), int(f[11]), int(f[12])]
    except OSError:
        pass
    return [""] * 5


def find_scope(unit):
    uid = os.getuid()
    base = f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service"
    for p in (f"{base}/app.slice/{unit}.scope", f"{base}/{unit}.scope"):
        if os.path.isdir(p):
            return p
    hits = glob.glob(f"/sys/fs/cgroup/**/{unit}.scope", recursive=True)
    return hits[0] if hits else None


def scope_mem(path):
    if not path:
        return ["", ""]
    try:
        d = dict(l.split() for l in open(os.path.join(path, "memory.stat")))
        return [int(d.get("file", 0)), int(d.get("anon", 0))]
    except (OSError, ValueError):
        return ["", ""]


def psi_avg10(path):
    try:
        m = re.search(r"^some .*avg10=([0-9.]+)", open(os.path.join(path, "io.pressure")).read(), re.M)
        return m.group(1) if m else ""
    except OSError:
        return ""


def prod_cgroups(names):
    out = []
    for n in names[:4]:
        r = subprocess.run(["docker", "inspect", "--format", "{{.Id}}", n], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            out.append(f"/sys/fs/cgroup/system.slice/docker-{r.stdout.strip()}.scope")
    return out


def discover_prod():
    """Containers whose name matches $SLOTSTREAM_PROD_RE. Empty or unset = no foreign sampling."""
    pat = os.environ.get("SLOTSTREAM_PROD_RE", "")
    if not pat:
        return []
    r = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True)
    return [n for n in r.stdout.split() if re.search(pat, n)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=0.02)
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--disk", default=os.environ.get("SLOTSTREAM_NVME_DEVICE") or "nvme0n1")
    ap.add_argument("--prod", default=None)
    a = ap.parse_args(argv)
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    names = a.prod.split(",") if a.prod else discover_prod()
    try:
        cgs = prod_cgroups(names)
    except OSError:
        cgs = []
    t0 = time.perf_counter()
    scope = None
    n = 0
    with open(a.out, "w", buffering=1) as f:
        f.write("t,rd_ios,rd_sectors,rd_ms,inflight,io_ms,scope_file,scope_anon,psi0,psi1,psi2,psi3\n")
        while not _STOP:
            now = time.perf_counter()
            if scope is None:
                scope = find_scope(a.scope)
            psi = [psi_avg10(c) for c in cgs] + [""] * (4 - len(cgs))
            row = [f"{now - t0:.3f}"] + disk_row(a.disk) + scope_mem(scope) + psi[:4]
            f.write(",".join(str(x) for x in row) + "\n")
            n += 1
            if a.count and n >= a.count:
                break
            nxt = t0 + n * a.interval
            time.sleep(max(0.0, nxt - time.perf_counter()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
