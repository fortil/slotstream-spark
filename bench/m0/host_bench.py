#!/usr/bin/env python3
"""M0 host-side microbenchmark: launch / sync cost and the process footprint of torch+triton.

  host_bench.py [--quick] [--out DIR]
  host_bench.py --footprint-child      (internal: run by the parent in a fresh process)

Writes $CAP/result.json (or --out DIR/result.json):
  {launch_us, event_sync_us, spin_query_us, ensure_sim_us, tau_us, gap_us:{gap:us}, footprint_bytes, ...}
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

GAPS_US = (0, 200, 5000)


def vm_hwm_bytes():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    return 0


def nvml_used_bytes(pid):
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
            if p.pid == pid and p.usedGpuMemory is not None:
                return int(p.usedGpuMemory), "pynvml"
    except Exception:
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0] == str(pid) and parts[1].isdigit():
                return int(parts[1]) * 1024 * 1024, "nvidia-smi"
    except Exception:
        pass
    return None, None


def footprint_child():
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
        o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = o < n
        tl.store(o_ptr + o, tl.load(x_ptr + o, mask=m) + tl.load(y_ptr + o, mask=m), mask=m)

    n = 1 << 16
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    o = torch.empty_like(x)
    add[(triton.cdiv(n, 1024),)](x, y, o, n, BLOCK=1024)
    a = torch.randn(2560, 2560, device="cuda", dtype=torch.bfloat16)
    (a @ a).sum().item()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        o.add_(1)
    g.replay()
    torch.cuda.synchronize()
    used, src = nvml_used_bytes(os.getpid())
    print(json.dumps({"vmhwm_bytes": vm_hwm_bytes(), "nvml_bytes": used, "nvml_source": src}))


def measure_footprint():
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "--footprint-child"],
                       capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        raise RuntimeError("footprint child failed: " + p.stderr[-500:])
    d = json.loads(p.stdout.strip().splitlines()[-1])
    nv = d["nvml_bytes"]
    # GPU memory of the process when NVML reports it; on a unified-memory GPU NVML may not,
    # then the resident-set high-water mark is the footprint.
    d["footprint_bytes"] = (nv + d["vmhwm_bytes"]) if nv else d["vmhwm_bytes"]
    d["footprint_source"] = "nvml+vmhwm" if nv else "vmhwm"
    return d


def med(vals):
    return statistics.median(vals)


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--footprint-child", action="store_true")
    a = ap.parse_args(argv)
    if a.footprint_child:
        footprint_child()
        return 0

    import torch
    n_launch = 500 if a.quick else 10000
    n_sync = 200 if a.quick else 2000
    pc = time.perf_counter_ns

    x = torch.zeros(1024, device="cuda")
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            x.add_(1)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        for _ in range(4):
            x.add_(1)
    g.replay()
    torch.cuda.synchronize()

    # launch_us: CPU time of g.replay()
    lat = []
    for i in range(n_launch):
        t0 = pc()
        g.replay()
        lat.append((pc() - t0) / 1e3)
        if i % 50 == 49:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    launch_us = med(lat)

    # event_sync_us: ev.synchronize() after a tiny kernel; spin_query_us: busy ev.query()
    ev = torch.cuda.Event()
    sync_l, spin_l = [], []
    for _ in range(n_sync):
        g.replay()
        ev.record()
        t0 = pc()
        ev.synchronize()
        sync_l.append((pc() - t0) / 1e3)
    for _ in range(n_sync):
        g.replay()
        ev.record()
        t0 = pc()
        while not ev.query():
            pass
        spin_l.append((pc() - t0) / 1e3)
    event_sync_us, spin_query_us = med(sync_l), med(spin_l)

    # ensure_sim_us: 10 dict lookups/updates and one append
    d = {i: i for i in range(64)}
    log = []
    sim = []
    for _ in range(n_sync):
        t0 = pc()
        for j in range(10):
            d[j] = d.get(j + 1, 0) + 1
        log.append(1)
        sim.append((pc() - t0) / 1e3)
    ensure_sim_us = med(sim)

    # g3: replay + spin after an idle gap
    gap_us = {}
    for gap in GAPS_US:
        vals = []
        for _ in range(100 if a.quick else 500):
            end = pc() + gap * 1000
            while pc() < end:
                pass
            t0 = pc()
            g.replay()
            ev.record()
            while not ev.query():
                pass
            vals.append((pc() - t0) / 1e3)
        gap_us[str(gap)] = med(vals)

    result = {
        "launch_us": launch_us, "event_sync_us": event_sync_us, "spin_query_us": spin_query_us,
        "ensure_sim_us": ensure_sim_us,
        "tau_us": spin_query_us + ensure_sim_us + launch_us,
        "gap_us": gap_us,
        "versions": {"torch": torch.__version__, "cuda": torch.version.cuda},
    }
    fp = measure_footprint()
    result["footprint_bytes"] = fp["footprint_bytes"]
    result["footprint"] = fp
    print(json.dumps(result))

    out = a.out or os.environ.get("CAP")
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "result.json"), "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
