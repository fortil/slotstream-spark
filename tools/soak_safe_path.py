"""Safe-path soak: random expert reads through the real pool/ensurer, bounded rate, CRC checks, hwmon pauses.

    soak_safe_path.py --reads N [--no-gpu] [--file F] --rate-gbps 3 --out FILE

Each step ensures 10 random experts (Demand, through slotio) plus 0-20 background reads, launches a GPU checksum
over the pinned slots (concurrent GPU reads) and, 1 read in 1,000, compares the slot bytes against an independent
O_DIRECT read. A batch that does not return within 2 s is a timeout and ends the run. The total rate stays under
--rate-gbps and the run pauses 30 s while the NVMe hwmon temperature is at or above the device limit (temp1_max - 5,000 mC).
The background reads go through a plain O_DIRECT preadv in a thread: the slotio bindings expose no
Speculative-priority entry point outside the ensurer.
"""
import argparse
import json
import mmap
import os
import random
import signal
import sys
import threading
import time
import zlib
from pathlib import Path

BATCH_IDS = 10
MAX_SPEC = 20
JOIN_TIMEOUT_S = 2.0
CRC_EVERY = 1000
PAUSE_S = 30.0
HWMON = "/sys/class/hwmon/hwmon1/temp1_input"


def temp_limit_mc():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from bench import eligibility
    return eligibility.nvme_limit_mc()[0]


def read_temp_mc(path=HWMON):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def pct(sorted_vals, q):
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


class DirectReader:
    """Independent O_DIRECT reads of whole records (aligned buffer)."""

    def __init__(self, path, record_bytes, align=4096):
        self.fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECT", 0))
        self.record_bytes = record_bytes
        size = (record_bytes + align - 1) // align * align
        self.buf = mmap.mmap(-1, size)
        self.size = size

    def read(self, off):
        n = os.preadv(self.fd, [self.buf], off)
        if n < self.record_bytes:
            raise OSError(f"short read {n} at {off}")
        return bytes(memoryview(self.buf)[:self.record_bytes])


class PoolBackend:
    """Real slotio pool + ensurer over an experts file with its manifest next to it."""

    def __init__(self, path, chunks=2, slots=None):
        from slotstream_spark.runtime.pool import ExpertPool
        path = Path(path)
        man = json.loads((path.parent / "experts.manifest.json").read_text())
        self.record_bytes = int(man["record_bytes"])
        self.layers, self.experts = int(man["layers"]), int(man["experts"])
        spc = (1 << 30) // self.record_bytes
        n = slots or min(chunks * spc, 25088)
        self.pool = ExpertPool(path.parent, n, experts_name=path.name)
        self.path = str(path)
        self.align = 4096
        import numpy as np
        self._progress = np.zeros(1, dtype=np.uint64)          # pins of generations below it retire
        self.pool.set_progress_ptr(self._progress.ctypes.data)
        self._gen = 0

    def ensure(self, layer, ids):
        self._gen += 1
        self._progress[0] = self._gen                           # the previous step is finished
        slots = self.pool.ensure(layer, ids, self._gen)
        t = self.pool.last_timing
        return [int(s) for s in slots], int(t["misses"]), int(t["read_wall_ns"])

    def slot_bytes(self, slot):
        return bytes(self.pool.pool.slot_view(slot))

    def gpu_sums(self, slots):
        import torch

        class _Ptr:
            def __init__(self, ptr, n):
                self.__cuda_array_interface__ = {"shape": (n,), "typestr": "|u1", "data": (ptr, False), "version": 2}

        out = []
        for s in slots:
            t = torch.as_tensor(_Ptr(int(self.pool.pool.slot_dev(s)), self.record_bytes), device="cuda")
            out.append(t.to(torch.int64).sum())
        return out


def run(backend, args, *, now=time.monotonic, sleep=time.sleep, temp=read_temp_mc, stop=None):
    rng = random.Random(args.seed)
    rec = backend.record_bytes
    direct = DirectReader(backend.path, rec)
    gpu = not args.no_gpu
    lat_ms, res = [], dict(reads=0, bytes=0, errors=0, timeouts=0, checksum_mismatches=0, pauses=0, crc_checks=0,
                           gpu_checks=0, spec_reads=0, demand_reads=0, hits=0)
    limit = args.temp_limit_mc
    t0 = now()
    stop = stop or threading.Event()
    spec_state = {"n": 0}

    def spec_job(offs):
        try:
            for o in offs:
                direct.read(o)
            spec_state["n"] = len(offs)
        except Exception as e:                                  # noqa: BLE001
            res["errors"] += 1
            res["last_error"] = "background read: " + repr(e)

    next_crc = CRC_EVERY
    while res["reads"] < args.reads and not stop.is_set():
        tc = temp(args.hwmon)
        while tc is not None and tc >= limit and not stop.is_set():
            res["pauses"] += 1
            sleep(PAUSE_S)
            tc = temp(args.hwmon)
        layer = rng.randrange(backend.layers)
        ids = rng.sample(range(backend.experts), BATCH_IDS)
        nspec = rng.randint(0, MAX_SPEC)
        offs = [(rng.randrange(backend.layers) * backend.experts + rng.randrange(backend.experts)) * rec for _ in range(nspec)]
        box = {}

        def work():
            try:
                import numpy as np
                slots, misses, rw = backend.ensure(layer, np.array(ids, dtype=np.int32))
                box["r"] = (slots, misses, rw)
                if gpu:
                    box["g"] = backend.gpu_sums(slots)
                    import torch
                    torch.cuda.synchronize()
                spec_job(offs)
            except Exception as e:                              # noqa: BLE001
                box["e"] = repr(e)

        tb = now()
        th = threading.Thread(target=work, daemon=True)
        th.start()
        th.join(JOIN_TIMEOUT_S)
        if th.is_alive():
            res["timeouts"] += 1
            break
        if "e" in box:
            res["errors"] += 1
            res["last_error"] = box["e"]
            break
        slots, misses, rw = box["r"]
        lat_ms.append(rw / 1e6 if rw else (now() - tb) * 1e3)
        res["demand_reads"] += misses
        res["hits"] += BATCH_IDS - misses
        res["spec_reads"] += spec_state["n"]
        n_new = misses + spec_state["n"]
        res["reads"] += n_new
        res["bytes"] += n_new * rec
        if res["reads"] >= next_crc:
            next_crc += CRC_EVERY
            k = rng.randrange(BATCH_IDS)
            off = (layer * backend.experts + ids[k]) * rec
            res["crc_checks"] += 1
            have = backend.slot_bytes(slots[k])
            if zlib.crc32(have) != zlib.crc32(direct.read(off)):
                res["checksum_mismatches"] += 1
            if gpu:
                res["gpu_checks"] += 1
                want = int(sum(have))
                if int(box["g"][k]) != want:
                    res["checksum_mismatches"] += 1
        # rate cap: total bytes / elapsed <= rate
        due = res["bytes"] / (args.rate_gbps * 1e9)
        el = now() - t0
        if el < due:
            sleep(due - el)
    el = max(now() - t0, 1e-9)
    s = sorted(lat_ms)
    res.update(seconds=round(el, 3), gbps=round(res["bytes"] / el / 1e9, 3), p50_read_ms=round(pct(s, .5), 3),
               p99_read_ms=round(pct(s, .99), 3), max_read_ms=round(s[-1], 3) if s else 0.0, batches=len(s),
               terminated=stop.is_set(), record_bytes=rec, target_reads=args.reads, gpu=gpu,
               rate_gbps_limit=args.rate_gbps, temp_limit_mc=limit)
    return res


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reads", type=int, required=True)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--file", help="experts file (manifest beside it); default $SLOTSTREAM_PACKED/experts.bin")
    ap.add_argument("--rate-gbps", type=float, default=3.0)
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--chunks", type=int, default=2)
    ap.add_argument("--hwmon", default=HWMON)
    ap.add_argument("--temp-limit-mc", type=int)
    a = ap.parse_args(argv)
    if a.temp_limit_mc is None:
        a.temp_limit_mc = temp_limit_mc()
        if a.temp_limit_mc is None:
            print("soak: nvme limit unreadable (hwmon temp1_max)", file=sys.stderr)
            sys.exit(2)
    return a


def out_path(a):
    if a.out:
        return Path(a.out)
    cap = os.environ.get("CAP")
    if not cap:
        raise SystemExit("soak_safe_path: --out or $CAP required")
    return Path(cap) / "result.json"


def main(argv=None):
    a = parse(argv)
    path = a.file or str(Path(os.environ["SLOTSTREAM_PACKED"]) / "experts.bin")
    dest = out_path(a)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    backend = PoolBackend(path, a.chunks)
    res = run(backend, a, stop=stop)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps(res, sort_keys=True))
    ok = (res["reads"] >= a.reads and not res["errors"] and not res["timeouts"] and not res["checksum_mismatches"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
