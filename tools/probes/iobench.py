#!/usr/bin/env python3
# Banco de lectura de un solo uso (slotstream-spark, 2026-10-10).
# Solo lectura de archivos existentes. Solo stdlib. Un proceso por unidad de
# profundidad de cola (QD = N procesos con pread sincrono), sin GIL de por medio.
# Offsets "never-repeat" dentro de cada corrida: permutacion afin sobre ranuras
# no solapadas, cada ranura se lee como mucho una vez.
import argparse, ctypes, ctypes.util, json, math, mmap, os, resource, sys, time
from array import array
from multiprocessing import Process, Pipe

PAGE = 4096


def mem_available_kb():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1])
    return -1


def diskstats(dev='nvme0n1'):
    with open('/proc/diskstats') as f:
        for line in f:
            p = line.split()
            if p[2] == dev:
                return {'rd_ios': int(p[3]), 'rd_merges': int(p[4]), 'rd_sect': int(p[5]), 'rd_ms': int(p[6])}
    return None


def residency(files):
    """Paginas en page cache por archivo, via mmap+mincore (no provoca lecturas)."""
    libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    out = []
    for fn in files:
        size = os.path.getsize(fn)
        fd = os.open(fn, os.O_RDONLY)
        try:
            addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
            if addr in (None, ctypes.c_void_p(-1).value):
                raise OSError(ctypes.get_errno(), 'mmap')
            npages = (size + PAGE - 1) // PAGE
            vec = ctypes.create_string_buffer(npages)
            if libc.mincore(ctypes.c_void_p(addr), size, vec) != 0:
                raise OSError(ctypes.get_errno(), 'mincore')
            res = sum(1 for b in vec.raw if b & 1)
            libc.munmap(ctypes.c_void_p(addr), size)
        finally:
            os.close(fd)
        out.append({'file': fn, 'size': size, 'resident_pages': res, 'pct': round(100.0 * res / npages, 2)})
    return out


def build_slots(files, slot, limit_bytes):
    sizes = [os.path.getsize(f) for f in files]
    counts = []
    for s in sizes:
        usable = min(s, limit_bytes) if limit_bytes else s
        counts.append(max(0, (usable - slot) // slot + 1) if usable >= slot else 0)
    return sizes, counts


def worker(conn, files, counts, slot, bs, direct, fadv_random, start_ns, deadline_ns, j, nworkers, step, phase, jitter, repeat):
    flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)
    fds = [os.open(f, flags) for f in files]
    if fadv_random:
        for fd in fds:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
    total = sum(counts)
    cum = []
    acc = 0
    for c in counts:
        cum.append(acc)
        acc += c
    buf = mmap.mmap(-1, ((bs + PAGE - 1) // PAGE) * PAGE)
    mv = memoryview(buf)[:bs]
    lat = array('q')
    nbytes = 0
    last = start_ns
    k = j
    import random
    rng = random.Random(1000 + j)
    while time.monotonic_ns() < start_ns:
        pass
    while True:
        if not repeat and k >= total:
            break
        if repeat:
            g = rng.randrange(total)
        else:
            g = (phase + k * step) % total
        # localizar archivo
        lo, hi = 0, len(cum) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if cum[mid] <= g:
                lo = mid
            else:
                hi = mid - 1
        fi = lo
        off = (g - cum[fi]) * slot
        if jitter:
            off += rng.randrange(0, slot - bs + 1)
        t0 = time.monotonic_ns()
        if t0 >= deadline_ns:
            break
        n = os.preadv(fds[fi], [mv], off)
        t1 = time.monotonic_ns()
        lat.append(t1 - t0)
        nbytes += n
        last = t1
        k += nworkers
    for fd in fds:
        os.close(fd)
    conn.send_bytes(lat.tobytes())
    conn.send((nbytes, last))
    conn.close()


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, max(0, int(math.ceil(p / 100.0 * len(sorted_vals))) - 1))
    return sorted_vals[i]


def run(args):
    avail = mem_available_kb()
    if avail < 20 * 1024 * 1024:
        print(json.dumps({'abort': 'MemAvailable < 20 GB', 'MemAvailable_kB': avail}))
        sys.exit(3)
    files = args.files
    slot = args.slot or ((args.bs + PAGE - 1) // PAGE) * PAGE
    sizes, counts = build_slots(files, slot, args.limit_bytes)
    total = sum(counts)
    step = int(total * 0.6180339887) | 1
    while math.gcd(step, total) != 1:
        step += 2
    phase = int(args.phase * total) % total
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    d0 = diskstats()
    start_ns = time.monotonic_ns() + 300_000_000
    deadline_ns = start_ns + int(args.dur * 1e9)
    procs, pipes = [], []
    for j in range(args.qd):
        a, b = Pipe(duplex=False)
        p = Process(target=worker, args=(b, files, counts, slot, args.bs, args.direct, args.fadv_random,
                                         start_ns, deadline_ns, j, args.qd, step, phase, args.jitter, args.repeat))
        p.start()
        procs.append(p)
        pipes.append(a)
    lats = array('q')
    nbytes = 0
    last = start_ns
    for a in pipes:
        lats.frombytes(a.recv_bytes())
        nb, lt = a.recv()
        nbytes += nb
        last = max(last, lt)
    for p in procs:
        p.join()
    t_end = time.monotonic_ns()
    d1 = diskstats()
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall = max(last - start_ns, 1)
    s = sorted(lats)
    ops = len(s)
    dd = {k: d1[k] - d0[k] for k in d0}
    res = {
        'label': args.label,
        'mode': 'O_DIRECT' if args.direct else ('buffered+FADV_RANDOM' if args.fadv_random else 'buffered'),
        'bs': args.bs, 'slot': slot, 'qd': args.qd, 'dur_s': round(wall / 1e9, 3), 'repeat': args.repeat,
        'files': len(files), 'slots_total': total,
        'ops': ops, 'bytes': nbytes,
        'GBps': round(nbytes / (wall / 1e9) / 1e9, 3),
        'GiBps': round(nbytes / (wall / 1e9) / (1 << 30), 3),
        'IOPS': round(ops / (wall / 1e9), 1),
        'lat_us': {'min': round(s[0] / 1e3, 1) if s else None,
                   'p50': round(pct(s, 50) / 1e3, 1) if s else None,
                   'p90': round(pct(s, 90) / 1e3, 1) if s else None,
                   'p99': round(pct(s, 99) / 1e3, 1) if s else None,
                   'p999': round(pct(s, 99.9) / 1e3, 1) if s else None,
                   'max': round(s[-1] / 1e3, 1) if s else None,
                   'mean': round(sum(s) / ops / 1e3, 1) if s else None},
        'cpu_s': {'user': round(r1.ru_utime - r0.ru_utime, 2), 'sys': round(r1.ru_stime - r0.ru_stime, 2)},
        'diskstats_delta': dd,
        'dev_avg_req_KiB': round(dd['rd_sect'] * 512 / dd['rd_ios'] / 1024, 1) if dd['rd_ios'] else None,
        'dev_GBps': round(dd['rd_sect'] * 512 / (wall / 1e9) / 1e9, 3),
        'MemAvailable_kB_before': avail, 'MemAvailable_kB_after': mem_available_kb(),
    }
    line = json.dumps(res)
    print(line)
    with open(os.path.expanduser('~/iobench-results.jsonl'), 'a') as f:
        f.write(line + '\n')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('res')
    r.add_argument('files', nargs='+')
    b = sub.add_parser('run')
    b.add_argument('--label', required=True)
    b.add_argument('--bs', type=int, required=True)
    b.add_argument('--slot', type=int, default=0)
    b.add_argument('--qd', type=int, default=1)
    b.add_argument('--dur', type=float, default=5.0)
    b.add_argument('--direct', action='store_true')
    b.add_argument('--fadv-random', action='store_true')
    b.add_argument('--jitter', action='store_true', help='offset aleatorio dentro de la ranura (lecturas no alineadas)')
    b.add_argument('--repeat', action='store_true', help='offsets aleatorios con repeticion (prueba de cache caliente)')
    b.add_argument('--limit-bytes', type=int, default=0, help='limita cada archivo a sus primeros N bytes')
    b.add_argument('--phase', type=float, default=0.0)
    b.add_argument('files', nargs='+')
    args = ap.parse_args()
    if args.cmd == 'res':
        for row in residency(args.files):
            print(json.dumps(row))
    else:
        if args.dur > 30:
            sys.exit('dur > 30 s no permitido')
        run(args)


if __name__ == '__main__':
    main()
