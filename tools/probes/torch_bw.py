#!/usr/bin/env python3
"""Ancho de banda de memoria del GB10 con torch (dentro de un contenedor existente).

Tope: 1 GiB por buffer, <= 3 GiB vivos a la vez, cada medida <= ~5 s.
"""
import json, time, torch

GiB = 1 << 30
N = 1 * GiB
REPS = 20
dev = torch.device("cuda:0")
res = {
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "arch_list": torch.cuda.get_arch_list(),
    "device": torch.cuda.get_device_name(0),
    "capability": torch.cuda.get_device_capability(0),
}
p = torch.cuda.get_device_properties(0)
res["props"] = {k: getattr(p, k) for k in dir(p) if not k.startswith("_") and isinstance(getattr(p, k), (int, float, str, bool))}
free, total = torch.cuda.mem_get_info()
res["mem_get_info_GiB"] = [round(free / GiB, 2), round(total / GiB, 2)]


def timed(fn, reps=REPS):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn(); torch.cuda.synchronize()
    s.record()
    for _ in range(reps):
        fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / 1e3 / reps


a = torch.empty(N, dtype=torch.uint8, device=dev).fill_(1)
b = torch.empty(N, dtype=torch.uint8, device=dev)
t = timed(lambda: b.copy_(a))
res["d2d_copy_1GiB"] = {"s": t, "GBps_bytes_copied": N / t / 1e9, "GBps_read_plus_write": 2 * N / t / 1e9}
af = a.view(torch.float32)
t = timed(lambda: af.sum())
res["d_read_sum_fp32_1GiB"] = {"s": t, "GBps": N / t / 1e9}
t = timed(lambda: b.fill_(3))
res["d_write_fill_1GiB"] = {"s": t, "GBps": N / t / 1e9}
del b

h_pin = torch.empty(N, dtype=torch.uint8, pin_memory=True).fill_(2)
t = timed(lambda: a.copy_(h_pin, non_blocking=True))
res["h2d_pinned_1GiB"] = {"s": t, "GBps": N / t / 1e9}
t = timed(lambda: h_pin.copy_(a, non_blocking=True))
res["d2h_pinned_1GiB"] = {"s": t, "GBps": N / t / 1e9}
del h_pin

h_pag = torch.empty(N, dtype=torch.uint8).fill_(4)
t = timed(lambda: a.copy_(h_pag), reps=5)
res["h2d_pageable_1GiB"] = {"s": t, "GBps": N / t / 1e9}
t = timed(lambda: h_pag.copy_(a), reps=5)
res["d2h_pageable_1GiB"] = {"s": t, "GBps": N / t / 1e9}
del h_pag

# copia de un registro de experto (2 764 800 B) host pinned -> device, a lo slotstream
E = 2764800
src = torch.empty(64 * E, dtype=torch.uint8, pin_memory=True).fill_(5)
dst = torch.empty(64 * E, dtype=torch.uint8, device=dev)
def one_by_one():
    for i in range(64):
        dst[i * E:(i + 1) * E].copy_(src[i * E:(i + 1) * E], non_blocking=True)
t = timed(one_by_one, reps=10)
res["h2d_pinned_64x_expert_record"] = {"s_per_64": t, "us_per_record": t / 64 * 1e6, "GBps": 64 * E / t / 1e9}
def d2d_one_by_one():
    for i in range(63):
        dst[(i + 1) * E:(i + 2) * E].copy_(dst[i * E:(i + 1) * E])
t = timed(d2d_one_by_one, reps=10)
res["d2d_63x_expert_record"] = {"s_per_63": t, "us_per_record": t / 63 * 1e6, "GBps": 63 * E / t / 1e9}

res["max_memory_allocated_GiB"] = round(torch.cuda.max_memory_allocated() / GiB, 3)
print(json.dumps(res, indent=1, default=str))
