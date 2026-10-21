#!/usr/bin/env python3
"""Sonda de solo lectura: vuelca todos los cudaDeviceAttr del GPU 0 via ctypes.

Los valores del enum se leen de /usr/local/cuda/include/driver_types.h para no
depender de numeros escritos a mano. Crea un contexto CUDA (unos cientos de MB)
y lo destruye al salir. Sin asignaciones grandes.
"""
import ctypes, re, json, sys

HDR = "/usr/local/cuda/include/driver_types.h"
LIB = "/usr/local/cuda/lib64/libcudart.so.13"

src = open(HDR).read()
m = re.search(r"enum\s+__device_builtin__\s+cudaDeviceAttr\s*\{(.*?)\};", src, re.S)
body = m.group(1)
attrs = []
for name, val in re.findall(r"(cudaDevAttr\w+)\s*=\s*(\d+)", body):
    attrs.append((name, int(val)))

rt = ctypes.CDLL(LIB)
def chk(rc, what):
    if rc != 0:
        rt.cudaGetErrorString.restype = ctypes.c_char_p
        raise SystemExit(f"{what} -> {rc} {rt.cudaGetErrorString(rc).decode()}")

v = ctypes.c_int()
chk(rt.cudaRuntimeGetVersion(ctypes.byref(v)), "cudaRuntimeGetVersion"); runtime = v.value
chk(rt.cudaDriverGetVersion(ctypes.byref(v)), "cudaDriverGetVersion"); driver = v.value
chk(rt.cudaGetDeviceCount(ctypes.byref(v)), "cudaGetDeviceCount"); ndev = v.value

out = {"runtime": runtime, "driver": driver, "device_count": ndev, "attrs": {}}
for name, val in attrs:
    r = ctypes.c_int(-999)
    rc = rt.cudaDeviceGetAttribute(ctypes.byref(r), val, 0)
    out["attrs"][name] = r.value if rc == 0 else f"err{rc}"

free = ctypes.c_size_t(); total = ctypes.c_size_t()
chk(rt.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)), "cudaMemGetInfo")
out["memgetinfo_free"] = free.value
out["memgetinfo_total"] = total.value

name = ctypes.create_string_buffer(256)
cu = ctypes.CDLL("libcuda.so.1")
dev = ctypes.c_int()
cu.cuInit(0); cu.cuDeviceGet(ctypes.byref(dev), 0)
cu.cuDeviceGetName(name, 256, dev)
out["name"] = name.value.decode()
tm = ctypes.c_size_t(); cu.cuDeviceTotalMem_v2(ctypes.byref(tm), dev)
out["cuDeviceTotalMem"] = tm.value

json.dump(out, sys.stdout, indent=1)
print()
