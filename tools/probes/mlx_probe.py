#!/usr/bin/env python3
"""Sonda MLX-CUDA en GB10: dispositivo, quantize/quantized_matmul afin 4-bit gs64/gs32
con formas de experto de Qwen3.8-Flash-Next, gather_qmm decode/prefill y mlx-lm.
Memoria pico prevista < 2 GiB. Sin descargas (HF_HUB_OFFLINE=1)."""
import json, time, importlib, os, sys
t_start = time.time()
import mlx.core as mx

out = {}

def rec(k, fn):
    try:
        out[k] = fn()
    except Exception as e:
        out[k] = f"ERR {type(e).__name__}: {str(e)[:400]}"

rec("mlx", lambda: mx.__version__)
rec("default_device", lambda: str(mx.default_device()))
rec("cuda_is_available", lambda: mx.cuda.is_available())
rec("metal_is_available", lambda: mx.metal.is_available())
rec("device_info", lambda: {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in mx.device_info().items()})
rec("smoke_add", lambda: (mx.array([1.0, 2.0]) + 1).tolist())
t0 = time.time(); rec("smoke_matmul_bf16", lambda: float((mx.ones((512, 512), dtype=mx.bfloat16) @ mx.ones((512, 512), dtype=mx.bfloat16)).sum().item())); out["first_matmul_s"] = round(time.time() - t0, 3)


def bench(fn, iters=50, warm=5):
    for _ in range(warm):
        mx.eval(fn())
    mx.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    mx.synchronize()
    return (time.perf_counter() - t) / iters * 1e6


def bench_batched(fn, n=32, iters=10):
    mx.eval([fn() for _ in range(n)]); mx.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        mx.eval([fn() for _ in range(n)])
    mx.synchronize()
    return (time.perf_counter() - t) / iters / n * 1e6


H, F = 2560, 640
qmm = {}
for gs in (64, 32):
    for (N, K, name) in ((F, H, "gate/up 2560->640"), (H, F, "down 640->2560")):
        key = f"gs{gs} {name}"
        try:
            w = mx.random.normal((N, K)).astype(mx.bfloat16)
            wq, s, b = mx.quantize(w, group_size=gs, bits=4)
            mx.eval(wq, s, b)
            wd = mx.dequantize(wq, s, b, group_size=gs, bits=4)
            entry = {"wq": [list(wq.shape), str(wq.dtype)], "scales": [list(s.shape), str(s.dtype)],
                     "biases": [list(b.shape), str(b.dtype)],
                     "bytes_q+s+b": int(wq.nbytes + s.nbytes + b.nbytes),
                     "max_abs_err_dequant": round(mx.abs(wd - w).max().item(), 4)}
            for B in (1, 256):
                x = mx.random.normal((B, K)).astype(mx.bfloat16)
                f = lambda: mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=gs, bits=4)
                y = f(); ref = x @ wd.T
                rel = (mx.abs(y - ref).max() / mx.abs(ref).max()).item()
                us = bench(f)
                usb = bench_batched(f)
                fl = 2 * B * N * K
                entry[f"B{B}"] = {"us_eval_each": round(us, 1), "us_amortized_32": round(usb, 2),
                                  "TFLOPs_amortized": round(fl / usb / 1e6, 3),
                                  "weight_GBps_amortized": round(entry["bytes_q+s+b"] / usb / 1e3, 1),
                                  "rel_err_vs_dequant_matmul": round(rel, 4)}
            qmm[key] = entry
        except Exception as e:
            qmm[key] = f"ERR {type(e).__name__}: {str(e)[:400]}"
out["quantized_matmul"] = qmm
out["peak_GiB_after_qmm"] = round(mx.get_peak_memory() / 2**30, 3)

# gather_qmm con un pool de E expertos (forma SwitchGLU de mlx-lm): top-10
gq = {"has_gather_qmm": hasattr(mx, "gather_qmm")}
E, TOPK = 128, 10
for gs in (64,):
    try:
        w = mx.random.normal((E, F, H)).astype(mx.bfloat16)
        wq, s, b = mx.quantize(w, group_size=gs, bits=4)
        del w
        mx.eval(wq, s, b)
        gq["pool"] = {"E": E, "wq": list(wq.shape), "GiB": round((wq.nbytes + s.nbytes + b.nbytes) / 2**30, 3)}
        for T in (1, 8, 256):
            x = mx.random.normal((1, T, 1, 1, H)).astype(mx.bfloat16)
            idx = mx.random.randint(0, E, (1, T, TOPK)).astype(mx.uint32)
            f = lambda: mx.gather_qmm(x, wq, s, b, rhs_indices=idx, transpose=True, group_size=gs, bits=4)
            y = f(); mx.eval(y)
            r = {"out_shape": list(y.shape), "us_unsorted": round(bench(f, iters=20), 1)}
            # camino ordenado como SwitchGLU (_gather_sort)
            flat = idx.flatten(); order = mx.argsort(flat); inv = mx.argsort(order)
            xs = x.flatten(0, -3)[order // TOPK]
            ids = flat[order]
            fs = lambda: mx.gather_qmm(xs, wq, s, b, rhs_indices=ids, transpose=True, group_size=gs, bits=4, sorted_indices=True)
            mx.eval(fs())
            r["us_sorted"] = round(bench(fs, iters=20), 1)
            # referencia densa con los mismos FLOPs
            xd = mx.random.normal((T * TOPK, H)).astype(mx.bfloat16)
            fd = lambda: mx.quantized_matmul(xd, wq[0], s[0], b[0], transpose=True, group_size=gs, bits=4)
            r["us_dense_same_flops"] = round(bench(fd, iters=20), 1)
            fl = 2 * T * TOPK * F * H
            r["TFLOPs_unsorted"] = round(fl / r["us_unsorted"] / 1e6, 3)
            r["TFLOPs_sorted"] = round(fl / r["us_sorted"] / 1e6, 3)
            r["TFLOPs_dense"] = round(fl / r["us_dense_same_flops"] / 1e6, 3)
            gq[f"T{T}_top{TOPK}_gs{gs}"] = r
    except Exception as e:
        gq[f"gs{gs}"] = f"ERR {type(e).__name__}: {str(e)[:400]}"
out["gather_qmm"] = gq
out["peak_GiB"] = round(mx.get_peak_memory() / 2**30, 3)

# mlx-lm
ml = {}
try:
    import mlx_lm
    ml["version"] = getattr(mlx_lm, "__version__", "?")
    from mlx_lm import utils as U
    d = os.path.join(os.path.dirname(mlx_lm.__file__), "models")
    ml["qwen_models"] = sorted(f[:-3] for f in os.listdir(d) if "qwen" in f and f.endswith(".py"))
    ml["has_gated_delta"] = os.path.exists(os.path.join(d, "gated_delta.py"))
    ml["remap_qwen4_exp"] = U.MODEL_REMAPPING.get("qwen4_exp")
    try:
        U._get_classes({"model_type": "qwen4_exp"})
        ml["get_classes_qwen4_exp"] = "OK"
    except Exception as e:
        ml["get_classes_qwen4_exp"] = f"{type(e).__name__}: {e}"
    try:
        U._get_classes({"model_type": "qwen3_next"})
        ml["get_classes_qwen3_next"] = "OK"
    except Exception as e:
        ml["get_classes_qwen3_next"] = f"{type(e).__name__}: {e}"
except Exception as e:
    ml["ERR"] = f"{type(e).__name__}: {e}"
out["mlx_lm"] = ml
out["elapsed_s"] = round(time.time() - t_start, 1)
print(json.dumps(out, indent=1, default=str))
