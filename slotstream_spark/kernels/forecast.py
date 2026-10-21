# SPDX-License-Identifier: MIT
# Port of RouterTapCorrection.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""Forecast tap of the decode graph (plan step 73; RouterTapCorrection.swift, Model.swift buildAttentionForecast).

After layer l's attention the hyper-connection streams `hyp` already hold everything layer l+1 will read except layer l's
routed experts and l+1's own attention. The tap runs the target's own `mlp_hyper_connection` mix and its router on those
streams, in FP32, and keeps the top-24 ids and logits of the target. With the learned correction (rank 128, FP16 factors,
the wide product in FP16 and the narrow one in FP32) the logits are

    logits + ((mixed - mu[T]) a[T]) b[T] + delta[T]

It moves forecast logits only: routing, expert computation and outputs are untouched.
"""
import hashlib
import json
import struct
from pathlib import Path

import torch
import torch.nn.functional as F

SCHEMA = "slotstream-tap-correction-v1"
SHIPPED_REL = "lookahead/tap-correction-attention-rank128-v1.safetensors"
SHIPPED_BYTES = 37_540_708
SHIPPED_SHA256 = "37b00d3a32d1e1889a1794bbb8e97905a157a77c0508db620c1a11f2a895f7f5"
K_FORECAST = 24
_DTYPES = {"F16": torch.float16, "F32": torch.float32, "I32": torch.int32}


class TapCorrectionError(ValueError):
    pass


def read_header(path):
    """The safetensors header of a correction file, checked like RouterTapCorrection.readHeader (nothing is loaded).
    -> dict(targets, first_target, hidden, rank, experts, tap, file_bytes, tensors{name: (dtype, shape, begin, end)},
    base) with `base` the byte offset of the data section."""
    p = Path(path)
    with open(p, "rb") as f:
        pre = f.read(8)
        if len(pre) != 8:
            raise TapCorrectionError(f"{p}: not a safetensors file")
        (n,) = struct.unpack("<Q", pre)
        if not 0 < n < 1 << 24:
            raise TapCorrectionError(f"{p}: header length {n} is invalid")
        raw = f.read(n)
    try:
        obj = json.loads(raw)
    except ValueError as e:
        raise TapCorrectionError(f"{p}: header is not JSON") from e
    meta = obj.get("__metadata__", {})
    if meta.get("schema") != SCHEMA:
        raise TapCorrectionError(f"{p}: schema must be {SCHEMA}")
    tap = meta.get("tap", "")
    if tap not in ("attention", "attention-readout"):
        raise TapCorrectionError(f"{p}: must be fitted on the attention or attention-readout tap, not {tap!r}")

    def shape(name, dtype):
        e = obj.get(name)
        if not isinstance(e, dict) or e.get("dtype") != dtype or "shape" not in e:
            raise TapCorrectionError(f"{p}: needs tensor {name} as {dtype}")
        return list(e["shape"])

    a, b = shape("a", "F16"), shape("b", "F16")
    mu, delta, targets = shape("mu", "F32"), shape("delta", "F32"), shape("targets", "I32")
    if not (len(a) == 3 and len(b) == 3 and len(mu) == 2 and len(delta) == 2 and len(targets) == 1 and a[0] > 0 and a[2] > 0
            and b[0] == a[0] and mu[0] == a[0] and delta[0] == a[0] and targets[0] == a[0]
            and b[1] == a[2] and mu[1] == a[1] and delta[1] == b[2]):
        raise TapCorrectionError(f"{p}: tensor shapes are inconsistent")
    try:
        first = int(meta.get("first_target", ""))
    except ValueError:
        first = 0
    if first < 1:
        raise TapCorrectionError(f"{p}: must declare first_target >= 1")
    tensors = {k: (v["dtype"], v["shape"], v["data_offsets"][0], v["data_offsets"][1])
               for k, v in obj.items() if k != "__metadata__"}
    return {"targets": a[0], "first_target": first, "hidden": a[1], "rank": a[2], "experts": b[2], "tap": tap,
            "file_bytes": p.stat().st_size, "tensors": tensors, "base": 8 + n}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def locate(model_dir, pinned_sha256=SHIPPED_SHA256):
    """-> (path | None, reason): the shipped correction under `model_dir` if present, a readable attention-tap file and
    its sha256 equals `pinned_sha256` (None skips the digest)."""
    p = Path(model_dir) / SHIPPED_REL
    if not p.is_file():
        return None, f"no correction at {SHIPPED_REL}"
    try:
        h = read_header(p)
    except TapCorrectionError as e:
        return None, f"correction header rejected: {e}"
    if h["tap"] != "attention":
        return None, f"correction was fitted on the {h['tap']} tap"
    digest = file_sha256(p)
    if pinned_sha256 and digest != pinned_sha256:
        return None, f"correction has digest {digest[:16]}, not the measured {pinned_sha256[:16]}"
    return p, f"measured correction {digest[:16]} at {SHIPPED_REL}"


class TapCorrection:
    """The factors on the device: a [T, hidden, rank] and b [T, rank, experts] FP16, mu [T, hidden] and delta [T, experts]
    FP32 (resident, ~37.5 MB)."""

    def __init__(self, a, b, mu, delta, first_target=1, identity="in-memory"):
        self.a, self.b = a.to(torch.float16), b.to(torch.float16)
        self.mu, self.delta = mu.to(torch.float32), delta.to(torch.float32)
        self.first_target, self.identity = int(first_target), identity
        self.targets = self.a.shape[0]
        self.hidden, self.rank, self.experts = self.a.shape[1], self.a.shape[2], self.b.shape[2]
        self.b32 = self.b.float()            # the narrow product runs in FP32
        self.resident_bytes = sum(t.numel() * t.element_size() for t in (self.a, self.b, self.mu, self.delta))

    @classmethod
    def load(cls, path, device="cuda", hidden=None, experts=None, targets=None):
        h = read_header(path)
        if h["first_target"] != 1 or (targets is not None and h["targets"] != targets) \
                or (experts is not None and h["experts"] != experts) or (hidden is not None and h["hidden"] != hidden):
            raise TapCorrectionError(
                f"{path}: geometry (targets {h['first_target']}..{h['first_target'] + h['targets'] - 1}, hidden "
                f"{h['hidden']}, experts {h['experts']}) does not match the model (targets 1..{targets}, hidden "
                f"{hidden}, experts {experts})")
        raw = bytearray(Path(path).read_bytes())
        out = {}
        for name, (dt, shp, b0, b1) in h["tensors"].items():
            n = 1
            for d in shp:
                n *= d
            out[name] = torch.frombuffer(raw, dtype=_DTYPES[dt], count=n, offset=h["base"] + b0).view(shp).clone()
        if out["targets"].tolist() != list(range(1, h["targets"] + 1)):
            raise TapCorrectionError(f"{path}: targets must run 1..{h['targets']}")
        for k in ("a", "b", "mu", "delta"):
            if not torch.isfinite(out[k].float()).all():
                raise TapCorrectionError(f"{path}: {k} is not finite")
        dev = {k: v.to(device) for k, v in out.items() if k != "targets"}
        return cls(dev["a"], dev["b"], dev["mu"], dev["delta"], h["first_target"], file_sha256(path))

    def covers(self, target):
        return self.first_target <= target < self.first_target + self.targets

    def apply(self, target, mixed, logits):
        """Corrected logits [R, experts] from the target's mixed router input [R, hidden] and the tap's own logits."""
        i = target - self.first_target
        centered = (mixed.float() - self.mu[i]).to(torch.float16)
        narrow = (centered @ self.a[i]).float()
        return logits + (narrow @ self.b32[i] + self.delta[i])


def router_weight(seg, target):
    """Dense bf16 [experts, hidden] router matrix of layer `target`."""
    w = seg.W(f"layers.{target}.mlp.gate.weight")
    w = getattr(w, "w", w)
    if not isinstance(w, torch.Tensor):
        raise TypeError("the forecast tap needs a dense router matrix")
    return w


def tap(seg, target, hyp, correction=None):
    """(mixed, raw logits fp32 [R, experts], logits fp32 used for the forecast) for `target` on streams `hyp`."""
    mixed, _ = seg.hc_mix(f"layers.{target}.mlp_hyper_connection", hyp, inject=False)
    raw = F.linear(mixed.float(), router_weight(seg, target).float())
    used = raw if correction is None or not correction.covers(target) else correction.apply(target, mixed, raw)
    return mixed, raw, used.contiguous()


def forecast_into(seg, target, hyp, ids_out, logits_out, correction=None):
    """Top-24 of the tap into `ids_out` int32 [R, 24] / `logits_out` fp32 [R, 24] (device or pinned). Graph capturable."""
    from slotstream_spark.kernels.topk_stable import topk_stable
    _, _, used = tap(seg, target, hyp, correction)
    idx = torch.empty(used.shape[0], logits_out.shape[-1], dtype=torch.int32, device=used.device)
    val = torch.empty(used.shape[0], logits_out.shape[-1], dtype=torch.float32, device=used.device)
    topk_stable(used, logits_out.shape[-1], out_idx=idx, out_val=val)
    ids_out.copy_(idx, non_blocking=True)
    logits_out.copy_(val, non_blocking=True)
