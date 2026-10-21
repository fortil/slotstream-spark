"""Stand-ins for the parts of vLLM that the vendored Qwen4Exp ops import.

Not an upstream file (slotstream-spark). It replaces: `vllm.platforms.current_platform`,
`vllm.triton_utils`, `vllm.envs`, `vllm.utils.math_utils.cdiv`, `direct_register_custom_op`,
the Triton warmup helpers, and the attention-backend / KV-cache interfaces that `qsa_cache.py`
subclasses. Nothing here runs vLLM; the engine drives the kernels directly.

PDL (programmatic dependent launch) stays off: `is_arch_support_pdl()` is False, so every
`launch_pdl` flag compiles out the `gdc_wait` / `gdc_launch_dependents` calls.
`persistent_topk` / `cooperative_topk` (vLLM CUDA ops) are replaced by `kernels.topk_stable`.
"""
import enum
import logging
import types
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import torch
import triton
import triton.language as tl

HAS_TRITON = True
PAD_SLOT_ID = -1
logger = logging.getLogger("qwen4_exp_ops")


def cdiv(a: int, b: int) -> int:
    return -(a // -b)


def is_arch_support_pdl() -> bool:
    return False


class _Platform:
    """Subset of vLLM's `current_platform` used by the vendored files."""

    @staticmethod
    def is_arch_support_pdl() -> bool:
        return is_arch_support_pdl()

    @staticmethod
    def get_device_capability():
        return tuple(torch.cuda.get_device_capability())

    @classmethod
    def has_device_capability(cls, capability: int) -> bool:
        major, minor = cls.get_device_capability()
        return major * 10 + minor >= capability

    @classmethod
    def is_device_capability_family(cls, capability: int) -> bool:
        return cls.get_device_capability()[0] == capability // 10


current_platform = _Platform()

# Chunk size for the prefill indexer logits (MB). vLLM reads VLLM_SPARSE_INDEXER_MAX_LOGITS_MB.
envs = types.SimpleNamespace(VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128)


# ---- custom-op registration (hc.py) -------------------------------------------------------
_LIB = None


def direct_register_custom_op(op_name, op_func, mutates_args=None, fake_impl=None,
                              target_lib=None, dispatch_key="CUDA", tags=()):
    """Register `op_func` as `torch.ops.vllm.<op_name>` (what vLLM's helper does)."""
    global _LIB
    if _LIB is None:
        _LIB = torch.library.Library("vllm", "FRAGMENT")
    lib = target_lib or _LIB
    schema = torch.library.infer_schema(op_func, mutates_args=mutates_args or ())
    lib.define(op_name + schema, tags=tags)
    lib.impl(op_name, op_func, dispatch_key)
    if fake_impl is not None:
        torch.library.register_fake(f"vllm::{op_name}", fake_impl, lib=lib)


# ---- top-k (qsa_indexer._topk) -----------------------------------------------------------
def persistent_topk(logits, visible_blocks, block_indices, workspace, block_topk, width):
    """Same contract as the vLLM op: the `block_topk` best columns of each row among the first
    `visible_blocks[row]`, indices into `block_indices` int32 [rows, block_topk]; a row with fewer
    visible columns than `block_topk` selects all of them and pads with -1."""
    del workspace
    from slotstream_spark.kernels.topk_stable import topk_stable

    scores = logits[:, :width]
    if scores.shape[1] < block_topk:
        pad = torch.full((scores.shape[0], block_topk - scores.shape[1]), float("-inf"),
                         dtype=scores.dtype, device=scores.device)
        scores = torch.cat([scores, pad], dim=1)
    scores = scores.contiguous()
    topk_stable(scores, block_topk, visible_blocks.to(torch.int32), block_indices, None)


# ---- warmup helpers (only the warmup_* functions use them; they are never called) ---------
class TritonWarmupTensor:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Triton warmup is not used by slotstream-spark")


def triton_scalar_specialization_rep(*args, **kwargs):
    raise NotImplementedError("Triton warmup is not used by slotstream-spark")


# ---- vLLM attention/KV-cache interfaces subclassed by qsa_cache.py ------------------------
CacheDType = str
VllmConfig = Any
CacheConfig = Any
T = TypeVar("T")


class AttentionLayerBase:
    def bind_kv_cache(self, kv_cache):
        self.kv_cache = kv_cache


class AttentionBackend:
    pass


class AttentionCGSupport(enum.Enum):
    NEVER = 0
    UNIFORM_SINGLE_TOKEN_DECODE = 1
    UNIFORM_BATCH = 2
    ALWAYS = 3


class AttentionMetadata:
    pass


class AttentionMetadataBuilder(Generic[T]):
    pass


class KVCacheLayout(enum.Enum):
    BLNHC = 0
    BLHNC = 1


class KVCacheSpec:
    pass


class AttentionSpec(KVCacheSpec):
    pass


class CircularBufferSpec(AttentionSpec):
    pass


class MLAAttentionSpec(AttentionSpec):
    pass


@dataclass
class CommonAttentionMetadata:
    """The fields `build_qsa_metadata` reads, for one or more requests on the device.

    query_start_loc int32 [R+1]; seq_lens int32 [R]; slot_mapping int64 [tokens] (main cache);
    block_table_tensor int32 [R, pages]; *_cpu copies for the host-side scalars.
    """
    query_start_loc: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    seq_lens: torch.Tensor
    num_reqs: int
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    block_table_tensor: torch.Tensor
    slot_mapping: torch.Tensor

    def token_to_req_indices(self, out):
        lens = torch.diff(self.query_start_loc)
        idx = torch.repeat_interleave(torch.arange(self.num_reqs, device=lens.device, dtype=torch.int32), lens)
        out[: idx.numel()].copy_(idx)
        return out


def split_decodes_and_prefills(*args, **kwargs):
    raise NotImplementedError("the engine splits decode and prefill itself")
