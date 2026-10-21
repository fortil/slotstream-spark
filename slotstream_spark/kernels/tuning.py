"""Fixed per-shape launch configuration for the dense GEMV kernels.

config_for depends only on (n, k, bits, gs), never on the number of rows R: a different split
or tile order per R would change the fp32 summation order and break "3-row verify == 3 decodes".
bits = 16 means the bf16 dense variant (gs ignored).

Lookup order: the frozen table `tuning_gb10.json` (written by bench/tune_kernels.py, step 55),
then OVERRIDES, then the formula. The table supersedes OVERRIDES for every shape it names.
"""
import json
from pathlib import Path

BLOCK_N = 16
BLOCK_K = 256          # elements per tile; a multiple of gs 32 and 64
TARGET_PROGRAMS = 192  # keep the GPU busy for skinny shapes (GB10 has 48 SMs)
SMEM_LIMIT = 101_376   # bytes per block on sm_12x

TABLE_PATH = Path(__file__).with_name("tuning_gb10.json")

# Explicit overrides, keyed by (n, k, bits, gs); fallback when the frozen table does not name the shape.
# (2560, 2560, 4, 64): the step-23 tuned config (m0-gemv.md, capture 20261011T190550-m0-gemv, cold_graph_config of
# 2560x2560: kv 3, BLOCK_N 32, BLOCK_K 256, SPLIT_K 1, 8 warps; num_stages 3 is the bench default for that config,
# gemv_bench.py:165). Adopted 2026-10-12 after M0'. The engine 4-bit kernel now runs the same body as that bench
# kernel (word-wise nibbles, one scale/bias per group, unrolled K), so the figure describes the same code.
OVERRIDES: dict[tuple, dict] = {
    (2560, 2560, 4, 64): {"block_n": 32, "block_k": 256, "split_k": 1, "num_warps": 8, "num_stages": 3},
}

MOE_DEFAULT = {"bn": 4, "br": 32, "bk": 512, "bkd": 128, "num_warps": 4}

# Tensor-core GEMM (gemm_affine.py) for the dense prefill rows. The config is a function of (n, k, bits) alone, never
# of the row count: a fixed tile and K order keep each output row independent of M.
GEMM_DEFAULT = {"block_m": 64, "block_n": 64, "block_k": 64, "num_warps": 4, "num_stages": 2}
_G128 = {"block_m": 128, "block_n": 64, "block_k": 128, "num_warps": 8, "num_stages": 2}
_G64 = {"block_m": 64, "block_n": 64, "block_k": 64, "num_warps": 4, "num_stages": 2}
_G128S3 = {"block_m": 128, "block_n": 64, "block_k": 64, "num_warps": 4, "num_stages": 3}
# Best measured config per prefill shape, measured on the diag-f5 GEMM run of 2026-10-13; the frozen table below is the
# only artifact kept.
# Rule (pre-registered): an override whose compiled metadata.shared exceeds SMEM_LIMIT, or that fails to compile at its
# shape, is replaced by GEMM_DEFAULT for that shape.
GEMM_OVERRIDES: dict[tuple[int, int, int], dict] = {
    (10240, 2560, 4): dict(_G128), (6144, 2560, 4): dict(_G128), (12288, 2560, 4): dict(_G128),
    (640, 2560, 4): dict(_G128), (2560, 2560, 4): dict(_G128),
    (2560, 6144, 4): dict(_G64), (320, 10240, 4): dict(_G64), (512, 2560, 4): dict(_G64),
    (10240, 320, 4): dict(_G128S3), (2560, 640, 4): dict(_G128S3),
}


def shape_key(n: int, k: int, bits: int, gs: int) -> str:
    return f"{n}x{k}x{bits}x{gs}"


def _load_table(path: Path = TABLE_PATH) -> dict:
    if not path.exists():
        return {"shapes": {}, "moe": None, "gemm": {}}
    d = json.loads(path.read_text())
    return {"shapes": {k: dict(v["config"]) for k, v in d.get("shapes", {}).items()},
            "moe": dict(d["moe"]["config"]) if d.get("moe") else None,
            "gemm": {k: dict(v["config"]) for k, v in d.get("gemm", {}).items()}}


_TABLE = _load_table()


def config_for(n: int, k: int, bits: int, gs: int) -> dict:
    key = shape_key(n, k, bits, gs)
    if key in _TABLE["shapes"]:
        return dict(_TABLE["shapes"][key])
    okey = (n, k, bits, gs)
    if okey in OVERRIDES:
        return dict(OVERRIDES[okey])
    progs = -(-n // BLOCK_N)
    chunks = -(-k // BLOCK_K)
    split_k = min(chunks, max(1, -(-TARGET_PROGRAMS // progs)))
    return {"block_n": BLOCK_N, "block_k": BLOCK_K, "split_k": split_k,
            "num_warps": 4, "num_stages": 2}


def formula_config(n: int, k: int) -> dict:
    """The untuned formula (no table, no override); the tuner's baseline."""
    progs = -(-n // BLOCK_N)
    chunks = -(-k // BLOCK_K)
    split_k = min(chunks, max(1, -(-TARGET_PROGRAMS // progs)))
    return {"block_n": BLOCK_N, "block_k": BLOCK_K, "split_k": split_k,
            "num_warps": 4, "num_stages": 2}


def moe_config() -> dict:
    """MoE gate/up (bn, bk) and down (br, bkd) tiles; one table entry, independent of R.
    A table entry without `bkd` (the step-55 table) gets MOE_DEFAULT's bkd."""
    c = dict(_TABLE["moe"] or MOE_DEFAULT)
    c.setdefault("bkd", MOE_DEFAULT["bkd"])
    return c


def gemm_config_for(n: int, k: int, bits: int) -> dict:
    """GEMM tile config for a weight shape. Table (a future tune_kernels.py --gemm section), then GEMM_OVERRIDES,
    then GEMM_DEFAULT. A function of (n, k, bits) alone: never of the row count."""
    key = f"{n}x{k}x{bits}"
    if key in _TABLE["gemm"]:
        return dict(_TABLE["gemm"][key])
    okey = (n, k, bits)
    if okey in GEMM_OVERRIDES:
        return dict(GEMM_OVERRIDES[okey])
    return dict(GEMM_DEFAULT)
