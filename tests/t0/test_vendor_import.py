"""The vendored ops import without vLLM or fla, PDL is off and no vLLM CUDA op is referenced (needs torch + triton)."""
import re
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

ROOT = Path(__file__).resolve().parents[2]
VLLM = ROOT / "third_party" / "vendor" / "vllm-187a0eb9" / "qwen4_exp_ops"
FLA = ROOT / "third_party" / "vendor" / "fla-0.5.2" / "fla_v052"


def test_adapters_import():
    from slotstream_spark.kernels import vendored
    for name in ("qsa_decode", "qsa_prefill", "qsa_store_kv", "hc_mix", "gdn_decode", "gdn_prefill"):
        assert callable(getattr(vendored, name))


def test_vendored_modules_import():
    from slotstream_spark.kernels import vendored
    m = vendored._ops()
    for name in ("hc", "qsa", "indexer", "prepare", "cache"):
        assert m[name] is not None
    chunk, fused = vendored._fla()
    assert callable(chunk) and callable(fused)


def test_pdl_is_off():
    from slotstream_spark.kernels import vendored
    shims = vendored._ops()["shims"]
    assert shims.is_arch_support_pdl() is False
    assert shims.current_platform.is_arch_support_pdl() is False
    assert vendored._ops()["cache"]._metadata_launch_pdl() is False


def test_no_vllm_cuda_ops_left():
    for f in VLLM.glob("*.py"):
        text = f.read_text()
        assert "torch.ops._C" not in text, f.name
        assert "persistent_topk" not in text or f.name in ("_shims.py", "qsa_indexer.py"), f.name
        assert not re.search(r"^\s*(from|import) vllm\b", text, re.M), f.name
    assert "torch.ops._C" not in (VLLM / "qsa_indexer.py").read_text()
    assert "_shims.persistent_topk" in (VLLM / "qsa_indexer.py").read_text()


def test_fla_uses_its_own_package_name():
    for f in FLA.rglob("*.py"):
        assert not re.search(r"^\s*(from|import) fla\b", f.read_text(), re.M), f
