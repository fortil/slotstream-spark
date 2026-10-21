import pytest


def _cuda_ok():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if _cuda_ok():
        return
    skip = pytest.mark.skip(reason="needs torch with a CUDA GPU")
    for item in items:
        if "cuda" in item.keywords:
            item.add_marker(skip)
