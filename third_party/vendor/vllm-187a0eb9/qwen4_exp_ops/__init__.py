"""Qwen4Exp QSA and hyper-connection kernels vendored from vLLM 187a0eb9 (see ../NOTICE).

Import the submodules (`qsa`, `qsa_indexer`, `qsa_prepare`, `qsa_cache`, `hc`) directly; they need
torch and triton and a CUDA device only when called.
"""
