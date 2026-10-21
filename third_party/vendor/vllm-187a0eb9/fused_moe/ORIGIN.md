# Origin of slotstream_spark/kernels/fused_moe_affine.py

Upstream file: `vllm/model_executor/layers/fused_moe/fused_moe.py`, function `fused_moe_kernel_gptq_awq`
(lines 64-296 of that file). Apache-2.0, Copyright contributors to the vLLM project.

Source used: the `vllm` package inside the local image `vllm/vllm-openai:v0.28.0`, path
`/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/fused_moe.py` (read with
`docker run --rm --entrypoint sh <image> -c "cat <path>"`). The kernel was not checked against commit 187a0eb9 itself:
that tree is not available offline, so the image copy is the recorded source.

sha256 of the upstream file (1,853 lines): d5955a3460746b66740b024f470aeca79bbebd10872c327f765f2d7fcb805f28

Modifications (all in `fused_moe_affine.py`): MLX affine 4-bit gs64 dequant (`q*s + b`, element i in bits 4i of each
u32) instead of the GPTQ/AWQ zero-point path; expert records addressed through a table of base pointers instead of a
stacked `[E, N, K]` tensor; gate and up fused with the SiLU product; fixed tiles (64x64x64, 4 warps, 2 stages); the
routed weight applied in the down launch; the `_C` `moe_align_block_size` replaced by the torch counting sort in
`moe_align.py`; no atomics. License text: `../LICENSE`.
