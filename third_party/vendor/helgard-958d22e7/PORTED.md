# Code ported from helgard-orlm/qwen-flash-next-8gb

Every file with ported code starts with:

`# Portions ported from helgard-orlm/qwen-flash-next-8gb@958d22e7ba0317750ccc38862b9667b1034a4dd1 <path>:<lines> (Apache-2.0, section 4); modified.`

| Piece | Origin @958d22e7 | Destination | Status |
|---|---|---|---|
| Layer-by-layer reference check | tools/ref_check.py:96-224 | tools/ref_layerwise.py | ported |
| N-gram ids and hash | qwen_engine.py:178-258 + tests/test_ngram.py | slotstream_spark/model/ngram_hash.py + tests/t0/test_ngram_hash.py | ported |
| Perplexity ablation | tools/ppl_ablate.py | tools/ppl_ablate.py | ported |
| MoE expert kernels | qmoe.py:24-83 | slotstream_spark/kernels/moe_gemv_affine.py (affine q·s+b; explicit rank-order sum) | ported |
| CUDA graph capture pattern (one shared graph pool, capture_error_mode="thread_local") | qwen_delta_graph.py:17-19, 54-55 | slotstream_spark/runtime/graphs.py (per-segment decode graphs G(0)..G(48), 1 and 3 rows) | ported |
| Reference forward (rmsnorm, GDN chunk and recurrent, gated residual, indexer, attention, PLE, MoE skeleton) | qwen_engine.py:91-174 and qwen_engine.py:684-902 | slotstream_spark/model/reference_torch.py (folded norms multiply by w, weights through injected accessors, dtype follows the input) | ported |

Later steps add rows for the remaining pieces.
