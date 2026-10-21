# Probes

These micro-benchmarks and CUDA sources keep the bytes they had on 2026-10-10, Spanish comments included. `SHA256SUMS` and `tests/t0/test_probes_sha.py` pin them, and the measurement records cite those hashes, so translating a comment would break the evidence. One exception: on 2026-10-20 the default results path in `iobench.py` changed from a private directory name to `~/iobench-results.jsonl` before publication. That line only picks where the probe appends its JSON lines, the io2 and m8 runs were made with the earlier bytes, and the hash in `SHA256SUMS` changed with it.
