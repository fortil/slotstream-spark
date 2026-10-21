# io2: IO-2 read benchmarks on our own checkpoint shards
Date: 2026-10-11
Capture: db/sources/runs/2026/10/20261011T183603-io2/
Command: tools/spark/run.sh --heavy io2 10240 900 --exclusive -- tools/capture.sh io2 -- bash tools/spark/io2.sh (iobench.py run --dur 5, files model-00001..00010 of the pinned checkpoint, never a production file)

19 runs. Buffered runs drop the page cache of their files before and after; the d2 runs keep it warm on purpose.

| label | mode | GB/s | p50 us | p99 us |
|---|---|---|---|---|
| b2-819200-qd1 | O_DIRECT | 2.44 | 299.5 | 829.7 |
| b2-819200-qd12 | O_DIRECT | 10.427 | 910.1 | 1301.1 |
| b2-819200-qd32 | O_DIRECT | 10.417 | 2497.6 | 2888.8 |
| b2-51200-qd1 | O_DIRECT | 0.188 | 199.9 | 768.3 |
| b2-51200-qd12 | O_DIRECT | 2.115 | 296.9 | 499.6 |
| b2-51200-qd32 | O_DIRECT | 5.548 | 295.3 | 534.9 |
| c1-4096-qd1 | O_DIRECT | 0.019 | 199.6 | 588.4 |
| c1-4096-qd32 | O_DIRECT | 0.688 | 197.5 | 405.1 |
| c1-512-qd1 | O_DIRECT | 0.002 | 199.4 | 398.3 |
| c1-1024-qd1 | O_DIRECT | 0.005 | 199.5 | 393.8 |
| c2-ng100-qd1 | buffered+FADV_RANDOM | 0.0 | 199.0 | 567.6 |
| c2-ng100-qd32 | buffered+FADV_RANDOM | 0.016 | 198.5 | 420.9 |
| c3-ng100-qd1 | buffered | 0.0 | 199.2 | 584.4 |
| c3-ng100-qd32 | buffered | 0.016 | 198.4 | 424.7 |
| d1-exp-buf-qd1 | buffered | 0.874 | 2901.2 | 5890.0 |
| d1-exp-buf-qd12 | buffered | 7.182 | 4178.1 | 6136.3 |
| d1-exp-buf-qd32 | buffered | 10.257 | 8478.6 | 17666.7 |
| d2-hot-qd1 | buffered | 27.808 | 97.8 | 111.0 |
| d2-hot-qd12 | buffered | 88.877 | 294.6 | 814.2 |

b2 confirms repacking is mandatory: 800 KiB direct reads reach 10.4 GB/s at qd12, while 50 KiB reads reach 2.1 GB/s at qd12 and 5.5 GB/s at qd32, so scattered per-expert reads from the original layout cannot reach the sequential rate. The 100-byte n-gram reads (c2, c3) are limited by the 4 KiB device page: 0.016 GB/s at qd32, which is about 156k reads per second.

Verdict: INFO — 19 runs recorded; c1 512 B vs 1024 B p50 ratio 1.0005 gives row_bytes=100
