# w131k-envelope: prefill transient and launch time at 65,536 and 131,072 tokens
Date: 2026-10-21
Capture: db/sources/runs/2026/10/20261021T184845-w131k-engine/
Command: tools/prefill_envelope.py --contexts 65536,131072 (phase `envelope` of bench/campaign/w131k_engine_job.sh)

| cell | value | fits when | fits |
|---|---|---|---|
| pass 256 x context 65536 (product 16,842,752) | transient 172,224,512 B, attention 8.9 ms | transient <= 5,324,800,000 B and attention <= 200 ms | yes |
| pass 512 x context 65536 (product 33,816,576) | transient 207,085,568 B, attention 11.1 ms | transient <= 5,324,800,000 B and attention <= 200 ms | yes |
| pass 1024 x context 65536 (product 68,157,440) | transient 278,904,832 B, attention 16.5 ms | transient <= 5,324,800,000 B and attention <= 200 ms | yes |
| pass 2048 x context 65536 (product 138,412,032) | transient 422,543,360 B, attention 26.2 ms | transient <= 5,324,800,000 B and attention <= 200 ms | yes |
| pass 4096 x context 65536 (product 285,212,672) | transient 709,820,416 B, attention 38.6 ms | transient <= 5,324,800,000 B and attention <= 200 ms | yes |
| pass 256 x context 131072 (product 33,619,968) | transient 172,224,512 B, attention 13.7 ms | transient <= 5,324,800,000 B and attention <= 200 ms | yes |
| pass 512 x context 131072 (product 67,371,008) | transient 207,085,568 B, attention 19.5 ms | transient <= 5,324,800,000 B and attention <= 200 ms | yes |
| pass 1024 x context 131072 (product 135,266,304) | transient 278,904,832 B, attention 33.8 ms | transient <= 5,324,800,000 B and attention <= 200 ms | yes |
| pass 2048 x context 131072 (product 272,629,760) | transient 422,543,360 B, attention 58.2 ms | transient <= 5,324,800,000 B and attention <= 200 ms | yes |
| pass 4096 x context 131072 (product 553,648,128) | transient 709,820,416 B, attention 108.3 ms | transient <= 5,324,800,000 B and attention <= 200 ms | yes |

- largest product that fits on this grid: 553,648,128
- contexts [65536, 131072], passes [256, 512, 1024, 2048, 4096]
- longest launch 108.3 ms, all launches within the watchdog: True
- stored envelope qk_product 150,994,944 (f5-prefill-envelope.json, not rewritten)

Verdict: INFO — grid beyond 32,768 tokens recorded; the stored envelope was not rewritten
