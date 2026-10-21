# m-lb4: real CUDA memory of the llama.cpp yardstick against the envelope
Date: 2026-10-12
Capture: db/sources/runs/2026/10/20261012T071331-lb2/
Command: part of lb2; session.sh reads NVML (nvidia-smi compute apps) and the scope's memory.current/memory.stat after /health is 200

| config | NVML MiB | memory.current MiB | file / anon / shmem MiB | MemoryMax MiB (table) | MemFree at start MiB | op_offload | root-direct file MiB |
|---|---|---|---|---|---|---|---|
| q41-a-d5 p1 | 5,396 | 47,019 | 46,453 / 400 / 621 | 47,616 (47,616) | 72,505 | false | 10,324 |
| q41-b-d5 p1 | 48,918 | 3,583 | 2,924 / 403 / 621 | 4,096 (4,096) | 72,693 | false | 10,954 |

Buffer lines, q41-a-d5: CUDA0 model buffer 3,197.34 MiB, CPU_Mapped 6,623.81 + 37,803.92 + 30,586.78 + 30,517.72 MiB, KV 768 + 192 MiB, recurrent state 112.57 MiB, compute 925.50 MiB (+ 574.98 MiB CUDA_Host). Full lines in mlb4.json.

The session ran with the default CUDA reserve (8,704 MiB) for the MemoryMax table; the real NVML figure for arm A is 5,396 MiB, so the same formula with the measured CUDA gives MemoryMax = min(57,344 - 5,396 - 1,024, 72,505 - 10,240 - 5,396 - 4,096) = min(50,924, 52,773) = 50,924 MiB. The MemFree clamp did not bind (72,505 - 10,240 - 8,704 - 4,096 = 49,465 > 47,616). envelope_exceeded is false for both arms (memory.current + NVML = 52,415 and 52,501 MiB, under 57,344).

Verdict: PASS — arm A holds 47,019 MiB in the scope plus 5,396 MiB of CUDA, inside the 57,344 MiB HUELLA
