# Architecture

This page describes how the engine runs one model, Qwen3.8-Flash-Next in MLX affine 4-bit (103.8 GB on disk), on a machine whose memory is smaller than the model. The routed experts stay on NVMe and are read into a pool of GPU-visible host memory as the router asks for them. The design follows slotstream, the Mac engine that does the same with Metal; where a mechanism or its wording comes from there, the file is named and the source is credited in `THIRD_PARTY_NOTICES.md`.

Every figure below sits next to the record that holds it. A record is a file in `db/records/` with the command, the capture and a verdict line (the campaign table is in `measurements.md` and the terms are in `glossary.md`). Values taken from code are marked with the file. Estimates are labeled as estimates.

## Components

| Component | Where | Job |
|---|---|---|
| server | `slotstream_spark/server/` | HTTP on 127.0.0.1: Anthropic Messages, OpenAI Chat, Responses, Ollama dialects. One generation at a time, a bounded queue, 503 while the model loads. |
| front | `slotstream_spark/front.py` | Small socket-activated process that starts the engine on the first request and answers 503 with `Retry-After` while it is not up. |
| engine | `slotstream_spark/engine/` | Generation loop, MTP drafts and rollback, prefill sweep, prefix cache in memory and on disk, sampler. |
| memory | `slotstream_spark/memory/` | Byte ledger, planner, governor, footprint accounting, signal thread. |
| runtime | `slotstream_spark/runtime/` | CUDA graphs per segment, the expert pool wrapper, pinned readback. |
| model | `slotstream_spark/model/` | Forward pass: 48 layers, of which 36 are gated delta net (GDN) and 12 are selected attention (QSA), plus hyper-connections, n-gram embedding, MoE, MTP head, `lm_head`. |
| kernels | `slotstream_spark/kernels/` | Triton kernels for the affine 4-bit GEMV and the MoE, plus vendored attention and GDN kernels. |
| slotio | `slotio/` | C++17 extension with its own threads: the pool, the O_DIRECT reader lanes, n-gram rows, `ensure`, CLOCK. |
| guard | `deploy/guard.sh`, `slotstream_spark/guard/` | Supervisor for every heavy run. See [guard](guard.md). |

The process sets `oom_score_adj` to 1000 at start, listens on 127.0.0.1 only and holds its own model lock.

Files on NVMe (sizes in `install-spark.md` and `model-and-repack.md`): `experts.bin` with every routed expert as one fixed-size record, `ngram.bin` with the n-gram rows, `trunk.safetensors` with everything else, `mtp_experts.bin` and `mtp.safetensors` for the MTP head. The original checkpoint is only read during repacking.

## Memory map

The ledger in `slotstream_spark/memory/ledger.py` counts every term as an integer number of bytes. The terms and their sizes:

| Term | Kind of memory | Size | Source |
|---|---|---|---|
| Trunk (GDN, QSA, indexer, hyper-connections, routers, shared expert, PLE projection, `lm_head`) | `cudaMalloc` through preallocated torch tensors | 2,566,424,320 B | `ledger.py` `TRUNK` |
| Expert pool | host memory the GPU reads, in chunks of 1 GiB, 388 slots each | chunks x 2^30 B | `config/modes.py` |
| Expert record | one slot | 2,764,800 B | [pool.md](../db/records/measurements/pool.md) |
| Staging for the sweep | host memory the GPU reads, groups of 32 records | 88,473,600 B per group, 4 groups charged | `ledger.py` |
| Staging, read scratch and row caches | host | 556,947,200 B in total (4 groups plus 203,052,800 B) | `ledger.py` `SCRATCH_ROWS` |
| Read scratch of the safe path | anonymous host memory, 4096-aligned, never mapped by the GPU | one record per lane, 12 lanes = 33,177,600 B | this page, inside the term above |
| KV cache and indexer keys | `cudaMalloc` | 27,648 B per token of the served window (905,969,664 B at the default 32,768 tokens, 3,623,878,656 B at 131,072) | `ledger.py` `KV_PER_TOKEN` |
| GDN state and conv state | `cudaMalloc` | 115,642,368 B per conversation, of which 113,246,208 B is the recurrent state | `ledger.py` `GDN_STATE` |
| MTP head | `cudaMalloc` | 1,470,946,816 B, resident in D2 and D5 | `ledger.py` `MTP` |
| Verify record ring (MTP resident) | `cudaMalloc` | three copies of the GDN and conv state | `ledger.py` `MTP_RECORD` |
| Prefill pass workspaces | `cudaMalloc` | 1,300,000 B per token of the pass | `ledger.py` `PASS_PER_TOKEN` |
| Read scope | `cudaMalloc` | 234,881,024 B | `ledger.py` `READ_SCOPE` |
| Sidecar of the lookahead correction | host | 37,540,708 B | `ledger.py` `SIDECAR` |
| Runtime (CUDA context, torch, Triton) | mixed | 2,638,217,216 B (2,516 MiB), measured by the engine process | `config/platform.toml` `measured.runtime_footprint_mib`, [m0prime.md](../db/records/measurements/m0prime.md) |
| Margin | none | 1,000,000,000 B | `ledger.py` `MARGIN` |

The rule behind the table: everything large is reserved at start with the size the planner fixes. Nothing grows while serving. The pool grows and shrinks only between requests and only by whole chunks.

Per mode, the ledger adds a prefix-cache term (500,000,000 B in D1, 1,000,000,000 B in D2 and D5, at the default window; above it the term scales with the window and a snapshot term is added, see [memory-modes](memory-modes.md), "Context window") and the prefill pass at the mode's token limit (1,024, 2,048 and 4,096 tokens). With the measured runtime term, `Ledger(mode).fixed_bytes()` returns 9,886,822,500 B for D1, 13,188,969,316 B for D2 and 15,851,369,316 B for D5 (computed from the code on the date of this page, no record). The constants in `config/modes.py` (8,821,658,084 B, 12,123,804,900 B and 14,786,204,900 B) are the design values, which assumed a 1,750,000,000 B runtime. The two sets differ by 1,065,164,416 B in D5. That is why the design ceiling for D5 is 49 chunks and the D5 campaign ledger counts 48 chunks (ledger 67,390,976,868 B in [f9-campaign-d5.md](../db/records/measurements/f9-campaign-d5.md)).

How a pool of N chunks is spent: a chunk holds 388 slots, so D5 at 48 chunks holds 18,624 slots and at 49 chunks 19,012 slots, which is about 388 experts per layer out of 512 for 48 layers. The pool is a global cache keyed by (layer, expert), not a per-layer table.

## Decode path

One token is a chain of 49 graph segments, G(0) to G(48), with a host step between each pair. The segments are captured once per row count (1 row for decode, 3 rows for a verify) with CUDA graphs and do not change when the pool changes, because the MoE kernel looks up the slot at run time: it reads `slot_ptr[map[layer][expert]]`.

```
token t
  host:  n-gram ids -> rows (cache or pread) -> pinned buffer ; embed row -> pinned buffer
  G(0):  embed, PLE (layer 1), hyper-connection, mixer(0), router(0), shared expert(0),
         forecast for layer 1 -> top-10 ids and weights in pinned memory, then seq = 0
  for layer l in 0..47:
      host:  bounded wait on seq(l)            (no stream synchronize)
      host:  ensure(l, ids)                    (hits are pinned; misses are read and published)
      host:  prefetch(l+1) from the forecast   (only candidates above the margin)
      host:  launch G(l+1)
      G(l+1): MoE(l) reads the slots, sums in router-rank order, hyper-connection write,
              mixer(l+1), router(l+1), shared expert(l+1), forecast(l+2), then seq(l+1)
  G(48): final hyper-connection, lm_head -> fp32 logits in pinned memory
  host:  sampler (CPU, bit-exact port of the reference sampler) -> token t+1
```

The comments in `slotstream_spark/runtime/graphs.py` document the same segments. `Readback.wait` is a bounded spin on the sequence word. The engine never calls `cudaStreamSynchronize` in the token loop. On the Mac, the expert scatter took 20 % of decode time, and slotstream traced the slow part to one full GPU synchronization per layer, 48 per token (`MEASUREMENTS.md:2808-2822` of slotstream `8bd57938`); this design removes them but keeps one wait per layer, because the experts of layer l are unknown before its router has run.

### `ensure` and CLOCK

`ensure(layer, ids)` is transactional:

1. Experts already in the pool are pinned for the current generation. The pin generation of layer l at step t is `t * 49 + l`, and the sequence word is also the progress pointer that releases pins of earlier layers.
2. For each miss, a victim slot is chosen by CLOCK before any read starts. CLOCK is global over the pool, with one reference bit per slot. Pinned slots are skipped.
3. The victim's map entry is cleared, then the record is read with `pread` and O_DIRECT, 2,764,800 B per expert. The queue holds at most 12 reads; demand reads go first, speculative reads next, sweep reads last.
4. When all reads of the layer have joined, the map entries are published and the layer launches.

The CLOCK hit rate is a simulation over router traces, not a measurement of the engine: from 0.3625 at 13 experts per layer to 0.9508 at 200 and 0.9925 at 396 ([h-curve.md](../db/records/measurements/h-curve.md), 48 traces, 2,926,080 accesses). The engine's own hit rate in D5 was 0.9917 for the 2,735-token prompt and 0.9607 for the 37-token one ([partition-d5.md](../db/records/measurements/partition-d5.md)).

### Safe path

Every read goes through the safe path: `pread` into an anonymous host scratch that the GPU never maps, registers or reads, then a CPU `memcpy` into the slot (or into the sweep staging). The fast path would `pread` straight into GPU-visible memory. It is out of scope here (`platform.toml` `memory.write_path = "safe"`).

The reason is a precedent from the Mac engine. A pilot run hung for 42 minutes inside one `pread` into GPU-visible staging after a warmup request that had prefetched straight into slots, and the machine needed a restart (`MEASUREMENTS.md:4646` of slotstream `8bd57938`). The cause was never proven. slotstream answered by reading each piece into a host scratch and copying it from the reader thread (`ExpertStore.swift:334-342`), and this engine copies that rule. The price is one `memcpy` per record. It measured 0.1238 ms with 8 lanes ([pool.md](../db/records/measurements/pool.md), `platform.toml` `measured.t_memcpy_ms`), and the copy of the last record of each layer with misses stays on the critical path.

### Pool memory type

Three kinds of pool memory were tried on one 1 GiB chunk ([pool.md](../db/records/measurements/pool.md), decision in [pool-memory.md](../db/records/decisions/pool-memory.md)):

| Type | Gather by GPU, GB/s | Reachable from Triton |
|---|---|---|
| `host_alloc` | 226.63 | yes |
| `vmm_host_numa` | 245.57 | yes |
| `anon_thp` | 161.74 | no |

The engine uses `vmm_host_numa`. Each chunk counts against `MemAvailable` by about 1 GiB whatever the type; `vmm_host_numa` shows up in NVML and barely in the cgroup, so footprint accounting must not rely on the cgroup alone (see the next section).

### Footprint accounting

`slotstream_spark/memory/footprint.py` counts each allocation once. A unified allocation can appear in both `VmRSS` and NVML `used_memory`. `nonpool_bytes` subtracts every unified byte from both counters; `once_bytes` adds each unified byte back one time. The governor uses the larger of the ledger's expected peak and the measured `once_bytes`, and never `cudaMemGetInfo` or the cgroup, which do not see CUDA allocations (the signal thread is in `memory/signals.py`).

## Prefill

A prompt of 256 tokens or more goes through the sweep, not through the decode path. The sweep reads each layer's experts once for the whole prompt:

```
all n-gram rows of the scope are prefetched on 32 lanes before layer 1
for each layer:
    mixer in chronological tiles (QSA prefill or GDN prefill), router over the whole scope
    counting sort of rows by expert, resident experts first
    for each group of 32 experts:
        resident   -> expert_map points at the pool slot (no copy, CLOCK bit untouched)
        not resident -> pread to scratch + memcpy to staging (queue 12)
        fused MoE with expert_map restricted to the group
        GPU computes group i while the reader fills group i+1 (at most 2 staging groups live)
the pool is not written by the sweep, except that the last pass admits the max(1, slots/48) most used experts per layer
every prefill launch is time-bounded (kernel exec timeout watchdog)
```

The sweep mirrors `Layers.swift:1704-1837`, `PrefillReadPolicy.swift:19-30` and `Context.swift:114-160` of slotstream. In D5, a cold 8K prompt reached a median 23.831 s to first token ([f5-prefill.md](../db/records/measurements/f5-prefill.md), 5 cold samples, TTFT limit 30 s). The read scope is capped at 8192 tokens, which keeps the bytes requested for a long prompt bounded. The pool is pre-filled from a per-layer frequency table (`config/expert_freq_gb10.json`, built from the traces of the F3 session: [f3-session.md](../db/records/measurements/f3-session.md)); `platform.toml` `pool.prefill = "table"`.

## Kernel rules

These are the rules every kernel has to meet, and each has a test against the oracle and against itself:

- Classic Triton with `mma.sync` (MMA v2). No TMA scatter and no `.shared::cluster` gather, which do not exist on sm_12x. Tiles stay inside 99 KiB of shared memory per block.
- Dequantization `q * s + b` with fp32 accumulation, no fast-math flags and exact `l2norm`.
- No atomics in the MoE reduction. Expert outputs are summed in router-rank order.
- Split-K is fixed per shape and does not depend on the number of rows. That makes a 3-row verify bitwise equal to three 1-row decodes. The gate test `tests/gates/test_h3.py` checks it, with pools of 13 and 48 experts per layer and the D5 maximum ([h3.md](../db/records/measurements/h3.md)).

Vendored kernels (attention and hyper-connection from vLLM, GDN from FLA) live under `third_party/vendor` and are loaded through `slotstream_spark/kernels/vendored.py`.

GEMV speed was measured as call time inside a CUDA graph with 128 MiB written between calls to evict L2. For 2560 x 2560 the tuned kernel reached 193.75 GB/s, 79.7 % of the 243 GB/s memory bandwidth ([m0-gemv.md](../db/records/measurements/m0-gemv.md)). The engine's own kernel figure at step 55 was lower, 135.54 GB/s, and `platform.toml` carries 184.44 GB/s from the final tuning (`measured.gemv_2560_gbs`).

## Pool invariance

The graph and the output do not depend on how much of the model is resident. The layer-0 gate ran the same prompt with pools of 13, 88, 181 and 512 experts per layer and got bitwise equal outputs (SHA-256 `c4aae857...fd41e9`, [h1.md](../db/records/measurements/h1.md)). The full-model check repeats it at 13 and 48 experts per layer and at the D5 maximum, with MTP verification equal to three single-row steps ([h3.md](../db/records/measurements/h3.md)). A smaller pool changes speed, never tokens.

## Speculative decoding (MTP)

The MTP head proposes 2 draft tokens. A verify step runs the same graphs in their 3-row variant. On a rejection, the engine restores the GDN state saved by `gdn_step_record`, trims the KV cache and rebuilds the n-gram context. The head costs 1,470,946,816 B plus the record ring, so it is on by policy only in D5 (`config/policy.toml`: D1 and D2 are `off`, D5 is `resident`). The A/B between MTP off and resident in D5 is in [f6-mtp-d5.md](../db/records/decisions/f6-mtp-d5.md), with decode medians in [f9-campaign-d5.md](../db/records/measurements/f9-campaign-d5.md).

Two other ideas were measured and are kept off:

- Lookahead (prefetching the next layer's experts from a forecast) is off in `platform.toml`. In the D5 A/B it gave a paired median of 0.8761 over 4 clean pairs against an adoption bar of 1.02, and the run stopped early for futility ([f6-lookahead-d5.md](../db/records/decisions/f6-lookahead-d5.md)). It never changes routing; a wrong forecast only wastes reads.
- A single token-wide CUDA graph in which the GPU waits on a flag per layer is off (`[f8] token_graph = false`). The partition that opened this work found 11.33 % of token time in host work and waits ([f8-entry.md](../db/records/decisions/f8-entry.md)), above the 10 % entry line. The A/B then gave a paired median of 0.9914 over 6 clean pairs against a bar of 1.05, and the 8-hour soak was skipped ([f8-token-graph.md](../db/records/decisions/f8-token-graph.md)).

## Performance model

Per token: `t_token = t_comp + t_host + (1 - phi) * (t_io_exp + t_copy)`.

- `t_comp` is dense bytes over the dense bandwidth plus expert bytes over the pool bandwidth, with an efficiency eta. The measured eta is 0.797 (193.75 GB/s over 243 GB/s, [m0-gemv.md](../db/records/measurements/m0-gemv.md)).
- `t_io_exp` is 48 layers times the expected miss cost: the first miss of a layer pays the p50 of a queue-depth-1 read, the rest arrive at the aggregate rate.
- `t_copy` is 48 x (1 - h^10) x `t_memcpy`, with `t_memcpy` = 0.1238 ms ([pool.md](../db/records/measurements/pool.md)).
- `t_host` is 48 x tau. The engine's tau was 5.744 us in the bench loop and 14.112 us in the four-layer run ([m0prime.md](../db/records/measurements/m0prime.md)); the D5 decode partition measured 21.44 us at p50 for all-hit layer steps ([partition-d5.md](../db/records/measurements/partition-d5.md)).
- `phi` is the share of reads hidden by lookahead: 0 here, since lookahead is off.

The recalibrated model is in [h0.md](../db/records/decisions/h0.md), with its own label: "Estimated, not measured". It gives 29.0 to 35.1 tok/s for D5 without lookahead, assuming a hit rate h of 0.96 to 0.98.

Measured wall-time partition of D5 decode (256 new tokens, MTP not part of that run, [partition-d5.md](../db/records/measurements/partition-d5.md)):

| Phase | Share of wall, 2,735-token prompt | Share of wall, 37-token prompt |
|---|---|---|
| GPU wait (compute) | 83.50 % | 70.45 % |
| Expert miss read | 7.18 % | 19.31 % |
| Ensure of hits | 2.16 % | 2.55 % |
| Copy tail | 0.95 % | 2.38 % |
| Other host work and waits | 6.23 % | 5.32 % |
| Decode speed (informational) | 21.231 tok/s | 18.432 tok/s |

"Other host work and waits" adds sample, n-gram hash, embed read, n-gram wait, launch and `other` for each column. The same record lists each phase on its own.

Measured end results, with the conditions in the records: D5 decode with MTP resident reached 29.374 tok/s as the median of 5 samples in the decode-8k cell ([f9-campaign-d5.md](../db/records/measurements/f9-campaign-d5.md)). D1 decode was 9.1985 tok/s as the median over the clean A/A runs ([h3.md](../db/records/measurements/h3.md), informative) and 9.37 tok/s in the D1 campaign, whose 32K cold-prefill cell failed in all 3 samples ([f9-campaign-d1.md](../db/records/measurements/f9-campaign-d1.md)).

## Start order

The engine starts in the order slotstream uses:

1. Verify the weights against the pinned hashes (`model.lock.json`, `SHA256SUMS.model`). The hash is computed once and sealed.
2. Measure memory and read the pool-size limits the governor allows (see [memory-modes](memory-modes.md)).
3. Plan: the planner turns the cap into whole chunks, the prefix budget and the prefill pass size.
4. Reserve everything the ledger lists. Nothing is allocated later.
5. Load the trunk, calling `posix_fadvise(DONTNEED)` on the files after reading so that the page cache does not hold a second copy.
6. Capture the graphs.
7. Run a health check, then serve.

If the cap is below the floor (fixed bytes plus 2 chunks, 776 slots), the engine refuses to start or exits.

## IO arbiter

`slotio` has an arbiter (`slotio/src/arbiter.cpp`) that reads `/proc/diskstats` for reads by other processes and the drive temperature, and lowers its queues accordingly. The 56,850 mC figure is a record, not a limit: it is the highest Composite temperature seen without throttling in a ramp that never reached a plateau ([m8.md](../db/records/measurements/m8.md); `platform.toml` `measured.nvme_temp_no_throttle_mc`). A later scratch run saw decode speed unchanged between 45.85 and 59.85 degC (`measured.nvme_temp_no_effect_mc`, informational).

## Prefix cache

The engine keeps recent prompt states in memory (budget per mode in the ledger above; at a window above the default, the window itself in tokens) and, optionally, on disk under `prefix.dir` with a quota of 20,000,000,000 B (`platform.toml`). The in-memory cache is dropped before the pool shrinks, so retained states never outlive the memory they were budgeted from (`memory/governor.py`, `apply_shrink`). The follow-up TTFT in D5 was a median 1.213 s ([f5-prefill.md](../db/records/measurements/f5-prefill.md), limit 12 s).

## Server behavior

- One generation runs at a time. Up to 4 more requests wait (`server.queue_max`); a further one gets 503.
- While the model loads, the front answers 503 with `Retry-After: 10` in the shape of the dialect the request used, and starts the engine only if the governor's start rule allows it ([h4-drills.md](../db/records/measurements/h4-drills.md): first request 503/10, then 200 after 6.1 s).
- Streams send a keepalive after 10 s with nothing to send (`keepalive_s` in `server/app.py`). Output is bounded.
- Images in a request are refused with 400 in each dialect. Routes and variables are in `server-api.md`.
- The engine exits after 1800 s without requests (`SLOTSTREAM_IDLE_EXIT_S`). The front itself exits after 300 s with no engine and no connections. The next request starts both again through the socket.

## Numeric contract

The engine is judged layer by layer against a transformers oracle (`tools/ref_layerwise.py`). Per-layer limits are in `slotstream_spark/config/numeric_contract.toml` and follow `L_k = 2 * s_k + margin`, where `s_k` is the spread between the oracle in bf16 and in fp32, and a wrong-expert control must sit at least 4 times above the spread. For layer 0 the relative L2 error of the engine was 0.0062751 against a limit of 0.032884 ([h1.md](../db/records/measurements/h1.md)). The sampler is a port of the reference sampler, runs on the CPU and uses a splitmix64 stream, so equal seeds give equal tokens on every machine. `exact-q4_1-reference.md` covers the separate exact reference used for perplexity checks.

## What is not here

Lookahead, the token graph and the fast write path are out. D2 has no decode measurement. The 32K cold-prefill cell failed in D1. The memory modes and the governor are in [memory-modes](memory-modes.md).
