# slotstream-spark

<p align="center">
  <a href="https://github.com/fortil/slotstream-spark"><img alt="Repository fortil/slotstream-spark on GitHub" src="https://img.shields.io/badge/repo-fortil%2Fslotstream--spark-181717?logo=github&logoColor=white"></a>
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-blue"></a>
  <img alt="Language: Python 3.12" src="https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="Project type: research engine" src="https://img.shields.io/badge/type-research%20engine-lightgrey">
  <img alt="Built on PyTorch 2.13" src="https://img.shields.io/badge/PyTorch-2.13-EE4C2C?logo=pytorch&logoColor=white">
  <img alt="Built on Triton 3.7.1" src="https://img.shields.io/badge/Triton-3.7.1-76B900?logoColor=white">
  <a href="https://github.com/fortil/slotstream-spark/actions/workflows/ci.yml"><img alt="CI workflow status" src="https://github.com/fortil/slotstream-spark/actions/workflows/ci.yml/badge.svg"></a>
</p>

Qwen3.8-Flash-Next on one NVIDIA DGX Spark, with the routed experts streamed from the NVMe.

This is the engine that [slotstream](https://github.com/carloslfu/slotstream) by Carlos Galarza does on a Mac, rebuilt for CUDA. The checkpoint is `pipenetwork/Qwen3.8-Flash-Next-MLX-4bit` at revision `aa7c790e804bbf9d491ddb109c3d61bc4a555f7c`: [103.8 GB on disk](db/records/measurements/checkpoint.md), a 125B-parameter mixture of experts with about 6B active per token (the [model card](third_party/qwen38-flash-next-metadata/MODEL_CARD.md) names it 125B-A6B), plus a [32 GB n-gram table](docs/model-and-repack.md) and a vision tower. The dense weights stay in memory and the routed experts are read from the NVMe into a pool of slots as the router asks for them.

**In the D5 memory mode on October 14, 2026, with the host's other services paused, the engine decoded at 29.37 tok/s (median of 5 runs, MTP head resident) and answered a cold 8K-token prompt in 22.033 s.**

It is a research engine, not a product. It runs one model on one machine, and the numbers under Speed, Quality and Memory and coexistence each have a claim record that `tools/claims_gate.py` checks against this page; other figures repeat those numbers or link to the record or page they come from.

## Who it is for

- People who want to run this checkpoint on a Spark and read how the memory is budgeted.
- People porting an expert-streaming design to another GPU, who want a worked example with its measurements, including the ones that failed.
- Operators who need an engine that shares a host with other services and gives memory back when they ask for it.

## Will it run on my machine?

Only one machine was measured: a Spark. Requirements as the code states them:

- NVIDIA GB10 (compute capability 12.1, `sm_121`) with 128 GB of unified memory.
- About 104 GB of NVMe for the downloaded checkpoint and about 104 GB more for the repacked files ([install-spark](docs/install-spark.md) has the exact sizes).
- Python 3.12, torch `2.13.0+cu130`, triton `3.7.1` (pinned in `requirements.lock`), a CUDA 13 toolchain, CMake.
- Memory envelopes per mode, ceiling and need in MiB: D1 19,866 and 62,644; D2 26,112 and 56,413; D5 65,126 and 16,896 ([memory-modes](docs/memory-modes.md)). The ceiling is the most the engine may hold. The need is the free memory it leaves for other workloads.

Other GPUs, other NVMe drives and other amounts of RAM were not measured.

## Speed

Decode speed is tokens generated per second after the first token. TTFT is the time from the request to the first token. Unless a row says otherwise, figures are from the D5 mode on October 14, 2026, with the host's other services paused.

**In the D5 memory mode on October 14, 2026, with the host's other services paused, the engine decoded at 29.37 tok/s (median of 5 runs, MTP head resident) and answered a cold 8K-token prompt in 22.033 s.**

| Workload | Median decode speed | Runs |
|---|---|---|
| Decode after an 8K prompt, MTP head resident | 29.37 tok/s | 5 |
| Decode with the MTP head off | 25.949 tok/s | 16 paired |

| Prompt | Median TTFT |
|---|---|
| Cold, 2048 tokens | 6.80 s |
| Cold, 8192 tokens | 22.033 s |
| Follow-up, 137 new tokens on an 8192-token prefix | 1.051 s |
| Cold, 31998 tokens | 77.47 s |

For the 29.37 tok/s row and the TTFT table: D5 mode with 48 chunks of 1 GiB in the pool, pre-filled from a frequency table; MTP head resident; lookahead off; token graph off; checkpoint revision `aa7c790e804bbf9d491ddb109c3d61bc4a555f7c`; medians of 5 runs; the host's other services paused. The MTP-off row is a median over 16 paired runs with the head not loaded. Records: [measurements](docs/measurements.md).

### By memory mode

| Mode | Median decode speed | Status |
|---|---|---|
| D1 | 9.37 tok/s | informative, median of 3, the cold 32K cell failed |
| D2 | not measured | |
| D5 | 29.37 tok/s | gated, median of 5 |

### Against llama.cpp

Against unmodified llama.cpp with Q4_1 and the experts cached on the GPU (arm B): decode 29.37 tok/s = 89.3% of arm B 32.9; cold 8K prefill 372.1 tok/s = 11.6x arm B 32.1. llama.cpp's prefill ran on the CPU in that arm. The engine is faster to the first token and slower in decode.

### What the numbers do not show

- One machine and one checkpoint.
- Medians of short runs, not a long soak.
- The host's other services were paused, so how the engine behaves next to a loaded neighbor is not measured here.
- The pool was pre-filled before the runs. A cold pool starts slower.
- 23 tok/s was a floor set before the runs, not a result.
- No other GPU, NVMe drive or amount of RAM was measured.

## Quality

The 4-bit checkpoint is compared with the transformers oracle and with llama.cpp on the same text, and with a 15-question behaviour probe.

| Check | Result |
|---|---|
| Perplexity on 32 wikitext windows, within 2 % of the transformers oracle | engine 4.7216, oracle 4.7265 |
| Perplexity against llama.cpp on the exact repack, same windows | engine 4.7216, llama.cpp 4.7239 |
| Pooled top-1 agreement with the oracle on the 8 reserved prompts, floor 0.88976 | 0.9545 (210/220) |
| Behaviour probe | probe 15/15 |

The repack of the checkpoint is exact except the 7 named exceptions: a few scale and bias tensors rounded by at most 2^-24, and `ple_conv1d` written as F32. The 194 hyper-connection matrices are stored as BF16 in the llama.cpp reference, not as Q4_1. The details are in [exact-q4_1-reference](docs/exact-q4_1-reference.md).

Over 145 windows the engine gives 5.7338 against a published 5.3914. That figure is informative only: the text and window protocol of the published number are not known, so it cannot serve as a criterion.

## Real workload

A nightly text-analysis pipeline calls an OpenAI-compatible chat endpoint with long prompts, one call at a time, some with `response_format` `json_object`. Three records replay that shape against the engine in mode D5 on October 16 and 17, 2026. The setup behind a LiteLLM proxy is in [integration-litellm](docs/integration-litellm.md).

**Call profile** ([workload-profile](db/records/measurements/workload-profile.md)). One call per shape, N = 3 calls, `max_tokens` 6000, engine stopped before the run, the host's other services running.

| Call | Prompt tokens | Completion tokens | Total seconds |
|---|---|---|---|
| json16k, `json_object` | 15740 | 562 | 64.778 |
| json25k, `json_object` | 25906 | 22 | 62.369 |
| text8k, plain text | 8154 | 1347 | 76.44 |

Record verdict: cold start 20.1 s, all calls <= 270 s (json16k 64.8 s, json25k 62.4 s, text8k 76.4 s). The first request on a stopped engine was answered after 20.137 s in 1 attempt. 0 calls went over the pipeline's 300 s attempt timeout, and 3 of 3 calls returned status 200. The profile checks the status and a finished response; it does not parse the JSON of the two `json_object` calls.

**Memory handover** ([workload-rehearsal](db/records/measurements/workload-rehearsal.md)). One stored day replayed with the same pipeline, 13 report sections, the host's resident services stopped for the run and restarted after it: no oom-kill, residents Up after the engine stopped, with 14 of 14 containers up. The engine was ready 17 s after each of its two starts. The engine and the resident services never ran together in this run, so it says nothing about memory pressure with both up.

**Against another model** ([workload-comparison](db/records/measurements/workload-comparison.md)). The same stored day, 13 sections, 1 reading segment, 3 supervisor calls and 0 escalated items, run once with gpt-oss-120b served by vLLM and once with the engine.

| Gate | gpt-oss-120b served by vLLM | Engine |
|---|---|---|
| Citation validation | 0.0% | 23.1% |
| Sections without cuts | 0 | 3 |
| Reading cuts | 13 | 10 |
| Synthesis and summary cuts | 20 | 18 |
| Mean reading characters | 146.2 | 274.2 |

Both models had 0 failed reading segments of 1 and 0 warnings. Polarity concordance is 0 of 0, not measurable, because no item was escalated. Record verdict: the engine completed the stored day with no failed segment; the comparison is inconclusive at 13 sections. This page does not claim the engine is better or worse than the reference. Both citation rates are low.

**Production nights** ([workload-nights](db/records/measurements/workload-nights.md)). Since October 17 the engine is the pipeline's reasoning model in production, started and stopped by the pipeline around its reading window. On the nights of October 19 and 20: 26 of 26 reading segments ok, 90.1 s to 133.6 s, median 113.25 s, with 7534 to 14921 prompt tokens per segment. That was 1203 and 1488 mentions read, 26 and 30 calls, 0 failed calls, and no GPU allocation error in the kernel log while the engine ran. On the first night the engine was left running when the resident services started, and one of them failed a GPU allocation and restarted; that night is not counted.

### What this does not show

- One pipeline, one language, one stored day, and two production nights.
- Page cache probably warm from earlier runs; a cold cache after a reboot loads slower.
- No labeled evaluation set. Quality is judged by structural gates and a polarity concordance, not by humans.
- Three calls per profile, so no percentile; the latencies are single calls.

## Memory and coexistence

The engine does not take a fixed amount of memory. A governor recomputes the allowed pool size every second and grows or shrinks the pool by whole 1 GiB chunks ([memory-modes](docs/memory-modes.md)).

During the final job: no oom-kill, MemAvailable >= 8 GiB, with a minimum MemAvailable of 59853 MiB. The host's other services were paused by the user during the final runs, so the criteria for their latency, memory pressure and quiet windows were recorded as not applicable. The criteria for oom, free memory and container count were gates and passed.

Three switches have a recorded decision:

- Lookahead of experts stays off: paired median 0.8761 against an adoption bar of 1.02, stopped after 4 clean pairs.
- The MTP head stays resident: paired median 1.1609 against a bar of 1.05.
- The token graph stays off: paired median 0.9914 against a bar of 1.05, over 6 clean pairs.

## Install

The full path, in order, is in [install-spark](docs/install-spark.md). In short, on the Spark:

```
bash tools/spark/bootstrap.sh
bash tools/spark/setup_env.sh
bash tools/build_slotio.sh
bash tools/spark/dl_checkpoint.sh
tools/spark/jobs.sh start repack 6144 7200 -- tools/capture.sh repack -- bash tools/spark/repack_all.sh
```

Downloads and the repack take hours and about [210 GB of disk](docs/install-spark.md). Nothing in this repository installs system packages or changes the host without you running it.

## Use it

```
tools/spark/py.sh -m slotstream_spark.cli doctor
bash tools/spark/install_units.sh
```

The front process listens on `127.0.0.1:18434` and starts the engine on the first request. The server speaks the Anthropic Messages, OpenAI Chat, OpenAI Responses and Ollama dialects over one model ([server-api](docs/server-api.md)). It listens on loopback only; from a Mac, `tools/mac/tunnel.sh up` forwards the port and `tools/mac/launch.sh claude` starts Claude Code connected to it. Text only: a request with an image gets a 400.

## How it works

The idea comes from slotstream. Its README puts it this way: "Slotstream keeps shared weights in memory and reads the needed experts from SSD into a cache" (`README.md:238-239` of slotstream `8bd57938`). This engine keeps the same split and the same memory arithmetic, and adds what CUDA on unified memory needs:

- The pool lives in unified memory allocated through the virtual memory API. One 1 GiB chunk was gathered at 245.57 GB/s ([pool](db/records/measurements/pool.md)).
- `slotio`, a C++17 extension with its own threads, owns the pool, the O_DIRECT reader lanes, the n-gram rows, the `ensure` call and the CLOCK eviction.
- Every read takes the safe path: `pread` into a host scratch the GPU never maps, then a copy into the slot. Reading straight into GPU-visible memory is not used, because of a hang that slotstream recorded ([architecture](docs/architecture.md)).
- Lookahead of experts is implemented and measured, and it is off (result above).
- The governor shrinks fast and grows slowly, and exits at its floor rather than starve the host.

The design is split into components in [architecture](docs/architecture.md); terms are in [glossary](docs/glossary.md).

## Built on PyTorch and Triton

The model code is PyTorch with Triton kernels instead of Swift on MLX. slotstream is Swift end to end and runs only on Apple hardware; there is no Swift runtime for this GPU stack, and the attention and delta-net kernels come from vLLM and flash-linear-attention, which are Python and Triton. The cost is host work in the token loop: the partition that opened the token-graph experiment found 11.33 % of token time in host work and waits ([f8-entry](db/records/decisions/f8-entry.md)).

## Status and limits

- Text only. The vision tower is not loaded.
- One generation at a time, with a queue of 4.
- Context window 32,768 tokens by default. `SLOTSTREAM_CONTEXT_WINDOW=131072` raises it to 131,072, where one 78K conversation is kept in the prefix cache at a time; the figures above were measured at the default window.
- One checkpoint, one machine, no authentication: loopback only, reach it through an ssh tunnel.
- The only comparison with llama.cpp is the one above.
- Not a product: no releases, no wheels, no support promise.

## FAQ

**Why not load the whole model?** It is [103.8 GB](db/records/measurements/checkpoint.md) on a host with 128 GB shared with other services. The engine must leave memory free for them.

**Does it change outputs?** The repack is exact except the named exceptions, and quality is checked against the oracle above.

**Can I run it without other services on the host?** Yes. An empty router URL and an empty container count turn both host checks off ([install-spark](docs/install-spark.md)).

**Why is the first request slow?** The front starts the engine on demand and answers 503 with `Retry-After` while it loads. A cold prompt also pays the prefill.

**Where do I report a bad number?** Open an issue with the record path and the command you ran.

## Measurements and claims

Every measurement is a record in `db/records/measurements/`, every decision one in `db/records/decisions/`, and raw captures live under `db/sources/` ([db/DB.md](db/DB.md)). A claim record in `db/records/claims/` names the exact text of a number, the records that support it and the pages that show it. `tools/claims_gate.py` checks the claim needles against this page and fails when a number in Speed, Quality or Memory and coexistence has no claim. It does not check every figure on the page.

## Acknowledgements

- [slotstream](https://github.com/carloslfu/slotstream) by Carlos Galarza: the design, the memory arithmetic, the record format, the dialects, the measurement discipline and the claims contract. Without it this would not have been possible.
- helgard-orlm, whose `qwen-flash-next-8gb` engine reads the same experts from NVMe on an 8 GB GPU and was the source of several ported pieces.
- vLLM and flash-linear-attention, for the attention and delta-net kernels.
- The Qwen team and pipenetwork, for the model and the MLX checkpoint.
- llama.cpp, used as the reference for the exact repack and the speed comparison.
- nanobind and doctest, for the native module and its tests.

Licenses and ported files: [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md).

## Who made this

William Penagos wrote and measured this on a single Spark.

## License

MIT, see [LICENSE](LICENSE). The model weights are not part of this repository and follow the license of the Qwen checkpoint.
