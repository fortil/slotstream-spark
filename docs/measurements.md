# Measurements

This page lists every figure the project publishes, with the condition it was measured under and the record that holds it. It is a surface of the claims gate: each row of the first table carries the exact text of a claim needle, and `tools/claims_gate.py` fails if the text is missing here or in the record. The README shows only a subset of these rows. Nothing numeric is meant to appear in the README without a row here.

The design and the way of presenting numbers (unit first, conditions next to the headline, medians, what the numbers do not show) follow [slotstream](https://github.com/carloslfu/slotstream) at commit `8bd57938`; see [sources](sources.md).

## How to read a record

Records live in `db/records/`. A measurement record (`db/records/measurements/`) names its date, the capture directory with the raw files, the command that produced it, a table of gates (value, limit, ok) and a one-line verdict. A decision record (`db/records/decisions/`) names the decision, its evidence, the rule it was judged by and how to reverse it. A claim record (`db/records/claims/`) binds one published text to the records that support it.

Verdicts: `PASS` means every gate of the record passed, `INFO` means the record states numbers without a gate, `FAIL` means a gate failed. A capture directory under `db/sources/runs/` holds the raw output of the run. Captures that are not kept in this repository are listed in `db/sources/ARCHIVED.txt`; their records stay.

Unless a row says otherwise, decode speed is the number of generated tokens after the first, divided by the sum of the inter-token times, in tokens per second, and "median" is the median of the samples named in the row. All runs were on one NVIDIA DGX Spark (GB10, 128 GB unified memory) with the model [Qwen3.8-Flash-Next](model-and-repack.md) as MLX affine 4-bit weights repacked for streaming.

## Claims

One row per claim record. The Needle column is the text the gate looks for, unsplit. N is the number of samples or pairs behind the figure.

| Claim | Needle | N and condition | Record |
|---|---|---|---|
| D5 decode, MTP head resident | 29.37 tok/s | median of 5 samples, decode-8k cell, pool pre-filled, measured 2026-10-14; the record's floor is 23 tok/s | [f9-campaign-d5](../db/records/measurements/f9-campaign-d5.md) |
| D5 decode, MTP off | 25.949 tok/s | median of 16 clean pairs, arm A of the MTP A/B, lookahead off | [f6-decode-d5](../db/records/measurements/f6-decode-d5.md) |
| D1 decode (informative) | 9.37 tok/s | median of 3 prompts, decode-8k cell; informative, not gated; the cold 32K cell of that campaign failed | [f9-campaign-d1](../db/records/measurements/f9-campaign-d1.md) |
| Cold 2K prompt, time to first token | 6.80 s | 2048 tokens, campaign N=5, published, not gated | [f9-campaign-d5](../db/records/measurements/f9-campaign-d5.md) |
| Cold 8K prompt, time to first token | 22.033 | median of 5, 8192 prompt tokens, every cold sample had `reused_tokens == 0`; limit 30 s | [f9-campaign-d5](../db/records/measurements/f9-campaign-d5.md) |
| Cold 32K prompt, time to first token | 77.47 s | 31998 tokens, campaign N=5, published, not gated | [f9-campaign-d5](../db/records/measurements/f9-campaign-d5.md) |
| Follow-up after an 8192-token prefix | 1.051 | median of 5, 8192 tokens reused, 137 new; limit 12 s | [f9-campaign-d5](../db/records/measurements/f9-campaign-d5.md) |
| Against llama.cpp arm B | decode 29.37 tok/s = 89.3% of arm B 32.9; cold 8K prefill 372.1 tok/s = 11.6x arm B 32.1 | arm B is unmodified llama.cpp, Q4_1, 43,520 MiB of experts cached on the GPU; its prefill runs on the CPU | [f9-campaign-d5](../db/records/measurements/f9-campaign-d5.md), [lb2](../db/records/measurements/lb2.md) |
| Pool ledger in D5 | 48 chunks | 48 chunks of up to 1 GiB, 388 slots of 2,764,800 B each; 49 chunks is 19,012 (19012) slots, the D5 design ceiling | [f9-campaign-d5](../db/records/measurements/f9-campaign-d5.md), [partition-d5](../db/records/measurements/partition-d5.md) |
| Perplexity against the oracle | engine 4.7216, oracle 4.7265 | 32 wikitext-2 windows, 65,504 tokens, oracle in bfloat16, ratio -0.10 %, gate limit 2 %, passed | [f9-quality](../db/records/measurements/f9-quality.md) |
| Top-1 agreement with the oracle | 0.9545 (210/220) | pooled over the 8 reserved prompts, 220 positions; calibrated bar 0.88976 | [f9-quality](../db/records/measurements/f9-quality.md) |
| Behaviour probe | probe 15/15 | `tools/quality/quality_probe.sh` against the live D5 server | [f9-quality](../db/records/measurements/f9-quality.md) |
| Perplexity against llama.cpp | engine 4.7216, llama.cpp 4.7239 | same 32 windows, prefill on the CPU (`prefill_device=cpu`), difference 0.05 % | [f9-reference](../db/records/measurements/f9-reference.md) |
| Exactness of the reference repack | exact except the 7 named exceptions | sweep over 3215 tensors, absolute error at most 2^-24 on the 7 named ones | [f9-reference](../db/records/measurements/f9-reference.md) |
| Memory impact of the job | no oom-kill, MemAvailable >= 8 GiB | f9-engine job window, production paused by the user; latency, PSI and quiet-window criteria are not applicable there | [f9-impact](../db/records/measurements/f9-impact.md) |
| Lowest MemAvailable in the job | 59853 | MiB, from the 1 s samples of the f9-engine job | [f9-impact](../db/records/measurements/f9-impact.md) |
| Lookahead stays off | paired median 0.8761 | 4 clean pairs, adoption bar 1.02, stopped for futility; decision KEEP_OFF | [f6-lookahead-d5](../db/records/decisions/f6-lookahead-d5.md) |
| MTP head stays resident | paired median 1.1609 | 16 clean pairs, 14 of 16 positive, bar 1.05, outputs identical; decision ADOPT_RESIDENT | [f6-mtp-d5](../db/records/decisions/f6-mtp-d5.md) |
| Token graph stays off | paired median 0.9914 | 6 clean pairs, bar 1.05 not met, soak skipped because the A/B did not pass; decision KEEP_OFF | [f8-token-graph](../db/records/decisions/f8-token-graph.md) |
| Cold 78K prompt at a 131,072-token window, time to first token | 221.89 s | 78000 tokens, campaign N=5, 38 chunks, measured 2026-10-21; published, not gated | [w131k-campaign-d5](../db/records/measurements/w131k-campaign-d5.md) |
| Follow-up after a 76800-token prefix, 131,072-token window | 7.389 | median of 5, 76800 tokens reused, 1337 new; limit 12 s, reuse floor 70000 | [w131k-campaign-d5](../db/records/measurements/w131k-campaign-d5.md) |
| D5 decode at a 131,072-token window | 26.918 | median of 5, decode-8k cell, 37 chunks, MTP head resident; informative beside the 29.37 tok/s of the default window; floor 23 tok/s | [w131k-campaign-d5](../db/records/measurements/w131k-campaign-d5.md) |
| Pool ledger in D5 at a 131,072-token window | 38 chunks | 37 chunks in the decode cell and 38 in the TTFT cells, fixed bytes 25,270,020,452 B; 40 at the D5 ceiling, calculated | [w131k-campaign-d5](../db/records/measurements/w131k-campaign-d5.md) |
| Memory impact of the window job | no oom-kill, MemAvailable >= 8 GiB | w131k-engine job window, production paused by the user; latency, PSI and quiet-window criteria are not applicable there | [w131k-impact](../db/records/measurements/w131k-impact.md) |
| Lowest MemAvailable in the window job | 25582 | MiB, from the 1 s samples of the w131k-engine job | [w131k-impact](../db/records/measurements/w131k-impact.md) |

## Decode and first-token campaign

The D5 campaign ran 5 samples per cell on 2026-10-14 ([f9-campaign-d5](../db/records/measurements/f9-campaign-d5.md)). Gates and limits from that record:

| Cell | Median | Limit | Passed |
|---|---|---|---|
| D5 decode, decode-8k (tok/s) | 29.374 | at least 23 | yes |
| Cold 8K first token (s) | 22.033 | at most 30 | yes |
| Follow-up first token (s) | 1.051 | at most 12 | yes |

Individual samples, same record: decode-8k 32.08, 30.37, 28.63, 29.37 and 27.39 tok/s; cold 8K first token 21.60, 22.03, 22.25, 21.94 and 22.18 s. The cold 2K prefill ran at 301.8 tok/s and the cold 32K prefill at 413.3 tok/s (published, not gated).

The D1 campaign of the same job ran 3 samples and is informative only ([f9-campaign-d1](../db/records/measurements/f9-campaign-d1.md)): decode-8k 9.37 tok/s, cold 2K first token 8.42 s, cold 8K first token 24.50 s with a follow-up of 14.46 s. All three cold 32K samples failed with "the stream ended without a final frame", so D1 has no 32K figure.

## Context window 131,072

The same campaign runner, with the engine started at a 131,072-token window (`SLOTSTREAM_CONTEXT_WINDOW=131072`), ran 5 samples per cell on 2026-10-21 ([w131k-campaign-d5](../db/records/measurements/w131k-campaign-d5.md)). The `ttft-cold-78k` cell cuts a 78000-token prompt from its own frozen text (`bench/prompts/ttft_long.json`) and sends a second turn on the same server. Gates and limits from that record:

| Cell | Median | Limit | Passed |
|---|---|---|---|
| D5 decode, decode-8k (tok/s), 37 chunks | 26.918 | at least 23 | yes |
| Cold 8K first token (s) | 22.407 | at most 30 | yes |
| Follow-up first token (s), 8K | 1.187 | at most 12 | yes |
| Cold 78K first token (s) | 221.89 | published, not gated | |
| Follow-up first token (s), 78K | 7.389 | at most 12 | yes |
| Follow-up reused tokens, 78K, least sample | 76800 | at least 70000 | yes |
| Peak of the worst cell against the ledger (B) | 56438362112 | at most 66072209764 (38 chunks) | yes |

Individual samples, same record: decode-8k 30.79, 26.92, 27.49, 25.29 and 24.37 tok/s; cold 8K first token 22.53, 22.49, 22.41, 22.39 and 22.37 s; cold 78K first token 221.91, 222.41, 220.81, 221.89 and 221.24 s (prefill 351.9 tok/s); follow-up 78K 7.39, 7.39, 7.37, 7.39 and 7.37 s with 76800 tokens reused and 1337 new in every sample. The pool held 37 chunks in the decode cell and 38 in the TTFT cells; 40 is the calculated count at the D5 ceiling. The job's impact ([w131k-impact](../db/records/measurements/w131k-impact.md)): no oom-kill, lowest MemAvailable 25582 MiB, 14 containers up. The decision to serve 131,072 in production and its reversal rule are in [context-window](../db/records/decisions/context-window.md).

Before this campaign could run, two limits had to go: the stable top-k kernel of the indexer accepted rows of 16,384 compressed blocks (65,536 positions), and the first request of a server left every pool slot pinned after a prefill of several passes, so its first decode step raised `slotio.Exhausted` (CHANGELOG, Unreleased).

## Prefill envelope

The F5 run measured prefill with 5 cold samples ([f5-prefill](../db/records/measurements/f5-prefill.md), 2026-10-13, earlier than the campaign, so its numbers differ from the table above): cold 8K first token median 23.831 s, follow-up median 1.213 s, cold 8K prefill 344.2 tok/s, 10.72 times the llama.cpp arm B prefill of 32.1 tok/s. One 2K pass took 8.36 s (245.7 tok/s) and one 32K pass 92.74 s (353.5 tok/s).

Prefill pass size matters for expert reads. At 16K tokens with a read scope of 0, a pass of 4096 read 89017 records and ran at 384.2 tok/s, and a pass of 1024 read 304421 records and ran at 161.5 tok/s (same record). The CUDA transient and launch time per pass are in [f5-prefill-envelope](../db/records/measurements/f5-prefill-envelope.md): the largest launch measured was 49.6 ms and the measured grid covers contexts up to 32,768 tokens only. A second grid at 65,536 and 131,072 tokens is in [w131k-envelope](../db/records/measurements/w131k-envelope.md): every cell up to a 4096-token pass at 131,072 tokens fits (transient 709,820,416 B, attention 108.3 ms), so the largest product that fits on that grid is 553,648,128 against the stored 150,994,944; the stored envelope was not rewritten, which is why a 78000-token prompt still runs 31 passes.

## Decode A/B decisions

Three features were measured as paired A/B runs and judged by a rule fixed before the run. The rule needs a median at or above the bar, at least 80 % positive pairs, at least 5 clean pairs and identical outputs.

| Feature | Arms | Pairs run | Paired median | Bar | Decision |
|---|---|---|---|---|---|
| MTP head | off against resident | 16 | 1.1609 | 1.05 | adopt resident ([record](../db/records/decisions/f6-mtp-d5.md)) |
| Lookahead | off against attention-corrected | 4 | 0.8761 | 1.02 | keep off ([record](../db/records/decisions/f6-lookahead-d5.md)) |
| Token graph | segmented against graph | 6 | 0.9914 | 1.05 | keep off ([record](../db/records/decisions/f8-token-graph.md)) |

With the MTP head resident, the verify rounds accepted a first draft token 3405 times in 3803 (0.895) and a second one 3006 times in 3403 (0.883), 2.68 tokens per round ([f6-mtp-d5](../db/records/decisions/f6-mtp-d5.md)).

## Session noise and pool invariance

The F3 session ran 16 A/A pairs, all clean; the median absolute deviation between two identical arms was 0.004216 and the maximum 0.011606 ([h3](../db/records/measurements/h3.md)). Every adoption bar above is larger than that noise. In the same record the D5 decode median over the clean A/A runs was 19.677 tok/s, before the MTP head and the later kernel work; the later campaign figure is the one in the claims table.

Pool size does not change the output. Tokens and logits were bitwise equal with pools of 13, 88, 181 and 512 slots per layer, and 1000 expert reads matched an independent `O_DIRECT` read byte for byte ([h1](../db/records/measurements/h1.md)). The h3 record repeats the check at the D5 maximum with a 3-row verify equal to three 1-row steps.

## Hit rate by pool size

Simulation over the router traces of the F3 session (48 traces, 2,926,080 accesses), not a measurement of the engine ([h-curve](../db/records/measurements/h-curve.md)). N is experts per layer, held in one global cache keyed by layer and expert.

| N | Records | LRU hit rate | CLOCK hit rate |
|---|---|---|---|
| 13 | 624 | 0.3521 | 0.3625 |
| 48 | 2304 | 0.6235 | 0.6327 |
| 105 | 5040 | 0.8093 | 0.8085 |
| 200 | 9600 | 0.9426 | 0.9508 |
| 396 | 19008 | 0.9927 | 0.9925 |

## Reads, pool memory and kernels

- Expert layout: 800 KiB direct reads reached 10.4 GB/s at queue depth 12, and 50 KiB reads reached 2.1 GB/s at depth 12, which is why the experts are repacked into contiguous records ([io2](../db/records/measurements/io2.md), 19 runs, `O_DIRECT` unless the label says buffered).
- Pool memory type: a `vmm_host_numa` chunk gathered at 245.57 GB/s against 226.63 GB/s for `host_alloc`; `anon_thp` is not addressable by the kernels and gathered at 161.74 GB/s ([pool](../db/records/measurements/pool.md), one 1 GiB chunk of 388 slots).
- GEMV: a 2560 x 2560 affine 4-bit GEMV timed inside a CUDA graph with the L2 cache flushed between calls reached 193.75 GB/s, 79.7 % of the 243 GB/s memory bandwidth; the 248320 x 2560 output projection reached 232.83 GB/s ([m0-gemv](../db/records/measurements/m0-gemv.md)). The record also lists the plain `do_bench` figures, which are lower and informational.
- Wall-time partition of D5 decode: named phases plus "other" add up to the wall time, with "other" at most 10 % ([partition-d5](../db/records/measurements/partition-d5.md)).

## Real workload

Figures from the real-workload records of October 16, 17 and 20, 2026, mode D5. Each needle is also in the README section of the same name. The comparison is informative and inconclusive at 13 sections.

| Claim | Needle | N and condition | Record |
|---|---|---|---|
| Status of the profile calls | 3 of 3 | N = 3 calls with status 200; JSON of the two `json_object` calls not parsed; 0 failed reading segments of 1 in the comparison | [workload-profile](../db/records/measurements/workload-profile.md), [workload-comparison](../db/records/measurements/workload-comparison.md) |
| Cold start and call latencies | cold start 20.1 s, all calls <= 270 s (json16k 64.8 s, json25k 62.4 s, text8k 76.4 s) | one call per shape, `max_tokens` 6000, engine stopped before the run, host services running; 0 calls over 300 s | [workload-profile](../db/records/measurements/workload-profile.md) |
| Memory handover | no oom-kill, residents Up after the engine stopped | one stored day, 13 report sections, 14 of 14 containers up; engine and residents never ran together | [workload-rehearsal](../db/records/measurements/workload-rehearsal.md) |
| Against gpt-oss-120b served by vLLM | the engine completed the stored day with no failed segment; the comparison is inconclusive at 13 sections | 13 sections, 3 supervisor calls, 0 escalated items; citation validation 0.0% against 23.1% | [workload-comparison](../db/records/measurements/workload-comparison.md) |
| Production nights | 26 of 26 reading segments ok, 90.1 s to 133.6 s, median 113.25 s | nights of October 19 and 20, 2026; 1203 and 1488 mentions; 0 failed calls; no allocation error while the engine ran | [workload-nights](../db/records/measurements/workload-nights.md) |

## Numeric contract

The engine is judged against the transformers oracle by limits fixed before any engine run: per layer, `L_k = 2*s_k + 0.01`, where `s_k` is the oracle's own bfloat16-against-float32 spread. The logits limits were calibrated because the oracle's own bfloat16 pair scored top-1 0.9449 over 254 positions, below the first fixed bar. Details, the per-sequence rules and the calibrated tables are in [numeric-contract](../db/records/decisions/numeric-contract.md). Layer 0 matched the oracle at relative L2 0.0062751 against a limit of 0.032884 ([h1](../db/records/measurements/h1.md)).

## What failed or was withdrawn

- Lookahead lost in every pair that ran (4 of 4 below 1), so it stays off. Only 4 of 16 planned pairs were run because the 80 % positive bar could no longer be reached ([f6-lookahead-d5](../db/records/decisions/f6-lookahead-d5.md)).
- The token graph did not reach its bar. Its soak ran for 0 s of the 28,800 s wanted, so the soak result is "not run", not "passed" ([f8-token-graph](../db/records/decisions/f8-token-graph.md)).
- Against the llama.cpp yardstick the engine is slower in decode: 89.3 % of arm B with the MTP head, 78.9 % without it ([f6-decode-d5](../db/records/measurements/f6-decode-d5.md)). An earlier same-day pair ratio of 0.6150 was taken with a smaller arm B cache ([stopgo-f3](../db/records/decisions/stopgo-f3.md)).
- The cold 32K cell of the D1 campaign failed three times (see above).
- The perplexity over 145 windows is 5.7338 against a published reference of 5.3914, 6.35 % higher; that reference's text and window protocol are not published, so the comparison is informative and never a pass ([f9-quality](../db/records/measurements/f9-quality.md)).
- The latency, PSI and quiet-window criteria of the production-impact report are not applicable in the f9 job because production was paused ([f9-impact](../db/records/measurements/f9-impact.md)). Impact on a live neighbor workload is not measured by that record.

## How to add a measurement

1. Run the work through `deploy/guard.sh` ([guard](guard.md)) so the capture carries the guard files.
2. Write the record in `db/records/measurements/` with the date, the capture directory, the command, a gate table and a verdict line, then run `tools/db_check.py db/records/measurements/*.md`.
3. To publish a figure, add a claim in `db/records/claims/` with the needle copied from the record, then put the same text on this page and, if it is shown there, in the README.
4. Run `tools/claims_gate.py` and `tools/docs_check.py`. A figure that is not in a record, or is written differently in a document, fails.
