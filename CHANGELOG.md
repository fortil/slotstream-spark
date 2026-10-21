# Changelog

## Unreleased

- The context window is configurable: `SLOTSTREAM_CONTEXT_WINDOW` or `serve --context-window`, default 32,768, a multiple of 64 up to the model's 262,144. Above the default the ledger charges the KV cache at the window, the in-memory prefix cache is sized in tokens of the window and no longer charges the request being served, so a 78,000-token conversation keeps its state between turns; the front's start floor follows the same window through `serve.env`, which both units load. The engine asks the cache before copying a state and skips the copies it would refuse. For the measurement: the campaign runner gets `--window` and a `ttft-cold-78k` cell with its own frozen text, `tools/decide.py` judges every peak at the cell's window and gains the `w131k-campaign` and `w131k-envelope` judges, `tools/prefill_envelope.py --contexts` measures other grids without rewriting the stored envelope, and `tools/impact_report.py` names its record after the job label.
- The front no longer starts the engine for a `GET`. A poller that listed the models every second would have started the engine whenever the start rule allowed it and held the request open while it loaded. A `GET` now gets an immediate 503 with the code `engine_stopped`, `loading` or `not_allowed`, so a caller can tell a stopped engine from a refused start.
- The guard drill's case k calls `kill-all.sh --force`. Under pause mode (`NO_WINDOWS=1`) the plain call does nothing, so case k failed whenever production was paused. `deploy/tests/killall_t0.sh` covers it.

## 0.1.0 - 2026-10-15

First public version. Every figure below is in a record; the conditions are in the record.

- Decode of Qwen3.8-Flash-Next with the routed experts streamed from the NVMe into a pool of slots, in three memory modes (D1, D2, D5). D5 decoded at 29.37 tok/s, median of 5 runs ([f9-campaign-d5](db/records/measurements/f9-campaign-d5.md)).
- D1 mode measured at 9.37 tok/s, informative only ([f9-campaign-d1](db/records/measurements/f9-campaign-d1.md)).
- Prefill and time to first token: cold 8K-token prompt in 22.033 s ([f9-campaign-d5](db/records/measurements/f9-campaign-d5.md), [f5-prefill](db/records/measurements/f5-prefill.md)).
- Quality against the oracle and against llama.cpp on the same windows ([f9-quality](db/records/measurements/f9-quality.md)).
- MTP head resident for speculative decode ([mtp](db/records/measurements/mtp.md), [f6-mtp-d5](db/records/decisions/f6-mtp-d5.md)).
- Token graph decision: paired median 0.9914 against the bar of 1.05, so it stays off ([f8-token-graph](db/records/decisions/f8-token-graph.md)).
- Lookahead tried and not adopted ([f6-lookahead-d5](db/records/decisions/f6-lookahead-d5.md)).
- A front process that starts the engine on the first request and answers 503 while it loads; Anthropic Messages, OpenAI Chat, OpenAI Responses and Ollama dialects over one model ([h5-agents](db/records/measurements/h5-agents.md)).
- `deploy/guard.sh` memory guard and its drill ([guard-drill](db/records/measurements/guard-drill.md), [docs/guard.md](docs/guard.md)).
- Real-workload profile and rehearsal for a nightly batch pipeline, with a comparison against gpt-oss-120b served by vLLM that is inconclusive at 13 sections ([workload-profile](db/records/measurements/workload-profile.md), [workload-comparison](db/records/measurements/workload-comparison.md)).
- Claims gate: every README number has a claim record whose needle appears in a measurement record ([docs/measurements.md](docs/measurements.md)).
- Identifiers renamed to English (for example `HUELLA` to `FOOTPRINT`); old records keep their original keys, and the reader accepts both ([glossary](docs/glossary.md)).
