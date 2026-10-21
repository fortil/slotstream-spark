# Server API

The engine speaks four wire dialects over one model: Anthropic Messages, OpenAI Chat Completions, OpenAI Responses and Ollama. Existing SDKs and coding tools connect without an adapter. The routing and request handling follow `Server.swift` of [slotstream](https://github.com/carloslfu/slotstream) at commit `8bd57938`; the code is in `slotstream_spark/server/app.py` and `slotstream_spark/server/dialects/`.

## Ports and processes

| Process | Address | Role |
|---|---|---|
| front (`slotstream_spark/front.py`) | `127.0.0.1:18434` | the address clients use; starts the engine on demand and relays bytes without buffering |
| engine (`python -m slotstream_spark.cli serve`) | `127.0.0.1:18435` | the HTTP server above the model |

The engine listens on loopback only and refuses to bind any other address (`slotstream_spark/server/http_limits.py`, `bind_host`). The front listens on loopback too, with one optional exception: a second address added by a socket drop-in for containers on the same host (see "Extra bind" below). There is no authentication on loopback, so reach it from another machine through an ssh tunnel (`tools/mac/tunnel.sh`, see [install-spark](install-spark.md)), never by opening the port.

The front is started by a systemd socket unit (`deploy/slotstream-spark.socket`, which hands the connection to `deploy/slotstream-spark-front.service`) and exits after 5 minutes with no engine and no connection. The engine exits after 1800 s without a request unless `SLOTSTREAM_IDLE_EXIT_S` says otherwise. The policy that decides whether a request may start the engine is in [memory-modes](memory-modes.md).

## Routes

| Method and path | Dialect | Notes |
|---|---|---|
| `POST /v1/messages` | Anthropic | streaming and non-streaming, tool use, thinking blocks split from the answer |
| `POST /v1/messages/count_tokens` | Anthropic | tokenizes only; never takes the generation gate |
| `POST /v1/chat/completions` | OpenAI Chat | streaming and non-streaming, tool calls |
| `POST /v1/responses` | OpenAI Responses | streaming and non-streaming |
| `GET`, `DELETE /v1/responses/{id}` | OpenAI Responses | always 404 `stored_state_unsupported`: the server stores no responses |
| `GET /v1/models` | OpenAI | the served model id, `context_window` (the window the engine was started with) and `max_output_tokens` |
| `POST /api/chat`, `POST /api/generate` | Ollama | streaming and non-streaming |
| `GET /api/tags`, `POST /api/show`, `GET /api/ps`, `GET /api/version` | Ollama | metadata only |
| `GET /metrics` | Prometheus text | request counters per route and status, queue length, busy flag, engine snapshot |
| `GET /` | plain text | `slotstream-spark is running` |

Any other path answers 404 in the error shape of the route's dialect, so an SDK can parse it.

## Limits and refusals

- One generation runs at a time. Up to 4 more requests wait in order (`[server] queue_max` in `slotstream_spark/config/platform.toml`). A fifth gets 503 `queue_full` with `Retry-After: 1`. Waiting is cancelable: a client that disconnects leaves the queue without taking the slot.
- While the engine loads and `SLOTSTREAM_FRONT_WAIT_S` is 0 or has run out, the front answers 503 with `Retry-After: 10` in the error shape of the dialect that was spoken (Anthropic clients get `overloaded_error`). When the start policy refuses, the front answers 503 and starts nothing.
- A `GET` never starts the engine and is never held. While the engine is not ready it gets an immediate 503 whose error `code` is `engine_stopped` (the start rule would allow a start), `loading` (with `Retry-After: 10`) or `not_allowed` (the start rule refuses). A poller can read the state this way without waking the engine. A refused `GET` is not written to the front's log.
- A request body is limited to 32 MiB. A larger one gets 413 `request_too_large`, and the server drains the refused body within bounded bytes and time so the client can read the answer.
- The server holds at most 32 concurrent connections and a backlog of 16 (`uvicorn_options` in `http_limits.py`). Requests are charged against a memory budget before they are read; when it is exhausted the answer is 503 `insufficient_memory`.
- The server runs text only. An image in any dialect gets 400 with code `invalid_image` and the message "this server was started with --vision off; images are not accepted". The refusal comes before tokenizing and without taking the gate.
- Every request body is validated by its dialect before anything is tokenized; invalid JSON gets 400 `invalid_request_error`.
- Cross-origin requests are checked against loopback origins only.

## Environment

| Variable | Meaning | Default |
|---|---|---|
| `SLOTSTREAM_MODE` | memory mode `D1`, `D2` or `D5` | `D5` |
| `SLOTSTREAM_CONTEXT_WINDOW` | served context in tokens, a multiple of 64 from 8,192 to 262,144; `serve --context-window` wins over it; the front reads it too for its start floor ([memory-modes](memory-modes.md), "Context window") | `32768` |
| `SLOTSTREAM_MODEL` | directory of the checkpoint metadata and shards | set by `tools/spark/env.sh` |
| `SLOTSTREAM_PACKED` | directory of the packed experts, n-gram table, trunk and MTP head | set by `tools/spark/env.sh` |
| `SLOTSTREAM_PREFIX_DIR` | persistent prefix tier (`[prefix]` in `platform.toml`, quota 20,000,000,000 B) | `~/slotstream-spark/prefix` |
| `SLOTSTREAM_RUNNING_URL` | model-router `/running` endpoint the governor and the front watch; empty means no router | empty |
| `SLOTSTREAM_EXTRA_BIND` | address the front also binds when it is started by hand (under systemd the socket drop-in adds it); empty means loopback only | empty |
| `SLOTSTREAM_API_KEY` | bearer key required on the extra address only; loopback never asks | empty (no key) |
| `SLOTSTREAM_FRONT_WAIT_S` | seconds the front holds a request while the engine loads before it answers 503; 0 restores the immediate 503 | 240 |
| `SLOTSTREAM_IDLE_EXIT_S` | idle seconds before the engine exits | 1800 |
| `SLOTSTREAM_MTP` | `off` or `resident`, overrides the MTP policy of the mode for this process | from `policy.toml` |
| `SLOTSTREAM_LOOKAHEAD` | overrides `mode` of `[lookahead]` in `platform.toml` for this process | off (see [measurements](measurements.md)) |
| `SLOTSTREAM_TOKEN_GRAPH` | `on` or `off`, overrides `[f8] token_graph` | off |
| `SLOTSTREAM_POOL_PREFILL` | `off` or `table`: pre-fill the pool from the expert frequency table at start | see `slotstream_spark/cli.py` |
| `SLOTSTREAM_MAX_CHUNKS` | cap on pool chunks, below the mode's own cap; used by drills only | none |

The memory knobs and what the governor does with them are in [memory-modes](memory-modes.md). Host-specific values (router URL, container count, NVMe device) go in `private/env.local.sh`, documented in `tools/spark/env.local.sh.example`.

## Extra bind

`deploy/slotstream-spark.socket.d/extra-bind.conf.example` is a drop-in that adds a second `ListenStream` (the address of the container network gateway, left as a placeholder) so containers on the same host can reach the front. The front takes every activated socket (`LISTEN_FDS`); a request that arrives on a non-loopback socket needs `Authorization: Bearer $SLOTSTREAM_API_KEY` when the key is set, and gets 401 otherwise. The engine port is never exposed. The steps are in the RUNBOOK ("Containers on the same host").

While the engine loads, a request is held up to `SLOTSTREAM_FRONT_WAIT_S` seconds and then relayed; if the engine is still down after that, the 503 with `Retry-After` described above is returned. A refusal by the start rule is answered at once.

## Examples

Start the engine through the front by asking for the model list. A 503 with `Retry-After` means it is loading; retry until it answers 200.

```
curl -s -i http://127.0.0.1:18434/v1/models
```

Anthropic Messages, non-streaming:

```
curl -s http://127.0.0.1:18434/v1/messages \
  -H 'content-type: application/json' \
  -d '{"model":"qwen","max_tokens":64,"messages":[{"role":"user","content":"Say hello."}]}'
```

OpenAI Chat Completions, streaming:

```
curl -sN http://127.0.0.1:18434/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"qwen","stream":true,"messages":[{"role":"user","content":"Say hello."}]}'
```

The `model` field is accepted as sent; the served id is the one `/v1/models` reports. Metrics:

```
curl -s http://127.0.0.1:18434/metrics
```

## Coding tools

`tools/mac/launch.sh claude` and `tools/mac/launch.sh codex` bring the tunnel up, wait up to 600 s for `/v1/models` (503 is accepted while the front starts the engine), build the connection from what the server reports and start the tool with it for that run only. `--dry-run` prints the plan with secrets redacted and starts nothing. `SLOTSTREAM_URL` replaces the target, which is how the tests point it at a stub. The design follows `CodingToolLaunch.swift` of slotstream.
