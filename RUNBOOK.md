# slotstream-spark operator runbook

## Overview

The engine runs Qwen3.8-Flash-Next (MLX affine 4-bit checkpoint) on the DGX Spark. The dense weights stay in memory
and the routed experts are read from the NVMe into a pool of slots. A small front process (`slotstream_spark/front.py`)
listens on 127.0.0.1:18434 and starts the engine on the first request. The engine is a PyTorch and Triton process with a
C++ extension (`slotio`). Everything listens on loopback only, except the optional extra bind of the front below; reach it from the Mac through the SSH tunnel.

The Spark also hosts the production services of another project. This project may only use its own files, one model
process at a time, and the memory the guard grants. The rules are in `AGENTS.md`; the guard that enforces them is
`deploy/guard.sh`. Any heavy run goes through `tools/spark/run.sh --heavy`.

## Measured figures

What to expect from a healthy D5 server. Each line is backed by a record in `db/records/` and carries a claim needle that
`tools/claims_gate.py` checks against its record. Production was paused by the user during the final runs, so the production latency, PSI and
quiet-window criteria were recorded as not applicable.

- Decode with the MTP head resident: 29.37 tok/s. Without MTP: 25.949 tok/s.
- Cold 8K TTFT 22.033 s, follow-up on an 8192-token prefix 1.051 s, cold 32K TTFT 77.47 s.
- Perplexity on 32 wikitext windows, within 2 % of the transformers oracle: engine 4.7216, oracle 4.7265. The 145-window
  figure of 5.7338 against a published 5.3914 is informative only; the protocol of that reference is unknown.
- Pooled top-1 agreement with the oracle on the 8 reserved prompts: 0.9545 (210/220). Behaviour probe: probe 15/15.
- Memory during the f9 job: no oom-kill, MemAvailable >= 8 GiB, with a minimum of 59853 MiB.
- Lookahead stays off (paired median 0.8761). The MTP head stays resident (paired median 1.1609). The token graph stays
  off (paired median 0.9914, bar 1.05, 6 clean pairs).
- D5 pool: 48 chunks of 1 GiB at the default 32,768-token window. Cold 2K TTFT 6.80 s. D1 decode 9.37 tok/s (informative, median of 3).
- At the 131,072-token window (`serve.env`): decode 26.918 tok/s with 37 chunks (informative), cold 78000-token TTFT 221.89 s,
  follow-up on a 76800-token prefix 7.389 s, 38 chunks in the pool during the TTFT cells, lowest MemAvailable of the job 25582 MiB.
- Reference repack: exact except the 7 named exceptions, with `ple_conv1d` written as F32 and the 194 hyper-connection
  matrices stored as BF16 in the llama.cpp reference. Engine against llama.cpp on the same windows:
  engine 4.7216, llama.cpp 4.7239.
- Against llama.cpp arm B: decode 29.37 tok/s = 89.3% of arm B 32.9; cold 8K prefill 372.1 tok/s = 11.6x arm B 32.1.

## Modes

`SLOTSTREAM_MODE` selects the memory envelope. The default is D5.

- `D1`: the smallest envelope (11 design chunks at most).
- `D2`: the middle envelope (14 chunks at most).
- `D5`: the default (49 chunks at most); the figures in README.md were measured in D5.

The engine unit sets `SLOTSTREAM_MODE=D5`; the front reads the same variable from its own environment for the start
floor. `python -m slotstream_spark.cli doctor` prints the context window and, for each mode, the memory cap, the design
chunks and the planned chunks for the memory free at that moment.

### Context window

The engine serves 32,768 tokens of context by default. `SLOTSTREAM_CONTEXT_WINDOW` (or `--context-window` on `serve`)
sets another window: a multiple of 64 between 8,192 and 262,144, the model's limit. Production uses 131,072 for the
desktop app's long prompts, set in `~/slotstream-spark/serve.env` (one `KEY=value` line, mode 600). Both the engine unit
and the front unit load that file: the front's start floor follows the window the engine will serve. Never put
`SLOTSTREAM_MODE` in it, an environment file overrides the unit's own line.

Above the default window the ledger charges the KV cache at the window and sizes the in-memory prefix cache in tokens
of the window, so a long conversation keeps its state between turns. In D5 at 131,072 tokens the fixed bytes are
25,270,020,452 B and the floor 26,148 MiB; the pool is 40 chunks at the ceiling (calculated) and held 37 to 38 chunks
during the campaign of 2026-10-21 (`db/records/measurements/w131k-campaign-d5.md`); the formula is in
[memory-modes](docs/memory-modes.md). The acceptance script, the campaign and the drills start the engine without
systemd, so they take the variable on their command line. Windows above 131,072 are accepted by the code and not
measured. The decision and its reversal rule are in `db/records/decisions/context-window.md`.

Two switches have a recorded decision: `SLOTSTREAM_MTP=off|resident` (default resident, `db/records/decisions/f6-mtp-d5.md`)
and `SLOTSTREAM_LOOKAHEAD=off` (default off, `db/records/decisions/f6-lookahead-d5.md`). The token graph stays off
(`db/records/decisions/f8-token-graph.md`).

## Schedule

The units live in `deploy/` and are linked into the user systemd with `tools/spark/install_units.sh`.

- `slotstream-spark-start.timer` starts `slotstream-spark.socket` at 04:00.
- `slotstream-spark-stop.timer` stops the socket, the front and the engine at 23:59.
- Nothing runs between 00:00 and 04:00. The guard refuses heavy work in that window.
- `deploy/crontab.backup` holds the matching crontab lines tagged `#slotstream-spark`.

When the user pauses production, `$R/state/PAUSED` exists on the Spark (`NO_WINDOWS=1` and `EXPECTED_CONTAINERS=<n>`).
While it exists the time windows and the expected-containers rule (`EXPECTED_CONTAINERS`) are lifted, and `deploy/kill-all.sh` does nothing unless it is
called with `--force`. Deleting the file restores every rule.

## Mac access

- `tools/mac/tunnel.sh up` opens the forward 127.0.0.1:18434; `status` exits 0 when `/api/version` answers 200 or 503
  (503 means the engine is still loading); `down` closes it.
- `tools/mac/launch.sh claude` and `tools/mac/launch.sh codex` bring the tunnel up, wait for `/v1/models` and start the
  tool already connected. Your own Claude Code and Codex configuration is not written. `--dry-run` prints the plan.
- The server speaks the OpenAI, Ollama, Anthropic and Responses dialects. Requests with images get a 400.

## Health

- `GET /api/ps` shows the loaded model, `GET /api/version` the server, `GET /metrics` the Prometheus counters.
- `python -m slotstream_spark.cli doctor` (through `tools/spark/py.sh` on the Spark) checks `slotstream_spark/config/platform.toml`, free memory,
  the artifacts and their seals, and reads the model router's `/running`.
- `tools/quality/quality_probe.sh --url http://127.0.0.1:18434` runs the 15-question probe; it must print 15/15.
- `tools/spark/jobs.sh status NAME` and `tools/spark/jobs.sh log NAME` show a long job.

## Coexistence with other model servers

The host may run a model router and other model servers. This project only calls `GET /running` on the host's model
router (the doctor does it, using `SLOTSTREAM_RUNNING_URL`). It never loads a model through the router or any gateway in
front of it, and it never writes `~/.cache/huggingface` or `~/LLMs`. The front reads the same `/running` answer before it
starts the engine and starts it only when that answer is readable and empty; when `/running` is unreadable it answers 503
and starts nothing. With no router URL set the check is off ([install-spark](docs/install-spark.md)).

## Containers on the same host (optional extra bind)

By default the front listens on 127.0.0.1 only. To let containers on the same host call it, add a second address to
the front socket: copy `deploy/slotstream-spark.socket.d/extra-bind.conf.example` to
`~/.config/systemd/user/slotstream-spark.socket.d/extra-bind.conf`, put the container network gateway address in place
of `<gateway-ip>`, then `systemctl --user daemon-reload` and `systemctl --user restart slotstream-spark.socket`.
Nothing installs this for you. Only the front port gets the extra address; the engine port stays on loopback.

- `SLOTSTREAM_API_KEY` (in `~/slotstream-spark/front.env`, `KEY=value` lines, mode 600): when set, requests on the extra
  address need `Authorization: Bearer <key>`; loopback stays open. Without a key the extra address is open to every
  container on the host.
- `SLOTSTREAM_FRONT_WAIT_S` (same file, default 240, 0 turns it off): a request that finds the engine loading, or
  starts it, is held until the engine answers instead of getting 503 + `Retry-After`. After the wait it gets the 503.
  Refusals (router `/running` not empty, outside the window, memory below the floor, a development run) stay
  immediate 503, so a client that retries behaves as before.
- Remove the extra address by deleting the drop-in, `daemon-reload` and restarting the socket.

## Emergency stop

1. Create the file `$R/run/STOP` on the Spark (`$R` is `~/slotstream-spark`). The guard cuts every run it supervises and
   `deploy/kill-all.sh` stops everything even in pause mode.
2. `bash deploy/kill-all.sh --force` stops our processes, containers and units now. It always exits 0.
3. `systemctl --user stop slotstream-spark.socket slotstream-spark-front.service slotstream-spark.service` stops only the
   service chain.

Production containers, compose files and other projects' units are never touched.

## Updating

1. On the Mac, with the project virtualenv active, run the T0 suite (`python -m pytest -q`).
2. `tools/spark/sync.sh` copies the tree to the Spark.
3. `tools/spark/check_units.sh` and `tools/spark/install_units.sh` re-link the units when they changed.
4. The persistent prefix cache is keyed by an identity hash of the code, the configuration, the weights, the geometry
   and the numeric arithmetic version (`slotstream_spark/engine/persistent_prefix.py`). A change in any of them gives
   a new identity, so old states are not reused: they are removed when found. No manual flush is needed.
5. `tools/acceptance.sh final` under the guard is the release check.

## JSON mode

`/v1/chat/completions` accepts `response_format: {"type": "json_object"}`. There is no grammar: the server turns
thinking off (whatever `reasoning_effort` or `think` say) and puts one line at the front of the system message, or
creates a system message: "Answer with a single valid JSON object and nothing else, with no code fence." Tokens are
not constrained, so a caller still validates the reply. `json_schema` and any other type stay a 400. The
non-streaming reply carries `usage.prompt_tokens`, `usage.completion_tokens` and `finish_reason`.

## Measurements and A/B

- Every measurement is a record in `db/records/measurements/` and every decision one in `db/records/decisions/`;
  `tools/db_check.py` validates them. Captures live under `db/sources/runs/`.
- `tools/decide.py` turns captures into records. `bench/ab.py` and `bench/f6_ab.sh` run paired A/B comparisons.
- The claims in README.md come from `db/records/claims/`; `tools/claims_gate.py` fails when a claim needle is missing from README.md,
  this runbook or its record, or when a number in the README's Speed, Quality or Memory and coexistence sections has no claim.
- Reference points: llama.cpp arm B (unmodified, Q4_1) is the yardstick for decode and cold prefill
  (`db/records/decisions/vara-d5.md`).

Every figure, its conditions and its record are listed in [docs/measurements.md](docs/measurements.md).

## Incidents

- oom-kill or `NV_ERR_NO_MEMORY` in `journalctl -k`: stop with `bash deploy/kill-all.sh --force`, note the time, check that
  all production containers are Up, and do not restart until the cause is in a record.
- Hung `pread` (a process in state D on the NVMe): the guard cuts the run after its budget; if it does not, use the
  emergency stop. Do not run `drop_caches`.
- CUDA context loss: the engine process exits; the front starts a new one on the next request. If it repeats, stop the
  service and read `journalctl -k`.
- Client timeouts on a cold prompt: a cold 8K prompt takes about 22 s and a 32K one about 77 s, so clients need
  a read timeout above 120 s. A follow-up on the same prefix takes about 1 s. At the 131,072-token window a cold
  78000-token prompt took 221.89 s and its follow-up 7.389 s: give those clients a read timeout of 600 s.
- Memory pressure from production: the front starts the engine only when the governor rule allows it, and answers 503
  with Retry-After otherwise.

## Rollback

To go back to the default context window only: remove `~/slotstream-spark/serve.env`, stop `slotstream-spark.service`
and restart `slotstream-spark-front.service`. The next start serves 32,768 tokens again; the prefix cache on disk is
kept, and the states it holds are reused where they still fit.

Disable the entry point and stop the chain:

1. `systemctl --user disable --now slotstream-spark-start.timer slotstream-spark-stop.timer`
2. `systemctl --user stop slotstream-spark.socket slotstream-spark-front.service slotstream-spark.service`
3. Remove the crontab lines tagged `#slotstream-spark`.

The model and packed artifacts under `~/slotstream-spark` can stay; they are not read while the socket is off.
