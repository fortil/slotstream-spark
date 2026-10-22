# Memory modes and the governor

The engine does not take a fixed amount of memory. It runs in one of three modes, each with a ceiling (the most it may hold) and a need (the free memory it must leave for other workloads on the same host). The governor recomputes the allowed size every second and grows or shrinks the expert pool by whole 1 GiB chunks. This page lists the numbers and the rules. How the ledger counts bytes is in [architecture](architecture.md); the run supervisor that enforces the same figures for heavy runs is in [guard](guard.md).

## The three modes

Ceilings and needs come from `slotstream_spark/config/__init__.py` (`MODES`) and are repeated in `deploy/guard.sh`. A test keeps the two copies equal (`tests/t0/test_guard_modes.py`).

| Mode | Ceiling, MiB | Need, MiB | Design fixed bytes | Ceiling chunks | MTP head |
|---|---|---|---|---|---|
| D1 | 19,866 | 62,644 | 8,821,658,084 | 11 | off |
| D2 | 26,112 | 56,413 | 12,123,804,900 | 14 | off |
| D5 | 65,126 | 16,896 | 14,786,204,900 | 49 | resident |

- Need is what the other workloads on the host must still be able to load while the engine runs. It is given with the margin their launcher already includes, and it is the reason the cap follows what is resident at the moment instead of being a fixed number.
- The fixed bytes are the ledger at the ceiling of each mode (`config/modes.py`). They assume a 1,750,000,000 B runtime term and the default 32,768-token context window. With the measured runtime term of 2,638,217,216 B ([m0prime.md](../db/records/measurements/m0prime.md), `platform.toml` `measured.runtime_footprint_mib`), `Ledger(mode).fixed_bytes()` gives 9,886,822,500 B for D1, 13,188,969,316 B for D2 and 15,851,369,316 B for D5 (computed from the code, no record). The governor uses the ledger value. A larger window raises every figure, see "Context window" below.
- MTP by mode is in `config/policy.toml`: D1 and D2 are `off` because they were not measured, D5 is `resident` ([f6-mtp-d5.md](../db/records/decisions/f6-mtp-d5.md)).
- The default is D5 (`memory.mode_default`), a user decision recorded in [memory-mode.md](../db/records/decisions/memory-mode.md). The reversal condition in that record: if the host's model router is used during the day (any non-empty `/running`), or a measured load transient exceeds 2,527 MiB, D1 or D2 apply.
- Mode `X` (all other workloads stopped by hand) is refused by the guard. No window of that kind is authorized.

### Chunks and slots

A chunk is 1 GiB (`CHUNK_BYTES`) and holds 388 slots of 2,764,800 B, so 11 chunks are 4,268 slots, 14 chunks are 5,432 slots and 49 chunks are 19,012 slots. The floor is 2 chunks (776 slots): the fixed bytes plus 2 pool chunks. Below the floor the engine does not start or it exits (`governor.floor_mib`: 11,477 MiB in D1, 14,626 MiB in D2 and 17,166 MiB in D5 with the measured ledger at the default window).

### Cap formula

The governor computes the cap for the empty-router case as:

```
cap = min(ceiling, MemAvailable + own footprint - need - margin)        margin = 512 MiB
chunks = min(ceiling chunks, (cap - fixed bytes) // 1 GiB)
```

When something is resident in the router, the need is already inside `MemAvailable`, so only the empty case charges it (`governor.cap_mib`). The `D1prime` variant (`memory.d1_variant`) would add a second bound from `MemFree`; this host runs `D1` because the check of the model server's memory comparison came out positive ([v1.md](../db/records/measurements/v1.md)).

Worked values, computed from the code with `MemAvailable` = 82,000 MiB, own footprint 0 and no resident in the router (no record):

| Mode | Cap, MiB | Chunks |
|---|---|---|
| D1 | 18,844 | 9 |
| D2 | 25,075 | 12 |
| D5 | 64,592 | 48 |

The guard uses a simpler expression for its `FOOTPRINT auto`: `min(ceiling, MemAvailable - need) - 256` MiB, and refuses a result under 16,384 MiB. `python -m slotstream_spark.config.modes footprint|cap|chunks MODE [AVAIL_MIB]` prints the same arithmetic from `config/modes.py`, which uses the design fixed bytes.

### Context window

The ledger's `kv` term is 27,648 B per token of the served window (`SLOTSTREAM_CONTEXT_WINDOW` or `serve --context-window`, default 32,768, a multiple of 64 up to the model's 262,144). Above the default the prefix cache scales with the window too: the `prefix` term becomes 27,648 B per token of the window plus 4 x 113,246,208 B (the GDN state of the four entries the cache may retain), and a `snapshot` term of 27,648 B per token of the window pays for the copy that exists while the cache decides whether to keep a state. At the default window nothing changes. In D5:

| Term | 32,768 tokens | 131,072 tokens |
|---|---|---|
| kv | 905,969,664 B | 3,623,878,656 B |
| prefix | 1,000,000,000 B | 4,076,863,488 B |
| snapshot | none | 3,623,878,656 B |
| fixed bytes, measured runtime | 15,851,369,316 B | 25,270,020,452 B |
| floor (2 chunks) | 17,166 MiB | 26,148 MiB |
| chunks at the ceiling | 48 | 40 |
| chunks at 82,000 MiB available | 48 | 39 |
| prefix cache budget | 36,168 tokens | 131,072 tokens |

The table is computed from the code. The campaign of 2026-10-21 at that window ([w131k-campaign-d5](../db/records/measurements/w131k-campaign-d5.md)) measured 37 chunks in the decode cell and 38 in the TTFT cells, with the governor growing the pool into the memory the host had free, and a worst-cell peak of 56,438,362,112 B against the ledger's 66,072,209,764 B at 38 chunks; the lowest MemAvailable of the job was 25,582 MiB ([w131k-impact](../db/records/measurements/w131k-impact.md)). With the scaled cache the engine no longer charges the request being served against the cache budget (its KV is already in `kv`): a prompt reserves its own length, a hit never evicts the entry it hands out, and the checkpoint of a whole prompt does not evict the pass-boundary snapshot of its own conversation. So one 78,000-token conversation keeps its snapshot (about 76,800 tokens, the last pass boundary) between turns, prompts of up to about 53,000 tokens fit beside it, and a second conversation of that size evicts it. The front computes its start floor from the same window, which is why both units load `serve.env` (RUNBOOK, "Context window"). The engine also asks the cache before copying a state and skips copies the cache would refuse.

## Governor rules

`slotstream_spark/memory/governor.py` holds the policy as pure functions (`decide`, `may_start`); `memory/signals.py` reads the world and applies the decision. The constants are in `slotstream_spark/config/governor.toml`.

1. **Start.** The engine does not start before 04:00 or after 23:45, nor with `/running` non-empty or unreadable, nor with a cap below the floor (`may_start_end_min` = 1425, `window_start_min` = 240). Pause mode lifts the two time checks only.
2. **Tick.** Each `tick_s` = 1 s it reads `MemAvailable`, `MemFree`, memory PSI and (every `running_poll_ms` = 250 ms, GET only) `/running`. Then it acts in this order:
   a. If a resident that needs the memory is starting, go to the floor at once, cancel the running generation if needed and drop the prefix cache.
   b. If the footprint is above the cap, drop the prefix cache and shrink by whole chunks, between requests.
   c. If `MemAvailable` is below `emergency_mib` = 10,240, shrink one chunk per tick, cancelling if needed. At the floor, exit.
   d. If the cap is below the floor, exit.
3. **OOM priority.** Write 1000 to `/proc/self/oom_score_adj` at start, also inside a container (`signals.write_oom_score_adj`). The guard checks the value on the container's PID.
4. **Shrink fast, grow slow.** Growth waits `grow_after_s` = 60 s without pressure, then adds `grow_step_chunks` = 1 chunk per step, and only if `cap - footprint` leaves one chunk plus the margin. `min_avail_mib` = 8192 is the line `MemAvailable` must not cross during a resident's load; if it would be crossed even at the floor, the engine exits.
5. **Idle exit and restart.** The engine exits after 1800 s without requests (`SLOTSTREAM_IDLE_EXIT_S`). A new request restarts it through the socket and the front, under the same start rule as rule 1, and never before `/running` is empty. The unit has no `Restart=`.
6. **Footprint.** The footprint the governor compares is the larger of the ledger's expected peak and the measured `once_bytes` (VmRSS + NVML + pinned, each unified allocation counted once, see [architecture](architecture.md)). It never uses `cudaMemGetInfo` or the cgroup.
7. **IO arbiter.** Speculative reads back off when another process reads the same drive, and the total queue backs off by temperature (`slotio/src/arbiter.cpp`).
8. **Start order.** Verify the weights, measure, plan, reserve, load the trunk with `DONTNEED`, capture graphs, health check, serve.

Rules 1 to 6 are exercised by `tests/t0/test_governor_policy.py` against fakes; the drills below run them against a stub router on the Spark.

### What the router reports (`/running`)

Each entry of `GET /running` of the host's model router is mapped to a class by fnmatch globs in `slotstream_spark/config/residents.toml`:

| Class | Default glob | What the governor does |
|---|---|---|
| A | `large-*` | D1/D2: floor while it starts, regrow when it is ready |
| B | `vision-*` | D1/D2: exit |
| `27B` | `candidate-27b` | D1/D2: exit |
| C and unknown names | `small-*`, `default_class = "C"` | D1/D2: exit |
| D5, any class | | floor at once, exit if the entry is still there after `d5.exit_after_s` = 5 s at the floor |

The globs are placeholders. Put the names your router reports in `slotstream_spark/config/residents.local.toml`, which is untracked and replaces the `[classes]` entries one class at a time. A failed GET counts as a resident of unknown kind, which is treated as present. A resident of class A is ready when its state says so and `MemAvailable` stayed inside a band of 512 MiB (`stable_band_mib`) for 60 s (`a_stable_s`); without a state field, after the measured load time (180 s default, `a1_load_default_s`) plus 60 s. If its load transient would leave less than 8,192 MiB of `MemAvailable` even at the floor, the engine exits instead of shrinking.

With no router (`SLOTSTREAM_RUNNING_URL` empty, the generic default), `/running` is treated as always empty and these rules never fire.

## Hit rate against pool size

The planner reads the hit-rate curve in `slotstream_spark/config/h_curve_gb10.json`. This is a simulation over router traces (48 traces, 2,926,080 accesses), not a measurement of the engine ([h-curve.md](../db/records/measurements/h-curve.md)). The rows nearest to each mode's pool:

| Experts per layer | Records | LRU hit rate | CLOCK hit rate | Nearest mode |
|---|---|---|---|---|
| 81 | 3,888 | 0.7481 | 0.7534 | D1 at 10 chunks, 3,880 slots |
| 105 | 5,040 | 0.8093 | 0.8085 | D2 at 13 chunks, 5,044 slots |
| 396 | 19,008 | 0.9927 | 0.9925 | D5 at the design ceiling of 49 chunks, 19,012 slots |

Measured in the engine: the pool hit rate in a D5 decode was 0.9917 for a 2,735-token prompt and 0.9607 for a 37-token prompt ([partition-d5.md](../db/records/measurements/partition-d5.md)).

## Status per mode

| Mode | State | Evidence |
|---|---|---|
| D5 | Default. Decode with MTP resident reached 29.374 tok/s, median of 5 samples in the decode-8k cell | [f9-campaign-d5.md](../db/records/measurements/f9-campaign-d5.md) |
| D1 | Recorded only. Decode 9.1985 tok/s (median over clean A/A runs, target at least 9), campaign decode-8k 9.37 tok/s; the 32K cold-prefill cell failed in all 3 samples | [h3.md](../db/records/measurements/h3.md), [f9-campaign-d1.md](../db/records/measurements/f9-campaign-d1.md) |
| D2 | Implemented and covered by unit tests. No decode, TTFT or MTP measurement exists. Its tok/s in [h0.md](../db/records/decisions/h0.md) is a model estimate (13.3 to 14.4 tok/s without lookahead), labeled there as not measured | [h0.md](../db/records/decisions/h0.md) |

## Pause mode

When the host's other workloads are paused on purpose, `$R/state/PAUSED` exists with these lines (`slotstream_spark/guard/paused.py`, bash twin `paused_load` in `tools/spark/env.sh`):

```
NO_WINDOWS=1
EXPECTED_CONTAINERS=<n>
```

- `NO_WINDOWS=1` lifts the 00:00 to 04:00 refusal, the 23:45 limit and the 23:59 stop. The stop script skips itself unless it is called with `--force` or the `STOP` file exists.
- `EXPECTED_CONTAINERS` is the container count the guard compares. When the variable is absent, the count is not checked. An explicit `EXPECTED_CONTAINERS` in the environment wins over the file.
- A file that exists but is malformed (empty, a line without `=`, an unknown key, a value that is not 0/1 or a non-negative integer) is an error: the guard refuses to run (exit 2) and the Python reader raises `PausedError`. It never falls back to the defaults.

## Drills

The governor was checked against a stub router and a bounded memory hog on the Spark ([h4-drills.md](../db/records/measurements/h4-drills.md), 2026-10-13):

| Check | Value | Limit |
|---|---|---|
| D5 floor after the router becomes non-empty | 0.797 s | at most 2 s |
| D5 exit while the router stays non-empty | 7.001 s | at most 15 s |
| D1 with a class A resident: floor | 0.377 s | at most 2 s |
| D1 with a class B resident: exit | 1.206 s | at most 2 s |
| D5 hog: slowest release after the hog step | 1.656 s | at most 2 s |
| D5 hog: lowest `MemAvailable` | 16.38 GiB | at least 8 GiB |

The same record lists the D2 floor, the D1 hog, the restart sequence and the 23:59 stop. Kernel log lines with `oom-kill` or `NV_ERR_NO_MEMORY` during those runs: 0.
