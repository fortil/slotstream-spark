# The guard

`deploy/guard.sh` supervises every heavy run: benchmarks, tests that load the model, repacking, long jobs. It checks that the machine can take the run before it starts, watches memory once a second while it runs, cuts the run when a limit is crossed and checks afterwards that nothing of the project is left behind. The goal is a run that cannot starve other workloads on the same host. The numbers per mode are in [memory-modes](memory-modes.md).

The guard applies to development and verification runs. The serving engine protects itself with the governor ([memory-modes](memory-modes.md)); the guard and the governor use the same ceilings and needs, kept equal by `tests/t0/test_guard_modes.py`.

## Commands

```
deploy/guard.sh precheck [flags] FOOTPRINT DUR
deploy/guard.sh run [flags] LABEL FOOTPRINT DUR -- CMD...
deploy/guard.sh postcheck LABEL
```

`FOOTPRINT` is a number of MiB or the word `auto`. `DUR` is the planned duration in seconds. Flags:

| Flag | Effect |
|---|---|
| `--nolock` | do not take the model lock (for runs that never load the model) |
| `--exclusive` | refuse if another run of this project is alive |
| `--allow-truncate` | the run may be cut at 23:40 instead of having to end by 23:45 |
| `--memcg SCOPE` | the command runs in a systemd user scope; its memory is watched through the cgroup |
| `--container` | the command is a container named `slotstream-spark-LABEL`, stopped and killed on a cut |
| `--mode MODE` | `D1`, `D2` or `D5` (default `SLOTSTREAM_MODE`, then `D5`); any other value is refused |

Exit codes: 0 ok, 75 refused, 76 cut, 77 postcheck failed, anything else is the command's own code.

## FOOTPRINT

The footprint is the memory the run is allowed to hold. A number is used as given. `auto` resolves to `min(ceiling, MemAvailable - need) - 256` MiB for the mode and is refused when the result is under 16,384 MiB. A footprint above the mode's ceiling is refused.

## Precheck

The checks run in this order and the first one that fails refuses the run with `REFUSED: <reason>` and exit 75. The JSON line the guard prints holds the mode, the footprint, `MemAvailable`, `MemFree`, the router state, the container count and `ok`.

1. **Window.** Not between 00:00 and 04:00, and the run must end before 23:45 (`now + DUR`), or before 23:40 with `--allow-truncate`. Skipped when the pause file says `NO_WINDOWS=1`.
2. **Model router idle.** If `SLOTSTREAM_RUNNING_URL` is set, its `/running` must answer `{"running":[]}`. An empty or unset URL skips the check.
3. **Memory.** The footprint is within the mode's ceiling, and `MemAvailable - footprint` is at least the mode's need.
4. **Other containers.** If `EXPECTED_CONTAINERS` is set, the number of running containers that are not named `slotstream-spark-*` must equal it. Empty means the count is not checked.
5. **Own containers.** A `slotstream-spark-*` container may run only if a live `run/*.run` file owns it.
6. **Model lock.** `flock -n` on `$R/.model.lock` must succeed, unless `--nolock`.
7. **Exclusive.** With `--exclusive`, no other `run/*.run` file may point at a live process group.
8. **Memory controller.** With `--memcg`, the user session must have the memory controller delegated.

## During the run

`run` repeats the precheck quietly, takes the lock, creates a capture directory `$R/runs/YYYY/MM/<timestamp>-<label>/`, writes 1000 to its own `oom_score_adj`, and starts the command in a new session under `timeout`. It exports `CAP`, `SLOTSTREAM_LABEL`, `SLOTSTREAM_MODE`, `SLOTSTREAM_FOOTPRINT_MIB`, `SLOTSTREAM_DEADLINE` and `SLOTSTREAM_R` to the command. It writes `guard.json` (flags, precheck, process group, start time) and a `run/LABEL.run` file with the group and container.

A watchdog checks once a second. It cuts the run for the first of these reasons (written to `cut` in the capture):

| Reason | Condition |
|---|---|
| `memavailable` | `MemAvailable` below `MEMAVAILABLE_MIN_MIB` (default 10,240) |
| `running` | the router's `/running` is no longer empty |
| `deadline` | the planned duration (cut to 23:45 unless the window is off) is over |
| `stop` | the file `$R/run/STOP` exists |
| `memcg_over` | with `--memcg`, scope memory plus the NVML memory of its processes exceeds the footprint |
| `memfree` | with `--memcg`, `MemFree` below `MEMFREE_MIN_MIB` (default 10,240) |
| `ctr_oom_adj` | with `--container`, the container's `oom_score_adj` is not 1000 |

A cut event is also logged in `guard-events.jsonl`. The cut itself stops the container (10 s grace, 2 s with `--memcg`), sends TERM to the process group, waits for that grace time, then kills the container and the group. With `--memcg` it waits up to 30 s for the scope to empty and then asks the kernel to drop the cached pages of the model files (`posix_fadvise` DONTNEED on the GGUF paths in `GUARD_DONTNEED`) and logs a `dontneed` event. The guard never drops the global page cache.

## Postcheck

After the command ends, the guard writes `guard-post.json`:

| Field | Meaning |
|---|---|
| `rc`, `cut` | command exit code and the cut reason or null |
| `group_alive`, `own_containers` | anything of ours left running |
| `containers_up`, `expected_containers` | other containers and the configured count (null when unset) |
| `running` | the router's answer, or `off` |
| `memavailable_mib`, `memfree_mib` | memory after the run |
| `klog`, `klog_lines` | kernel log since the start: `clean`, `oom` (lines with `oom-kill`, `out of memory` or `NV_ERR_NO_MEMORY`) or `unreadable` |
| `nvsmi_own_pids` | GPU processes still in our group |
| `paused` | whether pause mode was active |

The exit code is 76 if the run was cut, 77 if a process or container of ours is still alive or the kernel log shows an OOM, and otherwise the command's own code. `guard.sh postcheck LABEL` repeats the liveness check later from the `run/LABEL.run` file and removes it when nothing is left.

## Running things through it

On the development machine, `tools/spark/run.sh` sends a command to the Spark over ssh:

```
tools/spark/run.sh [--timeout S] -- CMD...                        light command, 300 s default, 3600 s maximum
tools/spark/run.sh --heavy LABEL FOOTPRINT_MIB DUR_S [flags] -- CMD...   precheck, then guard.sh run
```

A light command is not guarded; it still sets `oom_score_adj` 1000 and is refused between 00:00 and 04:00 unless pause mode lifts the window. A heavy command runs `guard.sh precheck` first and `guard.sh run` second. A refused precheck is exit 75. If the install is container-based (`ENV_KIND=container` in `$R/ENV`), `--container` is added.

`tools/spark/jobs.sh` wraps the same call for long, resumable jobs under `systemd --user`:

```
tools/spark/jobs.sh start NAME FOOTPRINT DUR [flags] -- CMD...
tools/spark/jobs.sh status NAME
tools/spark/jobs.sh wait NAME [--timeout S]
tools/spark/jobs.sh log NAME [N]
tools/spark/jobs.sh restart NAME
```

The unit runs with `OOMScoreAdjust=1000` and calls `guard.sh run`. Status lines are `queued`, `running`, `done 0`, `cut 76` or `failed <rc>`; a `running` line whose unit is gone becomes `failed 70`. `wait` exits 0 only for `done 0`, 1 for a failed or cut job and 124 on timeout.

## Settings

All of these are environment variables, set in the untracked `private/env.local.sh` (see `tools/spark/env.local.sh.example`) or on the command line:

| Variable | Default | Use |
|---|---|---|
| `SLOTSTREAM_RUNNING_URL` | empty | the model router's `/running` endpoint; empty means no router |
| `EXPECTED_CONTAINERS` | empty | container count to hold constant; empty means do not count |
| `SLOTSTREAM_NVME_DEVICE` | `nvme0n1` | drive that holds the packed experts |
| `SLOTSTREAM_MODE` | `D5` | memory mode |
| `MEMAVAILABLE_MIN_MIB`, `MEMFREE_MIN_MIB` | 10,240 | watchdog lines |
| `GUARD_ROOT` | `$R` (`~/slotstream-spark`) | root of `run/`, `runs/` and `state/` |

`GUARD_MEMINFO`, `GUARD_DOCKER`, `GUARD_KLOG`, `GUARD_NVSMI`, `GUARD_CGCTL` and `GUARD_NOW` replace the system sources in tests and drills.

## Pause mode and the 23:59 stop

With `$R/state/PAUSED` holding `NO_WINDOWS=1`, the time checks of the guard do not apply (see [memory-modes](memory-modes.md)). A malformed file makes the guard refuse with exit 2. `deploy/kill-all.sh` stops every process, container and unit of the project. Cron or a timer calls it at 23:59; in pause mode it does nothing unless called with `--force` or `$R/run/STOP` exists. Installing the cron line or the units is a manual step: see `install-spark.md`.

## Drills

`tools/spark/drill_guard.sh` runs six cases through the real `guard.sh` and `kill-all.sh`. The cases are named a to e and k. A shell test suite covers the same paths without a GPU (`deploy/tests/guard_t0.sh`, `deploy/tests/killall_t0.sh`). Result of the run of 2026-10-11 ([guard-drill.md](../db/records/measurements/guard-drill.md)):

| Case | Scenario | Expected | Result |
|---|---|---|---|
| a | planned duration 60 s, a 1,024 MiB process that ignores TERM | rc 76, `deadline` | passed, group gone within 1 s of the return |
| b | router turns non-empty after 8 s | rc 76, `running` | passed, guard returned after 18 s (flip at 8 s, 10 s TERM grace) |
| c | watchdog line set to `MemAvailable` + 1,024 MiB | rc 76, `memavailable` | passed |
| d | `--container`, a container that ignores TERM | rc 76, `deadline`, `oom_score_adj` 1000 | passed, container gone |
| e | `--memcg`, a 2,048 MiB process in a 4G scope, footprint 1,536 | rc 76, `memcg_over` | passed, 2.33 s from the process's TERM handler to the group gone |
| k | `kill-all.sh` with a process group and a container registered | exit 0 | passed in 42 s, group, container and run file gone |

Case k calls `kill-all.sh --force`. Without the flag, kill-all does nothing while `PAUSED` holds `NO_WINDOWS=1`, and on 2026-10-20 case k failed for that reason with production paused. `killall_t0.sh` runs kill-all with the arguments the drill passes, under a pause file.

These scripts need a Linux host with `flock` and `setsid`. They have not been run on the development machine, which is not a Linux host; the results above are from the Spark.
