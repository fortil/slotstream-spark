# guard-drill-public: guard and kill-all drill on the renamed tree
Date: 2026-10-20
Capture: db/sources/runs/2026/10/20261020T131410-guard-drill/
Command: bash tools/spark/drill_guard.sh

Same six cases as [guard-drill](guard-drill.md), run on commit `c92a589` (the English identifiers: `--container`, FOOTPRINT, `MEMAVAILABLE_MIN_MIB`, `cut_run`). Each case runs through the real `deploy/guard.sh run`; case k runs `deploy/kill-all.sh --force` with `KILLALL_SYSTEMCTL=true`. The host was in pause mode (`state/PAUSED` with `NO_WINDOWS=1`), 12 containers Up before and after, the engine's socket and front stopped for the run.

| case | scenario | expected cut | result |
|---|---|---|---|
| a | deadline `drill-a 2048 60`, hog ignoring TERM | rc 76, `deadline` | PASS, group gone within 1 s of return |
| b | `/running` stub flips to non-empty | rc 76, `running` | PASS, cut after 19 s |
| c | `MEMAVAILABLE_MIN_MIB` = MemAvailable + 1024 (threshold 117526 MiB) | rc 76, `memavailable` | PASS |
| d | `--container drill-d`, TERM-ignoring container | rc 76, `deadline` | PASS, container gone, oom_score_adj 1000 (HostConfig and /proc) |
| e | `--memcg`, hog over its scope | rc 76, `memcg_over` | PASS, TERM to group gone 2.25 s |
| k | kill-all with a setsid hog and a container registered in `drill-k.run` | exit 0 | PASS in 41 s, group, container and run file gone |

Notes: in pause mode `kill-all.sh` without `--force` stops nothing, so before `c92a589` case k failed whenever `state/PAUSED` existed; `deploy/tests/killall_t0.sh` now covers that path. Case k's kill-all also removed a `drill-b.run` file left by an earlier interrupted drill. The whole drill took 229 s.

Verdict: PASS — all 6 cases cut or stopped as expected on the renamed tree, nothing of ours left running
