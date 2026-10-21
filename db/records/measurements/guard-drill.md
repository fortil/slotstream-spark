# guard-drill: guard and kill-all drill on the Spark
Date: 2026-10-11
Capture: db/sources/runs/2026/10/20261011T172154-guard-drill/
Command: tools/spark/run.sh --timeout 600 -- bash tools/spark/drill_guard.sh

Each case runs through the real `deploy/guard.sh run` (case k through `deploy/kill-all.sh` with `KILLALL_SYSTEMCTL=true`). Production containers: 14 Up before and after.

| case | scenario | expected cut | result |
|---|---|---|---|
| a | deadline `drill-a 2048 60`, 1024 MiB hog ignoring TERM | rc 76, `deadline` | PASS, group gone within 1 s of return |
| b | `/running` stub flips to non-empty after 8 s | rc 76, `running` | PASS, guard returned after 18 s (flip at 8 s, 10 s TERM grace) |
| c | `UMBRAL_MIB` = MemAvailable + 1024 | rc 76, `memavailable` | PASS |
| d | `--contenedor drill-d`, TERM-ignoring container | rc 76, `deadline` | PASS, container gone, oom_score_adj 1000 (HostConfig and /proc) |
| e | `--memcg`, 2048 MiB hog in a 4G scope, huella 1536 | rc 76, `memcg_over` | PASS, TERM to group gone 2.33 s, `dontneed` event written before guard-post.json, scope inactive |
| k | kill-all with a setsid hog and a container registered in `drill-k.run` | exit 0 | PASS in 42 s, group, container and run file gone |

Notes: case e measures 2.33 s from the hog's SIGTERM handler to the group being gone. The guard sleeps 2 s between TERM and KILL; the remaining 0.33 s is process teardown plus the 50 ms poll, so the drill allows 2.5 s.

Verdict: PASS — all 6 cases cut or stopped as expected, nothing of ours left running
