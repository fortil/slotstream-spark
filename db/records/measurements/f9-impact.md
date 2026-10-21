# f9-impact: production impact over the f9-engine job window
Date: 2026-10-14
Capture: db/sources/runs/2026/10/20261014T212529-f9-engine/
Command: .venv/bin/python tools/impact_report.py --label f9-engine --write

| criterion | status | detail |
|---|---|---|
| 1 oom | PASS | {"detail": "klog clean"} |
| 2 memavail | PASS | {"min_mib": 59853} |
| 3 latency | NOT_APPLICABLE | {"detail": "not_applicable: production paused by the user"} |
| 4 psi (per cgroup) | NOT_APPLICABLE | {"detail": "not_applicable: production paused by the user"} |
| 7 quiet window | NOT_APPLICABLE | {"detail": "not_applicable: production paused by the user"} |
| 8 containers | PASS | {"containers_up": 1, "expected": 1, "own_containers": 0, "problems": [], "servers_checked_oom_score_adj": 21} |

- criteria 3, 4 and 7 are recorded as "not_applicable: production paused by the user" (user decision of 2026-10-14 17:40: the services were stopped by the user, so there is no live production to measure and no window to protect); 1, 2 and 8 are gates
Verdict: PASS — no oom-kill, MemAvailable >= 8 GiB, expected containers Up, no process of ours left
