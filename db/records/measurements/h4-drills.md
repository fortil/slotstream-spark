# h4-drills: /running stub, bounded hog, on-demand restart and the 23:59 stop
Date: 2026-10-13
Captures: db/sources/runs/2026/10/20261013T132716-drills2/, db/sources/runs/2026/10/20261013T133310-drills-light/
Command: tools/spark/run.sh --heavy drills auto 5400 -- tools/spark/py.sh -m pytest -q -m gate tests/gates/drills/test_running_stub.py tests/gates/drills/test_hog.py; tools/spark/run.sh --timeout 1800 -- tools/capture.sh drills-light -- tools/spark/py.sh -m pytest -q -m gate tests/gates/drills/test_restart.py tests/gates/drills/test_stop.py

| gate | value | limit | ok |
|---|---|---|---|
| D5 floor after the flip (s) | 0.797 | <= 2 | yes |
| D5 exit while /running stays non-empty (s) | 7.001 | <= 15 | yes |
| D1+A floor (s) | 0.377 | <= 2 | yes |
| D1+A ready: regrow one chunk per >= 60 s | 2 grows, gaps [61.742] | >= 2 grows, +1 chunk, gaps >= 60 s | yes |
| D1+B exit (s) | 1.206 | <= 2 | yes |
| D2+A floor (s) | 0.253 | <= 2 | yes |
| D5 hog: releases observed | 40 | >= 1 | yes |
| D5 hog: slowest release after the hog step (s) | 1.656 | <= 2 | yes |
| D5 hog: smallest MemAvailable rise at a release (GiB) | 1 | >= 0.9 | yes |
| D5 hog: MemAvailable minimum (GiB) | 16.38 | >= 8 | yes |
| D1 hog: releases observed | 4 | >= 1 | yes |
| D1 hog: slowest release after the hog step (s) | 1.426 | <= 2 | yes |
| D1 hog: smallest MemAvailable rise at a release (GiB) | 2.03 | >= 0.9 | yes |
| D1 hog: MemAvailable minimum (GiB) | 60.93 | >= 8 | yes |
| restart: first request 503 + Retry-After, then 200 | 503/10, 200 after 6.1 s | 503/10 then 200 | yes |
| restart: engine gone after the idle exit | 62.1 s, procs [] | none | yes |
| restart: request after the idle exit 503 + Retry-After, then 200 | 503/10, 200 after 6.1 s | 503/10 then 200 | yes |
| restart: /running non-empty -> 503 and no engine process | 503, procs [], unit inactive | 503, none | yes |
| stop unit: socket, front and engine inactive (s) | 0.19 | <= 15 | yes |
| stop unit: timers | TimersCalendar={ OnCalendar=*-*-* 04:00:00 ; next_elapse=Wed 2026-10-14 04:00:00 -05 } | TimersCalendar={ OnCalendar=*-*-* 23:59:00 ; next_elapse=Tue 2026-10-13 23:59:00 -05 } | 04:00:00 and 23:59:00 | yes |
| may_start 03:59 / 04:00 / 23:44 / 23:45 | False / True / True / False | False / True / True / False | yes |
| socket restored by the 04:00 service; no [Install] | active, install False | active, no | yes |
| oom-kill or NV_ERR_NO_MEMORY lines in journalctl -k | 0 | 0 | yes |
| Arbiter throttle_c named (device temp1_max - 5 C) | 77.85 | recorded | yes |

- D5 hog stopped by size_cap at 50534 MiB after 50 steps; engine cap 64870 MiB, chunks at start 48; engine peak (footprint.once_bytes over the 20 ms series, informational) 54.101 GiB
- D1 hog stopped by size_cap at 5530 MiB after 6 steps; engine cap 19866 MiB, chunks at start 10; engine peak (footprint.once_bytes over the 20 ms series, informational) 16.1 GiB

Verdict: PASS — every release and floor inside its limit, MemAvailable never below 8 GiB, no oom, restart and stop drills pass
