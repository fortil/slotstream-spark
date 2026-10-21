# workload-nights: the engine as the pipeline's reasoning model on two production nights
Date: 2026-10-20
Capture: db/sources/runs/2026/10/20261020T074630-workload-nights/
Command: none; the nightly pipeline ran on its own schedule with the engine behind the LiteLLM alias, and this record reads the pipeline's night logs and the host's kernel and unit journals afterwards
Rule: informative; every reading segment finishes, no oom-kill and no GPU allocation error in the kernel log while the engine runs, resident services Up after the engine stops

The nightly pipeline of October 18-19 and 19-20, 2026 (nights 2 and 3 of its use in production), with the engine in mode D5 started and stopped by the pipeline's control script around the reading window. The capture holds the pipeline's own logs, which name customers and stay offline; this record keeps only counts and timings.

| item | night 2 (Oct 19) | night 3 (Oct 20) |
|---|---|---|
| mentions read | 1203 of 1203 | 1488 of 1488 |
| reading segments | 11, all ok, finish reason stop | 15, all ok, finish reason stop |
| segment time | 90.1 s to 123.9 s, median 110.4 s | 94.3 s to 133.6 s, median 117.4 s |
| prompt tokens per segment | 7534 to 14921 | 7909 to 14816 |
| calls to the reasoning alias | 26 | 30 |
| engine up (unit journal) | 05:56:35 to 06:26:55 | 07:04:59 to 07:46:30 |
| oom-kill or NV_ERR_NO_MEMORY while the engine ran | 0 | 0 |

Both nights together: 26 of 26 reading segments ok, 90.1 s to 133.6 s, median 113.25 s, 0 failed calls, 2691 mentions.

Notes:
- After each night the resident services came back Up. One of them (an embeddings server) exits once about 70 s after it starts and its restart policy brings it back; the same start-exit pair shows in the journal on October 9, 10 and 14, before the engine was used, so it is not caused by the engine.
- Night 1 (October 17-18) is not in the table: the control script did not see its configuration and left the engine running when the residents started, and the kernel logged NV_ERR_NO_MEMORY at 02:15:42 while the embeddings server restarted. The same happened twice during daytime tests on October 18 with the engine up. After the configuration fix the handover on nights 2 and 3 had no allocation error.
- Four other NV_ERR_NO_MEMORY lines between October 16 and 20 came at times the engine was not running.
- Two nights, one pipeline, one language. No MemAvailable trace was kept for these nights.

Verdict: INFO — 26 of 26 reading segments ok on two production nights, no allocation error while the engine ran
