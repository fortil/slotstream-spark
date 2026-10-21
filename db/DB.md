# DB

Records the engine produces. Prose lives in `docs/`; numbers live here.

## Layout

- `sources/runs/YYYY/MM/<cap-id>/`: immutable capture packages, fetched from the Spark by `tools/spark/fetch.sh`.
- `records/measurements/`: one file per measurement.
- `records/decisions/`: one file per decision.
- `records/claims/`: one file per claim.
- `log.md`: append-only log.

## Measurement template

```
# <id>: <title>
Date: YYYY-MM-DD
Capture: db/sources/runs/YYYY/MM/<cap-id>/      (one or more lines)
Command: <command>
<numbers table>
Verdict: PASS|FAIL|INFO — <one line>
```

Exactly one `Verdict:` line. Every `Capture:` directory must exist and hold `guard-post.json` or `verdict.json`.

## Decision template

```
# <id>
Date: YYYY-MM-DD
Decision: <value>
Evidence: <repo paths, at least one that exists>
Reversal: <condition>
```
