# Contributing

## Scope

slotstream-spark is a research engine for one checkpoint (Qwen3.8-Flash-Next, MLX affine 4-bit) on one kind of machine (an NVIDIA DGX Spark). Fixes, measurements from other hardware, documentation corrections and ports of the expert-streaming design are welcome. Support for other models, a general-purpose serving stack and new kernels without a measured gain are out of scope.

## Numbers come with their method

A figure in the README, `docs/` or a pull request description needs the chain that produced it:

1. A capture: the raw output of a run, written by `tools/capture.sh` under `db/sources/runs/`.
2. A record in `db/records/measurements/` (or `decisions/`) that states the date, the capture, the exact command and the result. The template is in `db/DB.md`. Records are never deleted. A wrong record is superseded by a new one that says so.
3. A claim in `db/records/claims/` for every number the README judges (the Speed, Quality, Real workload and Memory and coexistence sections). The claim names the needle text that must appear in the supporting record.
4. The gate: `.venv/bin/python tools/claims_gate.py` fails when a README number has no claim or a needle is missing from its record.

Rules for the measurement itself:

- Compare variants as interleaved pairs and report the paired median, not two separate runs.
- Say N next to every median.
- Label estimates as estimates. A number that was not measured does not go in a table of measurements.
- Failed and withdrawn results are recorded where the successful ones are.

## Memory safety on a shared host

The engine is built to run next to other services. Heavy runs go through `deploy/guard.sh` (see `docs/guard.md`), which checks free memory and stops the run before it starves the neighbors. Do not bypass it for a run that loads the model. Keep one model process at a time, keep `oom_score_adj` at 1000 on processes you start, listen on 127.0.0.1 only, and never drop the page cache.

## Build and test

```
python3.12 -m venv .venv
.venv/bin/pip install -r requirements-mac.txt
bash tools/build_slotio.sh
.venv/bin/python -m pytest -q
```

The default run excludes tests marked `gate` (they need the model weights and run under `deploy/guard.sh` on the Spark) and the `cuda` marker needs a GPU. CI runs `pytest -q -m 'not gate and not cuda'`. Also run before you push:

```
.venv/bin/python tools/claims_gate.py
.venv/bin/python tools/runbook_check.py RUNBOOK.md
.venv/bin/python tools/docs_check.py
.venv/bin/python tools/db_check.py db/records/measurements/*.md db/records/decisions/*.md
```

Anything that needs the Spark (the acceptance run, the guard drill, the agent gate) is run by a person on the Spark. Say in the pull request which of those you ran and give the capture id.

## Style

- Public text (README, `docs/`, RUNBOOK, comments, commit messages) is plain English. No em dashes, no emoji, no inflated vocabulary, no closing paragraph that adds no fact. Every number sits next to its condition.
- Commit messages: imperative mood, subject under 72 characters, the body says why.
- Python follows the surrounding code. Do not reformat files you are not changing.

## Private data

Host names, logins, addresses, absolute local paths and production names must not appear in tracked files. `tools/private_terms_check.py` checks the tracked tree, and the pre-commit hook in `.pre-commit-config.yaml` runs it on staged files. To also block your own names, put one regular expression per line in a file and point `SLOTSTREAM_PRIVATE_TERMS` at it. Model weights and packed artifacts never enter git.
