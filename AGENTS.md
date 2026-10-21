# slotstream-spark

Qwen3.8-Flash-Next (MLX affine 4-bit, 103.8 GB) on an NVIDIA DGX Spark, with the routed experts streamed from NVMe the way slotstream does it on a Mac.

## Rules for every agent

- Model weights and packed artifacts never enter git (see `.gitignore`).
- Mac Python is `.venv/bin/python`. On the Spark, project Python is always `tools/spark/py.sh`.
- Commit messages: imperative, subject under 72 characters, body says why. No attribution lines, no em dashes, no emoji.
- Public files (README, `docs/`, RUNBOOK, comments, commit messages) are written in plain English: no inflated vocabulary, no filler closings, every number next to its condition.
- Local absolute paths, host names, addresses and private names must not appear in tracked files. `tools/private_terms_check.py` and a pre-push hook block them. Use relative paths or placeholders.
- Numbers go through `db/` and the claims gate: a figure in the README's Speed, Quality or Memory and coexistence sections, or in the claims table of `docs/measurements.md`, needs a claim record in `db/records/claims/`. Records the engine produces go in `db/`.

## Coexistence on a shared host

The engine can run next to other workloads on the same machine. Heavy runs go through `deploy/guard.sh`, which checks free memory and stops the run before it can starve the neighbors. The checks, flags and exit codes are in `docs/guard.md`. Host knobs (model router URL, container count, NVMe device) are listed in `tools/spark/env.local.sh.example`; set them in the untracked `private/env.local.sh`. One model process at a time, `oom_score_adj` 1000 on every process this project starts, listen on 127.0.0.1 only (one declared exception: an optional extra bind of the front for containers on the same host, see `docs/server-api.md`), never `--dangerously-skip-permissions`.

## Tasky: document destination

BASE: .tasky/history/
RETENTION_DAYS: 7

If `private/AGENTS.local.md` exists, read it: it holds the rules of the host this project is developed on.
