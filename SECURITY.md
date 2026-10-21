# Security

## Reporting a problem

Use GitHub's private vulnerability reporting: the "Report a vulnerability" button under the Security tab of this repository. Please do not open a public issue for a vulnerability. Include the commit, what you did and what happened.

## What runs where

- The engine and the front process listen on 127.0.0.1 only. The engine port is never exposed. One optional exception exists: a socket drop-in can add a second address for containers on the same host. Requests that arrive on that address need a bearer key when `SLOTSTREAM_API_KEY` is set (see `docs/server-api.md`).
- There is no authentication on loopback. From another machine, reach the port through an SSH tunnel (`tools/mac/tunnel.sh`). Do not forward it to a network.
- The code sends no telemetry. After the downloads the engine runs offline (`HF_HUB_OFFLINE=1`).
- The persistent prefix cache writes under `SLOTSTREAM_PREFIX_DIR` with a quota of 20,000,000,000 B.
- Nothing in this repository installs system packages. The install scripts are run by the person who owns the host.

## Supply chain

- The checkpoint is pinned to a revision and checked by `sha256sum -c` against `SHA256SUMS.model`. `model.lock.json` lists the pinned files and sizes, and the MTP head is accepted only when its hash matches the lock.
- Python and CUDA packages are pinned in `requirements.lock`.
- Vendored code lives under `third_party/vendor/`. Each tree keeps its license. The `fla` and `vllm` trees also keep an `expected.diff` that `diff.sh` checks byte for byte against the upstream release.
- The C++ test header `slotio/tests/doctest.h` is fetched from a tagged release and its hash is recorded in `slotio/tests/doctest.h.sha256`.

## Scope

In scope: the server, the front, the guard scripts and the tools in this repository. Out of scope: vulnerabilities in the checkpoint, PyTorch, Triton, CUDA or other upstream projects (report those upstream), and any setup that exposes the port on a network.
