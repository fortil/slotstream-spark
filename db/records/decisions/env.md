# Decision: Python environment on the Spark
Date: 2026-10-11
Decision: standalone
Evidence: db/sources/runs/2026/10/20261011T175214-triton-env/, db/sources/runs/2026/10/20261011T175243-triton-ctr/, requirements.lock

The host python3 is 3.12.3 without Python.h, so `tools/spark/setup_env.sh` took the python-build-standalone branch (3.12.15, ships its headers, `PYTHON_H=yes`). It installed torch 2.13.0+cu130, triton 3.7.1, transformers 5.19.0 and flash-linear-attention 0.5.2 (see requirements.lock, fetched from the Spark).

| check | environment | result |
|---|---|---|
| tests/t0/test_triton_smoke.py under pytest | standalone venv | 1 passed, exact fp32 vector add |
| tests/t0/triton_smoke.py in the vLLM image (`SLOTSTREAM_ENV_KIND=container`, `--contenedor`) | container | `triton smoke OK 2.13.0+cu130 3.7.1` |

The container run went through the docker branch of py.sh: the first attempt (capture 20261011T175223-triton-ctr) failed with `docker: invalid reference format` because the standalone ENV file leaves IMAGE empty; py.sh now falls back to the vLLM image when IMAGE is empty, and the second attempt passed.

Reversal: a missing wheel or header after a reinstall; then `SLOTSTREAM_ENV_KIND=container` with the vLLM image and `$R/pylib` becomes the default (the image already carries the same torch and triton).
