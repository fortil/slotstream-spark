# Install on a DGX Spark

This page takes a Spark from a fresh account to a running engine, in the order the scripts expect. Every script lives in `tools/spark/` and sources `tools/spark/env.sh`, which defines the directory layout under `$R` (default `~/slotstream-spark`):

| Path | Content |
|---|---|
| `$R/src` | this repository |
| `$R/model/qwen38-flash-next-mlx-4bit` | the downloaded checkpoint |
| `$R/model/packed` | the repacked artifacts the engine reads |
| `$R/runs/YYYY/MM/<timestamp>-<label>` | one capture per heavy run |
| `$R/state` | the optional `PAUSED` file ([guard](guard.md)) |
| `$R/hf` | a private Hugging Face home, so the account's own cache is never written |

## What you need

- An NVIDIA DGX Spark (GB10, compute capability 12.1) with 128 GB of unified memory and an NVMe drive. The only machine measured is one Spark; see the README for what that does and does not show.
- Disk: the checkpoint is 103,793,508,077 B over 24 files ([checkpoint.md](../db/records/measurements/checkpoint.md)). The packed artifacts add 104,343,133,720 B ([r0.md](../db/records/measurements/r0.md), sizes of the five files). Plan for about 210 GB, plus room for captures.
- Python 3.12, `systemd --user`, Docker for the optional container kind, CUDA 13 toolchain with `nvcc`, `cmake`, `rsync`.
- Network access for the downloads. After that the engine runs offline (`HF_HUB_OFFLINE=1` is set by `tools/spark/py.sh`).

Everything below is run by you on the host. Heavy steps go through `deploy/guard.sh` (directly or through `tools/spark/jobs.sh`), which refuses a run when free memory is short; see [guard](guard.md).

## 1. Host settings

Copy the example and edit it. The file is ignored by git and read on the Spark and on the Mac:

```
cp tools/spark/env.local.sh.example private/env.local.sh
```

The knobs are `SPARK_HOST` (the ssh target, Mac side), `SLOTSTREAM_RUNNING_URL`, `EXPECTED_CONTAINERS`, `SLOTSTREAM_NVME_DEVICE`, `SLOTSTREAM_HWMON_INPUT` and `SLOTSTREAM_TOOLS_VENV`. The defaults suit a Spark that runs nothing else: an empty URL and an empty container count turn both checks off. See [running without a model router](#running-without-a-model-router).

## 2. Put the tree on the Spark

From the Mac, with `SPARK_HOST` set:

```
tools/spark/sync.sh
```

It rsyncs the working tree to `$R/src` and prints `synced <id>`. It skips `build/`, `bench/traces/`, `tests/goldens/data/` and compiled `*.so`. The reference clone `third_party/slotstream` is a git submodule (`.gitmodules`); run `git submodule update --init` before syncing if the MTP conversion in step 7 should use it.

You can also clone the repository on the Spark and work in the clone. The scripts only assume that the tree is at `$SRC`.

## 3. Check the host

```
bash tools/spark/bootstrap.sh
```

It creates the directory layout and writes `$R/bootstrap.txt` with the facts the later steps depend on: whether the user manager delegates the `memory` and `io` cgroup controllers, whether `loginctl` linger is on, the Docker server version, the `nvcc` release, the system Python, the logical block size of the NVMe drive and the free memory. Read it before going on. A missing memory controller means `--memcg` runs cannot be watched through the cgroup.

## 4. Tools environment

```
bash tools/spark/setup_tools_env.sh
```

This makes a small venv (`$R/venv-tools`, or `SLOTSTREAM_TOOLS_VENV`) with the Hugging Face CLI. It also tries to install `mlx`, which is needed only to convert the MTP weights in step 7. When no `mlx` wheel exists for the platform the script logs `mlx: unavailable` and step 7 uses the pinned mirror instead.

## 5. Project environment

```
bash tools/spark/setup_env.sh
```

It builds the Python environment the engine runs in and verifies torch 2.13.0 with CUDA 13, triton 3.7.1 and transformers 5.19.0, then records the choice in `$R/ENV`. It tries three kinds in order: a venv (host Python has `Python.h`), a standalone Python from python-build-standalone, or a container (the `vllm/vllm-openai:v0.28.0` image with packages under `$R/pylib`).

The standalone kind is what the one measured Spark ended up with, because its system Python 3.12.3 had no `Python.h`. The container kind was checked there too and gave the same versions. Both are in [env.md](../db/records/decisions/env.md), with the lock in `requirements.lock`. Use `tools/spark/py.sh` for every project Python call on the Spark; it reads `$R/ENV`, sets the cache directories under `$R/tmp`, sets `oom_score_adj` to 1000 and puts the source tree and `build/slotio` on `PYTHONPATH`.

## 6. Build the native module

```
tools/spark/py.sh -c 'print(1)'      # smoke test of the environment
bash tools/build_slotio.sh
```

`tools/build_slotio.sh` checks the pinned `doctest.h` checksum, builds `slotio` with CMake and nanobind into `build/slotio`, and runs the CTest suite. It exits non-zero if a test fails. The build output is not in git, so repeat this step after every fresh clone.

## 7. Download and verify the checkpoint

```
bash tools/spark/dl_checkpoint.sh
```

It downloads `pipenetwork/Qwen3.8-Flash-Next-MLX-4bit` at revision `aa7c790e804bbf9d491ddb109c3d61bc4a555f7c` into `$R/model/qwen38-flash-next-mlx-4bit` (through `tools/spark/hf_fetch.sh`, inside a memory-limited systemd scope, into the private `HF_HOME`), then runs `sha256sum -c` against `SHA256SUMS.model`. It exits 0 only when 24 of 24 files print `OK`. The pinned file list with sizes is `model.lock.json`. The download is about 104 GB. If it was interrupted, run the script again; the Hugging Face CLI resumes.

The checkpoint does not carry the MTP head. Build it once:

```
bash tools/spark/mtp_build.sh
```

The script produces `mtp.safetensors` next to the checkpoint and accepts it only when its sha256 equals `model.lock.json` (`c80b58fa...`). It first converts the head from the official `Qwen/Qwen3.8-Flash-Next` weights with MLX, if the preflight record [4b-mlx.md](../db/records/measurements/4b-mlx.md) says `PASS` and `mlx` imports. Otherwise it downloads the pinned mirror and compares the hash. Both paths end with the same file or with an error.

## 8. Repack the weights

The engine does not read the checkpoint directly. `tools/spark/repack_all.sh` turns it into five files under `$R/model/packed`:

| File | Size, B | Content |
|---|---|---|
| `experts.bin` | 67,947,724,800 | 24,576 expert records (48 layers x 512), 2,764,800 B each |
| `mtp_experts.bin` | 1,415,577,600 | 512 records of the MTP layer |
| `ngram.bin` | 32,000,153,600 | 320,001,536 rows of 100 B for the n-gram table |
| `trunk.safetensors` | 2,924,300,312 | everything dense: 2,066 tensors |
| `mtp_trunk.safetensors` | 55,377,408 | the dense tensors of the MTP head: 59 tensors |

The sizes are from [r0.md](../db/records/measurements/r0.md). The formats are in [model-and-repack](model-and-repack.md). The script runs under `tools/capture.sh`, which sets `CAP`, and it runs as a long job:

```
tools/spark/jobs.sh start repack 6144 7200 -- tools/capture.sh repack -- bash tools/spark/repack_all.sh
tools/spark/jobs.sh status repack
tools/spark/jobs.sh wait repack --timeout 7200
```

The first number is the footprint in MiB the run may hold (6,144 is what the recorded repack used), the second the planned duration in seconds. The script is resumable: each of the five outputs is skipped when it exists and its manifest names the current checkpoint, and the writes are atomic (a temporary file renamed at the end). The last stage, `tools/verify_r0.py`, re-reads every packed record, row and tensor and compares the bytes with the checkpoint. The recorded run found 0 mismatches in all five artifacts and left no pages in the page cache ([r0.md](../db/records/measurements/r0.md)). Check the same on yours: the capture's `r0.json` must have `all_equal` true.

Reads and writes use `O_DIRECT` and `posix_fadvise` so that the repack does not fill the page cache that other workloads on the host rely on.

## 9. Verify the artifacts

```
bash tools/spark/verify_artifacts.sh
```

It writes `artifacts.json` and exits non-zero on any failure. It checks the 24 checkpoint files, that nothing was written to the account's own Hugging Face cache, and the `mtp.safetensors` hash. The same script also checks the llama.cpp build and the GGUF files used by the llama.cpp comparison arms ([lb0.md](../db/records/measurements/lb0.md)); those checks fail if you did not build that comparison setup. They are not needed to serve the model.

## 10. Check the engine

```
tools/spark/py.sh -m slotstream_spark.cli doctor
```

`doctor` validates `slotstream_spark/config/platform.toml`, prints free memory and, for each mode, the cap and the number of 1 GiB chunks the pool would get ([memory-modes](memory-modes.md)), reads the model router URL if one is set, and checks that the packed files and their manifests exist. It exits 1 when `SLOTSTREAM_MODEL` or `SLOTSTREAM_PACKED` is not set; `tools/spark/py.sh` sets both.

Then a first token, checked against the stored golden (this loads the model, so run it through the guard):

```
deploy/guard.sh run first-token auto 900 -- tools/spark/py.sh -m slotstream_spark.cli run --mode D5 --pool-epl 48 --golden-prompt dev0 --max-tokens 16 --greedy --check-first-token
```

## 11. Serve

The service is a front process that listens on `127.0.0.1:18434` and starts the engine on a request when the start rules allow it (window, memory cap above the floor, no development run holding the model). The front exits after 5 minutes with no engine and no connections. Install the units:

```
bash tools/spark/install_units.sh
```

This links the seven user units from `deploy/`, enables the start and stop timers, starts the socket inside the 04:00 to 23:59 window and merges the tagged lines of `deploy/crontab.backup` into your crontab. It changes your crontab and your user units, so read it first and run it yourself. Remove the cron lines and the timers if you do not want a daily window. The port is bound to loopback only unless you add the optional extra bind for containers (RUNBOOK, "Containers on the same host"); use the tunnel below to reach it from another machine.

To serve a context window other than the default 32,768 tokens, write `~/slotstream-spark/serve.env` with one line, for example `SLOTSTREAM_CONTEXT_WINDOW=131072`, mode 600. Both units load it: the engine for the window, the front for its start floor. Stop `slotstream-spark.service` and restart `slotstream-spark-front.service` for it to take effect; remove the file to go back. Never set `SLOTSTREAM_MODE` there. What the window costs is in [memory-modes](memory-modes.md), "Context window".

## 12. From the Mac

```
tools/mac/tunnel.sh up
tools/mac/tunnel.sh status
tools/mac/launch.sh claude
```

`tunnel.sh up` forwards `127.0.0.1:18434` over ssh and does nothing while the port already answers. `status` exits 0 when `/api/version` answers 200 or 503; 503 is the front reporting that the engine is loading. `launch.sh claude` and `launch.sh codex` bring the tunnel up, wait up to 600 s for `/v1/models`, and start the tool with the connection built from what the server reports. `--dry-run` prints the plan and starts nothing. Routes and limits of the server are in `server-api.md`.

## Running without a model router

The guard and the engine can run next to a model router (a server such as llama-swap that loads other models into the same memory). Without one, leave `SLOTSTREAM_RUNNING_URL` empty. Then:

- the guard skips the `/running` check ([guard](guard.md), precheck item 2);
- the front does not fetch `/running`; it treats the router as reporting no running models (`slotstream_spark/front.py`), so the other start rules still apply: the 04:00 to 23:45 window, a pool cap above the floor, and no development run holding the model;
- `cli doctor` prints `/running: off`.

Leave `EXPECTED_CONTAINERS` empty too, and the guard does not count the containers that are not its own. Set the number only when you want a heavy run to refuse if some other container went down. While `$R/state/PAUSED` exists, the expected count comes from that file unless `EXPECTED_CONTAINERS` is set in the environment, which wins ([guard](guard.md), [memory-modes](memory-modes.md), "Pause mode").

## If something fails

| Symptom | Check |
|---|---|
| `bootstrap.sh` reports `memcg_delegated=no` | the user manager does not delegate the memory controller; runs with `--memcg` cannot be watched, use a plain guard run |
| `dl_checkpoint.sh` prints fewer than 24 `OK` lines | rerun it; the log is `$R/model/verify-ckpt.txt` |
| `build_slotio.sh` fails on the checksum | `slotio/tests/doctest.h` was edited; restore it |
| `repack_all.sh` stops partway | rerun the same job; finished stages are skipped |
| the guard exits 75 | the precheck refused; the message after `REFUSED:` says why ([guard](guard.md)) |
| the front answers 503 | the engine is loading (the answer carries `Retry-After: 10`), or a start rule refused the request: outside the window, a running model reported by the router, or a development run holding the model |
