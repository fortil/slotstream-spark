# Glossary

Terms as this project uses them. Where a term has a former name in the records, it is given in brackets; the old words stay in historical record text and are not rewritten.

| Term | Meaning |
|---|---|
| slot | One place in the expert pool that holds one expert record of 2,764,800 B. A 1 GiB chunk holds 388 slots ([architecture](architecture.md)). |
| chunk | A 1 GiB block of the pool. The governor grows and shrinks the pool by whole chunks ([memory-modes](memory-modes.md)). |
| pool | The memory that holds the experts currently in use, filled from the NVMe by `pread` and addressed by the kernels through a per-layer map. |
| expert record | The fixed-size byte layout of one routed expert in `experts.bin` ([model-and-repack](model-and-repack.md)). |
| ensure | The `slotio` call that makes a set of experts resident: hits are marked, misses get a victim slot and a read. |
| CLOCK | The replacement rule for victim slots: one reference bit per slot, a global hand, pinned slots skipped. |
| pin | A mark that keeps a slot from being chosen as a victim while a layer still reads it. |
| sweep | Prefill reads: the prompt is processed in groups of 32 records through staging memory and the pool is not written. |
| read scope | The number of prompt tokens whose routing is decided before the experts are read. Capped at 8192 tokens ([architecture](architecture.md)). |
| lookahead | A forecast of the next layers' experts used to start reads early. It changes speed, never the output. Off in D5 ([measurements](measurements.md)). |
| MTP | Multi-token prediction: a small extra head that drafts the next tokens. The main model then checks them in one pass. |
| verify | The pass that checks MTP drafts. It runs 3 rows and is bitwise equal to three 1-row decodes. |
| D1, D2, D5 | The three memory modes, each with a ceiling and a need ([memory-modes](memory-modes.md)). D5 is the largest and the one the campaign measured. |
| envelope | The measured memory and time of one prefill pass, by pass size and context ([measurements](measurements.md)). |
| ceiling | The most host memory a mode may hold. |
| need | The free memory a mode must leave for other workloads on the same host. |
| cap | What the governor allows at this second: the ceiling, or less when the host is under pressure. |
| floor | The smallest pool the engine runs with: the fixed bytes plus 2 chunks. Below it the engine does not start or exits. |
| footprint [formerly HUELLA] | The memory a guarded run is allowed to hold, in MiB or `auto` ([guard](guard.md)). |
| threshold | A named limit a check compares against. Where an older script says UMBRAL, read `MEMAVAILABLE_MIN_MIB`. |
| yardstick [formerly vara] | The reference system a result is compared with: unmodified llama.cpp, Q4_1, arm B. The record `decisions/vara-d5.md` and its tables keep the old name as recorded. |
| arm B | llama.cpp with 43,520 MiB of experts cached on the GPU, prefill on the CPU ([measurements](measurements.md)). |
| capture | A directory under `db/sources/runs/` with the raw output of one run: command, host facts, guard files, logs. |
| record | A Markdown file in `db/records/` with a date, a capture, a command, a gate table and a verdict. |
| claim | A record that ties one published text (the needle) to the records that support it. |
| needle | The exact text of a claim. The gate looks for it, unsplit, on every surface and in every supporting record. |
| surface | A document where a claim's needle must appear: the README, the runbook and `docs/measurements.md`. |
| gate | A pass or fail check with a limit fixed before the run. Also the name of the claims gate, `tools/claims_gate.py`. |
| oracle | The reference implementation (transformers, bfloat16) that the engine's numbers are compared with. |
| golden | A stored output of a reference implementation that a test compares against. |
| DESIGN GATE | The message a calibration tool prints when a rule would have to be bent to pass. The work stops and the owner decides. |
| paired median | The median of the ratios B/A over A/B pairs run back to back, alternating the order. |
| safe path | Reads go to a host scratch the GPU never maps and are then copied into the slot. The fast path (read straight into GPU-visible memory) is not used. |
| front | The loopback process on port 18434 that starts the engine on demand and relays requests ([server-api](server-api.md)). |
| guard | `deploy/guard.sh`: the supervisor of heavy runs ([guard](guard.md)). |
| pause mode | A state file that tells the guard and the front the other workloads are stopped on purpose, so time windows and the container count are not enforced. |
