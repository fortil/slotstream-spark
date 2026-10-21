# Documentation

Reading order for someone who wants to understand, install and check the engine. Numbers are on [measurements](measurements.md) and in `db/records/`; the other pages link to them instead of repeating them.

1. [architecture](architecture.md): components, the memory map, the decode path, prefill, kernel rules.
2. [memory-modes](memory-modes.md): the D1, D2 and D5 modes, the governor and pause mode.
3. [model-and-repack](model-and-repack.md): the checkpoint and the layout of the repacked files.
4. [exact-q4_1-reference](exact-q4_1-reference.md): why the repack is exact and how it is checked against llama.cpp.
5. [install-spark](install-spark.md): from a clean Spark to a running engine, and the Mac side.
6. [server-api](server-api.md): ports, routes, limits and environment variables.
7. [guard](guard.md): the supervisor of heavy runs.
8. [measurements](measurements.md): every published figure, its conditions and its record.
9. [glossary](glossary.md): the terms used on these pages.
10. [sources](sources.md): slotstream, the reference clones, the model metadata and the captures.

Outside this folder: the [README](../README.md) in short form, the [runbook](../RUNBOOK.md) for operating the engine, `db/DB.md` for how records and captures are organised, and [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md) for licenses.

Checks: `tools/docs_check.py` verifies links and anchors, the character rules and the absence of local paths in these pages. `tools/claims_gate.py` verifies every published number. `tools/runbook_check.py` verifies the runbook.
