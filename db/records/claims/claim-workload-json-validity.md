---
type: claim
id: workload-json-validity
created: 2026-10-17
summary: Every call of the workload profile returned status 200 (3 of 3, N = 3)
basis: measured
needle: 3 of 3
supported_by: db/records/measurements/workload-profile.md
surfaces: README.md, docs/measurements.md
title: Every call of the workload profile returned status 200 (3 of 3, N = 3)
status: current
---
Profile of 2026-10-16, mode D5, engine stopped before the run, the host's other services running, one call per shape, N = 3 calls: json16k and json25k with response_format json_object (no grammar), text8k as plain text, max_tokens 6000. The profile checks status 200 and a finished response, so the claim is 3 of 3 calls with status 200. It does not parse the JSON of the two json_object calls. In the comparison record of 2026-10-17 the engine had 0 failed reading segments of 1, the same as gpt-oss-120b served by vLLM, on 13 report sections.
