# Decision: daytime memory mode
Date: 2026-10-11
Decision: mode_default=D5; d1_variant=D1; a_transient_mib=2527
Evidence: db/sources/runs/2026/10/20261011T175055-prod-reads/, db/records/measurements/v1.md, db/records/measurements/a1.md
Reversal: the host's model router used by day (any non-empty /running) or an a1 load with a transient above 2,527 MiB; D1 or D2 then apply and the guard falls back to them.

mode_default is D5 by the user decision of 2026-10-11 (nobody uses the model router by day). d1_variant is D1 because v1 is POSITIVE. a_transient_mib is the largest transient of the six observed loads.
