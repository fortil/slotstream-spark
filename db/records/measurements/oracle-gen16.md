# oracle-gen16: first 16-token generation of the oracle on the Spark
Date: 2026-10-12
Capture: db/sources/runs/2026/10/20261012T142519-oracle-chain/
Command: part of the oracle-chain job (tools/spark/oracle_chain.sh phase gen16); result in gen16.json

Prompt ids: 23 tokens (chat template, user turn about sunlight). Generated ids: 29108, 4009, 5435, 660, 7736, 314, 279, 9155, 19142, 11, 321, 948, 424, 27891, 8964, 579 (16 tokens, no EOS reached).

Decoded text: "Sunlight contains all colors of the visible spectrum, and when it enters Earth's"

The text is coherent English and not degenerate. impact_check on the capture: see oracle-calibration.md (oom clean, memavail and PSI from the same capture).

Verdict: PASS — 16 tokens generated, text coherent and not degenerate
