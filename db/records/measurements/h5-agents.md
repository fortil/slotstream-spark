# h5-agents: coding agents and Open WebUI through the tunnel
Date: 2026-10-21
Capture: db/sources/runs/2026/10/20261021T210744-h5-agents/
Command: tools/mac/tunnel.sh up && bash tools/coding_agents_gate.sh --out tmp/h5 --record db/records/measurements

| check | ok |
|---|---|
| claude: turn 1 wrote hello.txt with the exact text | yes |
| claude: turn 1 num_turns >= 2 | yes |
| claude: turn 2 (resume) leaves two lines | yes |
| claude: turn 2 cache read > 0 (usage.cache_read_input_tokens, else modelUsage sum with the server's reused_tokens) | yes |
| claude: turn 3 wrote out.txt == done | yes |
| claude: turn 3 images_rejected_total +1 | yes |
| claude: turn 3 /api/ps size unchanged within 1 MiB | yes |
| codex: apply_patch edit task changed exactly line 2 | yes |
| webui: tags answers 200 | yes |
| webui: show answers 200 | yes |
| webui: chat turn 1 streams and ends with done | yes |
| webui: title generation (not streamed) answers one body | yes |
| webui: chat turn 2 streams and ends with done | yes |
| webui: chat turn 2 cached_tokens > 0 | yes |
| webui: a request with images is a 400 | yes |
| webui: images_rejected_total grew by 1 | yes |
| host: 14 containers Up after the run | yes (14) |
| host: no oom-kill / NV_ERR_NO_MEMORY in the kernel log since the run started | yes (0 lines) |

- agent versions: claude 2.1.284 (Claude Code); codex codex-cli 0.141.0
- verdicts: claude pass; codex pass; webui pass
- claude usage per turn (last API call of each invocation): {"turn1": {"cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0}, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "inference_geo": "", "input_tokens": 74788, "iterations": [], "output_tokens": 229, "output_tokens_details": {"thinking_tokens": 0}, "server_tool_use": {"web_fetch_requests": 0, "web_search_requests": 0}, "service_tier": "standard", "speed": "standard"}, "turn2": {"cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0}, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 20480, "inference_geo": "", "input_tokens": 40030, "iterations": [], "output_tokens": 196, "output_tokens_details": {"thinking_tokens": 0}, "server_tool_use": {"web_fetch_requests": 0, "web_search_requests": 0}, "service_tier": "standard", "speed": "standard"}, "turn3": {"cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0}, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "inference_geo": "", "input_tokens": 54721, "iterations": [], "output_tokens": 190, "output_tokens_details": {"thinking_tokens": 0}, "server_tool_use": {"web_fetch_requests": 0, "web_search_requests": 0}, "service_tier": "standard", "speed": "standard"}}
- claude turn 2 cache read: {'model_usage': 73728, 'server_reused_delta': 57344.0, 'usage': 20480}
- claude turn seconds: {"turn1": 418.8, "turn2": 370.0, "turn3": 267.7}
- images_rejected_total delta in claude turn 3: 1.0; /api/ps size before and after: [63096009572, 63096009572]
- webui: turn 1 prompt tokens 4675, turn 2 cached_tokens 4096
- safety: claude ran with --permission-mode acceptEdits --allowedTools Read,Write,Edit --disallowedTools Bash and a CLAUDE_CONFIG_DIR under the output directory; codex with -s workspace-write, approval_policy never and a CODEX_HOME under the output directory; the throwaway projects live under the output directory
- server_tree_sha256: f36c8e85cfc00749354bab891025f46c15b6750b584547f9d109759419bf4162 (sha256 over `<path>\0<file sha256>\n` of slotstream_spark/server/**/*.py and slotstream_spark/front.py, sorted; tools/webui_sequence.py tree_sha256)
Verdict: PASS — claude pass, codex pass, webui pass, host clean
