"""The Mac tunnel and the agent launcher (step 88): dry runs against a stub server, no ssh, no agent, no network.

launch.sh builds the Claude Code environment and the Codex argv from what /v1/models reports (CodingToolLaunch.swift
claude and codex plans). A stub plays the front: 200 with the catalogue, or 503 while "loading".
"""
import http.server
import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "tools" / "mac" / "launch.sh"
TUNNEL = ROOT / "tools" / "mac" / "tunnel.sh"
MODEL = "qwen38-flash-next-mlx-4bit"
WINDOW, OUT = 32768, 8192
SECRET_KEY = "sk-ant-should-never-survive"

CATALOGUE = {"object": "list", "data": [{"id": MODEL, "object": "model", "created": 0, "owned_by": "slotstream-spark",
                                         "context_window": WINDOW, "max_output_tokens": OUT}]}


class Stub:
    """A loopback server: /v1/models answers `models_status` (200 with the catalogue, or another code), /api/version
    answers `version_status`. After `loading` 503s on /v1/models it answers 200."""

    def __init__(self, models_status=200, version_status=200, loading=0):
        stub = self
        self.models_status, self.version_status, self.loading = models_status, version_status, loading
        self.hits = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                stub.hits.append(self.path)
                if self.path == "/v1/models":
                    code = 503 if stub.loading > 0 else stub.models_status
                    if stub.loading > 0:
                        stub.loading -= 1
                    body = json.dumps(CATALOGUE if code == 200 else {"error": "x"}).encode()
                elif self.path == "/api/version":
                    code, body = stub.version_status, b'{"version":"0.1.0"}'
                else:
                    code, body = 404, b"{}"
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stub():
    s = Stub()
    yield s
    s.close()


def run(script, *args, url=None, extra_env=None, timeout=60):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "CLAUDE_", "MAX_THINKING", "API_TIMEOUT"))}
    env["SLOTSTREAM_LAUNCH_WAIT_S"] = "20"
    if url:
        env["SLOTSTREAM_URL"] = url
    env.update(extra_env or {})
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True, env=env, timeout=timeout,
                          cwd=str(ROOT))


def dry(stub, tool, *args, extra_env=None):
    r = run(LAUNCH, tool, *args, "--dry-run", url=stub.url, extra_env=extra_env)
    assert r.returncode == 0, r.stderr
    plan = json.loads(r.stdout)
    if tool == "codex":
        # the model catalog entry (step 89) is a fixed pair of words; the tests below compare the rest of the argv
        argv, i = plan["argv"], None
        for k, a in enumerate(argv):
            if a.startswith("model_catalog_json="):
                i = k - 1
        assert i is not None and argv[i] == "-c", argv
        plan["catalog_arg"] = argv[i + 1]
        plan["argv"] = argv[:i] + argv[i + 2:]
    return plan


def test_scripts_parse():
    for f in (LAUNCH, TUNNEL):
        assert subprocess.run(["bash", "-n", str(f)]).returncode == 0, f


# ------------------------------------------------------------------------------------------------ claude
def test_claude_environment_matches_the_table(stub):
    plan = dry(stub, "claude", extra_env={"ANTHROPIC_API_KEY": SECRET_KEY})
    env = plan["env"]
    assert env["ANTHROPIC_BASE_URL"] == stub.url
    assert env["ANTHROPIC_AUTH_TOKEN"] == "<redacted>"
    for name in ("ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
                 "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL"):
        assert env[name] == MODEL, name
    assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == str(WINDOW) and env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == str(OUT)
    assert env["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0" and env["MAX_THINKING_TOKENS"] == "0"
    assert env["API_TIMEOUT_MS"] == env["CLAUDE_STREAM_IDLE_TIMEOUT_MS"] == "1800000"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert plan["window"] == WINDOW and plan["max_output"] == OUT and plan["model"] == MODEL


def test_the_api_key_and_provider_switches_do_not_survive(stub):
    parent = {"ANTHROPIC_API_KEY": SECRET_KEY, "CLAUDE_CODE_USE_BEDROCK": "1", "CLAUDE_CODE_USE_VERTEX": "1"}
    r = run(LAUNCH, "claude", "--dry-run", url=stub.url, extra_env=parent)
    assert r.returncode == 0 and SECRET_KEY not in r.stdout + r.stderr
    plan = json.loads(r.stdout)
    assert "ANTHROPIC_API_KEY" not in plan["env"] and "ANTHROPIC_API_KEY" in plan["unset"]
    for k in ("CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_GATEWAY"):
        assert k in plan["unset"] and k not in plan["env"], k
    settings = json.loads(plan["argv"][1])
    assert plan["argv"][0] == "--settings"
    assert settings["apiKeyHelper"] == "" and settings["permissions"]["deny"] == ["WebSearch"]
    assert settings["env"]["ANTHROPIC_API_KEY"] == ""
    assert all(settings["env"][k] == "0" for k in plan["unset"] if k.startswith("CLAUDE_CODE_USE_"))
    assert settings["env"]["ANTHROPIC_BASE_URL"] == stub.url
    assert settings["env"]["ANTHROPIC_AUTH_TOKEN"] == "<redacted>"          # dry-run redacts it inside the JSON too


def test_the_users_own_settings_and_variables_win_where_they_should(stub):
    mine = json.dumps({"permissions": {"deny": ["Bash"]}, "theme": "dark", "env": {"FOO": "1", "ANTHROPIC_BASE_URL": "x"}})
    plan = dry(stub, "claude", "--settings", mine, "-p", "hello", extra_env={"MAX_THINKING_TOKENS": "16000"})
    assert "MAX_THINKING_TOKENS" not in plan["env"]                    # exported by the user: left alone
    assert plan["argv"].count("--settings") == 1 and plan["argv"][2:] == ["-p", "hello"]
    settings = json.loads(plan["argv"][1])
    assert settings["theme"] == "dark" and settings["permissions"]["deny"] == ["Bash", "WebSearch"]
    assert settings["env"]["FOO"] == "1" and settings["env"]["ANTHROPIC_BASE_URL"] == stub.url   # connection wins


def test_a_bad_model_id_is_refused():
    s = Stub()
    try:
        CATALOGUE["data"][0]["id"] = "bad id; rm -rf"
        r = run(LAUNCH, "claude", "--dry-run", url=s.url)
        assert r.returncode != 0 and "not one this launcher passes" in r.stderr
    finally:
        CATALOGUE["data"][0]["id"] = MODEL
        s.close()


# ------------------------------------------------------------------------------------------------ codex
PROVIDER = ('model_providers.slotstream={name="slotstream-spark",base_url="%s/v1",wire_api="responses",'
            'stream_idle_timeout_ms=1800000}')


def connection(stub):
    return ["-c", 'model_provider="slotstream"', "-c", PROVIDER % stub.url, "-c", f"model_context_window={WINDOW}",
            "-m", MODEL]


def test_codex_argv_goes_after_the_deepest_subcommand(stub):
    assert dry(stub, "codex")["argv"] == connection(stub)
    assert dry(stub, "codex", "exec", "do it")["argv"] == ["exec"] + connection(stub) + ["do it"]
    assert dry(stub, "codex", "exec", "resume", "--last")["argv"] == ["exec", "resume"] + connection(stub) + ["--last"]
    # a -c given above the subcommand moves to it, after the connection, so the user's keys win
    got = dry(stub, "codex", "-c", "a=1", "exec", "go")["argv"]
    assert got == ["exec"] + connection(stub) + ["-c", "a=1", "go"]


def test_codex_keeps_the_users_model_and_refuses_cloud_and_output_schema(stub):
    got = dry(stub, "codex", "exec", "-m", "mine", "go")["argv"]
    assert "-m" in got and got.count("-m") == 1 and got[got.index("-m") + 1] == "mine"
    for bad in (("cloud",), ("exec", "--output-schema", "s.json", "x"), ("--output-schema=s.json",)):
        r = run(LAUNCH, "codex", *bad, "--dry-run", url=stub.url)
        assert r.returncode == 1 and r.stdout == "", bad
        assert "cloud" in r.stderr or "output-schema" in r.stderr


def test_codex_dry_run_never_prints_a_secret(stub):
    plan = dry(stub, "codex", extra_env={"OPENAI_API_KEY": "sk-openai-secret"})
    assert plan["env"] == {} and "sk-openai-secret" not in json.dumps(plan)


# ------------------------------------------------------------------------------------------------ waiting
def test_loading_503s_are_waited_out():
    s = Stub(loading=2)
    try:
        r = run(LAUNCH, "claude", "--dry-run", url=s.url)
        assert r.returncode == 0 and s.hits.count("/v1/models") == 3
    finally:
        s.close()


def test_a_server_that_never_gets_ready_times_out():
    s = Stub(loading=10 ** 6)
    try:
        r = run(LAUNCH, "claude", "--dry-run", url=s.url, extra_env={"SLOTSTREAM_LAUNCH_WAIT_S": "3"})
        assert r.returncode == 1 and "was not ready after 3 s" in r.stderr
    finally:
        s.close()


def test_any_other_status_fails_at_once():
    s = Stub(models_status=404)
    try:
        r = run(LAUNCH, "claude", "--dry-run", url=s.url)
        assert r.returncode == 1 and "HTTP 404" in r.stderr and s.hits.count("/v1/models") == 1
    finally:
        s.close()


def test_usage_errors():
    assert run(LAUNCH).returncode == 2
    assert run(LAUNCH, "vim").returncode == 2


# ------------------------------------------------------------------------------------------------ tunnel
@pytest.mark.parametrize("status,code", [(200, 0), (204, 0), (503, 0), (404, 1), (500, 1)])
def test_tunnel_status(status, code):
    s = Stub(version_status=status)
    try:
        assert run(TUNNEL, "status", url=s.url).returncode == code
    finally:
        s.close()


def test_tunnel_status_without_a_listener():
    r = run(TUNNEL, "status", url="http://127.0.0.1:1")
    assert r.returncode == 1


def test_tunnel_up_is_idempotent_and_never_dials_when_the_port_answers(stub):
    for _ in range(2):
        r = run(TUNNEL, "up", url=stub.url)
        assert r.returncode == 0 and "already up" in r.stdout


def test_tunnel_up_with_an_overridden_dead_target_does_not_open_ssh():
    r = run(TUNNEL, "up", url="http://127.0.0.1:1")
    assert r.returncode == 1 and "not opening ssh" in r.stderr


def test_tunnel_down_touches_only_its_own_forward():
    r = run(TUNNEL, "down", extra_env={"SLOTSTREAM_TUNNEL_PORT": "1"})
    assert r.returncode == 0 and "not running" in r.stdout


def test_tunnel_usage():
    assert run(TUNNEL).returncode == 2


def test_the_tool_runs_in_the_callers_directory(stub, tmp_path):
    """Step 89 found the launcher starting the agent in the repo root: the tool must start where the caller stands."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "claude"
    fake.write_text("#!/bin/sh\npwd -P > \"$MARKER\"\n")
    fake.chmod(0o755)
    work = tmp_path / "work"
    work.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "CLAUDE_"))}
    env.update({"SLOTSTREAM_URL": stub.url, "SLOTSTREAM_LAUNCH_WAIT_S": "20", "MARKER": str(tmp_path / "cwd.txt"),
                "PATH": f"{bindir}:{env['PATH']}"})
    p = subprocess.run(["bash", str(LAUNCH), "claude", "-p", "x"], capture_output=True, text=True, env=env,
                       cwd=str(work), timeout=60)
    assert p.returncode == 0, p.stderr
    assert (tmp_path / "cwd.txt").read_text().strip() == os.path.realpath(work)


def test_codex_gets_a_catalog_that_declares_apply_patch(stub, tmp_path):
    """Step 89: without a catalog entry Codex does not offer the model apply_patch (unknown-model metadata)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "codex"
    fake.write_text("#!/bin/sh\n[ \"$1\" = --version ] && { echo 'codex-cli 9.8.7'; exit 0; }\necho \"$@\" > \"$MARKER\"\n")
    fake.chmod(0o755)
    instr = tmp_path / "instr.md"
    instr.write_text("Use apply_patch.")
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "CLAUDE_"))}
    env.update({"SLOTSTREAM_URL": stub.url, "SLOTSTREAM_LAUNCH_WAIT_S": "20", "MARKER": str(tmp_path / "argv.txt"),
                "PATH": f"{bindir}:{env['PATH']}", "SLOTSTREAM_LAUNCH_DIR": str(tmp_path / "launch"),
                "SLOTSTREAM_CODEX_INSTRUCTIONS": str(instr)})
    p = subprocess.run(["bash", str(LAUNCH), "codex", "exec", "go"], capture_output=True, text=True, env=env,
                       cwd=str(tmp_path), timeout=60)
    assert p.returncode == 0, p.stderr
    path = tmp_path / "launch" / "codex" / f"catalog-9.8.7-{WINDOW}.json"
    assert f"model_catalog_json=\"{path}\"" in (tmp_path / "argv.txt").read_text()
    entry = json.loads(path.read_text())["models"][0]
    assert entry["slug"] == MODEL and entry["apply_patch_tool_type"] == "freeform"
    assert entry["context_window"] == WINDOW and entry["base_instructions"] == "Use apply_patch."


def test_tunnel_output_does_not_pollute_the_tools_stdout(stub, tmp_path):
    """Step 89: `tunnel: up` on stdout broke `claude --output-format json`; the launcher sends it to stderr."""
    src = LAUNCH.read_text()
    assert "tools/mac/tunnel.sh up >&2" in src
