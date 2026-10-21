#!/bin/bash
# SPDX-License-Identifier: MIT
# Port of CodingToolLaunch.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
# Mac: start Claude Code or Codex already connected to the Spark front through the tunnel.
# launch.sh {claude|codex} [args...] [--dry-run]
#   - brings the tunnel up (tools/mac/tunnel.sh), then polls /v1/models for up to 600 s, accepting 503 while the
#     front starts the engine;
#   - builds the connection from what the server reports (served id, window, reply budget) and runs the tool with it;
#   - this run only: the user's own Claude Code / Codex configuration is not written;
#   - --dry-run prints the plan as JSON (secrets redacted) and starts nothing.
# SLOTSTREAM_URL overrides the target (a stub in tests; the tunnel is not touched then).
# SLOTSTREAM_LAUNCH_WAIT_S overrides the 600 s wait. Codex also gets a model catalog entry (apply_patch freeform, the
# served window) written under SLOTSTREAM_LAUNCH_DIR (default ~/.slotstream-spark/launch); its base instructions come
# from SLOTSTREAM_CODEX_INSTRUCTIONS (a file), the cache, or a download of Codex's own prompt for that version. No prompt leaves the Mac by a side route: the hosted-provider
# switches, the API key and the key helper are closed for Claude Code, and `codex cloud` is refused.
# Port of the claude and codex plans of CodingToolLaunch.swift. Bash 3.2-safe; the plan itself is Python (stdlib).
set -u
CALLER=$PWD                                  # the tool runs where the caller stands, not in the repo
cd "$(dirname "$0")/../.."
HERE=$PWD

TOOL=${1:-}
case "$TOOL" in
  claude|codex) shift ;;
  *) echo "usage: launch.sh {claude|codex} [args...] [--dry-run]" >&2; exit 2 ;;
esac
DRY=0
ARGS=()
for a in "$@"; do
  if [ "$a" = "--dry-run" ]; then DRY=1; else ARGS+=("$a"); fi
done

if [ -n "${SLOTSTREAM_URL:-}" ]; then
  URL=${SLOTSTREAM_URL%/}
else
  URL=http://127.0.0.1:${SLOTSTREAM_TUNNEL_PORT:-18434}
  tools/mac/tunnel.sh up >&2 || exit 1
fi

WAIT=${SLOTSTREAM_LAUNCH_WAIT_S:-600}
MODELS=$(mktemp "${TMPDIR:-/tmp}/slotstream-models.XXXXXX")
trap 'rm -f "$MODELS"' EXIT
deadline=$(( $(date +%s) + WAIT ))
while :; do
  code=$(curl -s -o "$MODELS" -w '%{http_code}' --max-time 10 "$URL/v1/models" 2>/dev/null) || code=000
  case "$code" in
    200) break ;;
    503|000) ;;                                          # loading (503) or the forward is still settling
    *) echo "launch: $URL/v1/models answered HTTP $code" >&2; exit 1 ;;
  esac
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "launch: the server at $URL was not ready after ${WAIT} s (last answer $code)" >&2
    exit 1
  fi
  sleep 2
done

read -r -d '' PLAN <<'PY' || true
import json, os, re, shutil, sys

tool, url, models_path, dry = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == "1"
user = sys.argv[5:]

def fail(msg, code=1):
    print("launch: " + msg, file=sys.stderr)
    sys.exit(code)

try:
    with open(models_path) as f:
        entry = json.load(f)["data"][0]
    model, window, out = entry["id"], int(entry["context_window"]), int(entry["max_output_tokens"])
except (ValueError, KeyError, IndexError, TypeError):
    fail(url + "/v1/models is not this server's catalogue")
try:
    os.unlink(models_path)                    # the shell cannot clean up once this process execs the tool
except OSError:
    pass
if not (1 <= len(model) <= 128 and re.fullmatch(r"[A-Za-z0-9._:/-]+", model)):
    fail("the served model id %r is not one this launcher passes to a tool" % model)

PROVIDER_SWITCHES = ["CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY",
                     "CLAUDE_CODE_USE_ANTHROPIC_AWS", "CLAUDE_CODE_USE_ANTHROPIC_GOOGLE_CLOUD",
                     "CLAUDE_CODE_USE_MANTLE", "CLAUDE_CODE_USE_GATEWAY"]
CLAUDE_DEFAULTS = {"MAX_THINKING_TOKENS": "0", "API_TIMEOUT_MS": "1800000",
                   "CLAUDE_STREAM_IDLE_TIMEOUT_MS": "1800000", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
SECRET = re.compile(r"(_TOKEN|_KEY|SECRET|PASSWORD|_AUTH)$", re.I)      # AUTH_TOKEN, API_KEY; not MAX_*_TOKENS

def option_value(args, name):
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return i, 2, args[i + 1]
        if a.startswith(name + "="):
            return i, 1, a[len(name) + 1:]
    return None

def claude_plan():
    conn = {"ANTHROPIC_BASE_URL": url, "ANTHROPIC_AUTH_TOKEN": "slotstream-local", "ANTHROPIC_MODEL": model,
            "ANTHROPIC_SMALL_FAST_MODEL": model, "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(window),
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(out), "CLAUDE_CODE_ATTRIBUTION_HEADER": "0"}
    for alias in ("OPUS", "SONNET", "HAIKU", "FABLE"):
        conn["ANTHROPIC_DEFAULT_%s_MODEL" % alias] = model
    env = dict(conn)
    for k, v in CLAUDE_DEFAULTS.items():
        if k not in os.environ:
            env[k] = v
    settings = {}
    given = option_value(user, "--settings")
    args = list(user)
    if given:
        i, width, value = given
        try:
            text = value if value.lstrip().startswith("{") else open(os.path.expanduser(value)).read()
            settings = json.loads(text)
        except (OSError, ValueError):
            fail("the Claude Code --settings you passed is neither a JSON object nor a readable JSON file")
        if not isinstance(settings, dict):
            fail("the Claude Code --settings you passed must be a JSON object")
        del args[i:i + width]
    for key in ("env", "permissions"):
        if key in settings and not isinstance(settings[key], dict):
            fail("the `%s` in the Claude Code --settings you passed must be an object" % key)
    senv = dict(settings.get("env", {}))
    senv.update(conn)
    senv["ANTHROPIC_API_KEY"] = ""
    for k in PROVIDER_SWITCHES:
        senv[k] = "0"
    settings["env"] = senv
    settings["apiKeyHelper"] = ""
    perms = dict(settings.get("permissions", {}))
    deny = list(perms.get("deny", []))
    if "WebSearch" not in deny:
        deny.append("WebSearch")
    perms["deny"] = deny
    settings["permissions"] = perms
    argv = ["--settings", json.dumps(settings, sort_keys=True)] + args
    notes = ["Claude Code will use %s with a %d-token window." % (model, window)]
    if window < 32768:
        notes.append("Claude Code's instructions and tools take about 15,000 tokens; this window is small for it.")
    return env, ["ANTHROPIC_API_KEY"] + PROVIDER_SWITCHES, argv, notes

CODEX_COMMANDS = {"": {"exec", "e", "review", "login", "logout", "mcp", "plugin", "mcp-server", "app-server",
                       "remote-control", "app", "completion", "update", "doctor", "sandbox", "debug", "apply", "a",
                       "resume", "archive", "delete", "migrate-rollouts", "unarchive", "fork", "cloud",
                       "exec-server", "features"},
                  "exec": {"resume", "fork", "review"}, "e": {"resume", "fork", "review"}}
CODEX_VALUE_OPTIONS = {"-c", "--config", "--enable", "--disable", "--remote", "--remote-auth-token-env", "-m",
                       "--model", "--local-provider", "-p", "--profile", "-s", "--sandbox", "-C", "--cd",
                       "--add-dir", "-a", "--ask-for-approval", "--output-schema", "--color", "-o",
                       "--output-last-message", "--base", "--commit", "--title"}

def codex_scan(words):
    commands, end, config_words, i = [], 0, [], 0
    while i < len(words):
        w = words[i]
        if w == "--":
            break
        if w in ("-i", "--image"):
            i += 1
            while i < len(words) and not words[i].startswith("-"):
                i += 1
            continue
        if w in ("-c", "--config"):
            config_words += [i, i + 1]
            i += 2
            continue
        if w.startswith("--config=") or (w.startswith("-c") and not w.startswith("--")):
            config_words.append(i)
            i += 1
            continue
        if w.startswith("-"):
            i += 2 if w in CODEX_VALUE_OPTIONS else 1
            continue
        if w not in CODEX_COMMANDS.get(commands[-1] if commands else "", set()):
            break
        commands.append(w)
        i += 1
        end = i
    return commands, end, config_words

def codex_arguments(connection, words):
    """CodingToolLaunch.codexArguments: Codex reads the -c settings of only the deepest command given any, so the
    connection goes right after it and every -c the user gave above it moves there too, after the connection."""
    _, end, config_words = codex_scan(words)
    if end == 0:
        return connection + words
    moving = {i for i in config_words if i < end}
    head = [words[i] for i in range(end) if i not in moving]
    moved = [words[i] for i in sorted(moving) if i < len(words)]
    return head + connection + moved + words[end:]

def launch_dir():
    return os.environ.get("SLOTSTREAM_LAUNCH_DIR") or os.path.join(os.path.expanduser("~"), ".slotstream-spark", "launch")

def codex_version():
    import subprocess
    try:
        out = subprocess.run(["codex", "--version"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    m = re.search(r"\d+\.\d+\.\d+\S*", out)
    return m.group(0) if m else "unknown"

FALLBACK_INSTRUCTIONS = ("You are a coding agent running in the Codex CLI. Work in the user's workspace: read files, "
                         "edit them with the apply_patch tool, run commands with the shell tool, and report briefly "
                         "what you changed.")

def codex_instructions(version):
    """Codex's own base prompt for this version (cached), SLOTSTREAM_CODEX_INSTRUCTIONS for a local file, or a short
    built-in text when the download fails."""
    local = os.environ.get("SLOTSTREAM_CODEX_INSTRUCTIONS")
    if local:
        with open(local) as f:
            return f.read()
    cache = os.path.join(launch_dir(), "codex", "instructions-%s.md" % version)
    try:
        with open(cache) as f:
            text = f.read()
        if text.strip():
            return text
    except OSError:
        pass
    import urllib.request
    try:
        src = "https://raw.githubusercontent.com/openai/codex/rust-v%s/codex-rs/models-manager/prompt.md" % version
        with urllib.request.urlopen(src, timeout=30) as r:
            text = r.read().decode()
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        with open(cache, "w") as f:
            f.write(text)
        return text
    except (OSError, ValueError):
        print("launch: Codex's base instructions could not be downloaded; using a short built-in text", file=sys.stderr)
        return FALLBACK_INSTRUCTIONS

def write_codex_catalog(path, version):
    """The model entry that tells Codex the served window and declares apply_patch (CodingToolLaunch.codexCatalog):
    without it Codex falls back to unknown-model metadata and offers the model no apply_patch tool."""
    entry = {"slug": model, "display_name": "Qwen3.8-Flash-Next via Slotstream-Spark",
             "description": "Local model served by slotstream-spark on the Spark.",
             "supported_reasoning_levels": [{"effort": "low", "description": "Brief thinking before each reply"},
                                            {"effort": "medium", "description": "Balanced thinking"},
                                            {"effort": "high", "description": "Thorough thinking, slower"}],
             "shell_type": "unified_exec", "visibility": "list", "supported_in_api": True, "priority": 1,
             "availability_nux": None, "upgrade": None, "support_verbosity": False, "default_verbosity": None,
             "apply_patch_tool_type": "freeform", "truncation_policy": {"mode": "bytes", "limit": 10000},
             "experimental_supported_tools": [], "supports_parallel_tool_calls": False, "supports_reasoning_summaries": False, "context_window": window, "max_context_window": window,
             "base_instructions": codex_instructions(version)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"models": [entry]}, f, indent=1)
        f.write("\n")

def codex_plan():
    commands, _, _ = codex_scan(user)
    if commands[:1] == ["cloud"]:
        fail("`codex cloud` runs tasks on OpenAI's servers, not on the Spark; run `codex cloud` directly.")
    if any(a == "--output-schema" or a.startswith("--output-schema=") for a in user):
        fail("`--output-schema` asks for a reply that follows a JSON schema, which this server does not support.")
    provider = ('{name="slotstream-spark",base_url="%s/v1",wire_api="responses",stream_idle_timeout_ms=1800000}' % url)
    settings = ['model_provider="slotstream"', "model_providers.slotstream=" + provider,
                "model_context_window=%d" % window]
    version = codex_version()
    catalog = os.path.join(launch_dir(), "codex", "catalog-%s-%d.json" % (version, window))
    settings.append("model_catalog_json=" + json.dumps(catalog))
    connection = [x for s in settings for x in ("-c", s)]
    if not any(a in ("-m", "--model") or a.startswith("--model=") for a in user):
        connection += ["-m", model]
    if not dry:
        write_codex_catalog(catalog, version)
    notes = ["Codex %s will use %s with a %d-token window." % (version, model, window)]
    return {}, [], codex_arguments(connection, user), notes

env, unset, argv, notes = claude_plan() if tool == "claude" else codex_plan()

def redact_text(s):
    try:
        obj = json.loads(s)
    except ValueError:
        return s
    def walk(o):
        if isinstance(o, dict):
            return {k: ("<redacted>" if SECRET.search(k) and isinstance(v, str) and v else walk(v)) for k, v in o.items()}
        return o
    return json.dumps(walk(obj), sort_keys=True)

if dry:
    print(json.dumps({"tool": tool, "url": url, "model": model, "window": window, "max_output": out,
                      "env": {k: ("<redacted>" if SECRET.search(k) else v) for k, v in sorted(env.items())},
                      "unset": sorted(unset), "argv": [redact_text(a) for a in argv], "notes": notes},
                     indent=1, sort_keys=True))
    sys.exit(0)

if shutil.which(tool) is None:
    hint = {"claude": "Install Claude Code from https://code.claude.com/docs/en/setup",
            "codex": "Install Codex with `npm install -g @openai/codex`"}[tool]
    fail("%s is not installed. %s" % (tool, hint), 127)
for n in notes:
    print("launch: " + n, file=sys.stderr)
child = {k: v for k, v in os.environ.items() if k not in unset}
child.update(env)
os.execvpe(tool, [tool] + argv, child)
PY

cd "$CALLER" || exit 1
exec python3 -c "$PLAN" "$TOOL" "$URL" "$MODELS" "$DRY" ${ARGS[@]+"${ARGS[@]}"}
