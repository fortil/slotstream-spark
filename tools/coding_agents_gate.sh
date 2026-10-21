#!/bin/bash
# Mac: the coding-agents gate H5. Claude Code, Codex and an Open WebUI request sequence talk to the Spark front
# through the tunnel (D5 front on 18434), each inside a throwaway project under --out.
#   coding_agents_gate.sh --out tmp/h5 --record db/records/measurements
# Refuses outside 04:00-23:00 (exit 75) unless the Spark has $R/state/PAUSED with NO_WINDOWS=1 (production paused by
# the user); the host check then expects EXPECTED_CONTAINERS from that file instead of 14. Never uses --dangerously-skip-permissions; Bash is never allowed to Claude
# Code (--allowedTools Read,Write,Edit, --disallowedTools Bash, --permission-mode acceptEdits); Codex runs with
# -s workspace-write inside its own throwaway project. The user's Claude Code and Codex configuration is not read or
# written: CLAUDE_CONFIG_DIR and CODEX_HOME live under --out. Only the two own units are started (socket) and, when the
# gate started them, stopped again; the engine runs behind the front, never under a guard run of its own.
# Result: <record>/h5-agents.md plus a capture directory db/sources/runs/YYYY/MM/<ts>-h5-agents/ (verdict.json).
# Exit: 0 = claude and webui passed (codex passed or was skipped), 1 = a check failed, 75 = outside the window.
set -u
cd "$(dirname "$0")/.."
ROOT=$PWD
OUT=""; REC=""
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT=${2:?}; shift 2 ;;
    --record) REC=${2:?}; shift 2 ;;
    *) echo "usage: coding_agents_gate.sh --out DIR --record DIR" >&2; exit 2 ;;
  esac
done
[ -n "$OUT" ] && [ -n "$REC" ] || { echo "usage: coding_agents_gate.sh --out DIR --record DIR" >&2; exit 2; }
case "$OUT" in /*) ;; *) OUT=$ROOT/$OUT ;; esac
case "$REC" in /*) ;; *) REC=$ROOT/$REC ;; esac

. tools/spark/env.sh
remote() { ssh $SPARK_SSH_OPTS "$SPARK_HOST" "export XDG_RUNTIME_DIR=/run/user/\$(id -u); $1"; }
# pause mode, read on the Spark by paused_load: "<active> <no_windows> <expected>", ERR when malformed, empty when unreachable
PAUSE=$(remote "cd $SRC && . tools/spark/env.sh && paused_load >/dev/null && echo \"\$PAUSED_ACTIVE \$PAUSED_NO_WINDOWS \$PAUSED_EXPECTED\" || echo ERR" 2>/dev/null | tail -1)
[ "$PAUSE" = ERR ] && { echo "coding_agents_gate: malformed state/PAUSED on the Spark" >&2; exit 1; }
NO_WINDOWS=$(echo "$PAUSE" | awk '{print $2+0}')
EXPECTED=$(echo "$PAUSE" | awk '{print $3}'); EXPECTED=${EXPECTED:-14}
export EXPECTED_CONTAINERS=$EXPECTED
hour=$((10#$(date +%H)))
if [ "$NO_WINDOWS" != 1 ] && { [ "$hour" -lt 4 ] || [ "$hour" -ge 23 ]; }; then
  echo "coding_agents_gate: refused outside 04:00-23:00 (hour $hour)" >&2
  exit 75
fi

PY=$ROOT/.venv/bin/python
URL=http://127.0.0.1:${SLOTSTREAM_TUNNEL_PORT:-18434}
mkdir -p "$OUT"
rm -rf "$OUT/claude-proj" "$OUT/codex-proj" "$OUT/claude-config" "$OUT/codex-home" "$OUT/launch"
STARTED_AT=$(date '+%Y-%m-%d %H:%M:%S')

# ---- before: units, containers, kernel-log marker ------------------------------------------------------------
SOCKET_BEFORE=$(remote 'systemctl --user is-active slotstream-spark.socket' 2>/dev/null | head -1)
CONTAINERS_BEFORE=$(remote 'docker ps -q | wc -l' | tr -d ' ')
REMOTE_SINCE=$(remote "date '+%Y-%m-%d %H:%M:%S'")
echo "gate: socket before: ${SOCKET_BEFORE:-unknown}; containers up: $CONTAINERS_BEFORE; remote since $REMOTE_SINCE"
[ "$CONTAINERS_BEFORE" = "$EXPECTED" ] || { echo "gate: expected $EXPECTED containers Up before the run, found $CONTAINERS_BEFORE" >&2; exit 1; }
if [ "$SOCKET_BEFORE" != active ]; then
  remote 'systemctl --user start slotstream-spark.socket' || { echo "gate: could not start the front socket" >&2; exit 1; }
fi
cleanup() {
  if [ "$SOCKET_BEFORE" != active ]; then
    remote 'systemctl --user stop slotstream-spark.socket slotstream-spark-front.service slotstream-spark.service' >/dev/null 2>&1
  fi
}
trap cleanup EXIT

tools/mac/tunnel.sh up || exit 1
# A GET never starts the engine (the front answers 503 engine_stopped), so one request that can generate wakes it: the
# front holds it while the engine loads and the dialect may refuse the body afterwards, which is fine. Then wait for
# /v1/models (503 while loading).
curl -s -o /dev/null --max-time 300 -X POST -H 'content-type: application/json' \
  -d '{"model":"wake","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' "$URL/v1/chat/completions" || true
for _ in $(seq 1 300); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$URL/v1/models" 2>/dev/null) || code=000
  [ "$code" = 200 ] && break
  sleep 3
done
[ "$code" = 200 ] || { echo "gate: the front did not serve /v1/models (last HTTP $code)" >&2; exit 1; }

# ---- the agents --------------------------------------------------------------------------------------------
$PY - "$OUT" "$ROOT" "$URL" <<'PY'
import json, os, shutil, subprocess, sys, time, urllib.request, zlib, struct
from pathlib import Path

out, root, url = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
launch = str(root / "tools/mac/launch.sh")
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
res = {"claude": {"checks": []}, "codex": {"checks": []}}

def get(path):
    with opener.open(url + path, timeout=30) as r:
        return r.read().decode()

def metric(name):
    for line in get("/metrics").splitlines():
        if line.startswith(name + " "):
            return float(line.split()[1])
    return 0.0

def ps_size():
    return int(json.loads(get("/api/ps"))["models"][0]["size"])

def check(who, name, ok, info=None):
    res[who]["checks"].append({"check": name, "ok": bool(ok), "info": info})
    print(("PASS " if ok else "FAIL ") + who + ": " + name + ("" if info is None else "  " + json.dumps(info)), flush=True)

def sh(cmd, cwd, env=None, timeout=1800):
    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True, timeout=timeout)
    return p

def png(path, size=96):
    raw = b"".join(b"\x00" + b"".join(bytes((x * 255 // size, y * 255 // size, 128)) for x in range(size)) for y in range(size))
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))

def git_init(d):
    d.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(d)], check=True)
    subprocess.run(["git", "-C", str(d), "config", "user.email", "h5@example.invalid"], check=True)
    subprocess.run(["git", "-C", str(d), "config", "user.name", "h5"], check=True)

# ---- Claude Code ---------------------------------------------------------------------------------------------
claude_ver = sh(["claude", "--version"], str(out)).stdout.strip() if shutil.which("claude") else None
res["claude"]["version"] = claude_ver
if claude_ver is None:
    check("claude", "claude is installed", False)
else:
    proj = out / "claude-proj"
    git_init(proj)
    env = {"CLAUDE_CONFIG_DIR": str(out / "claude-config")}
    (out / "claude-config").mkdir(exist_ok=True)
    perm = ["--permission-mode", "acceptEdits", "--allowedTools", "Read,Write,Edit", "--disallowedTools", "Bash",
            "--output-format", "json"]
    line1, line2 = "slotstream h5 line one", "slotstream h5 line two"

    def turn(label, prompt, extra=()):
        t0 = time.time()
        p = sh([launch, "claude", "-p", prompt] + list(extra) + perm, str(proj), env)
        (out / f"claude-{label}.stdout.json").write_text(p.stdout)
        (out / f"claude-{label}.stderr.txt").write_text(p.stderr)
        try:
            j = json.loads(p.stdout)
        except ValueError:
            j = {}
        res["claude"][label] = {"rc": p.returncode, "seconds": round(time.time() - t0, 1), "num_turns": j.get("num_turns"),
                                "is_error": j.get("is_error"), "session_id": j.get("session_id"), "usage": j.get("usage"),
                                "result_head": str(j.get("result", ""))[:200]}
        return j

    j1 = turn("turn1", "Create a file named hello.txt in the current directory whose content is exactly the text "
              f"{line1} and nothing else, then read hello.txt back with the Read tool and tell me what it contains.")
    f = proj / "hello.txt"
    got = f.read_text() if f.exists() else None
    check("claude", "turn 1 wrote hello.txt with the exact text", got is not None and got.rstrip("\n") == line1, got)
    check("claude", "turn 1 num_turns >= 2", (j1.get("num_turns") or 0) >= 2 and not j1.get("is_error"), j1.get("num_turns"))
    sid = j1.get("session_id")
    reused0 = metric("slotstream_reused_tokens_total")
    j2 = turn("turn2", f"Append a second line with the exact text {line2} to hello.txt using the Edit tool, keeping the "
              "first line, then read the file back.", ["--resume", sid or "missing-session"])
    reused1 = metric("slotstream_reused_tokens_total")
    got = f.read_text() if f.exists() else ""
    lines = [l for l in got.split("\n") if l != ""]
    check("claude", "turn 2 (resume) leaves two lines", lines == [line1, line2], lines)
    # `usage` of the result JSON is the last API call of the invocation; `modelUsage` sums all of its calls and the
    # server counts every reused token, so a last call that happened to find nothing to reuse is not a missing cache
    cread = ((j2.get("usage") or {}).get("cache_read_input_tokens") or 0)
    msum = sum((m or {}).get("cacheReadInputTokens", 0) for m in (j2.get("modelUsage") or {}).values())
    check("claude", "turn 2 cache read > 0 (usage.cache_read_input_tokens, else modelUsage sum with the server's reused_tokens)",
          cread > 0 or (msum > 0 and reused1 - reused0 > 0),
          {"usage.cache_read_input_tokens": cread, "modelUsage.cacheReadInputTokens": msum, "server_reused_tokens_delta": reused1 - reused0})
    res["claude"]["cache_read"] = {"usage": cread, "model_usage": msum, "server_reused_delta": reused1 - reused0}
    png(proj / "pic.png")
    rej0, size0 = metric("slotstream_images_rejected_total"), ps_size()
    j3 = turn("turn3", "Look at pic.png with the Read tool, then create out.txt containing exactly the word done and "
              "nothing else.")          # a fresh session: three resumed turns plus an image do not fit a 32K window
    rej1, size1 = metric("slotstream_images_rejected_total"), ps_size()
    o = proj / "out.txt"
    got = o.read_text() if o.exists() else None
    check("claude", "turn 3 wrote out.txt == done", got is not None and got.strip() == "done", got)
    check("claude", "turn 3 images_rejected_total +1", rej1 - rej0 == 1, {"before": rej0, "after": rej1})
    check("claude", "turn 3 /api/ps size unchanged within 1 MiB", abs(size1 - size0) <= 1 << 20, {"before": size0, "after": size1})
    res["claude"]["images_rejected_delta"] = rej1 - rej0
    res["claude"]["ps_size"] = [size0, size1]

# ---- Codex ----------------------------------------------------------------------------------------------------
if shutil.which("codex") is None:
    res["codex"]["skipped"] = "codex is not installed"
    print("codex: skipped (not installed)", flush=True)
else:
    res["codex"]["version"] = sh(["codex", "--version"], str(out)).stdout.strip()
    proj = out / "codex-proj"
    git_init(proj)
    (proj / "notes.txt").write_text("alpha\nbeta\ngamma\n")
    subprocess.run(["git", "-C", str(proj), "add", "notes.txt"], check=True)
    subprocess.run(["git", "-C", str(proj), "commit", "-q", "-m", "seed"], check=True)
    (out / "codex-home").mkdir(exist_ok=True)
    t0 = time.time()
    p = sh([launch, "codex", "exec", "-c", 'approval_policy="never"', "-s", "workspace-write",
            "Use the apply_patch tool to apply exactly this patch to notes.txt, then stop:\n"
            "*** Begin Patch\n*** Update File: notes.txt\n@@\n alpha\n-beta\n+BETA\n gamma\n*** End Patch"], str(proj), {"CODEX_HOME": str(out / "codex-home"), "SLOTSTREAM_LAUNCH_DIR": str(out / "launch")})
    (out / "codex.stdout.txt").write_text(p.stdout)
    (out / "codex.stderr.txt").write_text(p.stderr)
    got = (proj / "notes.txt").read_text()
    res["codex"]["run"] = {"rc": p.returncode, "seconds": round(time.time() - t0, 1)}
    check("codex", "apply_patch edit task changed exactly line 2", got == "alpha\nBETA\ngamma\n", got)
    res["codex"]["skipped"] = None

(out / "agents.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
failed = [c for who in ("claude", "codex") for c in res[who]["checks"] if not c["ok"]]
sys.exit(1 if failed else 0)
PY
AGENTS_RC=$?

# ---- the Open WebUI order -----------------------------------------------------------------------------------
$PY tools/webui_sequence.py --url "$URL" --json "$OUT/webui.json"
WEBUI_RC=$?

# ---- after ---------------------------------------------------------------------------------------------------
CONTAINERS_AFTER=$(remote 'docker ps -q | wc -l' | tr -d ' ')
KLOG=$(remote "journalctl -k --since '$REMOTE_SINCE' --no-pager 2>/dev/null | grep -Eci 'oom-kill|NV_ERR_NO_MEMORY|out of memory' || true" | tr -d ' ')
echo "gate: containers up after: $CONTAINERS_AFTER; oom/NV_ERR_NO_MEMORY kernel-log lines since start: ${KLOG:-?}"

# ---- record --------------------------------------------------------------------------------------------------
$PY - "$OUT" "$REC" "$ROOT" "$AGENTS_RC" "$WEBUI_RC" "$CONTAINERS_AFTER" "${KLOG:-0}" "$STARTED_AT" <<'PY'
import json, os, subprocess, sys, time
from pathlib import Path

out, rec, root = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
agents_rc, webui_rc, containers, klog, started = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]), sys.argv[8]
sys.path.insert(0, str(root / "tools"))
import webui_sequence

agents = json.loads((out / "agents.json").read_text()) if (out / "agents.json").exists() else {"claude": {"checks": []}, "codex": {"checks": []}}
webui = json.loads((out / "webui.json").read_text()) if (out / "webui.json").exists() else {"checks": [], "detail": {}}
claude_ok = bool(agents["claude"]["checks"]) and all(c["ok"] for c in agents["claude"]["checks"])
codex = agents["codex"]
codex_state = "skipped" if codex.get("skipped") else ("pass" if codex["checks"] and all(c["ok"] for c in codex["checks"]) else "FAIL")
webui_ok = bool(webui["checks"]) and all(c["ok"] for c in webui["checks"])
expected = int(os.environ.get("EXPECTED_CONTAINERS", "14"))      # 14 unless the Spark is under state/PAUSED
host_ok = containers == expected and klog == 0
ok = claude_ok and webui_ok and codex_state != "FAIL" and host_ok
tree = webui_sequence.tree_sha256(root)
now = time.localtime()
stamp = time.strftime("%Y%m%dT%H%M%S", now)
cap = root / "db/sources/runs" / time.strftime("%Y", now) / time.strftime("%m", now) / f"{stamp}-h5-agents"
cap.mkdir(parents=True, exist_ok=True)
for name in ("agents.json", "webui.json"):
    if (out / name).exists():
        (cap / name).write_text((out / name).read_text())
(cap / "verdict.json").write_text(json.dumps({"rc": 0 if ok else 1, "claude": claude_ok, "codex": codex_state, "webui": webui_ok,
                                              "containers_after": containers, "klog_lines": klog, "started": started,
                                              "server_tree_sha256": tree}, indent=1) + "\n")

def rows(who):
    return [f"| {who}: {c['check']} | {'yes' if c['ok'] else 'NO'} |" for c in agents[who]["checks"]]

usage = {k: agents["claude"].get(k, {}).get("usage") for k in ("turn1", "turn2", "turn3")}
lines = ["# h5-agents: coding agents and Open WebUI through the tunnel",
         time.strftime("Date: %Y-%m-%d", now),
         f"Capture: db/sources/runs/{cap.relative_to(root / 'db/sources/runs')}/",
         "Command: tools/mac/tunnel.sh up && bash tools/coding_agents_gate.sh --out tmp/h5 --record db/records/measurements",
         "",
         "| check | ok |", "|---|---|"]
lines += rows("claude") + rows("codex")
lines += [f"| webui: {c['check']} | {'yes' if c['ok'] else 'NO'} |" for c in webui["checks"]]
lines += [f"| host: {expected} containers Up after the run | {'yes' if containers == expected else 'NO'} ({containers}) |",
          f"| host: no oom-kill / NV_ERR_NO_MEMORY in the kernel log since the run started | {'yes' if klog == 0 else 'NO'} ({klog} lines) |",
          "",
          f"- agent versions: claude {agents['claude'].get('version')}; codex {codex.get('version') or codex.get('skipped')}",
          f"- verdicts: claude {'pass' if claude_ok else 'FAIL'}; codex {codex_state}; webui {'pass' if webui_ok else 'FAIL'}",
          "- claude usage per turn (last API call of each invocation): " + json.dumps(usage, sort_keys=True),
          f"- claude turn 2 cache read: {agents['claude'].get('cache_read')}",
          "- claude turn seconds: " + json.dumps({k: agents["claude"].get(k, {}).get("seconds") for k in ("turn1", "turn2", "turn3")}),
          f"- images_rejected_total delta in claude turn 3: {agents['claude'].get('images_rejected_delta')}; /api/ps size before and after: {agents['claude'].get('ps_size')}",
          f"- webui: turn 1 prompt tokens {webui['detail'].get('turn1_prompt_tokens')}, turn 2 cached_tokens {webui['detail'].get('turn2_cached_tokens')}",
          "- safety: claude ran with --permission-mode acceptEdits --allowedTools Read,Write,Edit --disallowedTools Bash and a CLAUDE_CONFIG_DIR under the output directory; codex with -s workspace-write, approval_policy never and a CODEX_HOME under the output directory; the throwaway projects live under the output directory",
          f"- server_tree_sha256: {tree} (sha256 over `<path>\\0<file sha256>\\n` of slotstream_spark/server/**/*.py and slotstream_spark/front.py, sorted; tools/webui_sequence.py tree_sha256)",
          f"Verdict: {'PASS' if ok else 'FAIL'} — claude {'pass' if claude_ok else 'FAIL'}, codex {codex_state}, webui {'pass' if webui_ok else 'FAIL'}, host {'clean' if host_ok else 'NOT clean'}"]
rec.mkdir(parents=True, exist_ok=True)
(rec / "h5-agents.md").write_text("\n".join(lines) + "\n")
print(lines[-1])
sys.exit(0 if ok else 1)
PY
exit $?
