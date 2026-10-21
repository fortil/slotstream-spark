#!/usr/bin/env python3
"""F3 session job (one guard run, label f3-session). Phases in order:

  1 traces   tools/trace_routing.py: routing traces in D5 graph mode (48 requests x 128 tokens)
  2 A/A      16 balanced pairs of fresh `cli bench decode --mode D5` runs (prompts short and 2k, 256 tokens)
  3 LB       >= 5 clean pairs, engine vs the D5 yardstick (llama.cpp Q4_1 arm B, cold prefill), up to 8 attempts
  4 D1       3 engine runs in D1 (10 chunks), informative

Writes $CAP/started.json, $CAP/result.json (rewritten after every phase and every pair) and $CAP/traces/. The paired
quantity is decode tok/s per prompt, (N-1)/sum(inter-token) on both sides, never TTFT. The yardstick arm comes from
tools.yardstick.yardstick() and must be `q41-b-d5`. Excluded pairs stay in the output with a reason.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from bench import stats  # noqa: E402
from slotstream_spark.host import eligibility  # noqa: E402
from slotstream_spark.config import modes  # noqa: E402


PROMPT_KEYS = ("short", "2k")
YARD_ARM = "q41-b-d5"
LB_CLIENT_ARGS = ["--no-cache-prompt", "--ids", "short,2k", "--reps", "3", "--warmup", "1"]
AA_PAIRS, LB_CLEAN, LB_ATTEMPTS, D1_RUNS, D1_SLOTS = 16, 5, 8, 3, 10 * 388
EST_ENGINE_S, EST_LB_PAIR_S = 240, 560
CACHE_FULL_MIB, CACHE_FULL_MEMFREE_MIB, CACHE_OFFSET_MIB, CACHE_MIN_MIB = 43520, 70465, 26945, 21760
CACHE_RULE = {"full_mib": CACHE_FULL_MIB, "memfree_full_mib": CACHE_FULL_MEMFREE_MIB, "offset_mib": CACHE_OFFSET_MIB,
              "min_mib": CACHE_MIN_MIB}


def lb_args(yardstick):
    """`q41-b-d5` -> ['--weights','q41','--arm','B','--mode','D5']; any other arm is a hard error."""
    arm = yardstick.get("arm")
    if arm != YARD_ARM:
        raise SystemExit(f"session: yardstick arm {arm!r} is not {YARD_ARM!r}")
    if yardstick.get("op_offload") is not False or yardstick.get("prefill_device") != "cpu":
        raise SystemExit("session: yardstick must have op_offload=false and prefill_device='cpu'")
    w, a, m = arm.split("-")
    return ["--weights", w, "--arm", a.upper(), "--mode", m.upper()]


def engine_cmd(out, mode="D5", slots=None, python=sys.executable):
    c = [python, "-m", "slotstream_spark.cli", "bench", "decode", "--mode", mode, "--prompts", "bench/lb/prompts.json",
         "--prompt-keys", ",".join(PROMPT_KEYS), "--tokens", "256", "--warmup", "1", "--out", str(out)]
    if slots:
        c += ["--pool-slots", str(slots)]
    return c


def engine_tps(result):
    """{prompt: tok/s} from a `cli bench decode` result; None for a missing value."""
    return {k: result["prompts"][k].get("tok_s") for k in PROMPT_KEYS}


def lb_tps(records):
    """{prompt: median decode_tps of the reps} from client.jsonl records; plus prefill tps, max cache_n, cold flag."""
    out, pf, cache_n, cold = {}, {}, 0, True
    for k in PROMPT_KEYS:
        reps = [r for r in records if r["prompt_id"] == k]
        vals = [r["decode_tps"] for r in reps if r.get("decode_tps") is not None]
        out[k] = stats.true_median(vals) if vals else None
        pv = [r["prefill_tps"] for r in reps if r.get("prefill_tps") is not None]
        pf[k] = stats.true_median(pv) if pv else None
        for r in reps:
            cache_n = max(cache_n, r["cache_n"] if r.get("cache_n") is not None else 10 ** 9)
            cold = cold and r.get("cold") is True
    return out, pf, cache_n, cold


def lb_pair_row(i, eng, lb, prefill, cache_n, cold, reasons, meta=None, snapshot=None):
    """One pair record. Reasons present, or a missing tok/s, exclude the pair; `warm_prefix` when cache_n != 0 or not cold."""
    reasons = list(reasons)
    if cache_n != 0 or not cold:
        reasons.append("warm_prefix")
    ratio_by = {}
    for k in PROMPT_KEYS:
        if eng.get(k) and lb.get(k):
            ratio_by[k] = eng[k] / lb[k]
        else:
            reasons.append(f"missing_tps:{k}")
    clean = not reasons and len(ratio_by) == len(PROMPT_KEYS)
    row = {"pair": i, "engine_tps": eng, "lb_tps": lb,
           "ratio": stats.true_median(list(ratio_by.values())) if len(ratio_by) == len(PROMPT_KEYS) else None,
           "ratio_by_prompt": ratio_by, "prefill_tps_lb": prefill, "cache_n_max": cache_n, "clean": clean,
           "reason": ",".join(dict.fromkeys(reasons)) if reasons else None}
    row.update(meta or {})
    row["snapshot"] = snapshot
    return row


class Session:
    def __init__(self, cap, *, runner=subprocess.run, snapshot=None, now=time.time, deadline=None, yardstick=None,
                 lb_runner=None, python=sys.executable, meminfo="/proc/meminfo", sleep=time.sleep):
        self.cap = Path(cap)
        self.runner, self.now = runner, now
        self.deadline = deadline
        self.snapshot = snapshot or self._snapshot
        self.python = python
        self.meminfo, self.sleep = meminfo, sleep
        if yardstick is None:
            from tools import yardstick as yardstick_mod
            yardstick = yardstick_mod.yardstick()
        self.yard = yardstick
        self.lb_flags = lb_args(yardstick)
        self.lb_runner = lb_runner or self.run_lb_side
        self.result = {"label": "f3-session", "yardstick_arm": yardstick["arm"], "phases": {}, "complete": False}

    # --- plumbing -----------------------------------------------------------------------------------------
    def _snapshot(self, est_s=EST_ENGINE_S):
        return eligibility.collect(int(os.environ.get("SLOTSTREAM_FOOTPRINT_MIB", "0")), est_s)

    def save(self):
        (self.cap / "result.json").write_text(json.dumps(self.result, indent=1, sort_keys=True) + "\n")

    def time_left(self):
        return None if self.deadline is None else self.deadline - self.now()

    def enough(self, need_s):
        left = self.time_left()
        return left is None or left > need_s + 120

    def run_engine(self, out, mode="D5", slots=None):
        """One fresh engine process -> (parsed result or None, error or None)."""
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.unlink(missing_ok=True)
        p = self.runner(engine_cmd(out, mode, slots, self.python), capture_output=True, text=True, cwd=str(ROOT))
        if p.returncode != 0:
            return None, f"rc={p.returncode}"
        try:
            return json.loads(out.read_text()), None
        except Exception as e:                                  # noqa: BLE001
            return None, f"unparsable:{type(e).__name__}"

    # --- phases -------------------------------------------------------------------------------------------
    def preflight(self):
        """Fail in seconds, not after two hours, when the device NVMe limit cannot be read."""
        lim, src = eligibility.nvme_limit_mc()
        if lim is None:
            raise SystemExit("session: nvme hwmon limit unreadable")
        self.nvme_limit = (lim, src)

    def phase_traces(self):
        d = self.cap / "traces"
        p = self.runner([self.python, "tools/trace_routing.py", "--out", str(d)], capture_output=True, text=True,
                        cwd=str(ROOT))
        ok = p.returncode == 0
        self.result["phases"]["traces"] = {"ok": ok, "rc": p.returncode, "dir": "traces"}
        self.save()
        return ok

    def phase_aa(self, pairs=AA_PAIRS):
        rows, d = [], self.cap / "aa"
        for i, order in enumerate(stats.balanced_order(pairs)):
            if not self.enough(2 * EST_ENGINE_S):
                self.result["phases"]["aa_stopped"] = "deadline"
                break
            vals, errs = {}, []
            for arm in order:
                r, e = self.run_engine(d / f"p{i:02d}-{arm}.json")
                vals[arm] = engine_tps(r) if r else None
                if e:
                    errs.append(f"{arm}:{e}")
            snap = self.snapshot()
            ok, why = eligibility.evaluate(snap)
            ratios = {k: (vals["B"][k] / vals["A"][k]) if vals.get("A") and vals.get("B") and vals["A"].get(k)
                      and vals["B"].get(k) else None for k in PROMPT_KEYS}
            rows.append({"pair": i, "order": order, "a": vals.get("A"), "b": vals.get("B"), "ratio": ratios,
                         "eligible": ok and not errs and all(v is not None for v in ratios.values()),
                         "reasons": list(why) + errs, "snapshot": snap})
            self.result["phases"]["aa"] = {"pairs": rows}
            self.save()
        clean = [r for r in rows if r["eligible"]]
        dev = [abs(v - 1.0) for r in clean for v in r["ratio"].values()]
        self.result["phases"]["aa"] = {"pairs": rows, "pairs_clean": len(clean), "median_abs_dev":
                                       stats.true_median(dev) if dev else None, "max_abs_dev": max(dev) if dev else None}
        self.save()

    def run_lb_side(self, pairdir):
        """Launch llamacpp.sh (arm B), wait for /health, run the cold-prefill client, stop the server.
        -> dict {records, reason, op_offload, memfree_start_mib, memavailable_start_mib, cache_mib, cache_table_mib,
        cache_full, memfree_min_mib, memfree_stop_mib}."""
        pairdir = Path(pairdir)
        pairdir.mkdir(parents=True, exist_ok=True)
        mf = modes.meminfo_mib("MemFree", self.meminfo)
        ma = modes.meminfo_mib("MemAvailable", self.meminfo)
        hp = subprocess.run(["bash", "deploy/lb/llamacpp.sh", "--print-footprint", *self.lb_flags, "--memfree-mib", str(mf)],
                            capture_output=True, text=True, cwd=str(ROOT)).stdout
        kv = dict(t.split("=", 1) for t in hp.split() if "=" in t)
        flags = subprocess.run(["bash", "deploy/lb/llamacpp.sh", "--print-flags", *self.lb_flags], capture_output=True,
                               text=True, cwd=str(ROOT)).stdout
        op_offload = "OP_OFFLOAD=true" in flags
        memfree_log = pairdir / "memfree.csv"
        port = str(self.yard.get("port", 8091))
        with open(pairdir / "server.log", "w") as log:
            srv = subprocess.Popen(["bash", "deploy/lb/llamacpp.sh", *self.lb_flags, "--memfree-mib", str(mf),
                                    "--memfree-stop", "14336", "--memfree-log", str(memfree_log), "--port", port],
                                   stdout=log, stderr=subprocess.STDOUT, cwd=str(ROOT), start_new_session=True)
        reason, records = None, None
        try:
            up = False
            for _ in range(900):
                if srv.poll() is not None:
                    break
                c = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "3",
                                    f"http://127.0.0.1:{port}/health"], capture_output=True, text=True).stdout
                if c == "200":
                    up = True
                    break
                time.sleep(1)
            if not up:
                reason = "memfree_stop" if Path(str(memfree_log) + ".stop").exists() else "server_failed"
            else:
                time.sleep(2)
                cp = subprocess.run([self.python, "bench/lb/client.py", "--port", port, *LB_CLIENT_ARGS, "--out",
                                     str(pairdir / "client.jsonl")], cwd=str(ROOT))
                if Path(str(memfree_log) + ".stop").exists():
                    reason = "memfree_stop"
                elif cp.returncode != 0:
                    reason = f"client_rc={cp.returncode}"
                else:
                    records = [json.loads(l) for l in open(pairdir / "client.jsonl") if l.strip()]
        finally:
            try:
                os.killpg(srv.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                srv.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(srv.pid, signal.SIGKILL)
        if reason is None and srv.returncode == 5:
            reason = "memfree_stop"
        mins = []
        try:
            for l in open(memfree_log).read().splitlines()[1:]:
                mins.append(int(l.split(",")[1]))
        except (OSError, ValueError, IndexError):
            pass
        stop = None
        try:
            stop = int(Path(str(memfree_log) + ".stop").read_text().split()[2])
        except (OSError, ValueError, IndexError):
            pass
        return {"records": records, "reason": reason, "op_offload": op_offload, "memfree_start_mib": mf,
                "memavailable_start_mib": ma, "cache_mib": int(kv["CACHE_MIB"]) if "CACHE_MIB" in kv else None,
                "cache_table_mib": int(kv["CACHE_TABLE_MIB"]) if "CACHE_TABLE_MIB" in kv else None,
                "cache_full": kv.get("CACHE_FULL") == "true" if "CACHE_FULL" in kv else None,
                "memfree_min_mib": min(mins) if mins else None, "memfree_stop_mib": stop}

    def phase_lb(self, clean_goal=LB_CLEAN, attempts=LB_ATTEMPTS):
        rows, op_offload = [], None
        for i, order in enumerate(stats.balanced_order(attempts)):
            if sum(r["clean"] for r in rows) >= clean_goal:
                break
            if not self.enough(EST_ENGINE_S + EST_LB_PAIR_S):
                self.result["phases"]["lb_stopped"] = "deadline"
                break
            eng, lbr, errs, pairdir = None, None, [], self.cap / "lb" / f"pair-{i:02d}"
            mf0, ma0 = modes.meminfo_mib("MemFree", self.meminfo), modes.meminfo_mib("MemAvailable", self.meminfo)
            if mf0 - CACHE_OFFSET_MIB < CACHE_MIN_MIB:
                snap = self.snapshot(EST_ENGINE_S + EST_LB_PAIR_S)
                meta = {"memfree_start_mib": mf0, "memavailable_start_mib": ma0, "cache_mib": None}
                row = lb_pair_row(i, {}, {}, {}, 0, True, ["memfree_start_low"], meta, snap)
                row["clean"], row["reason"] = False, "memfree_start_low"
                rows.append(row)
                self.result["lb"] = {"arm": self.yard["arm"], "op_offload": op_offload, "prefill_device": "cpu",
                                     "cache_rule": CACHE_RULE, "pairs": rows}
                self.save()
                self.sleep(120)
                continue
            for side in order:                                  # "A" = engine first, "B" = LB first (balanced)
                if side == "A":
                    r, e = self.run_engine(pairdir / "engine.json")
                    eng = engine_tps(r) if r else None
                    if e:
                        errs.append(f"engine:{e}")
                else:
                    lbr = self.lb_runner(pairdir)
            records, lb_reason, op_offload = lbr["records"], lbr["reason"], lbr["op_offload"]
            snap = self.snapshot(EST_ENGINE_S + EST_LB_PAIR_S)
            ok, why = eligibility.evaluate(snap)
            reasons = list(why) + errs + ([lb_reason] if lb_reason else [])
            if records:
                lb, pf, cache_n, cold = lb_tps(records)
            else:
                lb, pf, cache_n, cold = {}, {}, 0, True
            meta = {k: lbr.get(k) for k in ("memfree_start_mib", "memavailable_start_mib", "cache_mib", "cache_table_mib",
                                            "cache_full", "memfree_min_mib", "memfree_stop_mib")}
            row = lb_pair_row(i, eng or {}, lb, pf, cache_n, cold, reasons, meta, snap)
            rows.append(row)
            self.result["lb"] = {"arm": self.yard["arm"], "op_offload": op_offload, "prefill_device": "cpu",
                                 "cache_rule": CACHE_RULE, "pairs": rows}
            self.save()
        self.result.setdefault("lb", {"arm": self.yard["arm"], "op_offload": op_offload, "prefill_device": "cpu",
                                      "cache_rule": CACHE_RULE, "pairs": rows})
        self.result["lb"]["clean_pairs"] = sum(r["clean"] for r in rows)
        self.result["lb"]["cache_full_pairs"] = sum(1 for r in rows if r["clean"] and r.get("cache_mib") == CACHE_FULL_MIB)
        self.save()

    def phase_d1(self, runs=D1_RUNS):
        rows = []
        for i in range(runs):
            if not self.enough(EST_ENGINE_S):
                break
            r, e = self.run_engine(self.cap / "d1" / f"run{i}.json", mode="D1", slots=D1_SLOTS)
            rows.append({"run": i, "tps": engine_tps(r) if r else None, "error": e})
        self.result["phases"]["d1"] = {"informative": True, "runs": rows}
        self.save()

    def run(self):
        self.preflight()
        (self.cap / "started.json").write_text(json.dumps(
            {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "deadline": self.deadline, "yardstick": self.yard,
             "lb_flags": self.lb_flags,
             "nvme_temp_limit_mc": self.nvme_limit[0], "nvme_temp_limit_src": self.nvme_limit[1],
             "cache_rule": CACHE_RULE}, indent=1) + "\n")
        self.phase_traces()
        self.phase_aa()
        self.phase_lb()
        self.phase_d1()
        self.result["complete"] = True
        self.save()
        clean = self.result["lb"]["clean_pairs"]
        return 0 if clean >= LB_CLEAN else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cap", default=os.environ.get("CAP"))
    a = ap.parse_args(argv)
    if not a.cap:
        raise SystemExit("session.py needs CAP (run it under the guard)")
    dl = os.environ.get("SLOTSTREAM_DEADLINE")
    return Session(a.cap, deadline=float(dl) if dl else None).run()


if __name__ == "__main__":
    sys.exit(main())
