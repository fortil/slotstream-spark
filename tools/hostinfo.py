#!/usr/bin/env python3
"""Print a JSON snapshot of the Spark host state (stdlib only)."""
import glob
import json
import os
import subprocess
import time
import sys
import urllib.request

MEM_KEYS = ("MemTotal", "MemAvailable", "MemFree", "Cached", "AnonPages",
            "AnonHugePages", "Mlocked", "Shmem")


def meminfo():
    out = {}
    try:
        for line in open("/proc/meminfo"):
            k, _, v = line.partition(":")
            if k in MEM_KEYS:
                out[k + "_kB"] = int(v.split()[0])
    except OSError:
        pass
    return out


def pressure(name):
    try:
        return open(f"/proc/pressure/{name}").read().strip().splitlines()
    except OSError:
        return None


def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return r.stdout.strip()
    except Exception:
        return ""


def container_family(name):
    """Family label of a container name: asr, embeddings, qdrant, postgres or other (tools/impact_report.py matches these)."""
    low = name.lower()
    for fam in ("asr", "embedding", "qdrant", "postgres"):
        if fam in low:
            return fam + ("s" if fam == "embedding" else "")
    return "other"


def docker_ps():
    """Running containers as {family-N: status}; names are not recorded."""
    out = {}
    seen = {}
    for line in run(["docker", "ps", "--format", "{{.Names}}\t{{.Status}}"]).splitlines():
        name, _, status = line.partition("\t")
        fam = container_family(name)
        seen[fam] = seen.get(fam, 0) + 1
        out[f"{fam}-{seen[fam]}"] = status
    return out


def pause_state():
    """Pause mode ($R/state/PAUSED) and the container count that applies: {active, no_windows, expected_containers}."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    try:
        from slotstream_spark.guard import paused
        s = paused.read()
        exp = paused.expected_containers()
    except Exception as e:      # PausedError (malformed file) or an import problem: report it, never default silently
        return {"active": None, "error": str(e), "expected_containers": None}
    return {"active": s is not None, "no_windows": bool(s and s.no_windows), "expected_containers": exp}


def running_raw():
    url = os.environ.get("SLOTSTREAM_RUNNING_URL", "")
    if not url:
        return "off"
    try:
        return urllib.request.urlopen(url, timeout=5).read().decode()
    except Exception as e:
        return f"error: {e}"


def hwmon():
    out = {}
    for n in (1, 2, 3):
        for p in glob.glob(f"/sys/class/hwmon/hwmon1/temp{n}_input"):
            try:
                out[f"temp{n}"] = int(open(p).read()) / 1000.0
            except (OSError, ValueError):
                pass
    return out


def main():
    info = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "epoch": int(time.time()),
        "uname": " ".join([os.uname()[0], "host"] + list(os.uname()[2:])),   # host name left out
        "driver": run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "meminfo": meminfo(),
        "pressure": {n: pressure(n) for n in ("memory", "io", "cpu")},
        "docker": docker_ps(),
        "paused": pause_state(),
        "running": running_raw(),
        "hwmon1": hwmon(),
        "loadavg": open("/proc/loadavg").read().strip() if os.path.exists("/proc/loadavg") else "",
    }
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
