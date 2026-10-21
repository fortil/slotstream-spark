"""Eligibility of a measurement pair (plan section 6): a pair that fails any rule is kept in the output with its reason.

`evaluate(snap, ...)` is pure over a snapshot dict; `collect()` builds the snapshot on the Spark.
Snapshot keys: running (list), scopes ([{name, mem_pressure_avg10, io_pressure_pct}]), cap_mib, cell_cap_mib,
finish_hhmm_estimate (minutes since 00:00 at which the cell will finish), cut (bool), gpu_clock_nominal (bool),
sw_power_cap_delta (number), nvme_temp_mc, nvme_temp_limit_mc, nvme_temp_limit_src, memfree_mib, memavailable_mib,
taken_at, hour (0-23 now), foreign_read_mbps, no_windows (pause mode: the quiet window and the 23:45 deadline do not apply).
The NVMe limit is the device's own temp1_max minus NVME_LIMIT_MARGIN_MC, read at run time (`nvme_limit_mc`).
"""
import glob
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from slotstream_spark.guard import paused

MEM_PRESSURE_MAX = 0.5          # "approximately 0": percent of time with some stall (avg10)
IO_PRESSURE_MAX = 10.0
FOREIGN_READ_MAX_MBPS = 200.0
DEADLINE_MIN = 23 * 60 + 45
HWMON_ROOT = "/sys/class/hwmon"
NVME_LIMIT_MARGIN_MC = 5000


def hwmon_input(root=None, env=None):
    """Path of the NVMe temperature input: `SLOTSTREAM_HWMON_INPUT` when set, else the first `<root>/*/name` that reads
    `nvme`, as `<dir>/temp1_input`. "" when none is found; callers treat that as unreadable (a hard error, never a guess)."""
    e = os.environ if env is None else env
    if e.get("SLOTSTREAM_HWMON_INPUT"):
        return e["SLOTSTREAM_HWMON_INPUT"]
    for d in sorted(glob.glob(f"{root or HWMON_ROOT}/hwmon*")):
        try:
            if (Path(d) / "name").read_text().strip() == "nvme":
                return str(Path(d) / "temp1_input")
        except OSError:
            continue
    return ""


def nvme_limit_mc(hwmon_path=None):
    """-> (limit_mc, source). Limit = the device's temp1_max (sibling of temp1_input) minus the margin; (None, "unreadable") otherwise."""
    if hwmon_path is None:
        hwmon_path = hwmon_input()
    if not hwmon_path:
        return None, "unreadable"
    try:
        mx = int((Path(hwmon_path).parent / "temp1_max").read_text())
    except (OSError, ValueError):
        return None, "unreadable"
    return mx - NVME_LIMIT_MARGIN_MC, f"temp1_max-{NVME_LIMIT_MARGIN_MC}"


def evaluate(s):
    """-> (eligible, [reasons])."""
    why = []
    if s.get("running"):
        why.append("running_not_empty")
    for sc in s.get("scopes", []):
        if sc.get("mem_pressure_avg10", 0) > MEM_PRESSURE_MAX:
            why.append(f"memory_pressure:{sc.get('name')}")
        if sc.get("io_pressure_pct", 0) >= IO_PRESSURE_MAX:
            why.append(f"io_pressure:{sc.get('name')}")
    if s.get("cap_mib", 0) < s.get("cell_cap_mib", 0):
        why.append("cap_below_cell")
    if (s.get("finish_minute", 0) >= DEADLINE_MIN and not s.get("no_windows")) or s.get("cut"):
        why.append("past_deadline_or_cut")
    if not s.get("gpu_clock_nominal", True):
        why.append("gpu_clock_not_nominal")
    if s.get("sw_power_cap_delta", 0) != 0:
        why.append("sw_power_cap")
    t, lim = s.get("nvme_temp_mc"), s.get("nvme_temp_limit_mc")
    if t is None or lim is None:
        why.append("nvme_temp_unknown")
    elif t >= lim:
        why.append("nvme_hot")
    if 0 <= s.get("hour", 12) < 4 and not s.get("no_windows"):
        why.append("quiet_window")
    if s.get("foreign_read_mbps", 0) > FOREIGN_READ_MAX_MBPS:
        why.append("foreign_reads")
    return (not why), why


def _pressure(path):
    try:
        for line in Path(path).read_text().splitlines():
            if line.startswith("some"):
                return float(dict(kv.split("=") for kv in line.split()[1:])["avg10"])
    except (OSError, KeyError, ValueError):
        pass
    return 0.0


def _diskstats_sectors():
    out = {}
    for line in Path("/proc/diskstats").read_text().splitlines():
        f = line.split()
        if len(f) > 5 and f[2].startswith("nvme") and "p" not in f[2][4:]:
            out[f[2]] = int(f[5])
    return out


def _cap_mib(mode, meminfo="/proc/meminfo"):
    """The cap an engine of `mode` would get now (numeric): min(ceiling, MemAvailable - need). Mode D1/D2/D5."""
    from slotstream_spark.config import modes
    return int(modes.dynamic_cap_mib(modes.get(mode), modes.meminfo_mib("MemAvailable", meminfo)))


def collect(cell_cap_mib, est_seconds, running_url=None, hwmon=None,
            own_read_mbps=0.0, window_s=2.0, mode="D5", meminfo="/proc/meminfo"):
    """Snapshot on the Spark (read only). `own_read_mbps` is what the caller's own run is reading right now.
    `cap_mib` is the numeric cap of `mode` right now (never None); `cell_cap_mib` is what the cell needs."""
    if running_url is None:
        running_url = os.environ.get("SLOTSTREAM_RUNNING_URL", "")
    if not running_url:                                         # no model router to watch
        running = []
    else:
        try:
            running = json.load(urllib.request.urlopen(running_url, timeout=5)).get("running", [])
        except Exception:                                       # noqa: BLE001
            running = ["unreachable"]
    scopes = []
    for d in glob.glob("/sys/fs/cgroup/**/docker-*.scope", recursive=True):
        scopes.append({"name": Path(d).name, "mem_pressure_avg10": _pressure(f"{d}/memory.pressure"),
                       "io_pressure_pct": _pressure(f"{d}/io.pressure")})
    a = _diskstats_sectors()
    time.sleep(window_s)
    b = _diskstats_sectors()
    mbps = sum(b[k] - a.get(k, 0) for k in b) * 512 / 1e6 / window_s - own_read_mbps
    q = subprocess.run(["nvidia-smi", "--query-gpu=clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.active",
                        "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    sw = 1 if q.split(",")[0].strip().lower() == "active" else 0
    if hwmon is None:
        hwmon = hwmon_input()
    try:
        temp = int(Path(hwmon).read_text()) if hwmon else None
    except (OSError, ValueError):
        temp = None
    from slotstream_spark.config import modes
    limit, limit_src = nvme_limit_mc(hwmon)
    lt = time.localtime()
    return {"running": running, "scopes": scopes, "cap_mib": _cap_mib(mode), "cell_cap_mib": cell_cap_mib,
            "finish_minute": lt.tm_hour * 60 + lt.tm_min + est_seconds / 60, "cut": False, "gpu_clock_nominal": sw == 0,
            "sw_power_cap_delta": sw, "nvme_temp_mc": temp, "nvme_temp_limit_mc": limit,
            "nvme_temp_limit_src": limit_src, "memfree_mib": modes.meminfo_mib("MemFree", meminfo),
            "memavailable_mib": modes.meminfo_mib("MemAvailable", meminfo), "taken_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "hour": lt.tm_hour, "no_windows": paused.no_windows(),
            "foreign_read_mbps": max(0.0, mbps)}
