"""Reader of the reversible pause state file, `$R/state/PAUSED` on the Spark.

The file exists only while the user has production paused. It holds key=value lines:
  NO_WINDOWS=1             the time windows (00:00-04:00, 23:45, 23:59, 13:15 start, ...) do not apply
  EXPECTED_CONTAINERS=<n>  the expected production container count (default: none, the count is not checked)
Absent file = every rule as before. A file that exists but is malformed (empty, a line without `=`, an unknown key, a
value that is not 0/1 or a non-negative integer) raises `PausedError`; it never falls back to the defaults silently.
Blank lines and `#` comments are allowed. The bash twin is `paused_load` in tools/spark/env.sh.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_EXPECTED_CONTAINERS = None
KEYS = ("NO_WINDOWS", "EXPECTED_CONTAINERS")


class PausedError(ValueError):
    """The state file exists but cannot be trusted."""


@dataclass(frozen=True)
class Paused:
    no_windows: bool = False
    expected_containers: Optional[int] = None


def path(root=None, env=None):
    env = os.environ if env is None else env
    if env.get("SLOTSTREAM_PAUSED_FILE"):
        return Path(env["SLOTSTREAM_PAUSED_FILE"])
    base = root if root is not None else (env.get("SLOTSTREAM_R") or Path.home() / "slotstream-spark")
    return Path(base) / "state" / "PAUSED"


def parse(text, where="PAUSED"):
    no_windows, expected, n = False, None, 0
    for i, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        k, sep, v = line.partition("=")
        if not sep or k not in KEYS:
            raise PausedError(f"{where}:{i}: unknown or malformed line {raw!r}")
        if k == "NO_WINDOWS":
            if v not in ("0", "1"):
                raise PausedError(f"{where}:{i}: NO_WINDOWS must be 0 or 1, got {v!r}")
            no_windows = v == "1"
        else:
            if not v.isascii() or not v.isdigit():
                raise PausedError(f"{where}:{i}: EXPECTED_CONTAINERS must be a non-negative integer, got {v!r}")
            expected = int(v)
        n += 1
    if n == 0:
        raise PausedError(f"{where}: exists but has no key=value line")
    return Paused(no_windows, expected)


def read(root=None, env=None):
    """-> Paused, or None when the state file does not exist. Raises PausedError when it is malformed."""
    p = path(root, env)
    if not p.exists():
        return None
    try:
        text = p.read_text()
    except OSError as e:
        raise PausedError(f"{p}: unreadable: {e}") from e
    return parse(text, str(p))


def no_windows(root=None, env=None):
    s = read(root, env)
    return bool(s and s.no_windows)


def expected_containers(default=DEFAULT_EXPECTED_CONTAINERS, root=None, env=None):
    """Explicit env EXPECTED_CONTAINERS wins over the state file, which wins over `default`."""
    e = os.environ if env is None else env
    if e.get("EXPECTED_CONTAINERS"):
        try:
            return int(e["EXPECTED_CONTAINERS"])
        except ValueError as ex:
            raise PausedError(f"EXPECTED_CONTAINERS in the environment is not an integer: {e['EXPECTED_CONTAINERS']!r}") from ex
    s = read(root, e)
    return s.expected_containers if s and s.expected_containers is not None else default


def paused_active(root=None, env=None):
    return read(root, env) is not None
