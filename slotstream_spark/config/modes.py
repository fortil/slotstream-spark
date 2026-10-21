"""Memory modes D1/D2/D5: ceilings and needs come from slotstream_spark.config.MODES (never redefined here);
this module adds the fixed bytes of the ledger and the pool-size arithmetic.

  python -m slotstream_spark.config.modes footprint|cap|chunks MODE [AVAIL_MIB]
"""
import sys
from dataclasses import dataclass

from . import MODES

CHUNK_BYTES = 1 << 30
SLOTS_PER_CHUNK = 388
FLOOR_CHUNKS = 2

# plan-implementacion.md section 3.8 (ledger at the ceiling of each mode) and the chunk ceilings
_FIXED_BYTES = {"D1": 8_821_658_084, "D2": 12_123_804_900, "D5": 14_786_204_900}
_CEILING_CHUNKS = {"D1": 11, "D2": 14, "D5": 49}


@dataclass(frozen=True)
class Mode:
    name: str
    ceiling_mib: int
    need_mib: int
    fixed_bytes: int
    ceiling_chunks: int


def get(name):
    if name not in MODES:
        raise ValueError(f"unknown mode {name!r}; expected one of {sorted(MODES)}")
    m = MODES[name]
    return Mode(name, m["ceiling_mib"], m["need_mib"], _FIXED_BYTES[name], _CEILING_CHUNKS[name])


def dynamic_cap_mib(m, avail, own=0):
    """Cap in MiB for the engine's footprint: min(ceiling, MemAvailable + what we already hold - need)."""
    return min(m.ceiling_mib, avail + own - m.need_mib)


def chunks_for(m, cap_mib):
    """Whole 1 GiB pool chunks that fit under `cap_mib` after the fixed bytes, at most the mode's ceiling."""
    return min(m.ceiling_chunks, (cap_mib * (1 << 20) - m.fixed_bytes) // CHUNK_BYTES)


def meminfo_mib(key, path="/proc/meminfo"):
    with open(path) as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1]) // 1024
    raise KeyError(key)


def footprint_mib(m, meminfo="/proc/meminfo"):
    """What the guard resolves for FOOTPRINT `auto`: min(ceiling, MemAvailable - need) - 256 MiB."""
    return min(m.ceiling_mib, meminfo_mib("MemAvailable", meminfo) - m.need_mib) - 256


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 2 or argv[0] not in ("footprint", "cap", "chunks"):
        print(__doc__)
        return 2
    cmd, m = argv[0], get(argv[1])
    if cmd == "footprint":
        print(footprint_mib(m))
        return 0
    avail = int(argv[2]) if len(argv) > 2 else meminfo_mib("MemAvailable")
    cap = dynamic_cap_mib(m, avail)
    print(cap if cmd == "cap" else max(chunks_for(m, cap), 0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
