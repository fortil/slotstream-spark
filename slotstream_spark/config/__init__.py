"""Platform configuration: reader and validator for platform.toml."""

import tomllib
from pathlib import Path

PLATFORM_PATH = Path(__file__).with_name("platform.toml")

MODES = {
    "D1": {"ceiling_mib": 19866, "need_mib": 62644},
    "D2": {"ceiling_mib": 26112, "need_mib": 56413},
    "D5": {"ceiling_mib": 65126, "need_mib": 16896},
}

UNMEASURED = "unmeasured"

# key -> allowed values (a set), or a type for free-form values
ALLOWED = {
    "spark.memcg_delegated": {UNMEASURED, "yes", "no"},
    "spark.io_delegated": {UNMEASURED, "yes", "no"},
    "env.kind": {UNMEASURED, "venv", "standalone", "container"},
    "env.python": str,
    "env.torch": str,
    "env.triton": str,
    "env.python_h": {UNMEASURED, "yes", "no"},
    "memory.mode_default": {"D1", "D2", "D5"},
    "memory.d1_variant": {UNMEASURED, "D1", "D1prime"},
    "memory.write_path": {"safe"},
    "memory.pool_memory": {UNMEASURED, "host_alloc", "vmm_host_numa", "device_alloc"},
    "memory.a_transient_mib": int,
    "ngram.row_bytes": {-1, 100, 128},
    "ngram.read": {"odirect"},
    "kernels.plan": {UNMEASURED, "A", "B"},
    "runtime.wait_mechanism": {UNMEASURED, "spin_flag", "event_sync"},
    "runtime.memop_in_graph": {UNMEASURED, "supported", "unsupported"},
    "measured.gemv_2560_gbs": float,
    "measured.tau_us": float,
    "measured.runtime_footprint_mib": int,
    "measured.t_memcpy_ms": float,
    "measured.pool_gather_gbs": float,
    "measured.nvme_temp_no_throttle_mc": int,
    "measured.nvme_temp_no_effect_mc": int,
    "lookahead.mode": {"off", "attention", "attention-corrected", "boundary2"},
    "lookahead.threshold": float,
    "lookahead.top": int,
    "lookahead.issue_cap": int,
    "lookahead.slot_cap": int,
    "lookahead.min_evictable": int,
    "pool.prefill": {"off", "table"},
    "server.queue_max": int,
    "prefix.dir": str,
    "prefix.quota_bytes": int,
    "f8.token_graph": {True, False},
}

_UNDECIDED = {UNMEASURED, -1, 0, 0.0}


def load(path=None):
    with open(path or PLATFORM_PATH, "rb") as f:
        return tomllib.load(f)


def _flat(cfg):
    out = {}
    for section, body in cfg.items():
        if isinstance(body, dict):
            for k, v in body.items():
                out[f"{section}.{k}"] = v
        else:
            out[section] = body
    return out


def get(key, cfg=None):
    flat = _flat(cfg if cfg is not None else load())
    return flat[key]


def decided(key, cfg=None):
    v = get(key, cfg)
    if isinstance(v, bool):
        return True
    return v not in _UNDECIDED


def _type_ok(spec, v):
    if spec is float:
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if spec is int:
        return isinstance(v, int) and not isinstance(v, bool)
    return isinstance(v, spec)


def validate(cfg):
    errors = []
    flat = _flat(cfg)
    for key, v in flat.items():
        if key not in ALLOWED:
            errors.append(f"invalid: {key}={v!r} (unknown key)")
            continue
        spec = ALLOWED[key]
        ok = (v in spec) if isinstance(spec, set) else _type_ok(spec, v)
        if not ok:
            errors.append(f"invalid: {key}={v!r}")
    for key in ALLOWED:
        if key not in flat:
            errors.append(f"invalid: {key}=<missing>")
    return errors
