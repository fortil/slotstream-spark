import hashlib
from pathlib import Path

PROBES = Path(__file__).resolve().parents[2] / "tools/probes"


def _sums():
    out = {}
    for line in (PROBES / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            sha, name = line.split(None, 1)
            out[name.strip().lstrip("*")] = sha
    return out


def test_five_probes_listed():
    assert sorted(_sums()) == ["cuda_attrs.py", "iobench.py", "mlx_probe.py", "torch_bw.py", "uma_probe.cu"]


def test_probe_hashes():
    for name, sha in _sums().items():
        assert hashlib.sha256((PROBES / name).read_bytes()).hexdigest() == sha, name


def test_known_prefixes():
    s = _sums()
    assert s["iobench.py"].startswith("d7ca26fd")
    assert s["uma_probe.cu"].startswith("248cffd9")
    assert s["mlx_probe.py"].startswith("7c6c093f")
