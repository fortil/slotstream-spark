import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _lock():
    return json.loads((ROOT / "model.lock.json").read_text())


def test_counts_and_totals():
    d = _lock()
    assert len(d["files"]) == 24
    assert d["total_bytes"] == 103793508077
    assert d["shards_bytes"] == 103770199081
    assert d["total_bytes"] == sum(f["size"] for f in d["files"])


def test_mtp_separate():
    d = _lock()
    assert all("mtp" not in f["path"] for f in d["files"])
    assert d["mtp"]["path"] == "mtp.safetensors"
    assert d["mtp"]["size"] == 1470955171


def test_sums_match_json():
    d = _lock()
    lines = (ROOT / "SHA256SUMS.model").read_text().splitlines()
    assert lines == [f"{f['sha256']}  {f['path']}" for f in d["files"]]
    assert "mtp" not in "\n".join(lines)
