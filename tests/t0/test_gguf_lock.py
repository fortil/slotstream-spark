import json
from pathlib import Path

LOCK = json.loads((Path(__file__).resolve().parents[2] / "deploy/lb/gguf.lock.json").read_text())


def test_q41_pinned():
    q = LOCK["q41"]
    assert q["repo"] == "bartowski/Qwen3.8-Flash-Next-GGUF"
    assert q["revision"] == "928589fdb66c6ff07f22ac561e3fbce76553548f"
    files = q["files"]
    assert [f["size"] for f in files] == [39530968768, 39641550400, 32086056352]
    assert q["total_bytes"] == 111258575520 == sum(f["size"] for f in files)
    assert [f["sha256"] for f in files] == [
        "53b2e4e5c63adddb5107304c8a266df491c1d11a525613dbf6303f0001a39212",
        "cf2cb13fb3c809e355e2b63746a6a07e3434c1868c26c6a20f6aebf118b8fae7",
        "4faa132fd26392605642f9d16538318583cbad2926d5a81906296e9769cb1d13",
    ]


def test_iq3xxs_pinned():
    q = LOCK["iq3xxs"]
    assert q["revision"] == "ed59f92082b1e93c0e96d60a8b11aab089b52f09"
    sizes = sorted(f["size"] for f in q["files"])
    assert sizes == [28800138432, 47039860096]
    assert sum(sizes) == 75839998528 == q["total_bytes"]
    assert all(len(f["sha256"]) == 64 for f in q["files"])
    assert all("IQ3_XXS" in f["path"] for f in q["files"])
