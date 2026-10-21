import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import make_goldens as mg  # noqa: E402
import ppl_ablate as pa  # noqa: E402


def make_dump(root, sub, T=3, fill=0.0, flip=None):
    d = Path(root) / sub
    d.mkdir(parents=True)
    (d / "tokens.json").write_text(json.dumps({"id": d.name, "set": d.parent.name, "ids": list(range(T))}))
    (d / "meta.json").write_text(json.dumps({"dtype": "bfloat16", "experts_mode": "correct", "gdn_impl": "torch",
                                             "transformers": "5.19.0", "torch": "2.13.0"}))
    np.full((T, mg.HIDDEN), fill, dtype=np.float32).tofile(d / "h_in.f32")
    np.arange(T * mg.TOPK, dtype=np.int32).tofile(d / "router_00_ids.i32")
    if flip:
        b = bytearray((d / flip).read_bytes())
        b[0] ^= 1
        (d / flip).write_bytes(bytes(b))
    return d


def test_compare_identical_runs_are_reproducible(tmp_path):
    for r in "AB":
        make_dump(tmp_path / r, "dev/dev0")
    m = mg.compare(tmp_path / "A", tmp_path / "B", None)
    assert m["reproducible"] is True
    f = m["files"]["dev/dev0/h_in.f32"]
    assert f["dtype"] == "float32" and f["shape"] == [3, mg.HIDDEN] and f["bytes"] == 3 * mg.HIDDEN * 4
    assert m["files"]["dev/dev0/router_00_ids.i32"]["shape"] == [3, mg.TOPK]
    assert m["oracle"] == {"transformers": "5.19.0", "torch": "2.13.0", "gdn_impl": "torch", "dtype": "bfloat16"}


def test_one_differing_file_sha_is_not_reproducible(tmp_path):
    make_dump(tmp_path / "A", "dev/dev0")
    make_dump(tmp_path / "B", "dev/dev0", flip="h_in.f32")
    assert mg.compare(tmp_path / "A", tmp_path / "B", None)["reproducible"] is False


def test_missing_file_is_not_reproducible(tmp_path):
    make_dump(tmp_path / "A", "dev/dev0")
    b = make_dump(tmp_path / "B", "dev/dev0")
    (b / "router_00_ids.i32").unlink()
    assert mg.compare(tmp_path / "A", tmp_path / "B", None)["reproducible"] is False


def test_cli_compare_exit_codes(tmp_path):
    make_dump(tmp_path / "A", "dev/dev0")
    make_dump(tmp_path / "B", "dev/dev0")
    assert mg.main(["--compare", str(tmp_path / "A"), str(tmp_path / "B"), "--manifest", str(tmp_path / "m.json")]) == 0
    assert json.loads((tmp_path / "m.json").read_text())["reproducible"] is True
    make_dump(tmp_path / "C", "dev/dev0", flip="h_in.f32")
    assert mg.main(["--compare", str(tmp_path / "A"), str(tmp_path / "C"), "--manifest", str(tmp_path / "m2.json")]) == 1


def test_load_golden_reads_the_layout(tmp_path):
    make_dump(tmp_path, "dev/dev0", T=3, fill=2.0)
    g = mg.load_golden("dev0", tmp_path)
    assert g["tokens"] == [0, 1, 2] and g["h_in"].shape == (3, mg.HIDDEN) and float(g["h_in"][0, 0]) == 2.0
    assert g["router_00_ids"].shape == (3, mg.TOPK) and g["meta"]["dtype"] == "bfloat16"


def test_expected_file_set():
    names = mg.expected_files()
    assert len(names) == 6 + 3 * 48 and "layer_47_out.f32" in names and "router_00_w.f32" in names


def test_convert_dump_layout(tmp_path):
    T, tmp, dst = 2, tmp_path / "npy", tmp_path / "raw"
    tmp.mkdir()
    dst.mkdir()
    np.save(tmp / "h_in.npy", np.ones((T, mg.HIDDEN), np.float32))
    for L in range(mg.LAYERS):
        np.save(tmp / f"layer_{L}.npy", np.full((T, mg.HIDDEN), L, np.float32))
        np.save(tmp / f"router_ids_{L}.npy", np.full((T, mg.TOPK), L, np.int64))
        np.save(tmp / f"router_weights_{L}.npy", np.full((T, mg.TOPK), 0.1, np.float32))
    np.save(tmp / "ngram_ids.npy", np.zeros((T, mg.NGRAM_HEADS), np.int64))
    np.save(tmp / "ngram_rows.npy", np.zeros((T, mg.NGRAM_HEADS, mg.NGRAM_DIM), np.float32))
    np.save(tmp / "logits.npy", np.zeros((T, mg.VOCAB), np.float32))
    mg.convert_dump(tmp, dst, T)
    have = {p.name for p in dst.iterdir()}
    assert have == set(mg.expected_files()) - {"tokens.json", "meta.json"}
    assert np.fromfile(dst / "layer_07_out.f32", np.float32)[0] == 7.0
    assert np.fromfile(dst / "router_05_ids.i32", np.int32).dtype == np.int32


def test_prompts_schema():
    P = mg.load_prompts(ROOT / "tests" / "goldens" / "prompts.json")
    assert len(P) == 16 and len({p["id"] for p in P}) == 16
    assert [p["id"] for p in P if p["set"] == "dev"] == [f"dev{i}" for i in range(8)]
    assert [p["id"] for p in P if p["set"] == "reserved"] == [f"res{i}" for i in range(8)]
    assert all(0 < len(p["text"]) <= 160 for p in P)  # <= 64 tokens is checked with the real tokenizer at make time
    T = json.load(open(ROOT / "tests" / "goldens" / "ppl_texts.json"))
    assert set(T) == {"en", "es"} and all(300 <= len(v) <= 700 for v in T.values())


def test_ablation_criterion():
    ab = {"ppl_correct_max": 20.0, "swap_min_ratio": 5.0, "noexp_min_ratio": 5.0}
    ok, _ = pa.evaluate({"correct": {"en": 4.0}, "swap": {"en": 30.0}, "noexp": {"en": 25.0}}, ab)
    assert ok
    ok, why = pa.evaluate({"correct": {"en": 4.0}, "swap": {"en": 19.0}, "noexp": {"en": 25.0}}, ab)
    assert not ok and "swap" in why[0]
    ok, _ = pa.evaluate({"correct": {"en": 25.0}, "swap": {"en": 900.0}, "noexp": {"en": 900.0}}, ab)
    assert not ok


MAN = ROOT / "tests" / "goldens" / "manifest.json"


@pytest.mark.skipif(not MAN.exists(), reason="tests/goldens/manifest.json not fetched yet")
def test_tracked_manifest_is_complete():
    m = json.loads(MAN.read_text())
    assert m["reproducible"] is True
    dirs = {}
    for rel in m["files"]:
        s, pid, name = rel.split("/", 2)
        dirs.setdefault(f"{s}/{pid}", set()).add(name)
    assert len(dirs) == 16
    for d, names in dirs.items():
        assert names == set(mg.expected_files()), d
