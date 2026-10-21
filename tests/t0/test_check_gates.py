import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench/m0"))
import check_gates  # noqa: E402


def _run(tmp_path, gate, obj):
    p = tmp_path / "r.json"
    p.write_text(json.dumps(obj))
    return check_gates.main([gate, str(p)])


def _gemv(gbs):
    # "gbs" (do_bench) is informational; the gate reads "gbs_cold_graph"
    return {"shapes": {"2560x2560": {"gbs": 999.0, "gbs_cold_graph": gbs, "config": {}}}}


def test_gemv_pass_and_fail(tmp_path, capsys):
    assert _run(tmp_path, "gemv", _gemv(130.0)) == 0
    assert _run(tmp_path, "gemv", _gemv(121.5)) == 0
    assert _run(tmp_path, "gemv", _gemv(120.0)) == 2
    assert "DESIGN GATE FAILED (Decision §7 row 1)" in capsys.readouterr().out


def test_gemv_gates_on_cold_graph_not_do_bench(tmp_path):
    obj = {"shapes": {"2560x2560": {"gbs": 200.0, "gbs_cold_graph": 100.0}}}
    assert _run(tmp_path, "gemv", obj) == 2
    obj = {"shapes": {"2560x2560": {"gbs": 90.0, "gbs_cold_graph": 130.0}}}
    assert _run(tmp_path, "gemv", obj) == 0
    assert _run(tmp_path, "gemv", {"shapes": {"2560x2560": {"gbs": 200.0}}}) == 3


def test_host_pass_and_fail(tmp_path):
    assert _run(tmp_path, "host", {"tau_us": 120, "footprint_bytes": 2.0e9}) == 0
    assert _run(tmp_path, "host", {"tau_us": 151, "footprint_bytes": 2.0e9}) == 2
    assert _run(tmp_path, "host", {"tau_us": 100, "footprint_bytes": 2.6e9}) == 2


def _yardstick(tps, ttft8, exceeded=False, op_offload=None, memfree_stop=False):
    pr = {pid: {"decode_tps": {"median": tps}, "ttft_s": {"median": 1.0}} for pid in ("short", "2k")}
    pr["8k"] = {"decode_tps": {"median": tps}, "ttft_s": {"median": ttft8}}
    cfg = {"prompts": pr, "envelope_exceeded": exceeded, "memfree_stop": memfree_stop}
    if op_offload is not None:
        cfg["server"] = {"op_offload": op_offload}
    return {"q41-a-d5": cfg}


def test_yardstick(tmp_path):
    assert _run(tmp_path, "yardstick", _yardstick(10.0, 60.0)) == 0
    assert _run(tmp_path, "yardstick", _yardstick(25.0, 60.0)) == 0      # fast decode but slow TTFT
    assert _run(tmp_path, "yardstick", _yardstick(25.0, 20.0)) == 2      # meets the D5 objective
    assert _run(tmp_path, "yardstick", _yardstick(10.0, 60.0, exceeded=True)) == 2
    assert _run(tmp_path, "yardstick", {"iq3-r-d5": {}}) == 2       # q41-a-d5 missing
    assert _run(tmp_path, "yardstick", _yardstick(25.0, 60.0, op_offload=False)) == 2   # CPU prefill: decode only
    assert _run(tmp_path, "yardstick", _yardstick(22.9, 1.0, op_offload=False)) == 0
    assert _run(tmp_path, "yardstick", _yardstick(25.0, 60.0, op_offload=True)) == 0
    assert _run(tmp_path, "yardstick", _yardstick(10.0, 60.0, memfree_stop=True)) == 2


def test_bad_input(tmp_path):
    assert check_gates.main(["gemv", str(tmp_path / "missing.json")]) == 3
    assert check_gates.main(["nope", "x"]) == 3
