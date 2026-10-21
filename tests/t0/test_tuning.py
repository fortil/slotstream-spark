import json

from slotstream_spark.kernels import tuning

STEP48 = {"block_n": 32, "block_k": 256, "split_k": 1, "num_warps": 8, "num_stages": 3}
TABLE = json.loads(tuning.TABLE_PATH.read_text())
GATE_KEY = tuning.shape_key(2560, 2560, 4, 64)


def _in_space(c, head=False):
    assert c["block_n"] in (4, 8, 16, 32, 64, 128), c
    assert c["block_k"] in (64, 128, 256, 512), c
    assert c["block_k"] % 64 == 0, c
    assert c["split_k"] in ((1, 2, 4) if head else (1, 2, 4, 8)), c
    assert c["num_warps"] in (2, 4, 8), c
    assert c["num_stages"] in (1, 2, 3), c


def test_step48_override_is_kept_as_fallback():
    assert tuning.OVERRIDES[(2560, 2560, 4, 64)] == STEP48


def test_table_supersedes_override_for_gate_shape():
    assert tuning.config_for(2560, 2560, 4, 64) == TABLE["shapes"][GATE_KEY]["config"]


def test_returned_dict_is_a_copy():
    c = tuning.config_for(2560, 2560, 4, 64)
    ref = dict(c)
    c["block_n"] = 999
    assert tuning.config_for(2560, 2560, 4, 64) == ref


def test_untabled_shapes_keep_the_formula():
    assert tuning.config_for(2560, 2560, 8, 64)["split_k"] == 2
    assert tuning.config_for(1234, 2560, 4, 64) == tuning.formula_config(1234, 2560)


def test_table_is_inside_the_search_space_and_the_smem_limit():
    assert len(TABLE["shapes"]) >= 11
    for key, r in TABLE["shapes"].items():
        n = int(key.split("x")[0])
        _in_space(r["config"], head=n > 100000)
        assert r["shared_bytes"] <= tuning.SMEM_LIMIT, key
        assert r["rows3_bitwise"] is True, key
        assert r["rel_err"] <= 1e-4, key
        assert r["candidates"] <= 256 + 2, key
    for r in TABLE["shapes"].values():
        assert set(r["config"]) == {"block_n", "block_k", "split_k", "num_warps", "num_stages"}


def test_config_for_never_takes_the_row_count():
    import inspect
    assert list(inspect.signature(tuning.config_for).parameters) == ["n", "k", "bits", "gs"]
    assert list(inspect.signature(tuning.moe_config).parameters) == []


def test_gate_passed_and_platform_carries_the_engine_figure():
    gate = TABLE["gate"]
    assert gate["pass"] is True and gate["gbs_cold_graph"] >= 121.5
    assert gate["gbs_cold_graph"] == TABLE["shapes"][GATE_KEY]["gbs_cold_graph"]
    toml = (tuning.TABLE_PATH.parents[1] / "config" / "platform.toml").read_text()
    line = next(x for x in toml.splitlines() if x.startswith("gemv_2560_gbs"))
    assert float(line.split("=")[1].split("#")[0]) == gate["gbs_cold_graph"]
    assert 193.75 != gate["gbs_cold_graph"]


def test_moe_config_is_the_table_entry_and_valid():
    c = tuning.moe_config()
    entry = dict(TABLE["moe"]["config"])
    entry.setdefault("bkd", 128)   # the step-55 table predates this key; 77c rewrites it
    assert "bkd" in TABLE["moe"]["config"]   # 77c: the regenerated entry carries bkd
    assert c == entry
    assert set(c) == {"bn", "br", "bk", "bkd", "num_warps"}
    assert c["bn"] in (4, 8, 16) and c["br"] in (8, 16, 32)
    assert 640 % c["bn"] == 0 and 2560 % c["br"] == 0
    assert c["bk"] in (128, 256, 512) and c["bkd"] in (128, 256) and c["num_warps"] in (2, 4, 8)
    assert TABLE["moe"]["rows3_bitwise"] is True and TABLE["moe"]["shared_bytes"] <= tuning.SMEM_LIMIT
    assert tuning.moe_config() is not tuning.moe_config()


def test_gemm_config_never_takes_the_row_count():
    import inspect
    assert list(inspect.signature(tuning.gemm_config_for).parameters) == ["n", "k", "bits"]


def test_gemm_overrides_are_inside_the_space():
    assert tuning.GEMM_OVERRIDES
    for (n, k, bits), c in tuning.GEMM_OVERRIDES.items():
        assert c["block_m"] in (64, 128) and c["block_n"] in (64, 128), c
        assert c["block_k"] in (64, 128), c
        assert c["num_warps"] in (4, 8), c
        assert c["num_stages"] in (2, 3), c
        assert k % c["block_k"] == 0, (n, k, bits, c)
        assert tuning.gemm_config_for(n, k, bits) == c


def test_gemm_untabled_and_dense_shapes_take_the_default():
    assert tuning.gemm_config_for(1234, 2560, 4) == tuning.GEMM_DEFAULT
    assert tuning.gemm_config_for(512, 2560, 16) == tuning.GEMM_DEFAULT


def test_gemm_config_is_a_copy():
    c = tuning.gemm_config_for(10240, 2560, 4)
    c["block_m"] = 1
    assert tuning.gemm_config_for(10240, 2560, 4)["block_m"] == 128
    tuning.gemm_config_for(1, 64, 4)["block_m"] = 1
    assert tuning.GEMM_DEFAULT["block_m"] == 64


def test_frozen_table_has_no_gemm_section():
    assert tuning._load_table()["gemm"] == {}
