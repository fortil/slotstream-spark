import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import perf_model as pm  # noqa: E402

EXPECT = {
    "D1": ((9.2, 11.7), (11.3, 17.5)),
    "D2": ((10.4, 13.3), (12.6, 19.4)),
    "D5": ((18.4, 28.8), (19.8, 31.7)),
}


def test_plan_ranges_match_section_3_8():
    got = pm.plan_ranges()
    for mode, (nl, la) in EXPECT.items():
        for g, e in zip(got[mode][0] + got[mode][1], nl + la):
            assert abs(g - e) <= 0.1, (mode, g, e)


def test_t_comp_plan_bounds():
    assert abs(pm.t_comp_ms(0.5, 222.0) - 35.6) < 0.1
    assert abs(pm.t_comp_ms(0.7, 222.0) - 25.4) < 0.1


def test_measured_uses_platform_keys(tmp_path):
    p = tmp_path / "p.toml"
    p.write_text("[measured]\ngemv_2560_gbs = 121.5\ntau_us = 100\nt_memcpy_ms = 0.1\npool_gather_gbs = 222\n")
    eta, m, rows = pm.measured_table(p)
    assert abs(eta - 0.5) < 1e-9
    assert abs(rows["D1"]["no_lookahead"][0] - 9.2) < 0.3
