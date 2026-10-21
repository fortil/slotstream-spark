"""NVMe hwmon discovery against a fake sysfs tree: no real /sys needed."""
from slotstream_spark.host import eligibility as E


def fake(tmp_path, names):
    for i, n in enumerate(names):
        d = tmp_path / f"hwmon{i}"
        d.mkdir()
        (d / "name").write_text(n + "\n")
        (d / "temp1_input").write_text("41000\n")
        (d / "temp1_max").write_text("85000\n")
    return str(tmp_path)


def test_discovers_the_nvme_entry_whatever_its_index(tmp_path):
    root = fake(tmp_path, ["acpitz", "coretemp", "nvme"])
    assert E.hwmon_input(root, {}) == str(tmp_path / "hwmon2" / "temp1_input")


def test_env_wins_over_discovery(tmp_path):
    root = fake(tmp_path, ["nvme"])
    assert E.hwmon_input(root, {"SLOTSTREAM_HWMON_INPUT": "/x/temp1_input"}) == "/x/temp1_input"


def test_nothing_found_is_empty_and_the_limit_is_unreadable(tmp_path):
    root = fake(tmp_path, ["acpitz"])
    assert E.hwmon_input(root, {}) == ""
    assert E.nvme_limit_mc("") == (None, "unreadable")


def test_limit_comes_from_the_discovered_entry(tmp_path):
    root = fake(tmp_path, ["nvme"])
    path = E.hwmon_input(root, {})
    assert E.nvme_limit_mc(path) == (85000 - E.NVME_LIMIT_MARGIN_MC, f"temp1_max-{E.NVME_LIMIT_MARGIN_MC}")


def test_missing_root_is_empty(tmp_path):
    assert E.hwmon_input(str(tmp_path / "none"), {}) == ""
