"""The package imports with only the package on sys.path, from a cwd outside the repo (no tools/ or bench/ needed)."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_package_imports_without_repo_dirs(tmp_path):
    code = (
        "import sys\n"
        f"sys.path[:] = [p for p in sys.path if p not in ('', {str(ROOT)!r})]\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"  # only the package root, not tools/ or bench/
        "import slotstream_spark.cli, slotstream_spark.host.eligibility, slotstream_spark.memory.expert_freq\n"
        "import slotstream_spark.engine.pool_prefill, slotstream_spark.bench.prompts\n"
        "assert not any(p.endswith(('/tools', '/bench')) for p in sys.path), sys.path\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_no_sys_path_insert_in_the_cli_and_pool_prefill():
    for rel in ("slotstream_spark/cli.py", "slotstream_spark/engine/pool_prefill.py"):
        assert "sys.path.insert" not in (ROOT / rel).read_text(), rel


def test_shims_alias_the_package_modules():
    from bench import eligibility as shim
    from slotstream_spark.host import eligibility as real
    assert shim is real
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import expert_freq as ef
        from slotstream_spark.memory import expert_freq as real_ef
        assert ef is real_ef
    finally:
        sys.path.remove(str(ROOT / "tools"))
