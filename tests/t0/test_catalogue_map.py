"""Every slotstream T0 check name maps to a test here, a ctest file, or a reasoned `na` (step 88).

The names come from `Check("name", tier: .t0)` in third_party/slotstream/Sources/SlotstreamTestKit/*.swift (85 of them).
Targets: `pytest:<file>[::<class or test>]` (the file must define at least one test, and the named one must exist),
`ctest:<file>` (the file must exist under slotio/tests), `na:<reason>` (at least 20 characters).
"""
import ast
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "third_party" / "slotstream" / "Sources" / "SlotstreamTestKit"
MAP = Path(__file__).with_name("catalogue_map.toml")
EXPECTED_NAMES = 85
T0 = re.compile(r'Check\(\s*"([^"]+)"\s*,\s*tier:\s*\.t0\b')


def swift_names():
    if not KIT.is_dir():
        pytest.fail(f"{KIT} is missing (git submodule third_party/slotstream)", pytrace=False)
    names = []
    for f in sorted(KIT.glob("*.swift")):
        names += T0.findall(f.read_text())
    return names


def mapping():
    return tomllib.loads(MAP.read_text())["checks"]


def test_names_match_the_swift_sources():
    names = swift_names()
    assert len(names) == EXPECTED_NAMES == len(set(names)), len(names)
    m = mapping()
    assert set(m) == set(names), {"unmapped": sorted(set(names) - set(m)), "unknown": sorted(set(m) - set(names))}
    assert len(m) == 85


def defined_tests(path):
    """{name: kind} of the top-level test functions and classes (with their test methods) of a Python file."""
    tree = ast.parse(path.read_text())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test"):
            found[node.name] = "function"
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef) and n.name.startswith("test")]
            if methods:
                found[node.name] = "class"
    return found


@pytest.mark.parametrize("name,target", sorted(mapping().items()))
def test_target_exists(name, target):
    kind, _, rest = target.partition(":")
    if kind == "pytest":
        path, _, member = rest.partition("::")
        f = ROOT / path
        assert f.is_file(), f"{name}: {path} does not exist"
        found = defined_tests(f)
        assert found, f"{name}: {path} defines no test"
        if member:
            assert member in found, f"{name}: {path} has no test or test class {member!r} (has {sorted(found)[:6]}...)"
    elif kind == "ctest":
        assert (ROOT / "slotio" / "tests" / Path(rest).name).is_file() and rest.startswith("slotio/tests/"), \
            f"{name}: {rest} is not a file under slotio/tests"
    elif kind == "na":
        assert len(rest) >= 20, f"{name}: the na reason is too short: {rest!r}"
    else:
        pytest.fail(f"{name}: unknown target kind in {target!r}")


def test_na_reasons_name_their_category():
    m = mapping()
    gateway = [n for n in m if n.startswith("gateway-") or n == "chat-splice"]
    assert len(gateway) == 11 and all(m[n].startswith("na:") and "AI Gateway" in m[n] for n in gateway)
    quant = [n for n in m if n.startswith(("quantization-", "model-pack-")) or n in
             ("pack-memory", "automatic-pack-policy", "affine-standalone-metadata")]
    assert len(quant) == 13 and all(m[n].startswith("na:") and "single affine pack" in m[n] for n in quant)
    na = [n for n, t in m.items() if t.startswith("na:")]
    assert len(na) == 11 + 13 + 2, len(na)


def test_the_map_has_no_duplicate_keys():
    keys = re.findall(r'^"([^"]+)"\s*=', MAP.read_text(), re.M)
    assert len(keys) == len(set(keys)) == EXPECTED_NAMES
