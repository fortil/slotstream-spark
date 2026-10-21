"""Every slotstream port names its source and license; the third-party license files exist."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SWIFT_PORT = re.compile(r"port of (?:[^\n]*? )?[A-Za-z0-9_*]+\.swift", re.I)
SOURCE_LINE = re.compile(
    r"^# Port of .+ from slotstream \(MIT, Copyright \(c\) 2026 Carlos Galarza\); see THIRD_PARTY_NOTICES\.md\.$", re.M)
HELGARD = [
    "slotstream_spark/kernels/moe_gemv_affine.py", "slotstream_spark/model/ngram_hash.py",
    "slotstream_spark/model/reference_torch.py", "slotstream_spark/runtime/graphs.py",
    "tools/ppl_ablate.py", "tools/ref_layerwise.py",
]
# Ports whose docstring does not use the phrase "port of X.swift".
OTHER_PORTS = [
    "slotstream_spark/engine/sampler.py", "slotstream_spark/engine/lookahead.py", "slotstream_spark/engine/mtp.py",
    "slotstream_spark/engine/rollback.py", "slotstream_spark/engine/sweep.py", "slotstream_spark/kernels/forecast.py",
    "slotstream_spark/kernels/topk_stable.py", "slotstream_spark/memory/governor.py", "slotstream_spark/memory/ledger.py",
    "slotstream_spark/server/common.py", "slotstream_spark/server/http_limits.py", "slotstream_spark/server/think.py", "slotstream_spark/server/dialects/ollama.py",
    "tools/claims_gate.py", "tools/api_robustness.sh", "tools/quality/quality_probe.sh", "tools/mac/launch.sh",
]


def _head(rel):
    return "\n".join((ROOT / rel).read_text().split("\n")[:6])


def _tracked_or_new():
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard",
                          "slotstream_spark", "tools"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [f for f in out.split("\n") if f.endswith((".py", ".sh")) and (ROOT / f).is_file()]


def test_swift_ports_have_spdx_and_source_line():
    ports = [f for f in _tracked_or_new() if SWIFT_PORT.search((ROOT / f).read_text())]
    assert len(ports) >= 11, ports
    for f in ports:
        head = _head(f)
        assert "# SPDX-License-Identifier: MIT" in head, f
        assert SOURCE_LINE.search(head), f


def test_other_ports_have_spdx_and_source_line():
    for f in OTHER_PORTS:
        head = _head(f)
        assert "# SPDX-License-Identifier: MIT" in head, f
        assert SOURCE_LINE.search(head), f


def test_helgard_files_are_mit_and_apache():
    for f in HELGARD:
        assert "# SPDX-License-Identifier: MIT AND Apache-2.0" in _head(f), f


def test_vllm_derived_kernel_is_apache():
    assert "# SPDX-License-Identifier: Apache-2.0" in _head("slotstream_spark/kernels/fused_moe_affine.py")


def test_license_files_exist():
    assert (ROOT / "LICENSE").read_text().startswith("MIT License")
    for d in ("qwen38-flash-next-metadata", "transformers-5.19.0-qwen4_exp"):
        assert (ROOT / "third_party" / d / "LICENSE").is_file()
