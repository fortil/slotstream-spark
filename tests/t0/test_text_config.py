"""TextConfig against the real checkpoint config (no torch needed)."""
import json
from pathlib import Path

import pytest

from slotstream_spark.model.config import EXPECTED, TextConfig

CKPT = Path(__file__).resolve().parents[2] / "third_party" / "qwen38-flash-next-metadata"


def test_config_validates_real_checkpoint():
    if not (CKPT / "config.json").exists():
        pytest.skip("checkpoint config not available (third_party/qwen38-flash-next-metadata is missing)")
    cfg = TextConfig.from_checkpoint(CKPT)
    assert cfg.layer_types[3] == "full_attention" and cfg.layer_types[2] == "linear_attention"
    assert cfg.ple_layers == [1] and cfg.rotary_dim == 64
    tc = json.loads((CKPT / "config.json").read_text())["text_config"]
    assert set(EXPECTED) <= set(tc)
    tc["num_experts"] = 256
    with pytest.raises(ValueError):
        TextConfig.from_dict(tc)
    tc["num_experts"] = 512
    tc["layer_types"][3] = "linear_attention"
    with pytest.raises(ValueError):
        TextConfig.from_dict(tc)
