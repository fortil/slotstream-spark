"""Text configuration of Qwen3.8-Flash-Next, read from the checkpoint and checked against the
architecture the engine is written for (qwen4_exp.py:20-67)."""
import json
from dataclasses import dataclass
from pathlib import Path

EXPECTED = {
    "hidden_size": 2560,
    "num_hidden_layers": 48,
    "num_attention_heads": 24,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "num_experts": 512,
    "num_experts_per_tok": 10,
    "moe_intermediate_size": 640,
    "shared_expert_intermediate_size": 640,
    "hc_count": 4,
    "hc_lowrank": 320,
    "indexer_n_heads": 4,
    "indexer_head_dim": 128,
    "indexer_budget": 2048,
    "indexer_compress_ratio": 4,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "ple_layer_ids": [2],
    "partial_rotary_factor": 0.25,
    "rms_norm_eps": 1e-6,
    "vocab_size": 248320,
    "eos_token_id": 248044,
    "full_attention_interval": 4,
}


@dataclass(frozen=True)
class TextConfig:
    raw: dict

    @classmethod
    def from_checkpoint(cls, directory):
        cfg = json.loads((Path(directory) / "config.json").read_text())
        return cls.from_dict(cfg["text_config"])

    @classmethod
    def from_dict(cls, tc, validate=True):
        tc = dict(tc)
        if validate:
            for key, want in EXPECTED.items():
                if tc.get(key) != want:
                    raise ValueError(f"text_config.{key} = {tc.get(key)!r}, expected {want!r}")
            types = tc.get("layer_types")
            if types is None or len(types) != tc["num_hidden_layers"]:
                raise ValueError("text_config.layer_types missing or of the wrong length")
            for i, t in enumerate(types):
                want = "full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                if t != want:
                    raise ValueError(f"layer {i}: type {t!r}, expected {want!r}")
        return cls(tc)

    def __getitem__(self, key):
        return self.raw[key]

    def get(self, key, default=None):
        return self.raw.get(key, default)

    @property
    def layer_types(self):
        return list(self.raw["layer_types"])

    @property
    def rotary_dim(self):
        return int(self.raw["head_dim"] * self.raw["partial_rotary_factor"])

    @property
    def rope_theta(self):
        return self.raw["rope_parameters"]["rope_theta"]

    @property
    def ple_layers(self):
        """Layer indices (0-based) that carry a PLE: `ple_layer_ids` are 1-based (id 2 is index 1)."""
        return [i - 1 for i in self.raw["ple_layer_ids"]]
