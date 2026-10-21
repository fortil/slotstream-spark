import hashlib
import json
import os
from pathlib import Path

import pytest

from slotstream_spark.engine.tokenizer import LOCK, TEMPLATE, TemplateMismatch, Tokenizer

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "third_party" / "qwen38-flash-next-metadata" / TEMPLATE
CONV = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Why?"}]


def _model_dir():
    d = os.environ.get("SLOTSTREAM_MODEL")
    return Path(d) if d else None


@pytest.fixture(scope="module")
def tok():
    path = SRC if SRC.exists() else (_model_dir() / TEMPLATE if _model_dir() else None)
    if path is None or not path.exists():
        pytest.skip("no chat_template.jinja (third_party/ on a checkout, $SLOTSTREAM_MODEL on the Spark)")
    return Tokenizer(path)


def test_template_sha_matches_the_lock():
    path = SRC if SRC.exists() else (_model_dir() / TEMPLATE if _model_dir() else None)
    if path is None or not path.exists():
        pytest.skip("no chat_template.jinja")
    want = next(f["sha256"] for f in json.loads(LOCK.read_text())["files"] if f["path"] == TEMPLATE)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == want


def test_changed_template_is_a_hard_error(tmp_path):
    if not SRC.exists():
        pytest.skip("no third_party/qwen38-flash-next-metadata checkout")
    bad = tmp_path / TEMPLATE
    bad.write_bytes(SRC.read_bytes() + b"\n")
    with pytest.raises(TemplateMismatch, match="differs from model.lock.json"):
        Tokenizer(bad)


def test_lock_without_the_entry_is_a_hard_error(tmp_path):
    if not SRC.exists():
        pytest.skip("no third_party/qwen38-flash-next-metadata checkout")
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"files": []}))
    with pytest.raises(TemplateMismatch, match="no entry"):
        Tokenizer(SRC, lock_path=lock)


def test_two_turn_conversation_thinking_off(tok):
    out = tok.render_chat(CONV)
    assert out == ("<|im_start|>user\nHi<|im_end|>\n"
                   "<|im_start|>assistant\n<think>\n\n</think>\n\nHello!<|im_end|>\n"
                   "<|im_start|>user\nWhy?<|im_end|>\n"
                   "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_thinking_on_opens_the_think_block(tok):
    out = tok.render_chat(CONV[:1], enable_thinking=True)
    assert out.endswith("<|im_start|>assistant\n<think>\n")


def test_single_user_turn_helper(tok):
    assert tok.render_user_turn("Hi") == tok.render_chat(CONV[:1])


def test_tools_reach_the_template(tok):
    tools = [{"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}}]
    out = tok.render_chat(CONV[:1], tools=tools)
    assert '"name": "f"' in out and out.index("<tools>") < out.index("<|im_start|>user")


def test_empty_messages_raise_from_the_template(tok):
    with pytest.raises(Exception):
        tok.render_chat([])


def test_encode_decode_roundtrip_needs_tokenizer_json():
    d = _model_dir()
    if d is None or not (d / "tokenizer.json").exists():
        pytest.skip("$SLOTSTREAM_MODEL/tokenizer.json not present (Mac)")
    t = Tokenizer.from_checkpoint(d)
    text = t.render_user_turn("Why does a mixture-of-experts model run with few active weights?")
    ids = t.encode(text)
    assert ids and all(isinstance(i, int) for i in ids)
    assert t.decode(ids) == text


def test_encode_without_tokenizer_json_says_so(tok):
    with pytest.raises(RuntimeError, match="without tokenizer.json"):
        tok.encode("x")
