"""Tokenizer and chat template of the checkpoint.

`Tokenizer.from_checkpoint(dir)` loads `tokenizer.json` (tokenizers) and renders chat messages through the
checkpoint's own `chat_template.jinja` with jinja2. The template file must hash to its entry in `model.lock.json`:
a different template is a hard error, because a silently different prompt format changes every output.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCK = ROOT / "model.lock.json"
TEMPLATE = "chat_template.jinja"


class TemplateMismatch(RuntimeError):
    pass


def _lock_sha(lock_path, name):
    files = json.loads(Path(lock_path).read_text())["files"]
    for f in files:
        if f["path"] == name:
            return f["sha256"]
    raise TemplateMismatch(f"{name} has no entry in {lock_path}")


def _raise(msg):
    raise ValueError(msg)


def _tojson(x, ensure_ascii=False, indent=None, sort_keys=False, separators=None):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys, separators=separators)


class Tokenizer:
    def __init__(self, template_path, tokenizer_json=None, lock_path=LOCK):
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        raw = Path(template_path).read_bytes()
        got, want = hashlib.sha256(raw).hexdigest(), _lock_sha(lock_path, TEMPLATE)
        if got != want:
            raise TemplateMismatch(f"{template_path}: sha256 {got} differs from model.lock.json {want}")
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise
        self._template = env.from_string(raw.decode("utf-8"))
        self._tok = None
        if tokenizer_json is not None:
            from tokenizers import Tokenizer as HFTokenizer
            self._tok = HFTokenizer.from_file(str(tokenizer_json))

    @classmethod
    def from_checkpoint(cls, directory, lock_path=LOCK):
        d = Path(directory)
        tj = d / "tokenizer.json"
        return cls(d / TEMPLATE, tj if tj.exists() else None, lock_path)

    def _need_tok(self):
        if self._tok is None:
            raise RuntimeError("this Tokenizer was built without tokenizer.json (template only)")
        return self._tok

    def encode(self, text):
        return list(self._need_tok().encode(text, add_special_tokens=False).ids)

    def decode(self, ids):
        return self._need_tok().decode([int(i) for i in ids], skip_special_tokens=False)

    def render_chat(self, messages, *, enable_thinking=False, tools=None):
        """Chat text with the generation prompt appended (add_generation_prompt)."""
        return self._template.render(messages=messages, tools=tools, add_generation_prompt=True,
                                     enable_thinking=enable_thinking, add_vision_id=False)

    def render_user_turn(self, text, *, enable_thinking=False):
        return self.render_chat([{"role": "user", "content": text}], enable_thinking=enable_thinking)
