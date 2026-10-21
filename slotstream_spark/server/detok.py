"""Incremental detokenizing for the stream: text only once it can no longer change.

A byte-level BPE token can end in the middle of a UTF-8 sequence, so decoding token by token produces U+FFFD that a
later token repairs. `IncrementalDecoder` decodes a small window (the tokens since the last stable point), emits the
new characters only when the window does not end in U+FFFD, and holds back any tail that could still become a stop
sequence. Non-streaming requests do not need this: `decode_once` decodes the finished id list in one call.
"""

REPLACEMENT = "�"


def decode_once(decode, ids, stop=()):
    """The whole completion in one decode, cut at the earliest stop sequence. -> (text, matched stop or None)."""
    text = decode(list(ids))
    best = None
    for s in stop:
        if s and (i := text.find(s)) >= 0 and (best is None or i < best[0]):
            best = (i, s)
    return (text, None) if best is None else (text[:best[0]], best[1])


def _held_tail(text, stop):
    """Length of the longest suffix of `text` that is a proper prefix of some stop sequence."""
    hold = 0
    for s in stop:
        for n in range(min(len(s) - 1, len(text)), hold, -1):
            if text.endswith(s[:n]):
                hold = n
                break
    return hold


class IncrementalDecoder:
    """`decode(ids) -> str` is the tokenizer's own decode. Feed ids with `push`, end with `finish`."""

    def __init__(self, decode, stop=()):
        self._decode = decode
        self.stop = [s for s in stop if s]
        self._ids = []
        self._prefix = 0     # start of the window: tokens before it are settled
        self._read = 0       # tokens in [_prefix, _read) were already turned into text
        self._pending = ""   # stable text not yet emitted (a possible stop-sequence prefix)
        self.stopped = None  # the stop sequence that ended the text, if any
        self.text = ""       # everything emitted so far

    def _stable_delta(self):
        """New characters of the window, or '' while the window ends mid-character."""
        old = self._decode(self._ids[self._prefix:self._read]) if self._read > self._prefix else ""
        new = self._decode(self._ids[self._prefix:])
        if len(new) <= len(old) or new.endswith(REPLACEMENT):
            return ""
        delta = new[len(old):]
        self._prefix, self._read = self._read, len(self._ids)
        return delta

    def _release(self, final=False):
        if self.stopped is not None:
            return ""
        best = None
        for s in self.stop:
            if (i := self._pending.find(s)) >= 0 and (best is None or i < best[0]):
                best = (i, s)
        if best is not None:
            out, self._pending, self.stopped = self._pending[:best[0]], "", best[1]
        elif final:
            out, self._pending = self._pending, ""
        else:
            keep = _held_tail(self._pending, self.stop)
            out, self._pending = self._pending[:len(self._pending) - keep], self._pending[len(self._pending) - keep:]
        self.text += out
        return out

    def push(self, token_id):
        """-> text that is now safe to send ('' while a character or a stop sequence is incomplete)."""
        if self.stopped is not None:
            return ""
        self._ids.append(int(token_id))
        self._pending += self._stable_delta()
        return self._release()

    def finish(self):
        """Flush at the end: an incomplete character is sent as the replacement character, a held stop prefix as text."""
        if self.stopped is not None:
            return ""
        if self._read < len(self._ids):
            tail = self._decode(self._ids[self._prefix:])
            old = self._decode(self._ids[self._prefix:self._read]) if self._read > self._prefix else ""
            self._pending += tail[len(old):]
            self._read = len(self._ids)
        return self._release(final=True)
