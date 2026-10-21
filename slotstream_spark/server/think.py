# SPDX-License-Identifier: MIT
# Port of Server.swift from slotstream (MIT, Copyright (c) 2026 Carlos Galarza); see THIRD_PARTY_NOTICES.md.
"""Reasoning split (port of ThinkSplitter in Server.swift).

Qwen writes its reasoning first and closes it with `</think>`. Every dialect carries that apart from the answer.
One instance follows one response.
"""

TAG = "</think>"


class ThinkSplitter:
    def __init__(self):
        self.buf = ""
        self.closed = False
        self.answered = False

    def push(self, s):
        """One delta to (thinking, content). Until the tag arrives the last 7 characters are withheld, so a tag
        straddling two deltas is never emitted as reasoning text."""
        if self.closed:
            return "", self._answer(s)
        self.buf += s
        at = self.buf.find(TAG)
        if at >= 0:
            think, rest = self.buf[:at], self.buf[at + len(TAG):]
            self.buf = ""
            self.closed = True
            return think, self._answer(rest)
        keep = min(len(self.buf), len(TAG) - 1)
        emit = self.buf[:len(self.buf) - keep]
        self.buf = self.buf[len(self.buf) - keep:]
        return emit, ""

    def _answer(self, s):
        """The answer without the newlines the model writes after `</think>`; they can arrive in later deltas."""
        if self.answered:
            return s
        rest = s.lstrip("\n")
        if rest:
            self.answered = True
        return rest

    def flush(self):
        """Whatever is still withheld when generation ends. Reasoning that never closed is all thinking."""
        rest, self.buf = self.buf, ""
        return ("", self._answer(rest)) if self.closed else (rest, "")

    @staticmethod
    def split(text):
        """The same split over a whole non-streamed response."""
        at = text.find(TAG)
        if at < 0:
            return text, ""
        return text[:at], text[at + len(TAG):].lstrip("\n")
