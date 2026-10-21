"""Port of think-split-stream (AnthropicChecks.swift)."""
from slotstream_spark.server.think import ThinkSplitter

TEXT = "Seventeen times twenty-three.</think>\n\n391\n\nDone."


def test_whole_text_splits_at_the_tag_without_the_newlines_after_it():
    assert ThinkSplitter.split(TEXT) == ("Seventeen times twenty-three.", "391\n\nDone.")


def test_deltas_of_every_size_split_like_the_whole_text():
    whole = ThinkSplitter.split(TEXT)
    for size in range(1, 13):
        sp = ThinkSplitter()
        think = answer = ""
        for start in range(0, len(TEXT), size):
            t, a = sp.push(TEXT[start:start + size])
            think, answer = think + t, answer + a
        t, a = sp.flush()
        assert (think + t, answer + a) == whole, size


def test_a_tag_split_across_deltas_is_never_reasoning_text():
    assert ThinkSplitter().push("abcde</th") == ("ab", "")
    sp = ThinkSplitter()
    out = [sp.push(p) for p in ("abc</th", "ink>\nok")]
    assert "".join(t for t, _ in out) == "abc" and "".join(a for _, a in out) == "ok"


def test_newlines_after_the_answer_starts_are_kept():
    late = ThinkSplitter()
    parts = [late.push(p)[1] for p in ["a</think>", "\n", "\n", "b\n", "\nc"]]
    assert "".join(parts) == "b\n\nc"


def test_reasoning_that_never_closes_is_all_thinking():
    sp = ThinkSplitter()
    first = sp.push("still thinking")
    rest = sp.flush()
    assert (first[0] + rest[0], first[1] + rest[1]) == ("still thinking", "")


def test_without_a_tag_the_whole_non_streamed_text_is_thinking():
    assert ThinkSplitter.split("no tag") == ("no tag", "")
