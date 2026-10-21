"""Persistent prefix tier (step 85): format, policy, directory handling, save/restore round trip, all on host arrays.

Ports of the slotstream checks persistent-prefix-policy, -metadata-bounds, -clear, conversation-ids, -removal-failures,
-read-failures and the round trip (Diagnostics+PersistentPrefix*.swift, PersistentPrefix*Checks.swift), plus the
Linux-side rules of the plan: lineage reuse bound to the head name, I/O errors never delete, identity pieces.
"""
import fcntl
import hashlib
import json
import os
import stat
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from slotstream_spark.engine import persistent_prefix as pp
from slotstream_spark.engine.persistent_prefix import (Configuration, DirectoryInUse, Entry, Extent, FileError,
                                                        FixedArray, Identity, InaccessibleFile, Layout3,
                                                        PersistentPrefix, RowsRecord, SegmentEntry, SeqArray,
                                                        SequenceRecord, StateView, Value)

NOW = 10_000_000.0
DAY = 86_400.0
IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0
BIG_FREE = lambda: 10 ** 12  # noqa: E731


def seg(n):
    return f"{n:032x}.slotseg"


def records(segments, base=0):
    if not segments:
        return {}
    ext = [Extent(s, base + i * 10, base + (i + 1) * 10) for i, s in enumerate(segments)]
    return {"kv.3.keys": SequenceRecord("kv.3.keys", "bfloat16", [1, 2, 1024, 8], 2, base, len(segments) * 10, ext)}


def entry(file, tokens, identity="own", used=NOW, draft=True, continued=False, shared=False, segments=()):
    return Entry(file, identity, list(tokens), 100, used, has_draft=draft, continued=continued, shared=shared,
                 sequences=records(list(segments)))


# ============================================================================================ policy (ported)
class TestPolicy:
    ALL = [entry("a", [1, 2, 3], used=NOW - 3), entry("b", [1, 2, 3, 4, 5], used=NOW - 5),
           entry("c", [1, 2, 3, 4, 5, 6], used=NOW - 4, draft=False),
           entry("d", [1, 2, 3, 4, 5, 6, 7], identity="other", used=NOW - 6), entry("e", [9, 9, 9, 9], used=NOW - 1)]
    PROMPT = [1, 2, 3, 4, 5, 6, 7, 8]

    @staticmethod
    def best(entries, prompt, retained=0, draft=False, max_age=None):
        e = pp.best_match(entries, "own", prompt, retained, draft, NOW, max_age)
        return e.file if e else None

    def test_defaults(self):
        assert pp.DEFAULT_MAX_BYTES == 20_000_000_000 and pp.DEFAULT_MINIMUM_TOKENS == 1024
        assert pp.DEFAULT_MAX_AGE_DAYS == 30 and pp.DEFAULT_MAX_AGE == 30 * 86_400
        assert Configuration("/nonexistent").max_age == 30 * 86_400
        assert pp.MINIMUM_FREE_BYTES == 2_000_000_000 and pp.MAXIMUM_SEGMENTS == 32 and pp.FORMAT_VERSION == 3

    def test_best_match(self):
        a = self.ALL
        assert self.best(a, self.PROMPT) == "c"
        assert self.best(a, self.PROMPT, draft=True) == "b"
        assert self.best(a, self.PROMPT, retained=6) is None
        assert self.best(a, [1, 2, 3, 4, 5], draft=True) == "a"        # the prompt must be LONGER than the state
        assert self.best(a, [1, 2, 3]) is None                         # an equal prompt is not an extension
        assert self.best([a[3]], self.PROMPT) is None                  # another identity never matches
        assert self.best(a, [1, 2, 4, 5, 6, 7]) is None
        stale = entry("s", [1, 2, 3, 4, 5, 6], used=NOW - 31 * DAY)
        assert self.best([stale], self.PROMPT, max_age=30 * DAY) is None
        assert self.best([stale], self.PROMPT) == "s"

    def test_boundaries_and_pass_size(self):
        e = replace(entry("p", [1, 2, 3]), prefill_chunk=256)
        ok = pp.best_match([e], "own", [1, 2, 3, 4], 0, False, NOW, None, boundaries={3}, prefill_chunk=256)
        assert ok is e
        assert pp.best_match([e], "own", [1, 2, 3, 4], 0, False, NOW, None, boundaries={2}, prefill_chunk=256) is None
        assert pp.best_match([e], "own", [1, 2, 3, 4], 0, False, NOW, None, boundaries={3}, prefill_chunk=512) is None
        assert pp.best_match([e], "own", [1, 2, 3, 4], 0, False, NOW, None, boundaries={3}, prefill_chunk=None) is None
        old = entry("q", [1, 2, 3])                                   # a state with no recorded pass size
        assert pp.best_match([old], "own", [1, 2, 3, 4], 0, False, NOW, None, boundaries={3}, prefill_chunk=256) is None

    def test_splice_and_branches(self):
        a = self.ALL
        stale = entry("s", [1, 2, 3, 4, 5, 6], used=NOW - 31 * DAY)
        assert pp.longest_extension(a, "own", [1, 2], NOW, None) == [1, 2, 3, 4, 5, 6]
        assert pp.longest_extension([a[3]], "own", [1, 2], NOW, None) is None
        assert pp.longest_extension([stale], "own", [1, 2], NOW, 30 * DAY) is None
        assert pp.extensions(a + [stale], "own", [1, 2], NOW, 30 * DAY) == [[1, 2, 3], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]
        t = replace(a[0], splicing_tokens=[1, 2, 3, 42, 43])
        assert pp.extensions([t], "own", [1, 2, 3, 42], NOW, None) == [[1, 2, 3, 42, 43]]
        assert pp.extensions([t], "own", t.splicing_tokens, NOW, None) == []

    def test_redundant_ancestors_and_shared(self):
        a = self.ALL
        assert {e.file for e in pp.redundant_ancestors(a, "own", self.PROMPT)} == {"a", "b"}
        assert pp.redundant_ancestors([a[0]], "own", self.PROMPT) == []
        assert pp.redundant_ancestors(a, "own", [7, 7, 7, 7, 7]) == []
        shared = entry("g", [1, 2], used=NOW - 2, shared=True)
        assert {e.file for e in pp.redundant_ancestors(a + [shared], "own", self.PROMPT)} == {"a", "b"}

    def test_common_prefix(self):
        a = self.ALL
        stale = entry("s", [1, 2, 3, 4, 5, 6], used=NOW - 31 * DAY)
        assert pp.longest_common_prefix(a, "own", [1, 2, 3, 4, 5, 6, 9], NOW, None) == 6
        assert pp.longest_common_prefix(a, "own", [1, 2, 9], NOW, None) == 2
        assert pp.longest_common_prefix([a[3]], "own", self.PROMPT, NOW, None) == 0
        assert pp.longest_common_prefix([stale], "own", self.PROMPT, NOW, 30 * DAY) == 0
        assert pp.longest_common_prefix(a, "own", [1, 2, 3, 4, 5, 6, 9], NOW, None, diverging=True) == 0
        assert pp.longest_common_prefix(a, "own", [1, 2, 3, 4, 9], NOW, None, diverging=True) == 4
        assert pp.common_prefix_length([1, 2, 3], [1, 2, 4]) == 2 and pp.common_prefix_length([], [1]) == 0

    def test_system_boundary(self):
        h, t = [10, 11, 12], [20, 21]
        assert pp.system_prefix_boundary([10, 11, 12, 5, 6, 20, 21, 7, 20, 21], h, t) == 7
        assert pp.system_prefix_boundary([10, 11, 13, 5, 20, 21], h, t) is None
        assert pp.system_prefix_boundary([10, 11, 12, 5, 6, 20], h, t) is None

    def test_removal_classes(self):
        classes = [entry("x", [1, 2, 3], used=NOW - 1), entry("y", [1, 2, 3, 4], used=NOW - 9, continued=True),
                   entry("z", [7, 7, 7], used=NOW - 2), entry("w", [5, 5], identity="other", used=NOW),
                   entry("v", [6, 6, 6], used=NOW - 40 * DAY, continued=True)]

        def value(i):
            return pp.value_of(classes[i], classes, "own", NOW, 30 * DAY)

        assert [value(i) for i in range(5)] == [Value.PARENT, Value.CONVERSATION, Value.ONE_OFF, Value.FOREIGN, Value.EXPIRED]
        assert [v.label for v in sorted(Value)] == ["other build", "expired", "one-off", "parent", "conversation",
                                                    "shared prefix"]
        assert [e.file for e in pp.eviction_order(classes, "own", NOW, 30 * DAY)] == ["w", "v", "z", "x", "y"]

    def test_lineages_decide_what_is_shared(self):
        lin = [entry("p", [1, 2], used=NOW - 9, shared=True), entry("q", [1, 2, 3], used=NOW - 1, continued=True),
               entry("r", [1, 2, 4], used=NOW - 2, continued=True), entry("t", [1, 2, 3, 5], used=NOW, continued=True),
               entry("o", [8, 8], used=NOW - 3, shared=True)]
        v = lambda i: pp.value_of(lin[i], lin, "own", NOW, 30 * DAY)  # noqa: E731
        assert v(0) == Value.SHARED and v(1) == Value.PARENT and v(4) == Value.ONE_OFF
        assert [e.file for e in pp.eviction_order(lin, "own", NOW, 30 * DAY)] == ["o", "q", "r", "t", "p"]
        regen = [entry("m", [1, 2, 3], used=NOW - 5, continued=True), entry("n1", [1, 2, 3, 4], used=NOW - 1, continued=True),
                 entry("n2", [1, 2, 3, 6], used=NOW, continued=True)]
        assert pp.value_of(regen[0], regen, "own", NOW, None) == Value.SHARED

    def test_quota_over_shared_segments(self):
        s1, s2, s3 = seg(1), seg(2), seg(3)
        sizes = {s1: 1000, s2: 1000, s3: 1000}
        sh = [entry("h1", [1, 2], used=NOW - 1, segments=[s1]),
              entry("h2", [1, 2, 3], used=NOW - 2, continued=True, segments=[s1, s2]),
              entry("h3", [8, 8], used=NOW, segments=[s3])]

        def victims(quota, incoming, pinned=(), freed=()):
            v = pp.eviction_victims(sh, sizes, "own", quota, incoming, set(pinned), set(freed), NOW, None)
            return None if v is None else [e.file for e in v]

        assert pp.held_bytes(sh, sizes, set()) == 3300
        assert victims(3400, 100) == []
        assert victims(3000, 100) == ["h3"]
        assert victims(2150, 100) == ["h3", "h1", "h2"]
        assert victims(1200, 100, freed=["h1", "h2"]) == []
        assert victims(1100, 50, pinned=[s1]) == ["h3", "h1", "h2"]
        assert victims(1040, 50, pinned=[s1]) is None
        assert pp.unreferenced_segments([sh[1]], [s1, s2, s3]) == [s3]

    def test_write_fits_quota(self):
        f = pp.write_fits_quota
        assert not f(300, 0, 100, 350) and f(200, 0, 100, 350) and f(300, 100, 150, 350)
        assert not f(200, 100, 100, 150) and f(250, 0, 100, 350)
        assert not f(1, 0, 2 ** 63 - 1, 2 ** 63 - 1) and f(2 ** 63 - 1, 2 ** 63 - 1, 2 ** 63 - 1, 2 ** 63 - 1)

    def test_extents(self):
        s1, s2, s3 = seg(1), seg(2), seg(3)
        rec = records([s1, s2, s3])["kv.3.keys"]
        assert pp.tiles(rec)
        gap = replace(rec, extents=[rec.extents[0], Extent(s2, 11, 20), rec.extents[2]])
        assert not pp.tiles(gap)
        assert not pp.tiles(replace(rec, live=rec.live + 1))
        assert not pp.tiles(replace(rec, extents=[Extent("../" + s1, 0, 10)] + rec.extents[1:]))
        assert pp.clip(rec.extents, 15, 30) == [Extent(s2, 15, 20), Extent(s3, 20, 30)]
        assert pp.clip(rec.extents, 5, 25) == [Extent(s1, 5, 10), Extent(s2, 10, 20), Extent(s3, 20, 25)]
        assert pp.clip(rec.extents, 30, 30) == []
        assert pp.clip(rec.extents, 0, 35) is None and pp.clip(rec.extents, 31, 30) is None

    def test_dangling(self):
        s1, s2, s3 = seg(1), seg(2), seg(3)
        rows = RowsRecord("kv.3.keys", "bfloat16", [1, 2], [8], 0, 10, 8, 320, 0)

        def held(names, end=10):
            out = {}
            for i, n in enumerate(names):
                r = replace(rows, start=i * 10, end=(i * 10 + end) if i == len(names) - 1 else (i + 1) * 10)
                out[n] = SegmentEntry(n, "own", 400, {"kv.3.keys": r})
            return out

        head = entry("h", [1], segments=[s1, s2, s3])
        assert pp.dangling_reason(head, held([s1, s2, s3])) is None
        assert pp.dangling_reason(head, held([s1, s2])) is not None
        assert pp.dangling_reason(head, held([s1, s2, s3], end=9)) is not None

    def test_reuse(self):
        s1, s2, s3 = seg(1), seg(2), seg(3)
        parent = records([s1, s2])

        def child(name="kv.3.keys", dtype="bfloat16", base=0, live=0):
            return SequenceRecord(name, dtype, [1, 2, 1024, 8], 2, base, live, [])

        def reuse(c, par=parent, exists=(s1, s2, s3), maximum=32):
            return pp.reuse_plan([c], par, lambda s: s in exists, maximum)

        grown = reuse(child(live=25))
        assert grown.extents["kv.3.keys"] == parent["kv.3.keys"].extents and grown.first_new_row["kv.3.keys"] == 20
        assert grown.segments == {s1, s2} and not grown.compacted
        released = reuse(child(base=12, live=13))
        assert released.extents["kv.3.keys"] == [Extent(s2, 12, 20)] and released.first_new_row["kv.3.keys"] == 20
        assert reuse(child(base=22, live=3)).first_new_row["kv.3.keys"] == 22
        assert reuse(child(dtype="float16", live=25)).first_new_row["kv.3.keys"] == 0
        assert reuse(child(live=15)).first_new_row["kv.3.keys"] == 0                 # a rewound state
        assert reuse(child(live=25), exists=(s2,)).first_new_row["kv.3.keys"] == 0
        assert reuse(child(live=25), par=None).first_new_row["kv.3.keys"] == 0
        b = reuse(child(live=25), maximum=2)
        assert b.compacted and not b.segments and b.first_new_row["kv.3.keys"] == 0 and b.extents["kv.3.keys"] == []
        assert not reuse(child(live=25), maximum=3).compacted


# ============================================================================================= format (ported)
class TestFormat:
    def test_layout(self):
        kv = pp.layout([1, 2, 2048, 256], 2, 1500, 2)
        assert (kv.logical_bytes, kv.capacity_bytes, kv.leading, kv.row_bytes) == (2 * 1500 * 256 * 2, 2 * 2048 * 256 * 2, 2, 512)
        assert pp.layout([1, 2, 1024, 8], 2, 1025, 2) is None
        assert pp.layout([1, 2], 2, 0, 2) is None
        assert pp.layout([2 ** 62, 4], 0, 1, 8) is None
        assert pp.layout([1, 1 << 20, 1 << 15], 1, 1, 2) is None
        assert pp.layout([pp.INT32_MAX], 0, 0, 1) is not None and pp.layout([pp.INT32_MAX + 1], 0, 0, 1) is None
        assert pp.rows_geometry([1, 2], [8], 2, 10) == (2, 16, 320)
        assert pp.rows_geometry([2 ** 62], [4], 2, 3) is None

    def test_footer(self):
        size = 8 + 1000 + 300 + pp.FOOTER_BYTES
        f = pp.footer(1008, 300, 0xDEADBEEF)
        assert len(f) == pp.FOOTER_BYTES and f[24:] == b"SLOTEND1"
        assert pp.parse_footer(f, size) == (1008, 300, 0xDEADBEEF)
        assert pp.parse_footer(f, size + 1) is None
        bad = bytearray(f)
        bad[31] ^= 1
        assert pp.parse_footer(bytes(bad), size) is None
        big = pp.MAX_HEADER_BYTES + 1
        assert pp.parse_footer(pp.footer(8, big, 0), 8 + big + pp.FOOTER_BYTES) is None

    def test_names(self):
        n = pp.file_name("own", [1, 2, 3])
        assert n == pp.file_name("own", [1, 2, 3]) == pp.file_name("own", [1, 2, 3, 4][:3])
        assert n != pp.file_name("other", [1, 2, 3]) and n != pp.file_name("own", [1, 2, 4])
        expect = hashlib.sha256(b"own" + np.array([1, 2, 3], dtype="<i8").tobytes()).hexdigest()[:40] + ".slotprefix"
        assert n == expect
        assert pp.is_head_name(n) and not pp.is_segment_name(n)
        fresh = pp.new_segment_name()
        assert pp.is_segment_name(fresh) and not pp.is_head_name(fresh) and fresh != pp.new_segment_name()
        assert not pp.is_segment_name("../" + fresh) and not pp.is_segment_name(fresh.upper()) and not pp.is_head_name("x.slotprefix")
        assert pp.token_bytes([1, 256]) == bytes([1, 0, 0, 0, 0, 1, 0, 0])

    def test_crc(self):
        assert pp.crc32(b"123456789") == 0xCBF43926
        assert pp.crc32(b"56789", pp.crc32(b"1234")) == 0xCBF43926

    def test_header_json_is_sorted_and_round_trips(self):
        h = pp.Head("own", 3, continued=True, compactStateWindows=True, ngramContext=[1, 2],
                    linear=[pp.LinearRecord(0, [1, 2])],
                    attention=[pp.AttentionRecord(3, 3, pp.IndexerRecord(3, 0, True, 0, 4))],
                    arrays=[pp.ArrayRecord("tokens", "int32", [3], 0, 3, 8, 12, 7)],
                    sequences=[SequenceRecord("kv.3.keys", "bfloat16", [1, 2, 1024, 8], 2, 0, 3, [Extent(seg(1), 0, 3)])],
                    sequenceBytes=4, residentBytes=10)
        raw = pp.encode_header(h)
        d = json.loads(raw)
        assert list(d) == sorted(d) and d["format"] == 3 and d["kind"] == "head"
        assert "draft" not in d and "shared" not in d and "splicingTokens" not in d and "prefillChunk" not in d
        again = pp.head_from_json(d)
        assert again == h and pp.tiles(again.sequences[0])
        assert pp.encode_header(again) == raw
        s = pp.Segment("own", [RowsRecord("kv.3.keys", "bfloat16", [1, 2], [8], 0, 3, 8, 96, 9)])
        assert pp.segment_from_json(json.loads(pp.encode_header(s))) == s

    def test_identity(self):
        i = Identity({"x": "1", "y": "2"})
        assert i.digest == Identity({"y": "2", "x": "1"}).digest and len(i.digest) == 64
        assert i.digest != Identity({"x": "1", "y": "3"}).digest
        assert i.digest == hashlib.sha256(b"x=1\ny=2").hexdigest()

    def test_identity_pieces(self, tmp_path):
        pkg = tmp_path / "pkg"
        (pkg / "sub").mkdir(parents=True)
        (pkg / "a.py").write_text("x = 1\n")
        (pkg / "sub" / "b.py").write_text("y = 2\n")
        so = tmp_path / "slotio.so"
        so.write_bytes(b"\x7fELF-one")
        d1 = pp.code_digest(pkg, [so])
        assert d1 == pp.code_digest(pkg, [so])
        (pkg / "sub" / "b.py").write_text("y = 3\n")
        assert pp.code_digest(pkg, [so]) != d1
        d2 = pp.code_digest(pkg, [so])
        so.write_bytes(b"\x7fELF-two")
        assert pp.code_digest(pkg, [so]) != d2
        d3 = pp.code_digest(pkg, [so])
        (pkg / "__pycache__").mkdir()
        (pkg / "__pycache__" / "z.py").write_text("junk")
        assert pp.code_digest(pkg, [so]) == d3              # caches are skipped

        w = tmp_path / "w.bin"
        data = bytearray(os.urandom(9 << 20))
        w.write_bytes(bytes(data))
        base = pp.sampled_digest(w)
        assert base.startswith("w.bin:") and str(9 << 20) in base
        data[(4 << 20) + 5] ^= 1                            # the middle is not sampled
        data[(4 << 20) + 777] ^= 1
        w.write_bytes(bytes(data))
        assert pp.sampled_digest(w) == base
        data[7] ^= 1
        w.write_bytes(bytes(data))
        first = pp.sampled_digest(w)
        assert first != base
        data[-1] ^= 1
        w.write_bytes(bytes(data))
        assert pp.sampled_digest(w) != first
        small = tmp_path / "s.bin"
        small.write_bytes(b"abc")
        assert pp.sampled_digest(small).endswith(hashlib.sha256(b"abc").hexdigest())
        assert [p.name for p in pp.packed_files(tmp_path)] == ["s.bin", "slotio.so", "w.bin"]

    def test_backend_identity(self):
        full = {"backend": "torch", "torch": "2.9", "triton": "3.5", "driver": "580.1", "gpu_name": "GB10",
                "compute_capability": "12.1"}
        c = pp.backend_components(lambda: full)
        assert c["backend.gpu_name"] == "GB10" and c["backend.compute_capability"] == "12.1" and c["backend.driver"] == "580.1"
        for missing in ("gpu_name", "compute_capability", "driver", "torch", "triton"):
            part = {k: v for k, v in full.items() if k != missing}
            with pytest.raises(pp.IdentityError):
                pp.backend_components(lambda p=part: p)
        assert pp.backend_components(lambda: {"backend": "torch"}, require_gpu=False) == {"backend.backend": "torch"}
        kw = dict(code="c", config_sha="s", weights=["w:1:aa"], geometry=[1, 2], optimizations={"b": 1, "a": 2},
                  envelope_id="e", arithmetic_version=1)
        a = pp.identity_for(backend=c, **kw)
        assert a.digest == pp.identity_for(backend=c, **kw).digest
        other_gpu = pp.backend_components(lambda: dict(full, gpu_name="H100"))
        assert pp.identity_for(backend=other_gpu, **kw).digest != a.digest
        for key, val in (("code", "c2"), ("config_sha", "s2"), ("weights", ["w:1:bb"]), ("geometry", [1, 3]),
                         ("optimizations", {"a": 2, "b": 2}), ("envelope_id", "e2"), ("arithmetic_version", 2)):
            assert pp.identity_for(backend=c, **dict(kw, **{key: val})).digest != a.digest, key
        assert a.components["format"] == "3" and a.components["optimizations"] == '{"a":2,"b":1}'


# ================================================================================ container fixtures (ported)
def put_container(path, payloads, expected, header):
    return pp.write_container(path, payloads, expected, header)


class TestMetadataBounds:
    """Checksum-valid containers with invalid array metadata, through the real reader and the startup indexer."""

    def build(self, tmp_path):
        identity = Identity({"fixture": "metadata-bounds"})
        original, serial = {}, [0]
        limit = pp.INT32_MAX

        def write(extra=None, change=None):
            serial[0] += 1
            tokens = [serial[0]]
            name = pp.file_name(identity.digest, tokens)
            payloads = [pp.token_bytes(tokens)] + ([bytes(extra)] if extra is not None else [])

            def header(placed):
                arrays = []
                for i, (off, count, crc) in enumerate(placed):
                    arrays.append(pp.ArrayRecord("tokens" if i == 0 else "fixture", "int32" if i == 0 else "uint8",
                                                 [1 if i == 0 else len(payloads[i])], 0, 1 if i == 0 else len(payloads[i]),
                                                 off, count, crc))
                h = pp.Head(identity.digest, 1, arrays=arrays, residentBytes=len(extra) if extra is not None else 0)
                if change:
                    change(h)
                return pp.encode_header(h)

            put_container(tmp_path / name, payloads, [len(p) for p in payloads], header)
            original[name] = (tmp_path / name).read_bytes()
            return name

        valid = [write(), write([1, 2, 3, 4]), write([]), write([1], lambda h: setattr(h.arrays[1], "shape", [])),
                 write([1, 2], lambda h: (setattr(h, "sequences", [SequenceRecord("rows", "uint8", [4], 0, 0, 0, [])]),
                                          setattr(h, "sequenceBytes", 4), setattr(h, "residentBytes", 6))),
                 write([], lambda h: setattr(h.arrays[1], "shape", [0, limit]))]

        def seq(shape, **kw):
            return [SequenceRecord("rows", "uint8", shape, 0, kw.get("base", 0), kw.get("live", 0), [])]

        def with_seq(h, shape, seq_bytes, resident, **kw):
            h.sequences, h.sequenceBytes, h.residentBytes = seq(shape, **kw), seq_bytes, resident

        invalid = [
            ("oversized sequence dimension", write(change=lambda h: with_seq(h, [limit + 1], limit + 1, limit + 1))),
            ("oversized empty fixed dimension", write([], lambda h: setattr(h.arrays[1], "shape", [0, limit + 1]))),
            ("negative resident ledger", write(change=lambda h: setattr(h, "residentBytes", -1))),
            ("overstated resident ledger", write(change=lambda h: setattr(h, "residentBytes", 1))),
            ("understated fixed capacity", write([1, 2], lambda h: setattr(h, "residentBytes", 1))),
            ("overstated fixed capacity", write([1, 2], lambda h: setattr(h, "residentBytes", 3))),
            ("negative sequence ledger", write(change=lambda h: setattr(h, "sequenceBytes", -1))),
            ("overstated sequence ledger", write(change=lambda h: setattr(h, "sequenceBytes", 1))),
            ("understated sequence capacity", write(change=lambda h: with_seq(h, [4], 0, 4))),
            ("understated total capacity", write(change=lambda h: with_seq(h, [4], 4, 0))),
            ("invalid fixed dtype", write([1], lambda h: setattr(h.arrays[1], "dtype", "unknown"))),
            ("invalid fixed capacity", write([1], lambda h: setattr(h.arrays[1], "shape", [2]))),
            ("partial fixed array", write([1], lambda h: setattr(h.arrays[1], "length", 0))),
            ("first array overflow", write(change=lambda h: setattr(h.arrays[0], "byteCount", 2 ** 63 - 1))),
            ("later array overflow", write([1], lambda h: setattr(h.arrays[1], "byteCount", 2 ** 63 - 1))),
            ("cumulative overflow", write([1], lambda h: (setattr(h.arrays[0], "byteCount", 2 ** 63 - 9),
                                                          setattr(h.arrays[1], "offset", 2 ** 63 - 1),
                                                          setattr(h.arrays[1], "byteCount", 1)))),
            ("negative byte count", write(change=lambda h: setattr(h.arrays[0], "byteCount", -1))),
            ("minimum byte count", write(change=lambda h: setattr(h.arrays[0], "byteCount", -(2 ** 63)))),
            ("past actual payload", write(change=lambda h: setattr(h.arrays[0], "byteCount", h.arrays[0].byteCount + 1))),
            ("incomplete coverage", write(change=lambda h: setattr(h.arrays[0], "byteCount", h.arrays[0].byteCount - 1))),
            ("gap before later array", write([1], lambda h: setattr(h.arrays[1], "offset", h.arrays[1].offset + 1))),
            ("overlap before later array", write([1], lambda h: setattr(h.arrays[1], "offset", h.arrays[1].offset - 1))),
            ("duplicate name", write([1], lambda h: setattr(h.arrays[1], "name", "tokens"))),
            ("maximum offset", write(change=lambda h: setattr(h.arrays[0], "offset", 2 ** 63 - 1))),
            ("sequence end overflow", write(change=lambda h: setattr(h, "sequences", [
                SequenceRecord("rows", "uint8", [1], 0, 2 ** 63 - 1, 1, [])]))),
        ]
        return identity, original, valid, invalid

    def test_valid_heads_read_and_invalid_are_file_errors(self, tmp_path):
        identity, original, valid, invalid = self.build(tmp_path)
        for name in valid:
            kind, e = pp.read_file(tmp_path, name)
            assert kind == "head" and len(e.tokens) == 1
        for label, name in invalid:
            with pytest.raises(FileError):
                pp.read_file(tmp_path, name)
        rep = pp.inspect(tmp_path)
        assert len(rep["states"]) == len(valid) and rep["unreadable_files"] == len(invalid)
        for name, data in original.items():
            assert (tmp_path / name).read_bytes() == data, f"inspection touched {name}"

    def test_startup_keeps_valid_and_removes_malformed(self, tmp_path):
        identity, original, valid, invalid = self.build(tmp_path)
        tier = PersistentPrefix(Configuration(tmp_path, 1 << 20, 1, None), identity, free_bytes_fn=BIG_FREE)
        assert tier.stored_states == len(valid) and tier.maintenance.unreadable == len(invalid)
        for name in valid:
            assert (tmp_path / name).read_bytes() == original[name]
        for label, name in invalid:
            assert not (tmp_path / name).exists(), label
        tier.close()


# ======================================================================================== synthetic engine state
LAYERS = Layout3(frozenset({0}), frozenset({3}), False)
CAP = 128


def fill(shape, axis, live, salt):
    a = np.zeros(shape, dtype=np.uint16)
    for i in range(live):
        idx = [slice(None)] * len(shape)
        idx[axis] = i
        a[tuple(idx)] = (salt * 131 + i * 17 + np.arange(a[tuple(idx)].size).reshape(a[tuple(idx)].shape)) % 65000 + 1
    return a


def make_view(n, draft=False, salt=1, pad=0):
    """A committed state of `n` tokens. Row i is a function of i and `salt` only, so a longer state equals its parent
    on the parent's rows, the way a decoded continuation does."""
    pooled = n // 4
    v = StateView(n, ngram_context=[5, 6], compact_state_windows=False,
                  linear=[pp.LinearRecord(0, [7, 8])],
                  attention=[pp.AttentionRecord(3, n, pp.IndexerRecord(n, 0, False, pooled, 4))])
    v.fixed["linear.0.conv"] = FixedArray("float32", (np.arange(12, dtype=np.float32).reshape(1, 3, 4) + n))
    v.fixed["linear.0.ssm"] = FixedArray("float32", np.full((2, 4, 4), float(n), dtype=np.float32))
    v.fixed["linear.0.ple"] = FixedArray("bfloat16", (np.arange(2, dtype=np.uint16).reshape(1, 2) + n))
    if pad:
        v.fixed["pad"] = FixedArray("uint8", np.full((pad,), 3, dtype=np.uint8))
    v.sequences["kv.3.keys"] = SeqArray("bfloat16", fill([1, 2, CAP, 8], 2, n, salt), 2, 0, n)
    v.sequences["kv.3.values"] = SeqArray("bfloat16", fill([1, 2, CAP, 8], 2, n, salt + 1), 2, 0, n)
    v.sequences["indexer.3.raw"] = SeqArray("bfloat16", fill([1, CAP, 8], 1, n, salt + 2), 1, 0, n)
    v.sequences["indexer.3.pooled"] = SeqArray("bfloat16", fill([1, CAP // 4, 8], 1, pooled, salt + 3), 1, 0, pooled)
    if draft:
        d = n - 1
        v.draft = pp.DraftRecord(d, pp.IndexerRecord(d, 0, False, d // 4, 4))
        v.sequences["draft.kv.keys"] = SeqArray("bfloat16", fill([1, 2, CAP, 8], 2, d, salt + 4), 2, 0, d)
        v.sequences["draft.kv.values"] = SeqArray("bfloat16", fill([1, 2, CAP, 8], 2, d, salt + 5), 2, 0, d)
        v.sequences["draft.indexer.raw"] = SeqArray("bfloat16", fill([1, CAP, 8], 1, d, salt + 6), 1, 0, d)
        v.sequences["draft.indexer.pooled"] = SeqArray("bfloat16", fill([1, CAP // 4, 8], 1, d // 4, salt + 7), 1, 0, d // 4)
        v.fixed["draft.last_multi"] = FixedArray("float32", np.ones((1, 1, 4), dtype=np.float32) * n)
    return v


def toks(n, start=100):
    return list(range(start, start + n))


def open_tier(path, identity=None, **kw):
    cfg = Configuration(path, kw.pop("max_bytes", 50_000_000), kw.pop("minimum_tokens", 1), kw.pop("max_age", None))
    return PersistentPrefix(cfg, identity or Identity({"fixture": "roundtrip"}), free_bytes_fn=BIG_FREE, **kw)


def assert_same_state(got, want, live_only=True):
    assert got.token_count == want.token_count and got.ngram_context == want.ngram_context
    assert got.linear == want.linear and got.attention == want.attention and got.draft == want.draft
    assert set(got.fixed) == set(want.fixed) and list(got.sequences) == list(want.sequences)
    for k, f in want.fixed.items():
        assert got.fixed[k].dtype == f.dtype and np.array_equal(got.fixed[k].data, f.data), k
    for k, s in want.sequences.items():
        g = got.sequences[k]
        assert g.data.shape == s.data.shape and (g.axis, g.base, g.live) == (s.axis, s.base, s.live), k
        sl = [slice(None)] * g.data.ndim
        sl[s.axis] = slice(0, s.live)
        assert np.array_equal(g.data[tuple(sl)], s.data[tuple(sl)]), k
        sl[s.axis] = slice(s.live, None)
        assert not g.data[tuple(sl)].any(), f"{k}: dead capacity must be zero"


def files(path, ext):
    return sorted(p.name for p in Path(path).iterdir() if p.name.endswith("." + ext))


# ========================================================================================= round trip and lineage
class TestRoundTrip:
    def test_save_then_restore_is_exact(self, tmp_path):
        tier = open_tier(tmp_path)
        v, t = make_view(40, draft=True), toks(40)
        r = tier.save(v, t)
        assert r.outcome == "saved" and r.reused_bytes == 0 and r.bytes > 0 and not r.compacted
        assert v.lineage is not None and v.lineage.head == pp.file_name(tier.identity.digest, t)
        e = tier.best_match(t + [999], require_draft=True)
        assert e is not None and e.tokens == t and e.has_draft
        got = tier.restore(e, layout=LAYERS, include_draft=True)
        assert got.tokens == 40 and got.bytes > 0
        assert_same_state(got.state, v)
        # a plain restore carries no draft rows
        plain = tier.restore(e, layout=LAYERS, include_draft=False)
        assert plain.state.draft is None and "draft.kv.keys" not in plain.state.sequences
        tier.close()

    def test_restore_survives_a_restart(self, tmp_path):
        v, t = make_view(40), toks(40)
        a = open_tier(tmp_path)
        a.save(v, t)
        a.close()
        b = open_tier(tmp_path)
        assert b.stored_states == 1 and b.maintenance.files == 0
        assert_same_state(b.restore(b.best_match(t + [1]), layout=LAYERS).state, v)
        b.close()

    def test_allocate_hook_provides_the_buffers(self, tmp_path):
        tier = open_tier(tmp_path)
        v, t = make_view(32), toks(32)
        tier.save(v, t)
        calls = []

        def allocate(shape, dtype):
            calls.append((tuple(shape), dtype))
            return np.full(shape, 9, dtype=pp.NUMPY_DTYPE[dtype])      # dirty memory: restore must zero dead capacity

        got = tier.restore(tier.best_match(t + [1]), allocate, layout=LAYERS)
        assert calls and ((1, 2, CAP, 8), "bfloat16") in calls
        assert_same_state(got.state, v)

    def test_lineage_save_writes_only_new_rows(self, tmp_path):
        tier = open_tier(tmp_path)
        a, ta = make_view(40), toks(40)
        first = tier.save(a, ta)
        assert len(files(tmp_path, "slotseg")) == 1
        seg_a = files(tmp_path, "slotseg")[0]
        b, tb = make_view(60), toks(60)
        b.lineage = a.lineage                                           # b is a's state grown by 20 tokens
        second = tier.save(b, tb)
        assert second.outcome == "saved" and second.reused_bytes > 0
        assert second.bytes < first.bytes
        segs = files(tmp_path, "slotseg")
        assert len(segs) == 2
        new_seg = next(s for s in segs if s != seg_a)
        entry_b = next(h for h in tier.indexed_entries if h.tokens == tb)
        kv = entry_b.sequences["kv.3.keys"]
        assert [(e.segment, e.start, e.end) for e in kv.extents] == [(seg_a, 0, 40), (new_seg, 40, 60)]
        assert tier.indexed_segments[new_seg].rows["kv.3.keys"].start == 40
        assert tier.json()["delta_saves"] == 1 and tier.json()["reused_bytes"] == second.reused_bytes
        # the continued head restores exactly, from rows in two segments
        assert_same_state(tier.restore(entry_b, layout=LAYERS).state, b)
        # the parent stays (its last turn can be regenerated); a third, longer save replaces it
        assert tier.stored_states == 2
        c, tc = make_view(80), toks(80)
        c.lineage = b.lineage
        tier.save(c, tc)
        assert {h.tokens[-1] for h in tier.indexed_entries} == {tb[-1], tc[-1]}
        assert not (tmp_path / pp.file_name(tier.identity.digest, ta)).exists()
        tier.close()

    def test_a_restored_state_continues_the_same_way(self, tmp_path):
        tier = open_tier(tmp_path)
        a, ta = make_view(40), toks(40)
        tier.save(a, ta)
        restored = tier.restore(tier.best_match(ta + [1]), layout=LAYERS).state
        grown = make_view(64)
        grown.lineage = restored.lineage
        r = tier.save(grown, toks(64))
        assert r.reused_bytes > 0

    def test_rows_are_not_matched_by_token_equality_alone(self, tmp_path):
        """The falsifier: identical token prefixes without a lineage bound to the head and the tier reuse nothing."""
        tier = open_tier(tmp_path)
        a, ta = make_view(40), toks(40)
        tier.save(a, ta)
        tb = toks(60)
        no_lineage = make_view(60)
        r1 = tier.save(no_lineage, tb)
        assert r1.reused_bytes == 0
        tier.clear()
        tier.save(make_view(40), ta)
        foreign = make_view(60)
        foreign.lineage = replace(a.lineage, tier="another-process")
        assert tier.save(foreign, tb).reused_bytes == 0
        tier.clear()
        tier.save(make_view(40), ta)
        wrong_head = make_view(60)
        diverged = ta[:10] + [7777] + ta[11:]                           # lineage head belongs to different ids
        lin = tier.indexed_entries[0]
        wrong_head.lineage = pp.Lineage(tier.instance, pp.file_name(tier.identity.digest, ta), 40, dict(lin.sequences))
        assert tier.save(wrong_head, diverged + toks(20, 5000)).reused_bytes == 0
        tier.close()

    def test_rewind_below_the_boundary_drops_the_lineage(self, tmp_path):
        tier = open_tier(tmp_path)
        a = make_view(40)
        tier.save(a, toks(40))
        a.drop_lineage_below(41)
        assert a.lineage is not None
        a.drop_lineage_below(30)
        assert a.lineage is None

    def test_compaction_past_the_segment_bound(self, tmp_path):
        tier = open_tier(tmp_path)
        tier.maximum_segments = 3
        prev, results = None, []
        for n in (20, 30, 40, 50, 60):
            v = make_view(n)
            v.lineage = prev.lineage if prev else None
            results.append(tier.save(v, toks(n)))
            prev = v
        assert [r.compacted for r in results] == [False, False, False, True, False]
        assert results[3].reused_bytes == 0
        assert tier.json()["compactions"] == 1
        tier.close()

    def test_same_ids_again_is_present(self, tmp_path):
        tier = open_tier(tmp_path)
        v, t = make_view(32, draft=True), toks(32)
        assert tier.save(v, t).outcome == "saved"
        assert tier.save(v, t).outcome == "present"
        # a plain state does not replace a head that has a draft, but a draft does replace a plain head
        plain_t = toks(32, 900)
        assert tier.save(make_view(32), plain_t).outcome == "saved"
        assert tier.save(make_view(32, draft=True), plain_t).outcome == "saved"
        tier.close()

    def test_short_states_and_bad_states_are_skipped_not_raised(self, tmp_path):
        tier = open_tier(tmp_path, minimum_tokens=64)
        assert tier.save(make_view(32), toks(32)).outcome.startswith("skipped: shorter than 64")
        t2 = open_tier(tmp_path / "b", minimum_tokens=1)
        v = make_view(32)
        v.recording = True
        assert t2.save(v, toks(32)).outcome.startswith("skipped")
        v = make_view(32)
        assert t2.save(v, toks(31)).outcome.startswith("skipped")         # not at a committed boundary
        v = make_view(32)
        v.committed = False
        assert t2.save(v, toks(32)).outcome.startswith("skipped")
        assert t2.save(make_view(32), [-1] + toks(31)).outcome.startswith("skipped")
        assert t2.stored_states == 0 and t2.json()["skipped_saves"] == 4

    def test_free_space_floor(self, tmp_path):
        tier = PersistentPrefix(Configuration(tmp_path, 50_000_000, 1, None), Identity({"f": "1"}),
                                free_bytes_fn=lambda: 1_000_000_000)
        r = tier.save(make_view(32), toks(32))
        assert r.outcome.startswith("skipped: less than") and tier.stored_states == 0

    def test_shared_prefixes_are_kept_when_a_conversation_extends_them(self, tmp_path):
        tier = open_tier(tmp_path)
        sys_t = toks(32)
        tier.save(make_view(32), sys_t, shared=True)
        conv = make_view(48)
        tier.save(conv, toks(48), splicing=toks(50))
        shared_entry = next(h for h in tier.indexed_entries if h.tokens == sys_t)
        assert shared_entry.shared and tier.stored_states == 2
        assert tier.json()["shared_saves"] == 1 and tier.stored_shared_states == 1
        assert tier.longest_extension(toks(40)) == toks(50)
        # rewriting a shared head without the flag keeps what is known about it
        tier.save(make_view(32), sys_t, prefill_chunk=256)
        assert next(h for h in tier.indexed_entries if h.tokens == sys_t).shared
        tier.close()

    def test_eviction_order_under_a_tight_quota(self, tmp_path):
        # Each file carries ~65 KB of container overhead in its estimate, so the states are padded to dominate it.
        tier = open_tier(tmp_path, max_bytes=900_000)
        for i, base in enumerate((100, 1000, 2000)):
            r = tier.save(make_view(40, salt=i, pad=300_000), toks(40, base))
            assert r.outcome == "saved", r.outcome
            time.sleep(0.01)
        assert tier.stored_states == 2
        assert tier.stored_bytes <= tier.configuration.max_bytes
        assert tier.json()["evictions"] >= 1
        assert not any(h.tokens[0] == 100 for h in tier.indexed_entries)      # the least recently used one-off went
        # a write larger than the whole quota is skipped, not forced
        r = tier.save(make_view(40, pad=2_000_000), toks(40, 7000))
        assert r.outcome.startswith("skipped: a ") and "quota" in r.outcome
        tier.close()


# ============================================================================================== corruption, I/O
class TestFailures:
    def two_states(self, tmp_path):
        tier = open_tier(tmp_path)
        a = make_view(40)
        tier.save(a, toks(40))
        b = make_view(60)
        b.lineage = a.lineage
        tier.save(b, toks(60))
        return tier, a, b

    def test_crc_corruption_deletes_the_head_and_its_dependents(self, tmp_path):
        tier, a, b = self.two_states(tmp_path)
        seg_a = next(iter(tier.indexed_entries[0].sequences["kv.3.keys"].extents)).segment
        path = tmp_path / seg_a
        raw = bytearray(path.read_bytes())
        raw[len(pp.MAGIC) + 3] ^= 0x55                                  # a payload byte of the segment both heads use
        path.write_bytes(bytes(raw))
        entry_b = next(h for h in tier.indexed_entries if len(h.tokens) == 60)
        with pytest.raises(FileError):
            tier.restore(entry_b, layout=LAYERS)
        assert tier.stored_states == 0 and files(tmp_path, "slotprefix") == []
        assert files(tmp_path, "slotseg") == [] and tier.json()["restore_failures"] == 1
        assert tier.json()["rejected_files"] == 2
        tier.close()

    def test_corrupt_head_payload_is_removed_at_restore(self, tmp_path):
        tier = open_tier(tmp_path)
        v, t = make_view(40), toks(40)
        tier.save(v, t)
        head = tmp_path / pp.file_name(tier.identity.digest, t)
        e = tier.indexed_entries[0]
        raw = bytearray(head.read_bytes())
        raw[len(pp.MAGIC) + 40 * 4 + 2] ^= 1                           # inside the first fixed array
        head.write_bytes(bytes(raw))
        with pytest.raises(FileError):
            tier.restore(e, layout=LAYERS)
        assert not head.exists() and tier.stored_states == 0
        tier.close()

    def test_file_changed_on_disk_and_layer_mismatch_are_format_errors(self, tmp_path):
        tier = open_tier(tmp_path)
        tier.save(make_view(40), toks(40))
        e = tier.indexed_entries[0]
        with pytest.raises(FileError, match="layer structure"):
            tier.restore(e, layout=Layout3(frozenset({1}), frozenset({3}), False))
        assert tier.stored_states == 0
        tier2 = open_tier(tmp_path / "x")
        tier2.save(make_view(40), toks(40))
        e2 = tier2.indexed_entries[0]
        with pytest.raises(FileError, match="committed boundary"):
            tier2.restore(e2, layout=Layout3(frozenset({0}), frozenset({3}), True))

    @pytest.mark.skipif(IS_ROOT, reason="permission failures require an unprivileged user")
    def test_eacces_at_restore_keeps_the_files(self, tmp_path):
        tier = open_tier(tmp_path)
        v, t = make_view(40), toks(40)
        tier.save(v, t)
        head = tmp_path / pp.file_name(tier.identity.digest, t)
        e = tier.indexed_entries[0]
        os.chmod(head, 0)
        try:
            with pytest.raises(pp.ReadFailure):
                tier.restore(e, layout=LAYERS)
        finally:
            os.chmod(head, 0o600)
        assert head.exists() and [h.file for h in tier.indexed_entries] == [head.name] and tier.stored_segments == 1
        assert_same_state(tier.restore(e, layout=LAYERS).state, v)
        tier.close()

    @pytest.mark.skipif(IS_ROOT, reason="permission failures require an unprivileged user")
    @pytest.mark.parametrize("which", ["head", "segment"])
    def test_unreadable_file_at_startup_refuses_the_directory_and_deletes_nothing(self, tmp_path, which):
        identity = Identity({"fixture": "roundtrip"})
        tier = open_tier(tmp_path, identity)
        t = toks(40)
        tier.save(make_view(40), t)
        tier.close()
        head = pp.file_name(identity.digest, t)
        seg_name = files(tmp_path, "slotseg")[0]
        victim = tmp_path / (head if which == "head" else seg_name)
        before = {n: (tmp_path / n).read_bytes() for n in (head, seg_name)}
        os.chmod(victim, 0)
        try:
            with pytest.raises(InaccessibleFile) as ei:
                open_tier(tmp_path, identity)
            assert ei.value.name == victim.name
            with pytest.raises(pp.ReadFailure):
                pp.read_file(tmp_path, victim.name)
        finally:
            os.chmod(victim, 0o600)
        assert {n: (tmp_path / n).read_bytes() for n in before} == before
        retry = open_tier(tmp_path, identity)                           # the failed open released the directory lock
        assert retry.stored_states == 1 and retry.maintenance.files == 0
        retry.close()

    def test_truncated_head_or_segment_is_removed_with_its_dependents(self, tmp_path):
        for kind in ("head", "segment"):
            d = tmp_path / kind
            identity = Identity({"fixture": "roundtrip"})
            tier = open_tier(d, identity)
            t = toks(40)
            tier.save(make_view(40), t)
            tier.close()
            name = pp.file_name(identity.digest, t) if kind == "head" else files(d, "slotseg")[0]
            (d / name).write_bytes(b"\x00")
            tier = open_tier(d, identity)
            assert tier.maintenance.unreadable == 1 and tier.maintenance.files == 2 and tier.stored_states == 0
            assert files(d, "slotprefix") == [] and files(d, "slotseg") == []
            tier.close()

    def test_a_missing_segment_removes_its_head(self, tmp_path):
        identity = Identity({"fixture": "roundtrip"})
        tier = open_tier(tmp_path, identity)
        tier.save(make_view(40), toks(40))
        tier.close()
        (tmp_path / files(tmp_path, "slotseg")[0]).unlink()
        with pytest.raises(FileError):
            pp.read_file(tmp_path, pp.file_name(identity.digest, toks(40))[:-9] + "0" * 0 + ".slotseg")
        tier = open_tier(tmp_path, identity)
        assert tier.maintenance.incomplete == 1 and tier.stored_states == 0
        tier.close()

    def test_identity_change_ignores_and_deletes_old_files(self, tmp_path):
        old = Identity({"fixture": "roundtrip", "gpu": "GB10"})
        tier = open_tier(tmp_path, old)
        tier.save(make_view(40), toks(40))
        tier.close()
        n_files = len(os.listdir(tmp_path))
        new = Identity({"fixture": "roundtrip", "gpu": "H100"})
        tier = open_tier(tmp_path, new)
        assert tier.stored_states == 0 and tier.maintenance.other_builds == 2
        assert files(tmp_path, "slotprefix") == [] and files(tmp_path, "slotseg") == []
        assert n_files - len(os.listdir(tmp_path)) == 2
        tier.close()

    def test_other_format_versions_are_removed_on_open(self, tmp_path):
        identity = Identity({"fixture": "roundtrip"})
        name = pp.file_name(identity.digest, [1])

        def header(placed):
            return json.dumps({"format": 2, "kind": "head", "identity": identity.digest}).encode()

        pp.write_container(tmp_path / name, [b"xx"], [2], header)
        tier = open_tier(tmp_path, identity)
        assert tier.maintenance.other_builds == 1 and not (tmp_path / name).exists()

    def test_expired_and_dangling_and_orphans_and_tmp_on_open(self, tmp_path):
        identity = Identity({"fixture": "roundtrip"})
        tier = open_tier(tmp_path, identity)
        tier.save(make_view(40), toks(40))
        tier.save(make_view(40, salt=2), toks(40, 5000))
        tier.close()
        old_head = tmp_path / pp.file_name(identity.digest, toks(40))
        past = time.time() - 40 * DAY
        os.utime(old_head, (past, past))
        (tmp_path / ".junk.tmp").write_bytes(b"half a write")
        orphan = tmp_path / ("e" * 32 + ".slotseg")
        seg0 = tmp_path / files(tmp_path, "slotseg")[0]
        orphan.write_bytes(seg0.read_bytes())                              # valid container, but no head uses it
        tier = open_tier(tmp_path, identity, max_age=30 * DAY)
        m = tier.maintenance
        assert m.expired == 1 and m.incomplete == 1 and m.orphan_segments >= 1
        assert not orphan.exists() and not (tmp_path / ".junk.tmp").exists()
        assert tier.stored_states == 1
        tier.close()

    def test_lowered_quota_applies_at_open(self, tmp_path):
        identity = Identity({"fixture": "roundtrip"})
        tier = open_tier(tmp_path, identity)
        a = tier.save(make_view(40), toks(40))
        tier.save(make_view(40, salt=2), toks(40, 5000))
        tier.close()
        tier = open_tier(tmp_path, identity, max_bytes=int(a.bytes * 1.5))
        assert tier.maintenance.over_quota == 1 and tier.stored_states == 1
        tier.close()


# ================================================================================================ the directory
class TestDirectory:
    def test_mode_and_lock(self, tmp_path):
        d = tmp_path / "prefix"
        tier = open_tier(d)
        assert stat.S_IMODE(os.stat(d).st_mode) == 0o700
        with pytest.raises(DirectoryInUse):
            open_tier(d)
        with pytest.raises(DirectoryInUse):
            pp.clear_directory(d)
        tier.close()
        open_tier(d).close()                                              # released on close

    def test_configuration_is_validated(self, tmp_path):
        with pytest.raises(ValueError):
            PersistentPrefix(Configuration(tmp_path, 0), Identity({}), free_bytes_fn=BIG_FREE)
        with pytest.raises(ValueError):
            PersistentPrefix(Configuration(tmp_path, 1, 0), Identity({}), free_bytes_fn=BIG_FREE)
        with pytest.raises(ValueError):
            PersistentPrefix(Configuration(tmp_path, 1, 1, -5.0), Identity({}), free_bytes_fn=BIG_FREE)

    def test_clear_directory_counts_and_preserves(self, tmp_path):
        states = {"head.slotprefix": 3, "rows.slotseg": 5, ".interrupted.tmp": 7}
        kept = ["notes.txt", "notes.tmp", ".keep", ".lock"]
        for n, size in states.items():
            (tmp_path / n).write_bytes(bytes(size))
        for n in kept:
            (tmp_path / n).write_bytes(b"keep")
        files_n, bytes_n = pp.clear_directory(tmp_path)
        assert (files_n, bytes_n) == (3, 15)
        assert set(os.listdir(tmp_path)) == set(kept)
        for n in kept:
            assert (tmp_path / n).read_bytes() == b"keep"

    def test_clear_refuses_a_held_lock_and_reports_failed_unlinks(self, tmp_path):
        head = tmp_path / "held.slotprefix"
        head.write_bytes(bytes(11))
        fd = os.open(tmp_path / ".lock", os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with pytest.raises(DirectoryInUse):
                pp.clear_directory(tmp_path)
            assert head.read_bytes() == bytes(11)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        assert pp.clear_directory(tmp_path) == (1, 11)
        blocked = tmp_path / "blocked.slotprefix"
        blocked.mkdir()                                                   # unlink cannot remove a directory
        head.write_bytes(bytes(17))
        with pytest.raises(pp.ReadFailure) as ei:
            pp.clear_directory(tmp_path)
        assert str(blocked) in str(ei.value) and "removed 1 prefix cache file but" in str(ei.value)
        assert blocked.exists() and not head.exists()
        blocked.rmdir()
        head.write_bytes(bytes(13))
        assert pp.clear_directory(tmp_path) == (1, 13)                     # the failure released the lock

    def test_instance_clear_removes_unusable_files_too(self, tmp_path):
        tier = open_tier(tmp_path)
        tier.save(make_view(40), toks(40))
        (tmp_path / "x.slotprefix").write_bytes(b"junk")
        assert tier.clear() == 3
        assert tier.stored_states == 0 and tier.stored_bytes == 0 and tier.stored_segments == 0
        assert not [n for n in os.listdir(tmp_path) if pp.is_state_file(n)]

    def test_remove_states_overlapping_a_conversation(self, tmp_path):
        tier = open_tier(tmp_path)
        a = make_view(32)
        tier.save(a, toks(32))
        b = make_view(48)
        b.lineage = a.lineage
        tier.save(b, toks(48))
        tier.save(make_view(32, salt=3), toks(32, 9000))
        assert tier.remove_states_overlapping([]) == 0
        assert tier.remove_states_overlapping(toks(48)) == 2
        assert tier.stored_states == 1 and tier.stored_segments == 1

    def test_inspect_reports_without_touching(self, tmp_path):
        tier = open_tier(tmp_path)
        tier.save(make_view(40), toks(40))
        before = {n: (tmp_path / n).read_bytes() for n in os.listdir(tmp_path) if pp.is_state_file(n)}
        rep = pp.inspect(tmp_path)
        assert rep["in_use"] is True and len(rep["states"]) == 1 and rep["segments"] == 1
        assert rep["states"][0]["tokens"] == 40 and rep["total_bytes"] == tier.stored_bytes
        assert {n: (tmp_path / n).read_bytes() for n in before} == before
        tier.close()
        assert pp.inspect(tmp_path)["in_use"] is False


# ====================================================================================== conversation ids (ported)
def legacy_head(tmp_path, identity, tokens):
    name = pp.file_name(identity.digest, tokens)

    def header(placed):
        off, count, crc = placed[0]
        return pp.encode_header(pp.Head(identity.digest, len(tokens),
                                        arrays=[pp.ArrayRecord("tokens", "int32", [len(tokens)], 0, len(tokens), off, count, crc)]))

    pp.write_container(tmp_path / name, [pp.token_bytes(tokens)], [len(tokens) * 4], header)
    return name


class TestConversationIds:
    def test_ids_attach_survive_restart_and_obey_the_quota(self, tmp_path):
        identity = Identity({"fixture": "conversation-ids"})
        tokens, conversation = [1, 2, 3], [1, 2, 3, 4, 5, 6]
        legacy_head(tmp_path, identity, tokens)
        cfg = Configuration(tmp_path, 1 << 20, 1)
        tier = PersistentPrefix(cfg, identity, free_bytes_fn=BIG_FREE)
        assert tier.remember_conversation(conversation)
        assert tier.longest_extension([1, 2, 3, 4]) == conversation
        assert tier.best_match(conversation).tokens == tokens                # resume stays at the numerical checkpoint
        assert not tier.remember_conversation([9, 8, 7])
        assert not tier.remember_conversation([1, 2, 3, -1])
        size = tier.stored_bytes
        tier.close()
        tier = PersistentPrefix(cfg, identity, free_bytes_fn=BIG_FREE)
        assert tier.longest_extension([1, 2, 3, 4]) == conversation
        assert tier.stored_bytes == size
        assert tier.indexed_entries[0].splicing_tokens == conversation
        assert tier.remember_conversation(conversation)                      # already there: nothing rewritten
        tier.close()
        tight = Configuration(tmp_path, size, 1)
        tier = PersistentPrefix(tight, identity, free_bytes_fn=BIG_FREE)
        assert not tier.remember_conversation(conversation + [7] * 1000)
        assert tier.longest_extension([1, 2, 3, 4]) == conversation
        assert tier.remove_states_overlapping(conversation) == 1
        assert tier.longest_extension([1, 2, 3, 4]) is None
        tier.close()


# ======================================================================================== removal failures (ported)
class TestRemovalFailures:
    def fixture(self, tmp_path):
        identity = Identity({"fixture": "removal-failures"})
        tokens, segment = [11], "a" * 32 + ".slotseg"
        head = pp.file_name(identity.digest, tokens)

        def head_header(placed):
            off, count, crc = placed[0]
            h = pp.Head(identity.digest, 1, arrays=[pp.ArrayRecord("tokens", "int32", [1], 0, 1, off, count, crc)],
                        sequences=[SequenceRecord("rows", "uint8", [1], 0, 0, 1, [Extent(segment, 0, 1)])],
                        sequenceBytes=1, residentBytes=1)
            return pp.encode_header(h)

        pp.write_container(tmp_path / head, [pp.token_bytes(tokens)], [4], head_header)

        def seg_header(placed):
            off, count, crc = placed[0]
            return pp.encode_header(pp.Segment(identity.digest, [RowsRecord("rows", "uint8", [], [], 0, 1, off, count, crc)]))

        pp.write_container(tmp_path / segment, [bytes([42])], [1], seg_header)
        return identity, tokens, head, segment

    def cfg(self, d):
        return Configuration(d, 1 << 20, 1, None)

    @pytest.mark.skipif(IS_ROOT, reason="permission failures require an unprivileged user")
    def test_startup_refuses_unsuccessful_cleanup(self, tmp_path):
        identity, tokens, head, segment = self.fixture(tmp_path)
        blocked = tmp_path / ".interrupted.tmp"
        blocked.mkdir()
        (blocked / "tiny").write_bytes(b"\x07")
        with pytest.raises(InaccessibleFile) as ei:
            PersistentPrefix(self.cfg(tmp_path), identity, free_bytes_fn=BIG_FREE)
        assert blocked.name in str(ei.value)
        assert (tmp_path / head).exists() and (tmp_path / segment).exists()
        (blocked / "tiny").unlink()
        blocked.rmdir()
        retry = PersistentPrefix(self.cfg(tmp_path), identity, free_bytes_fn=BIG_FREE)
        assert retry.stored_states == 1 and retry.maintenance.files == 0
        retry.close()

    @pytest.mark.skipif(IS_ROOT, reason="permission failures require an unprivileged user")
    @pytest.mark.parametrize("kind", ["head", "orphan", "clear"])
    def test_failed_deletes_stay_charged_and_recover(self, tmp_path, kind):
        identity, tokens, head, segment = self.fixture(tmp_path)
        tier = PersistentPrefix(self.cfg(tmp_path), identity, free_bytes_fn=BIG_FREE)
        events = []
        tier.on_event = events.append
        total = tier.stored_bytes
        seg_bytes = tier.indexed_segments[segment].bytes
        if kind == "orphan":
            (tmp_path / head).unlink()
        os.chmod(tmp_path, 0o500)
        try:
            removed = tier.clear() if kind == "clear" else tier.remove_states_overlapping(tokens)
            assert any("could not remove" in e for e in events)
            assert removed == (1 if kind == "orphan" else 0) or (kind == "orphan" and removed == 0)
            assert tier.stored_states == (0 if kind == "orphan" else 1)
            assert tier.stored_segments == 1
            assert tier.stored_bytes == (seg_bytes if kind == "orphan" else total)
            assert not tier.can_write(self.cfg(tmp_path).max_bytes - tier.stored_bytes + 1, "new-head")
            if kind != "orphan":
                assert tier.can_write(self.cfg(tmp_path).max_bytes - seg_bytes, head)
                assert not tier.can_write(self.cfg(tmp_path).max_bytes - seg_bytes + 1, head)
            assert tier.json()["removed_segments"] == 0
            assert (tmp_path / segment).exists()
        finally:
            os.chmod(tmp_path, 0o700)
        assert tier.clear() >= 1
        assert tier.stored_bytes == 0
        tier.close()

    def test_clearing_an_already_absent_head_forgets_it(self, tmp_path):
        identity, tokens, head, segment = self.fixture(tmp_path)
        tier = PersistentPrefix(self.cfg(tmp_path), identity, free_bytes_fn=BIG_FREE)
        (tmp_path / head).unlink()
        assert tier.clear() == 1                                           # only the segment file was really removed
        assert tier.stored_bytes == 0 and tier.stored_states == 0
        tier.close()
