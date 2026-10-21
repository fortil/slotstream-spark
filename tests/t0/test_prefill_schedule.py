import json

import pytest

from slotstream_spark.engine import prefill_schedule as ps

P = 32_833_536


@pytest.fixture(autouse=True)
def _bootstrap(monkeypatch):
    monkeypatch.setenv("SLOTSTREAM_ENVELOPE", "bootstrap")


def test_bootstrap_product_is_the_swift_constant():
    assert ps.qk_product() == P == 4096 * 8016


def test_max_position_per_pass_size():
    want = {4096: 3920, 2048: 13984, 1024: 31040, 512: 63616, 256: 128000}
    for c, pos in want.items():
        assert ps.fits(c, pos), c
        assert not ps.fits(c, pos + 1), c
        assert c * (pos + c) <= P < c * (pos + 1 + c)


def test_chunk_halves_and_floors():
    assert ps.chunk(0, 4096) == 4096
    assert ps.chunk(3920, 4096) == 4096
    assert ps.chunk(3921, 4096) == 2048
    assert ps.chunk(13985, 4096) == 1024
    assert ps.chunk(31041, 4096) == 512
    assert ps.chunk(63617, 4096) == 256
    assert ps.chunk(128000, 4096) == 256
    assert ps.chunk(128001, 4096) == 128         # 256 no longer fits: the floor drops to 64 and the pass halves to what fits
    assert ps.chunk(256_385, 4096) == 64
    assert ps.chunk(ps.MODEL_LIMIT, 4096) == 0


def test_tail_of_3864_accepted_after_a_4096_pass():
    assert ps.next(3864, 4096, 4096, tail_aware=True) == 3864
    assert ps.next(3864, 4096, 4096, tail_aware=False) == 2048     # the hypothetical full pass halves
    assert ps.passes(4096 + 3864, 0, 4096) == [4096, 3864]


def test_passes_cover_exactly_the_tokens():
    for n in (1, 255, 256, 2048, 8000, 9000, 9215, 20000):
        got = ps.passes(n, 0, 4096)
        assert sum(got) == n
        pos = 0
        for c in got:
            assert ps.fits(c, pos), (n, pos, c)
            pos += c


def test_prompt_end_is_never_a_boundary():
    for n in (2048, 4096, 8192, 9000, 10000):
        b = ps.resume_boundaries(n, 1024)
        assert n not in b
        assert all(0 < x < n for x in b)
    assert ps.resume_boundaries(4096, 1024) == {1024, 2048, 3072}


def test_late_regime_is_not_a_boundary():
    n = 140_000
    b = ps.resume_boundaries(n, 256)
    assert b and max(b) <= 128_000


def test_last_pass_end():
    assert ps.last_pass_end(5000, 0, 9000, 1024) == 4096
    assert ps.last_pass_end(100, 0, 9000, 1024) == 0
    assert ps.last_pass_end(-1, 0, 9000, 1024) is None
    assert ps.last_pass_end(9000, 0, 9000, 1024) == 9000


def test_scope_caps():
    # four or more identical 256/512/1024 passes form a scope, 256-token passes up to 8192 tokens, others up to 4096
    s = ps.automatic_scope(100_000, 0, 256)
    assert s == [256] * 32 and sum(s) == 8192
    s = ps.automatic_scope(100_000, 0, 512)
    assert s == [512] * 8 and sum(s) == 4096
    s = ps.automatic_scope(100_000, 0, 1024)
    assert s == [1024] * 4 and sum(s) == 4096
    # max_scope below the cap bounds it, and below four passes there is no scope
    assert ps.automatic_scope(100_000, 0, 256, max_scope=2048) == [256] * 8
    assert ps.automatic_scope(100_000, 0, 1024, max_scope=2048) is None
    # big passes and short runs never form a scope
    assert ps.automatic_scope(100_000, 0, 4096) is None
    assert ps.automatic_scope(100_000, 0, 2048) is None
    assert ps.automatic_scope(3 * 1024, 0, 1024) is None


def test_scope_does_not_cross_a_size_change():
    # 1024-token passes stop being 1024 after position 31040: the run ends there
    s = ps.automatic_scope(100_000, 31_040 - 2048, 1024)
    assert s == [1024, 1024] or s is None


def test_envelope_json_is_required_without_bootstrap(monkeypatch, tmp_path):
    monkeypatch.delenv("SLOTSTREAM_ENVELOPE")
    with pytest.raises(ps.EnvelopeMissing):
        ps.qk_product(tmp_path / "nope.json")
    j = tmp_path / "e.json"
    j.write_text(json.dumps({"qk_product": 123456}))
    assert ps.qk_product(j) == 123456
    j.write_text(json.dumps({"bytes_per_pair": 1}))
    with pytest.raises(ps.EnvelopeMissing):
        ps.qk_product(j)


def test_group_scopes_keeps_every_pass_and_leaves_the_last_alone():
    sizes = [1024] * 9
    sc = ps.group_scopes(sizes, 0, 8192)
    assert [x for g in sc for x in g] == sizes
    assert sc == [[1024] * 4, [1024] * 4, [1024]]            # 4096-token cap, final pass alone
    assert ps.group_scopes(sizes, 0, 0) == [[1024]] * 9        # read scope off
    assert ps.group_scopes([4096, 2048, 2048], 0, 8192) == [[4096], [2048], [2048]]
    assert ps.group_scopes([256] * 40, 0, 8192)[0] == [256] * 32
    assert ps.group_scopes([256] * 40, 0, 2048)[0] == [256] * 8
    assert ps.group_scopes([512] * 3 + [777], 0, 8192) == [[512], [512], [512], [777]]


def test_group_scopes_ends_at_a_checkpoint_a_pass_ends_on():
    sizes = [1024] * 9
    sc = ps.group_scopes(sizes, 0, 8192, checkpoints=(2048,))
    assert sc[0] == [1024, 1024]
    assert [x for g in sc for x in g] == sizes
    # a checkpoint inside a pass changes nothing
    assert ps.group_scopes(sizes, 0, 8192, checkpoints=(1500,)) == ps.group_scopes(sizes, 0, 8192)
