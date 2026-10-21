"""GraphSet.close() stops the token graph when one was built (carried minor of step 91/92)."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from slotstream_spark.runtime.graphs import GraphSet  # noqa: E402


def test_close_stops_the_token_graph_once():
    closed = []
    gs = SimpleNamespace(token_graph=SimpleNamespace(close=lambda: closed.append(1)))
    GraphSet.close(gs)
    GraphSet.close(gs)
    assert closed == [1] and gs.token_graph is None


def test_close_without_a_token_graph_is_a_no_op():
    GraphSet.close(SimpleNamespace())


def test_serve_closes_the_graphs_at_teardown():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "slotstream_spark" / "cli.py").read_text()
    assert src.index("server.run()") < src.index("graphs.close()") < src.index("sig.stop()")
