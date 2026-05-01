import pytest

from pm_rag import build_index, query
from pm_rag._demo import demo_events, demo_graph


def test_query_at_payment_settled_ranks_settled_handler_high() -> None:
    idx = build_index(demo_graph(), demo_events())
    hits = query(idx, ["order_received", "payment_settled"], k=5)
    assert any("payment_settled" in h.symbol for h in hits[:3])


def test_query_unknown_event_falls_back_uniform() -> None:
    idx = build_index(demo_graph(), demo_events())
    hits = query(idx, ["mystery_event"], k=3)
    assert len(hits) == 3
    # uniform seed → no single overwhelming winner
    assert hits[0].score < 0.9


def test_query_empty_prefix_raises() -> None:
    idx = build_index(demo_graph(), demo_events())
    with pytest.raises(ValueError):
        query(idx, [])


def test_query_returns_top_k() -> None:
    idx = build_index(demo_graph(), demo_events())
    hits = query(idx, ["payment_settled"], k=4)
    assert len(hits) == 4
    # Scores are non-increasing.
    for a, b in zip(hits, hits[1:], strict=False):
        assert a.score >= b.score


def test_unrelated_noise_node_ranks_low() -> None:
    idx = build_index(demo_graph(), demo_events())
    hits = query(idx, ["payment_settled"], k=10)
    ranks = {h.symbol: i for i, h in enumerate(hits)}
    # `format_amount` and `audit_event` are unrelated noise nodes
    assert ranks["utils.money.format_amount"] >= 5
