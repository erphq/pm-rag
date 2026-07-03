"""Edge-case tests for pm_rag.query.

Covers behavior not exercised by the main index tests: k > graph size,
prefix-uses-last-event-only, all-dangling graphs, and single-node graphs.
"""
from __future__ import annotations

import pytest

from pm_rag import CodeGraph, build_index, query
from pm_rag._demo import demo_events, demo_graph


def _always_first(events, symbols):
    """Mapping stub: every event maps to symbol index 0."""
    return {ev: [0] for ev in dict.fromkeys(events)}


def _tiny_graph() -> CodeGraph:
    """Three-node linear chain: 0 -> 1 -> 2."""
    return CodeGraph(
        nodes=["fn_a", "fn_b", "fn_c"],
        edges=[(0, 1, 1.0), (1, 2, 1.0)],
    )


def test_query_k_larger_than_graph_returns_all_nodes() -> None:
    """query(k=100) on a 3-node graph returns exactly 3 hits, not 100."""
    idx = build_index(_tiny_graph(), ["fn_a"], mapping_fn=_always_first)
    hits = query(idx, ["fn_a"], k=100)
    assert len(hits) == 3
    assert {h.symbol for h in hits} == {"fn_a", "fn_b", "fn_c"}


def test_query_uses_only_last_event_in_prefix() -> None:
    """A longer prefix produces the same result as a single-event prefix
    when the last event is the same.

    query() seeds PPR from trace_prefix[-1] only, so prepending earlier
    events to the prefix must not change the ranking or scores.
    """
    idx = build_index(demo_graph(), demo_events())
    hits_short = query(idx, ["payment_settled"], k=5)
    hits_long = query(idx, ["order_received", "payment_pending", "payment_settled"], k=5)
    assert [h.symbol for h in hits_short] == [h.symbol for h in hits_long]
    assert [h.score for h in hits_short] == pytest.approx(
        [h.score for h in hits_long], rel=1e-9
    )


def test_query_all_dangling_graph_seed_node_ranks_first() -> None:
    """PPR on a graph with no edges converges back to the seed vector.

    When p_t is all-zero, r_new = alpha * s + 0, which normalizes to s
    on the first iteration. The seed node therefore scores highest.
    """
    graph = CodeGraph(nodes=["entry", "other", "unrelated"], edges=[])
    idx = build_index(graph, ["my_event"], mapping_fn=_always_first)
    hits = query(idx, ["my_event"], k=3)
    assert hits[0].symbol == "entry"
    assert hits[0].score > hits[1].score


def test_query_single_node_graph_returns_one_hit() -> None:
    """A graph with one node always returns exactly one hit with score 1.0."""
    graph = CodeGraph(nodes=["only_fn"], edges=[])
    idx = build_index(graph, ["ev"], mapping_fn=_always_first)
    hits = query(idx, ["ev"], k=10)
    assert len(hits) == 1
    assert hits[0].symbol == "only_fn"
    assert pytest.approx(hits[0].score, rel=1e-6) == 1.0


def test_query_scores_non_negative_across_events() -> None:
    """PPR produces a probability vector; every returned score is >= 0."""
    idx = build_index(demo_graph(), demo_events())
    for event in demo_events():
        for h in query(idx, [event], k=5):
            assert h.score >= 0.0


def test_query_scores_non_increasing() -> None:
    """Hits are returned in non-increasing score order."""
    idx = build_index(demo_graph(), demo_events())
    hits = query(idx, ["payment_settled"], k=10)
    for a, b in zip(hits, hits[1:], strict=False):
        assert a.score >= b.score
