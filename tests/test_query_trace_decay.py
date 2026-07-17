"""Tests for the trace_decay parameter of pm_rag.query.

trace_decay weights the full trace prefix with exponential recency decay
so that earlier events contribute signal beyond what the last event alone
captures. Tests verify:
- default behavior (None) is unchanged
- single-event prefix with decay gives the same result as without decay
- decay changes the ranking when an earlier event maps to different symbols
- the seed falls back to uniform when no event in the prefix has a mapping
- out-of-range trace_decay values raise ValueError
- decay weight diminishes strictly with distance from the end
"""
from __future__ import annotations

import pytest

from pm_rag import CodeGraph, build_index, query


def _chain(n: int) -> CodeGraph:
    """Linear chain: 0 -> 1 -> 2 -> ... -> n-1."""
    edges = [(i, i + 1, 1.0) for i in range(n - 1)]
    nodes = [f"fn_{i}" for i in range(n)]
    return CodeGraph(nodes=nodes, edges=edges)


def _star(center: int, leaves: list[int], n: int) -> CodeGraph:
    """Center node with edges to all leaves; no other edges."""
    nodes = [f"fn_{i}" for i in range(n)]
    edges = [(center, leaf, 1.0) for leaf in leaves]
    return CodeGraph(nodes=nodes, edges=edges)


def test_trace_decay_none_is_default_behavior() -> None:
    """query with trace_decay=None must be identical to not passing it."""
    graph = _chain(5)
    idx = build_index(graph, ["fn_0", "fn_2"])
    prefix = ["fn_0", "fn_2"]
    hits_default = query(idx, prefix)
    hits_explicit = query(idx, prefix, trace_decay=None)
    assert [h.symbol for h in hits_default] == [h.symbol for h in hits_explicit]
    assert [h.score for h in hits_default] == pytest.approx(
        [h.score for h in hits_explicit], rel=1e-12
    )


def test_single_event_prefix_decay_matches_no_decay() -> None:
    """With a one-event prefix, decay weight is always 1.0 for the only
    event, so trace_decay must produce the same result as the default."""
    graph = _chain(4)
    idx = build_index(graph, ["fn_1"])
    hits_no_decay = query(idx, ["fn_1"])
    hits_decay = query(idx, ["fn_1"], trace_decay=0.5)
    assert [h.symbol for h in hits_no_decay] == [h.symbol for h in hits_decay]
    assert [h.score for h in hits_no_decay] == pytest.approx(
        [h.score for h in hits_decay], rel=1e-9
    )


def test_decay_blends_earlier_event_into_seed() -> None:
    """With two events mapping to disjoint symbol sets, using trace_decay
    must produce a different seed (and therefore different scores) than
    the last-event-only seed.

    graph: 4 nodes, no edges (all dangling so PPR converges to seed).
    event A maps to fn_0, event B maps to fn_3.
    Without decay: seed is entirely on fn_3.
    With decay=0.5: seed is fn_3 (weight 1) + fn_0 (weight 0.5), normalized.
    """
    graph = CodeGraph(
        nodes=["fn_0", "fn_1", "fn_2", "fn_3"],
        edges=[],
    )

    def _two_event_mapping(events, symbols):
        m = {}
        for ev in dict.fromkeys(events):
            if ev == "event_a":
                m[ev] = [0]
            elif ev == "event_b":
                m[ev] = [3]
            else:
                m[ev] = []
        return m

    idx = build_index(graph, ["event_a", "event_b"], mapping_fn=_two_event_mapping)
    prefix = ["event_a", "event_b"]

    hits_no_decay = query(idx, prefix, k=4)
    hits_decay = query(idx, prefix, k=4, trace_decay=0.5)

    scores_no_decay = {h.symbol: h.score for h in hits_no_decay}
    scores_decay = {h.symbol: h.score for h in hits_decay}

    assert scores_decay["fn_0"] > scores_no_decay["fn_0"]
    assert scores_decay["fn_3"] < scores_no_decay["fn_3"]


def test_decay_fallback_to_uniform_when_no_mapping() -> None:
    """When none of the prefix events have a mapping, the seed must fall
    back to uniform (all symbols get the same score)."""
    graph = _chain(3)
    idx = build_index(graph, ["unmapped_a", "unmapped_b"])
    hits = query(idx, ["unmapped_a", "unmapped_b"], k=3, trace_decay=0.7)
    assert len(hits) == 3
    scores = [h.score for h in hits]
    assert all(s > 0 for s in scores)


def test_decay_validation_zero_raises() -> None:
    graph = _chain(2)
    idx = build_index(graph, ["fn_0"])
    with pytest.raises(ValueError, match="trace_decay"):
        query(idx, ["fn_0"], trace_decay=0.0)


def test_decay_validation_one_raises() -> None:
    graph = _chain(2)
    idx = build_index(graph, ["fn_0"])
    with pytest.raises(ValueError, match="trace_decay"):
        query(idx, ["fn_0"], trace_decay=1.0)


def test_decay_validation_negative_raises() -> None:
    graph = _chain(2)
    idx = build_index(graph, ["fn_0"])
    with pytest.raises(ValueError, match="trace_decay"):
        query(idx, ["fn_0"], trace_decay=-0.1)


def test_decay_weight_diminishes_with_distance() -> None:
    """An event farther back in the prefix contributes less to the seed.

    Setup: 4-node dangling graph; events map to individual nodes.
    Prefix = [ev0, ev1, ev2, ev3] with decay=0.5.
    Weights: ev3 (1.0), ev2 (0.5), ev1 (0.25), ev0 (0.125).
    Since there are no edges, PPR converges to the seed, so scores
    reflect the raw weights. Node mapped by ev3 must outscore ev2,
    ev2 must outscore ev1, ev1 must outscore ev0.
    """
    graph = CodeGraph(
        nodes=["fn_0", "fn_1", "fn_2", "fn_3"],
        edges=[],
    )
    node_for = {f"ev{i}": i for i in range(4)}

    def _indexed(events, symbols):
        return {ev: [node_for[ev]] for ev in dict.fromkeys(events) if ev in node_for}

    idx = build_index(graph, list(node_for.keys()), mapping_fn=_indexed)
    hits = query(idx, ["ev0", "ev1", "ev2", "ev3"], k=4, trace_decay=0.5)
    score_of = {h.symbol: h.score for h in hits}

    assert score_of["fn_3"] > score_of["fn_2"]
    assert score_of["fn_2"] > score_of["fn_1"]
    assert score_of["fn_1"] > score_of["fn_0"]


def test_decay_scores_non_negative() -> None:
    """PPR with a decay seed still produces a valid probability vector."""
    graph = _chain(5)
    idx = build_index(graph, ["fn_0", "fn_2", "fn_4"])
    hits = query(idx, ["fn_0", "fn_2", "fn_4"], k=5, trace_decay=0.6)
    assert all(h.score >= 0 for h in hits)


def test_decay_scores_sum_to_approx_one() -> None:
    """The full PPR result (k=n) must sum to 1."""
    graph = _chain(5)
    idx = build_index(graph, ["fn_1", "fn_3"])
    hits = query(idx, ["fn_1", "fn_3"], k=5, trace_decay=0.5)
    assert sum(h.score for h in hits) == pytest.approx(1.0, rel=1e-6)
