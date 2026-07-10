"""Tests verifying the alpha restart-probability parameter in query() and evaluate().

alpha controls the PPR restart probability: higher values concentrate
more mass at the seed node, lower values let the walk diffuse further.
These tests verify that alpha is correctly propagated from the public
query() and evaluate() APIs into the underlying diffusion engine.
"""
from __future__ import annotations

import pytest

from pm_rag import (
    CodeGraph,
    build_index,
    evaluate,
    extract_cases,
    query,
)
from pm_rag._demo import demo_events, demo_graph, demo_traces


def _chain_graph() -> CodeGraph:
    """Three-node linear chain: seed_fn -> mid_fn -> tail_fn."""
    return CodeGraph(
        nodes=["seed_fn", "mid_fn", "tail_fn"],
        edges=[(0, 1, 1.0), (1, 2, 1.0)],
    )


def _seed_only(events, symbols):
    """Stub: every event maps to symbol index 0 (the seed node)."""
    return {ev: [0] for ev in dict.fromkeys(events)}


def test_query_higher_alpha_concentrates_at_seed() -> None:
    """Higher restart probability keeps more mass at the seed node."""
    idx = build_index(_chain_graph(), ["ev"], mapping_fn=_seed_only)
    low = {h.symbol: h.score for h in query(idx, ["ev"], alpha=0.05)}
    high = {h.symbol: h.score for h in query(idx, ["ev"], alpha=0.9)}
    assert high["seed_fn"] > low["seed_fn"]


def test_query_lower_alpha_diffuses_to_downstream() -> None:
    """Lower restart probability lets more mass reach downstream nodes."""
    idx = build_index(_chain_graph(), ["ev"], mapping_fn=_seed_only)
    low = {h.symbol: h.score for h in query(idx, ["ev"], alpha=0.05)}
    high = {h.symbol: h.score for h in query(idx, ["ev"], alpha=0.9)}
    assert low["tail_fn"] > high["tail_fn"]


def test_query_explicit_default_alpha_matches_implicit() -> None:
    """query() with explicit alpha=0.15 matches the default."""
    idx = build_index(_chain_graph(), ["ev"], mapping_fn=_seed_only)
    default_hits = query(idx, ["ev"])
    explicit_hits = query(idx, ["ev"], alpha=0.15)
    assert [h.symbol for h in default_hits] == [h.symbol for h in explicit_hits]
    for d, e in zip(default_hits, explicit_hits, strict=True):
        assert pytest.approx(d.score, rel=1e-9) == e.score


def test_query_scores_sum_to_one_regardless_of_alpha() -> None:
    """PPR output is always a valid probability vector (L1 norm = 1)."""
    idx = build_index(_chain_graph(), ["ev"], mapping_fn=_seed_only)
    for alpha in (0.01, 0.15, 0.5, 0.99):
        hits = query(idx, ["ev"], alpha=alpha)
        total = sum(h.score for h in hits)
        assert pytest.approx(total, rel=1e-6) == 1.0


def test_query_seed_ranks_first_at_high_alpha() -> None:
    """At high alpha the walk mostly restarts, concentrating mass at the seed."""
    idx = build_index(_chain_graph(), ["ev"], mapping_fn=_seed_only)
    for alpha in (0.5, 0.99):
        hits = query(idx, ["ev"], alpha=alpha)
        assert hits[0].symbol == "seed_fn"


def test_query_tail_accumulates_mass_at_low_alpha() -> None:
    """At very low alpha the walk rarely restarts; mass accumulates at the
    dangling tail node because it has no outgoing edges to redistribute it."""
    idx = build_index(_chain_graph(), ["ev"], mapping_fn=_seed_only)
    for alpha in (0.01, 0.1):
        hits = query(idx, ["ev"], alpha=alpha)
        assert hits[0].symbol == "tail_fn"


def test_evaluate_explicit_default_alpha_matches_implicit() -> None:
    """evaluate() with explicit alpha=0.15 matches the default."""
    idx = build_index(demo_graph(), demo_events())
    cases = extract_cases(demo_traces())
    s_default = evaluate(idx, cases)
    s_explicit = evaluate(idx, cases, alpha=0.15)
    assert s_default.top_k == s_explicit.top_k
    assert pytest.approx(s_default.mrr, rel=1e-9) == s_explicit.mrr


def test_evaluate_extreme_alpha_values_produce_valid_scores() -> None:
    """evaluate() at alpha extremes returns valid n and in-range scores."""
    idx = build_index(demo_graph(), demo_events())
    cases = extract_cases(demo_traces())
    for alpha in (0.01, 0.99):
        score = evaluate(idx, cases, alpha=alpha)
        assert score.n > 0
        for v in score.top_k.values():
            assert 0.0 <= v <= 1.0
        assert 0.0 <= score.mrr <= 1.0
