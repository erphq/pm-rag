import pytest

from pm_rag import build_index, query
from pm_rag._demo import demo_events, demo_graph
from pm_rag.mapping import compose_mappings, manual_mapping, regex_mapping


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


def test_build_index_custom_mapping_fn_is_used() -> None:
    graph = demo_graph()
    events = demo_events()
    # A mapping that always returns only the first node for every event.
    def first_only(evs: list[str], syms: list[str]) -> dict[str, list[int]]:
        return {ev: [0] for ev in dict.fromkeys(evs)}

    idx = build_index(graph, events, mapping_fn=first_only)
    for ev in events:
        assert idx.mapping[ev] == [0]


def test_build_index_default_matches_regex_mapping() -> None:
    graph = demo_graph()
    events = demo_events()
    idx_default = build_index(graph, events)
    idx_explicit = build_index(graph, events, mapping_fn=regex_mapping)
    assert idx_default.mapping == idx_explicit.mapping


def test_build_index_with_composed_strategy() -> None:
    graph = demo_graph()
    # Add a custom event that regex cannot match.
    custom_event = "custom_checkout_started"
    first_node = graph.nodes[0]
    composed = compose_mappings(
        regex_mapping,
        manual_mapping({custom_event: [first_node]}),
    )
    idx = build_index(graph, [custom_event], mapping_fn=composed)
    # regex finds nothing; manual fallback resolves to index 0.
    assert idx.mapping[custom_event] == [0]


def test_build_index_mapping_fn_none_uses_regex() -> None:
    graph = demo_graph()
    events = ["payment_settled"]
    idx = build_index(graph, events, mapping_fn=None)
    # regex_mapping matches "payment_settled" against nodes containing that substring.
    assert any("payment_settled" in graph.nodes[i] for i in idx.mapping["payment_settled"])
