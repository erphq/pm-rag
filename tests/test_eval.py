import pytest

from pm_rag import (
    LocalizationCase,
    build_index,
    evaluate,
    extract_cases,
)
from pm_rag._demo import demo_events, demo_graph, demo_traces


def test_extract_cases_counts() -> None:
    traces = [["a", "b", "c"], ["x", "y"], ["solo"]]
    cases = extract_cases(traces)
    # length-3 trace → 2 cases, length-2 → 1, length-1 → 0
    assert len(cases) == 3
    assert cases[0] == LocalizationCase(prefix=["a"], next_event="b")
    assert cases[1] == LocalizationCase(prefix=["a", "b"], next_event="c")
    assert cases[2] == LocalizationCase(prefix=["x"], next_event="y")


def test_extract_cases_handles_empty() -> None:
    assert extract_cases([]) == []
    assert extract_cases([[]]) == []


def test_evaluate_demo_returns_known_shape() -> None:
    idx = build_index(demo_graph(), demo_events())
    cases = extract_cases(demo_traces())
    score = evaluate(idx, cases)
    assert score.n > 0
    assert set(score.top_k.keys()) == {1, 3, 5, 10}
    for v in score.top_k.values():
        assert 0.0 <= v <= 1.0
    # top-k accuracy is monotonic non-decreasing in k
    assert score.top_k[1] <= score.top_k[3] <= score.top_k[5] <= score.top_k[10]


def test_evaluate_skips_unmappable_truth() -> None:
    idx = build_index(demo_graph(), demo_events())
    cases = [
        LocalizationCase(prefix=["order_received"], next_event="totally_unknown"),
    ]
    score = evaluate(idx, cases)
    assert score.n == 0
    assert all(v == 0.0 for v in score.top_k.values())


def test_evaluate_finds_downstream_in_top3() -> None:
    """PPR seeded at payment_settled should rank the payment-fulfillment
    chain (ship_order / delivery_confirmed / allocate_inventory) high
    enough that ship_order - a strong downstream attractor - appears
    in the top 3."""
    idx = build_index(demo_graph(), demo_events())
    cases = [
        LocalizationCase(
            prefix=["order_received", "payment_settled"],
            next_event="ship_order",
        ),
    ]
    score = evaluate(idx, cases, ks=[3])
    assert score.n == 1
    assert score.top_k[3] == 1.0


def test_evaluate_validates_ks() -> None:
    idx = build_index(demo_graph(), demo_events())
    cases = extract_cases([["a", "b"]])
    with pytest.raises(ValueError):
        evaluate(idx, cases, ks=[])
    with pytest.raises(ValueError):
        evaluate(idx, cases, ks=[0])
    with pytest.raises(ValueError):
        evaluate(idx, cases, ks=[-1])


def test_demo_traces_round_trip_through_eval() -> None:
    """Smoke test: running eval against the demo never raises and
    produces non-trivial output."""
    idx = build_index(demo_graph(), demo_events())
    cases = extract_cases(demo_traces())
    score = evaluate(idx, cases)
    # Demo is constructed so PPR should localize *something* -
    # top-10 over a 10-node graph should include the truth often.
    assert score.top_k[10] > 0.5
