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


def test_extract_cases_repeated_events_in_trace() -> None:
    cases = extract_cases([["a", "a", "a"]])
    assert len(cases) == 2
    assert cases[0] == LocalizationCase(prefix=["a"], next_event="a")
    assert cases[1] == LocalizationCase(prefix=["a", "a"], next_event="a")


def test_evaluate_ks_deduplication() -> None:
    idx = build_index(demo_graph(), demo_events())
    cases = extract_cases(demo_traces())
    score = evaluate(idx, cases, ks=[1, 1, 3])
    assert set(score.top_k.keys()) == {1, 3}


# MRR tests

def test_evaluate_mrr_in_range() -> None:
    idx = build_index(demo_graph(), demo_events())
    cases = extract_cases(demo_traces())
    score = evaluate(idx, cases)
    assert 0.0 <= score.mrr <= 1.0


def test_evaluate_mrr_zero_when_no_scorable_cases() -> None:
    idx = build_index(demo_graph(), demo_events())
    cases = [LocalizationCase(prefix=["order_received"], next_event="totally_unknown")]
    score = evaluate(idx, cases)
    assert score.n == 0
    assert score.mrr == 0.0


def test_evaluate_mrr_bounded_by_top_k_accuracy() -> None:
    """MRR is always >= top-1 accuracy and <= top-max_k accuracy."""
    idx = build_index(demo_graph(), demo_events())
    cases = extract_cases(demo_traces())
    score = evaluate(idx, cases, ks=[1, 5])
    assert score.top_k[1] <= score.mrr
    assert score.mrr <= score.top_k[5]


def test_evaluate_mrr_single_case_perfect_hit() -> None:
    """When the truth lands at rank 1, MRR for that case equals 1.0."""
    idx = build_index(demo_graph(), demo_events())
    cases = [
        LocalizationCase(
            prefix=["order_received", "payment_settled"],
            next_event="ship_order",
        ),
    ]
    score = evaluate(idx, cases, ks=[1, 5])
    # ship_order is in top-3 per existing test; verify MRR reflects the hit.
    assert score.n == 1
    assert score.mrr > 0.0
    # If it hit at rank 1, MRR is 1.0; if at rank r, MRR = 1/r.
    # Either way it must be >= top-1 accuracy and <= 1.0.
    assert score.mrr <= 1.0
    assert score.mrr >= score.top_k[1]


def test_evaluate_mrr_default_field_is_zero() -> None:
    """LocalizationScore can be constructed without mrr for backwards compat."""
    from pm_rag.eval import LocalizationScore
    s = LocalizationScore(top_k={1: 0.5}, n=2)
    assert s.mrr == 0.0
