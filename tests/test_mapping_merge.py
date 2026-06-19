"""Tests for merge_mappings: union-based strategy combination."""
from __future__ import annotations

from pm_rag.mapping import compose_mappings, manual_mapping, merge_mappings, regex_mapping


def test_zero_strategies_returns_empty_per_event() -> None:
    merged = merge_mappings()
    m = merged(["ev"], ["a", "b"])
    assert m == {"ev": []}


def test_single_strategy_matches_direct_call() -> None:
    symbols = ["foo_handler", "bar_handler"]
    merged = merge_mappings(regex_mapping)
    direct = regex_mapping(["foo", "bar"], symbols)
    result = merged(["foo", "bar"], symbols)
    assert result == direct


def test_union_combines_results_from_all_strategies() -> None:
    # regex matches "ev" in index 0; manual adds index 1 which regex misses
    symbols = ["ev_handler", "separate_manual_only"]
    manual_extra = manual_mapping({"ev": ["separate_manual_only"]})
    merged = merge_mappings(regex_mapping, manual_extra)
    m = merged(["ev"], symbols)
    assert set(m["ev"]) == {0, 1}


def test_union_deduplicates_when_strategies_overlap() -> None:
    symbols = ["a", "b"]
    manual1 = manual_mapping({"ev": ["a", "b"]})
    manual2 = manual_mapping({"ev": ["a"]})
    merged = merge_mappings(manual1, manual2)
    m = merged(["ev"], symbols)
    assert m["ev"] == [0, 1]  # "a" (idx 0) appears in both but deduplicated


def test_first_strategy_result_comes_first_in_output() -> None:
    # a_fn (idx 1) comes from strategy 1; b_fn (idx 0) comes from strategy 2
    symbols = ["b_fn", "a_fn"]
    manual1 = manual_mapping({"ev": ["a_fn"]})
    manual2 = manual_mapping({"ev": ["b_fn"]})
    merged = merge_mappings(manual1, manual2)
    m = merged(["ev"], symbols)
    assert m["ev"] == [1, 0]


def test_merge_vs_compose_contrast() -> None:
    """compose_mappings stops at first hit; merge_mappings unions all hits."""
    symbols = ["ev_handler", "separate_manual_only"]
    manual_extra = manual_mapping({"ev": ["separate_manual_only"]})
    composed = compose_mappings(regex_mapping, manual_extra)
    merged = merge_mappings(regex_mapping, manual_extra)
    m_compose = composed(["ev"], symbols)
    m_merge = merged(["ev"], symbols)
    # compose stops after regex finds index 0; manual (index 1) is never reached
    assert m_compose["ev"] == [0]
    # merge unions both: regex gives [0], manual gives [1]
    assert set(m_merge["ev"]) == {0, 1}


def test_duplicate_events_collapsed() -> None:
    merged = merge_mappings(regex_mapping)
    m = merged(["x", "x", "x"], ["x_handler"])
    assert list(m.keys()) == ["x"]
    assert m["x"] == [0]


def test_empty_events_returns_empty_dict() -> None:
    merged = merge_mappings(regex_mapping)
    assert merged([], ["a", "b"]) == {}


def test_empty_symbols_all_strategies_miss() -> None:
    merged = merge_mappings(regex_mapping)
    m = merged(["ev"], [])
    assert m == {"ev": []}


def test_all_strategies_miss_yields_empty_list() -> None:
    merged = merge_mappings(regex_mapping)
    m = merged(["totally_unknown"], ["a", "b"])
    assert m["totally_unknown"] == []


def test_event_order_preserved_in_result_keys() -> None:
    overrides = {"gamma": ["c_fn"], "alpha": ["a_fn"], "beta": ["b_fn"]}
    symbols = ["c_fn", "a_fn", "b_fn"]
    merged = merge_mappings(manual_mapping(overrides))
    events = ["gamma", "alpha", "beta"]
    m = merged(events, symbols)
    assert list(m.keys()) == events
    assert m["gamma"] == [0]
    assert m["alpha"] == [1]
    assert m["beta"] == [2]
