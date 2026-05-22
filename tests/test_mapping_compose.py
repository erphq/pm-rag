"""Tests for compose_mappings: strategy chaining and priority ordering.

These tests focus on compose_mappings behavior in isolation and in
combination. Interaction with individual strategies (regex, manual,
embedding, LLM) is covered in the dedicated per-strategy test files.
"""
from __future__ import annotations

from pm_rag.mapping import compose_mappings, manual_mapping, regex_mapping


def test_zero_strategies_returns_empty_for_all_events() -> None:
    composed = compose_mappings()
    m = composed(["ev"], ["a", "b"])
    assert m == {"ev": []}


def test_single_strategy_matches_direct_call() -> None:
    symbols = ["foo_handler", "bar_handler"]
    composed = compose_mappings(regex_mapping)
    direct = regex_mapping(["foo", "bar"], symbols)
    result = composed(["foo", "bar"], symbols)
    assert result == direct


def test_all_strategies_miss_returns_empty_list_per_event() -> None:
    composed = compose_mappings(regex_mapping)
    m = composed(["totally_unknown"], ["a", "b"])
    assert m == {"totally_unknown": []}


def test_second_fills_gap_third_result_not_used() -> None:
    symbols = ["a_handler", "b_handler", "c_handler"]
    manual1 = manual_mapping({"ev": ["b_handler"]})
    manual2 = manual_mapping({"ev": ["c_handler"]})
    composed = compose_mappings(regex_mapping, manual1, manual2)
    m = composed(["ev"], symbols)
    assert m["ev"] == [1]


def test_different_events_resolved_by_different_strategies() -> None:
    symbols = ["pay_handler", "legacy.old"]
    overrides = {"obscure": ["legacy.old"]}
    composed = compose_mappings(regex_mapping, manual_mapping(overrides))
    m = composed(["pay", "obscure", "no_match"], symbols)
    assert m["pay"] == [0]
    assert m["obscure"] == [1]
    assert m["no_match"] == []


def test_duplicate_events_collapsed_in_result() -> None:
    composed = compose_mappings(regex_mapping)
    m = composed(["x", "x", "x"], ["x_handler"])
    assert list(m.keys()) == ["x"]
    assert m["x"] == [0]


def test_empty_events_returns_empty_dict() -> None:
    composed = compose_mappings(regex_mapping)
    assert composed([], ["a", "b"]) == {}


def test_empty_symbols_all_strategies_miss() -> None:
    composed = compose_mappings(regex_mapping)
    m = composed(["ev"], [])
    assert m == {"ev": []}


def test_event_order_preserved_in_result_keys() -> None:
    symbols = ["c_fn", "a_fn", "b_fn"]
    overrides = {"alpha": ["a_fn"], "beta": ["b_fn"], "gamma": ["c_fn"]}
    composed = compose_mappings(manual_mapping(overrides))
    events = ["gamma", "alpha", "beta"]
    m = composed(events, symbols)
    assert list(m.keys()) == events
    assert m["gamma"] == [0]
    assert m["alpha"] == [1]
    assert m["beta"] == [2]
