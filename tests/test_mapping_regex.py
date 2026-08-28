"""Comprehensive tests for regex_mapping, including the top_k parameter."""
from __future__ import annotations

import pytest

from pm_rag import regex_mapping

# ---- basic behaviour (existing contract) ---------------------------------


def test_substring_match_is_case_insensitive() -> None:
    symbols = ["handlers.PAYMENT_SETTLED", "utils.money.format_amount"]
    m = regex_mapping(["payment_settled"], symbols)
    assert m["payment_settled"] == [0]


def test_no_match_yields_empty_list() -> None:
    m = regex_mapping(["no_such_event"], ["a.b.c", "x.y.z"])
    assert m["no_such_event"] == []


def test_duplicate_events_collapsed() -> None:
    m = regex_mapping(["pay", "pay", "pay"], ["pay_handler"])
    assert list(m.keys()) == ["pay"]
    assert m["pay"] == [0]


def test_regex_special_chars_in_event_are_escaped() -> None:
    # "a.b" must not match "axb" (dot is not a wildcard here)
    m = regex_mapping(["a.b"], ["a.b.handler", "axb_handler"])
    assert m["a.b"] == [0]
    assert 1 not in m["a.b"]


def test_single_event_matches_multiple_symbols() -> None:
    symbols = [
        "handlers.order.order_placed",
        "jobs.order.order_placed_audit",
        "tests.order_placed_mock",
        "utils.unrelated",
    ]
    m = regex_mapping(["order_placed"], symbols)
    assert m["order_placed"] == [0, 1, 2]


def test_multiple_events_independent_matches() -> None:
    symbols = [
        "handlers.payment.settled",
        "handlers.invoice.generated",
        "utils.shared_helper",
    ]
    m = regex_mapping(["settled", "generated"], symbols)
    assert m["settled"] == [0]
    assert m["generated"] == [1]


def test_empty_events_returns_empty_dict() -> None:
    m = regex_mapping([], ["a.b.c"])
    assert m == {}


def test_empty_symbols_all_events_map_to_empty() -> None:
    m = regex_mapping(["event_a", "event_b"], [])
    assert m == {"event_a": [], "event_b": []}


def test_event_matching_all_symbols() -> None:
    symbols = ["foo.bar", "foo.baz", "foo.qux"]
    m = regex_mapping(["foo"], symbols)
    assert m["foo"] == [0, 1, 2]


# ---- top_k parameter -----------------------------------------------------


def test_top_k_caps_number_of_results() -> None:
    symbols = [f"order_placed_handler_{i}" for i in range(10)]
    m = regex_mapping(["order_placed"], symbols, top_k=3)
    assert len(m["order_placed"]) == 3


def test_top_k_preserves_symbol_order() -> None:
    symbols = [f"evt_handler_{i}" for i in range(5)]
    m = regex_mapping(["evt"], symbols, top_k=3)
    # Indices must be the first three in ascending symbol-list order.
    assert m["evt"] == [0, 1, 2]


def test_top_k_larger_than_matches_returns_all() -> None:
    symbols = ["event_a_handler", "unrelated"]
    m = regex_mapping(["event_a"], symbols, top_k=100)
    assert m["event_a"] == [0]


def test_top_k_one_returns_single_result() -> None:
    symbols = ["pay_settle", "pay_settle_audit", "pay_settle_mock"]
    m = regex_mapping(["pay_settle"], symbols, top_k=1)
    assert m["pay_settle"] == [0]


def test_top_k_zero_raises_value_error() -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        regex_mapping(["x"], ["y"], top_k=0)


def test_top_k_negative_raises_value_error() -> None:
    with pytest.raises(ValueError):
        regex_mapping(["x"], ["y"], top_k=-1)


def test_top_k_none_is_default_no_cap() -> None:
    symbols = [f"settle_handler_{i}" for i in range(20)]
    m_default = regex_mapping(["settle"], symbols)
    m_explicit_none = regex_mapping(["settle"], symbols, top_k=None)
    assert m_default == m_explicit_none
    assert len(m_default["settle"]) == 20
