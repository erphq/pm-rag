"""Tests for the manual / YAML-override mapping strategy."""
from __future__ import annotations

from pm_rag.mapping import compose_mappings, manual_mapping, regex_mapping


def test_basic_override_resolves_to_correct_indices() -> None:
    symbols = [
        "handlers.payment.settled",
        "handlers.invoice.generate",
        "utils.money.format",
    ]
    strategy = manual_mapping({"payment_settled": ["handlers.payment.settled"]})
    m = strategy(["payment_settled"], symbols)
    assert m["payment_settled"] == [0]


def test_multiple_symbols_per_event() -> None:
    symbols = ["a", "b", "c"]
    strategy = manual_mapping({"ev": ["c", "a"]})
    m = strategy(["ev"], symbols)
    assert m["ev"] == [2, 0]


def test_unknown_symbol_name_silently_ignored() -> None:
    symbols = ["a", "b"]
    strategy = manual_mapping({"ev": ["a", "renamed_or_removed", "b"]})
    m = strategy(["ev"], symbols)
    assert m["ev"] == [0, 1]


def test_event_not_in_overrides_maps_to_empty() -> None:
    strategy = manual_mapping({"other": ["a"]})
    m = strategy(["missing_event"], ["a", "b"])
    assert m["missing_event"] == []


def test_empty_overrides_dict() -> None:
    strategy = manual_mapping({})
    m = strategy(["ev"], ["a", "b"])
    assert m["ev"] == []


def test_duplicate_events_are_collapsed() -> None:
    symbols = ["x_handler"]
    strategy = manual_mapping({"x": ["x_handler"]})
    m = strategy(["x", "x", "x"], symbols)
    assert list(m.keys()) == ["x"]
    assert m["x"] == [0]


def test_empty_symbol_list() -> None:
    strategy = manual_mapping({"ev": ["a"]})
    m = strategy(["ev"], [])
    assert m["ev"] == []


def test_order_of_overrides_is_preserved() -> None:
    symbols = ["c", "a", "b"]
    strategy = manual_mapping({"ev": ["c", "b", "a"]})
    m = strategy(["ev"], symbols)
    assert m["ev"] == [0, 2, 1]


def test_composes_under_compose_mappings_as_fallback() -> None:
    symbols = ["payment_settled_handler", "legacy.obscure_event_fn"]
    overrides = {"obscure_event": ["legacy.obscure_event_fn"]}
    composed = compose_mappings(regex_mapping, manual_mapping(overrides))
    m = composed(["payment_settled", "obscure_event"], symbols)
    # regex hits index 0 for payment_settled; manual picks up obscure_event
    assert m["payment_settled"] == [0]
    assert m["obscure_event"] == [1]


def test_regex_wins_when_it_matches_even_if_manual_also_has_entry() -> None:
    symbols = ["payment_settled_handler", "other"]
    overrides = {"payment_settled": ["other"]}
    composed = compose_mappings(regex_mapping, manual_mapping(overrides))
    m = composed(["payment_settled"], symbols)
    # regex matches index 0; manual override for index 1 is never reached
    assert m["payment_settled"] == [0]
