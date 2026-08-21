"""Tests for pm_rag.adapters.traces_from_json."""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import pytest

from pm_rag.adapters import traces_from_json


def _jio(data) -> io.StringIO:
    return io.StringIO(json.dumps(data))


# list-of-lists format


def test_list_of_lists_basic() -> None:
    traces = traces_from_json(_jio([["order_received", "payment_settled"], ["login"]]))
    assert traces == [["order_received", "payment_settled"], ["login"]]


def test_list_of_lists_empty_strings_skipped() -> None:
    traces = traces_from_json(_jio([["ev1", "", "  ", "ev2"]]))
    assert traces == [["ev1", "ev2"]]


def test_list_of_lists_non_string_element_raises() -> None:
    with pytest.raises(ValueError, match="not a string"):
        traces_from_json(_jio([["ev1", 42]]))


def test_list_of_lists_empty_list_returns_empty() -> None:
    assert traces_from_json(_jio([])) == []


# list-of-records format


def test_records_two_cases_sorted_by_timestamp() -> None:
    data = [
        {"case_id": "c1", "activity": "order_received", "timestamp": "10:00"},
        {"case_id": "c1", "activity": "payment_settled", "timestamp": "10:05"},
        {"case_id": "c2", "activity": "login", "timestamp": "11:00"},
    ]
    traces = traces_from_json(_jio(data))
    assert len(traces) == 2
    assert traces[0] == ["order_received", "payment_settled"]
    assert traces[1] == ["login"]


def test_records_out_of_order_reordered_by_sort_col() -> None:
    data = [
        {"case_id": "c1", "activity": "second", "timestamp": "10:05"},
        {"case_id": "c1", "activity": "first", "timestamp": "10:00"},
    ]
    assert traces_from_json(_jio(data))[0] == ["first", "second"]


def test_records_sort_by_none_preserves_document_order() -> None:
    data = [
        {"case_id": "c1", "activity": "second", "timestamp": "10:05"},
        {"case_id": "c1", "activity": "first", "timestamp": "10:00"},
    ]
    assert traces_from_json(_jio(data), sort_by=None)[0] == ["second", "first"]


def test_records_empty_activity_skipped() -> None:
    data = [
        {"case_id": "c1", "activity": "ev1"},
        {"case_id": "c1", "activity": ""},
        {"case_id": "c1", "activity": "ev2"},
    ]
    assert traces_from_json(_jio(data), sort_by=None) == [["ev1", "ev2"]]


def test_records_custom_column_names() -> None:
    data = [
        {"pid": "p1", "event": "login"},
        {"pid": "p1", "event": "checkout"},
        {"pid": "p2", "event": "login"},
    ]
    traces = traces_from_json(_jio(data), case_col="pid", activity_col="event", sort_by=None)
    assert traces == [["login", "checkout"], ["login"]]


def test_records_missing_case_col_raises() -> None:
    with pytest.raises(KeyError):
        traces_from_json(_jio([{"activity": "ev1"}]), case_col="case_id")


# file path support


def test_accepts_path_and_string_path() -> None:
    content = json.dumps([["order_received", "payment_settled"]])
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(content)
        path = tmp.name
    try:
        assert traces_from_json(path) == [["order_received", "payment_settled"]]
        assert traces_from_json(Path(path)) == [["order_received", "payment_settled"]]
    finally:
        Path(path).unlink()


# error cases


def test_non_list_root_raises() -> None:
    with pytest.raises(ValueError, match="root must be a list"):
        traces_from_json(_jio({"case_id": "c1"}))


def test_invalid_json_raises() -> None:
    with pytest.raises(json.JSONDecodeError):
        traces_from_json(io.StringIO("{not valid json"))


# compatibility


def test_output_compatible_with_extract_cases() -> None:
    from pm_rag import extract_cases

    traces = traces_from_json(
        _jio([
            ["order_received", "payment_settled", "ship_order"],
            ["order_received", "payment_settled"],
        ])
    )
    cases = extract_cases(traces)
    assert len(cases) == 3
    assert cases[0].next_event == "payment_settled"
    assert cases[1].next_event == "ship_order"
