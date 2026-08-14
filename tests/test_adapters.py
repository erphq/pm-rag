"""Tests for pm_rag.adapters.traces_from_csv."""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytest

from pm_rag.adapters import traces_from_csv


def _csv(*lines: str) -> io.StringIO:
    return io.StringIO("\n".join(lines))


def test_two_cases_sorted_by_timestamp() -> None:
    f = _csv(
        "case_id,activity,timestamp",
        "c1,order_received,2024-01-01T10:00",
        "c1,payment_settled,2024-01-01T10:05",
        "c2,order_received,2024-01-01T11:00",
    )
    traces = traces_from_csv(f)
    assert len(traces) == 2
    assert traces[0] == ["order_received", "payment_settled"]
    assert traces[1] == ["order_received"]


def test_out_of_order_rows_reordered_by_sort_col() -> None:
    f = _csv(
        "case_id,activity,timestamp",
        "c1,second_event,2024-01-01T10:05",
        "c1,first_event,2024-01-01T10:00",
    )
    assert traces_from_csv(f)[0] == ["first_event", "second_event"]


def test_sort_by_none_preserves_csv_row_order() -> None:
    f = _csv(
        "case_id,activity,timestamp",
        "c1,second_event,2024-01-01T10:05",
        "c1,first_event,2024-01-01T10:00",
    )
    assert traces_from_csv(f, sort_by=None)[0] == ["second_event", "first_event"]


def test_empty_csv_returns_empty() -> None:
    assert traces_from_csv(_csv("case_id,activity,timestamp")) == []


def test_empty_and_whitespace_activities_skipped() -> None:
    f = _csv(
        "case_id,activity,timestamp",
        "c1,order_received,1",
        "c1,,2",
        "c1,   ,3",
        "c1,payment_settled,4",
    )
    assert traces_from_csv(f) == [["order_received", "payment_settled"]]


def test_case_with_all_empty_activities_omitted() -> None:
    f = _csv("case_id,activity", "c1,", "c1, ")
    assert traces_from_csv(f, sort_by=None) == []


def test_custom_column_names() -> None:
    f = _csv("pid,event", "p1,login", "p1,checkout", "p2,login")
    traces = traces_from_csv(f, case_col="pid", activity_col="event", sort_by=None)
    assert traces == [["login", "checkout"], ["login"]]


def test_case_order_follows_first_appearance() -> None:
    f = _csv("case_id,activity", "z,ev_z", "a,ev_a")
    traces = traces_from_csv(f, sort_by=None)
    assert traces[0] == ["ev_z"]
    assert traces[1] == ["ev_a"]


def test_missing_sort_by_column_raises() -> None:
    with pytest.raises(ValueError, match="timestamp"):
        traces_from_csv(_csv("case_id,activity", "c1,ev"), sort_by="timestamp")


def test_missing_case_col_raises_key_error() -> None:
    with pytest.raises(KeyError):
        traces_from_csv(_csv("activity", "order_received"), case_col="case_id")


def test_accepts_path_and_string_path() -> None:
    content = "case_id,activity\nc1,order_received\nc1,payment_settled\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(content)
        path = tmp.name
    try:
        assert traces_from_csv(path, sort_by=None) == [["order_received", "payment_settled"]]
        assert traces_from_csv(Path(path), sort_by=None) == [["order_received", "payment_settled"]]
    finally:
        Path(path).unlink()


def test_output_compatible_with_extract_cases() -> None:
    from pm_rag import extract_cases

    f = _csv(
        "case_id,activity,timestamp",
        "c1,order_received,1",
        "c1,payment_settled,2",
        "c1,ship_order,3",
        "c2,order_received,1",
        "c2,payment_settled,2",
    )
    traces = traces_from_csv(f)
    cases = extract_cases(traces)
    assert len(cases) == 3
    assert cases[0].next_event == "payment_settled"
    assert cases[1].next_event == "ship_order"
