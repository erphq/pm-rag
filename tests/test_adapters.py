"""Tests for pm_rag.adapters.traces_from_csv."""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import pytest

from pm_rag.adapters import traces_from_csv


def _csv(*lines: str) -> io.StringIO:
    return io.StringIO("\n".join(lines))


# ---- basic reading -------------------------------------------------------


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


def test_sort_by_timestamp_reorders_out_of_order_rows() -> None:
    f = _csv(
        "case_id,activity,timestamp",
        "c1,second_event,2024-01-01T10:05",
        "c1,first_event,2024-01-01T10:00",
    )
    traces = traces_from_csv(f)
    assert traces[0] == ["first_event", "second_event"]


def test_sort_by_none_preserves_csv_row_order() -> None:
    f = _csv(
        "case_id,activity,timestamp",
        "c1,second_event,2024-01-01T10:05",
        "c1,first_event,2024-01-01T10:00",
    )
    traces = traces_from_csv(f, sort_by=None)
    assert traces[0] == ["second_event", "first_event"]


def test_empty_csv_header_only_returns_empty() -> None:
    f = _csv("case_id,activity,timestamp")
    traces = traces_from_csv(f)
    assert traces == []


def test_single_event_trace() -> None:
    f = _csv(
        "case_id,activity",
        "c1,only_event",
    )
    traces = traces_from_csv(f, sort_by=None)
    assert traces == [["only_event"]]


# ---- empty / whitespace activity filtering --------------------------------


def test_empty_activity_rows_skipped() -> None:
    f = _csv(
        "case_id,activity,timestamp",
        "c1,order_received,1",
        "c1,,2",
        "c1,payment_settled,3",
    )
    traces = traces_from_csv(f)
    assert traces == [["order_received", "payment_settled"]]


def test_whitespace_only_activity_skipped() -> None:
    f = _csv(
        "case_id,activity",
        "c1,   ",
        "c1,order_received",
    )
    traces = traces_from_csv(f, sort_by=None)
    assert traces == [["order_received"]]


def test_case_with_all_empty_activities_omitted() -> None:
    f = _csv(
        "case_id,activity",
        "c1,",
    )
    traces = traces_from_csv(f, sort_by=None)
    assert traces == []


# ---- custom column names --------------------------------------------------


def test_custom_case_and_activity_columns() -> None:
    f = _csv(
        "pid,event",
        "p1,login",
        "p1,checkout",
        "p2,login",
    )
    traces = traces_from_csv(f, case_col="pid", activity_col="event", sort_by=None)
    assert traces == [["login", "checkout"], ["login"]]


# ---- case ordering --------------------------------------------------------


def test_case_order_follows_first_appearance() -> None:
    f = _csv(
        "case_id,activity",
        "z,ev_z",
        "a,ev_a",
    )
    traces = traces_from_csv(f, sort_by=None)
    assert traces[0] == ["ev_z"]
    assert traces[1] == ["ev_a"]


# ---- error handling -------------------------------------------------------


def test_missing_sort_by_column_raises_value_error() -> None:
    f = _csv(
        "case_id,activity",
        "c1,ev",
    )
    with pytest.raises(ValueError, match="timestamp"):
        traces_from_csv(f, sort_by="timestamp")


def test_missing_case_col_raises_key_error() -> None:
    f = _csv(
        "activity,timestamp",
        "order_received,1",
    )
    with pytest.raises(KeyError):
        traces_from_csv(f, case_col="case_id")


# ---- file path variant ----------------------------------------------------


def test_accepts_path_argument() -> None:
    content = "case_id,activity\nc1,order_received\nc1,payment_settled\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        traces = traces_from_csv(tmp_path, sort_by=None)
        assert traces == [["order_received", "payment_settled"]]
    finally:
        tmp_path.unlink()


def test_accepts_string_path() -> None:
    content = "case_id,activity\nc1,ev\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        traces = traces_from_csv(tmp_path, sort_by=None)
        assert traces == [["ev"]]
    finally:
        Path(tmp_path).unlink()


# ---- integration with extract_cases --------------------------------------


def test_output_compatible_with_extract_cases() -> None:
    from pm_rag import extract_cases

    f = _csv(
        "case_id,activity,timestamp",
        "c1,order_received,1",
        "c1,payment_settled,2",
        "c1,ship_order,3",
    )
    traces = traces_from_csv(f)
    cases = extract_cases(traces)
    assert len(cases) == 2
    assert cases[0].next_event == "payment_settled"
    assert cases[1].next_event == "ship_order"


def test_multi_case_extract_cases_round_trip() -> None:
    from pm_rag import extract_cases

    f = _csv(
        "case_id,activity,timestamp",
        "c1,a,1",
        "c1,b,2",
        "c2,x,1",
        "c2,y,2",
        "c2,z,3",
    )
    traces = traces_from_csv(f)
    cases = extract_cases(traces)
    assert len(cases) == 3
