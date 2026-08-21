"""Trace-log adapters: read event traces from common file formats."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import IO


def traces_from_csv(
    file: str | Path | IO[str],
    *,
    case_col: str = "case_id",
    activity_col: str = "activity",
    sort_by: str | None = "timestamp",
) -> list[list[str]]:
    """Read event traces from a CSV file.

    Rows are grouped by ``case_col`` into traces and sorted within each
    trace by ``sort_by`` (lexicographic; ``None`` preserves row order).
    Rows with an empty or whitespace-only activity value are skipped; cases
    with no surviving activities are omitted. The result is compatible with
    ``pm_rag.eval.extract_cases``.

    Args:
        file: path-like or readable text file-like object.
        case_col: column name for the case identifier (default ``"case_id"``).
        activity_col: column name for the activity/event name
            (default ``"activity"``).
        sort_by: column to sort within each trace (default ``"timestamp"``).
            Pass ``None`` to keep CSV row order.

    Raises:
        KeyError: if ``case_col`` is absent from the CSV header.
        ValueError: if ``activity_col`` or ``sort_by`` (when not ``None``) is
            absent from the CSV header.
    """
    if isinstance(file, (str, Path)):
        with open(file, newline="", encoding="utf-8") as fh:
            return _read_csv(fh, case_col, activity_col, sort_by)
    return _read_csv(file, case_col, activity_col, sort_by)


def traces_from_json(
    file: str | Path | IO[str],
    *,
    case_col: str = "case_id",
    activity_col: str = "activity",
    sort_by: str | None = "timestamp",
) -> list[list[str]]:
    """Read event traces from a JSON file.

    Two formats are accepted:

    * **List-of-lists**: ``[["ev1", "ev2"], ["ev3"]]``. Each inner list is one
      trace in order. ``case_col``, ``activity_col``, and ``sort_by`` are ignored.
    * **List-of-records**: ``[{"case_id": "c1", "activity": "ev1", ...}, ...]``.
      Records are grouped by ``case_col`` and sorted by ``sort_by`` (``None``
      preserves document order), mirroring ``traces_from_csv`` behavior.

    Empty/whitespace activity strings are skipped. Cases with no surviving
    activities are omitted. The top-level JSON value must be an array.
    """
    if isinstance(file, (str, Path)):
        with open(file, encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("JSON root must be a list")
    if not data:
        return []

    if isinstance(data[0], list):
        return _read_json_lists(data)
    return _read_json_records(data, case_col, activity_col, sort_by)


def _read_json_lists(data: list) -> list[list[str]]:
    traces: list[list[str]] = []
    for i, trace in enumerate(data):
        if not isinstance(trace, list):
            raise ValueError(f"list-of-lists format: element {i} is not a list")
        activities: list[str] = []
        for j, ev in enumerate(trace):
            if not isinstance(ev, str):
                raise ValueError(
                    f"list-of-lists format: trace {i}, element {j} is not a string"
                )
            if ev.strip():
                activities.append(ev)
        if activities:
            traces.append(activities)
    return traces


def _read_json_records(
    data: list,
    case_col: str,
    activity_col: str,
    sort_by: str | None,
) -> list[list[str]]:
    rows_by_case: dict[str, list[dict]] = {}
    for row in data:
        if not isinstance(row, dict):
            raise ValueError("list-of-records format: each element must be a JSON object")
        case_id = str(row[case_col])
        rows_by_case.setdefault(case_id, []).append(row)

    traces: list[list[str]] = []
    for rows in rows_by_case.values():
        if sort_by is not None:
            rows = sorted(rows, key=lambda r: str(r.get(sort_by) or ""))
        activities = [
            act
            for r in rows
            if (act := str(r.get(activity_col) or "").strip())
        ]
        if activities:
            traces.append(activities)
    return traces


def _read_csv(
    fh: IO[str],
    case_col: str,
    activity_col: str,
    sort_by: str | None,
) -> list[list[str]]:
    reader = csv.DictReader(fh)
    rows_by_case: dict[str, list[dict]] = {}
    for row in reader:
        rows_by_case.setdefault(row[case_col], []).append(row)

    fieldnames = list(reader.fieldnames or [])
    if activity_col not in fieldnames:
        raise ValueError(f"activity_col {activity_col!r} not found in CSV header")
    if sort_by is not None and sort_by not in fieldnames:
        raise ValueError(f"sort_by column {sort_by!r} not found in CSV header")

    traces: list[list[str]] = []
    for rows in rows_by_case.values():
        if sort_by is not None:
            rows = sorted(rows, key=lambda r: r.get(sort_by) or "")
        activities = [act for r in rows if (act := str(r.get(activity_col) or "").strip())]
        if activities:
            traces.append(activities)
    return traces
