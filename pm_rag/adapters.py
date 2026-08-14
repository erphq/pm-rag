"""Trace-log adapters: read event traces from common file formats."""
from __future__ import annotations

import csv
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
        ValueError: if ``sort_by`` is not ``None`` and absent from the header.
    """
    if isinstance(file, (str, Path)):
        with open(file, newline="", encoding="utf-8") as fh:
            return _read_csv(fh, case_col, activity_col, sort_by)
    return _read_csv(file, case_col, activity_col, sort_by)


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

    if sort_by is not None and sort_by not in (reader.fieldnames or []):
        raise ValueError(f"sort_by column {sort_by!r} not found in CSV header")

    traces: list[list[str]] = []
    for rows in rows_by_case.values():
        if sort_by is not None:
            rows = sorted(rows, key=lambda r: r.get(sort_by) or "")
        activities = [act for r in rows if (act := str(r.get(activity_col) or "").strip())]
        if activities:
            traces.append(activities)
    return traces
