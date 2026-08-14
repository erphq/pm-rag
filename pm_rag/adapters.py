"""Trace-log adapters: read event traces from common file formats.

Currently supported:

- CSV: one row per event, with a case-identifier column and an
  activity column. Optional timestamp column controls ordering
  within each case.
"""
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

    Each unique value of ``case_col`` groups rows into one trace. Within
    each trace, rows are sorted by ``sort_by`` (lexicographic) when that
    column is provided; pass ``sort_by=None`` to preserve CSV row order.
    Rows whose activity value is empty or whitespace-only are skipped.
    Cases with no surviving activities are omitted from the result.

    The returned structure is a list of traces compatible with
    ``pm_rag.eval.extract_cases``.

    Args:
        file: path-like or readable text file-like object.
        case_col: column name for the case identifier (default ``"case_id"``).
        activity_col: column name for the activity/event name
            (default ``"activity"``).
        sort_by: column name to sort rows within each case
            (default ``"timestamp"``). Pass ``None`` to keep CSV row order.

    Returns:
        List of traces; each trace is a list of non-empty activity strings.

    Raises:
        KeyError: if ``case_col`` is absent from the CSV header.
        ValueError: if ``sort_by`` is not ``None`` and is absent from
            the CSV header.
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
        cid = row[case_col]
        rows_by_case.setdefault(cid, []).append(row)

    if sort_by is not None:
        fieldnames = list(reader.fieldnames or [])
        if sort_by not in fieldnames:
            raise ValueError(
                f"sort_by column {sort_by!r} not found in CSV header {fieldnames}"
            )

    traces: list[list[str]] = []
    for rows in rows_by_case.values():
        if sort_by is not None:
            rows = sorted(rows, key=lambda r: r.get(sort_by) or "")
        activities = [
            act
            for r in rows
            if (act := str(r.get(activity_col) or "").strip())
        ]
        if activities:
            traces.append(activities)
    return traces
