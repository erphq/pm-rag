"""Event → symbol mapping strategies. v0: regex (substring, case-insensitive)."""
from __future__ import annotations

import re
from collections.abc import Iterable


def regex_mapping(events: Iterable[str], symbols: list[str]) -> dict[str, list[int]]:
    """Map each unique event name to symbol indices whose name contains
    the event string (case-insensitive substring).

    Args:
        events: an iterable of event names (duplicates ignored).
        symbols: the symbol list (e.g. `CodeGraph.nodes`).

    Returns:
        A dict from event name to a list of matching symbol indices.
        Events with no match map to an empty list.
    """
    out: dict[str, list[int]] = {}
    seen: set[str] = set()
    for ev in events:
        if ev in seen:
            continue
        seen.add(ev)
        pattern = re.compile(re.escape(ev), re.IGNORECASE)
        out[ev] = [i for i, s in enumerate(symbols) if pattern.search(s)]
    return out
