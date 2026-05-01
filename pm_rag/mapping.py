"""Event → symbol mapping strategies.

Three strategies ship in v0.5:

1. ``regex_mapping`` — case-insensitive substring match (the v0
   default, fast and cheap, high precision when log strings actually
   embed function names).
2. ``embedding_mapping`` — user supplies an embedder callable
   ``embed_fn(text) -> Sequence[float]``. The mapping picks the top-k
   symbols by cosine similarity above a threshold. We never bundle a
   model — the caller decides which embedder to plug in (sentence-
   transformers, BAAI/bge-*, OpenAI's API, anything).
3. ``compose_mappings`` — try each strategy in order, take the first
   non-empty result per event. Lets you stack regex (cheap, precise)
   then embedding (broader recall) then a manual override.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence

EmbedFn = Callable[[str], Sequence[float]]
"""A pluggable embedder. Takes a text, returns a numeric vector."""


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


def embedding_mapping(
    events: Iterable[str],
    symbols: list[str],
    embed_fn: EmbedFn,
    *,
    threshold: float = 0.5,
    top_k: int = 5,
) -> dict[str, list[int]]:
    """Map each unique event to symbol indices whose embedding has
    cosine similarity ≥ `threshold` against the event's embedding.
    Up to `top_k` symbols per event, ranked by similarity descending.

    Args:
        events: an iterable of event names (duplicates ignored).
        symbols: the symbol list.
        embed_fn: callable ``text -> vector``. Vectors must all have
            the same length and be non-zero. The function is called
            once per unique event and once per symbol; cache results
            in the caller if needed.
        threshold: minimum cosine similarity to qualify (default 0.5).
            Set lower for higher recall; higher for higher precision.
        top_k: maximum symbols returned per event (default 5).

    Returns:
        A dict from event name to a list of matching symbol indices.

    Raises:
        ValueError: if `top_k <= 0` or the embedder returns inconsistent
            vector lengths.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    symbol_vectors = [_norm(embed_fn(s)) for s in symbols]
    out: dict[str, list[int]] = {}
    seen: set[str] = set()
    for ev in events:
        if ev in seen:
            continue
        seen.add(ev)
        ev_vec = _norm(embed_fn(ev))
        scored = [
            (i, _dot(ev_vec, sv))
            for i, sv in enumerate(symbol_vectors)
            if len(sv) == len(ev_vec)
        ]
        scored.sort(key=lambda x: -x[1])
        out[ev] = [i for i, score in scored[:top_k] if score >= threshold]
    return out


def compose_mappings(
    *strategies: Callable[[Iterable[str], list[str]], dict[str, list[int]]],
) -> Callable[[Iterable[str], list[str]], dict[str, list[int]]]:
    """Combine multiple mapping strategies. The composed mapping picks
    the FIRST non-empty result per event across `strategies`.

    Useful for stacking: cheap-and-precise (regex) → broader-but-
    fuzzier (embedding) → manual override.

    Each strategy must accept ``(events, symbols)`` and return the
    mapping dict.
    """

    def composed(events: Iterable[str], symbols: list[str]) -> dict[str, list[int]]:
        events_list = list(dict.fromkeys(events))
        result: dict[str, list[int]] = {ev: [] for ev in events_list}
        for strategy in strategies:
            partial = strategy(events_list, symbols)
            for ev in events_list:
                if not result[ev] and partial.get(ev):
                    result[ev] = list(partial[ev])
        return result

    return composed


def _norm(v: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v))
    if norm == 0:
        raise ValueError("embedder returned a zero vector")
    return [x / norm for x in v]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))
