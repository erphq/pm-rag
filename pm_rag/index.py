"""Joint index over a code graph and an event vocabulary, plus query."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from pm_rag.diffusion import personalized_pagerank
from pm_rag.graph import CodeGraph
from pm_rag.mapping import regex_mapping

MappingFn = Callable[[Iterable[str], list[str]], dict[str, list[int]]]
"""A mapping strategy: ``(events, symbols) -> {event: [symbol_idx, ...]}``."""


@dataclass
class Index:
    graph: CodeGraph
    mapping: dict[str, list[int]]
    p_t: np.ndarray


@dataclass(frozen=True)
class Hit:
    symbol: str
    score: float


def build_index(
    graph: CodeGraph,
    events: Iterable[str],
    *,
    mapping_fn: MappingFn | None = None,
) -> Index:
    """Build an index given the code graph and the set of event names
    that may appear in traces.

    Args:
        graph: the code graph.
        events: event names that may appear in traces.
        mapping_fn: strategy with signature
            ``(events, symbols) -> dict[str, list[int]]``. Defaults to
            ``regex_mapping``. Pass ``compose_mappings(...)`` to stack
            strategies (e.g. regex then embedding then manual override).
    """
    fn = mapping_fn if mapping_fn is not None else regex_mapping
    mapping = fn(events, graph.nodes)
    return Index(graph=graph, mapping=mapping, p_t=graph.transition_matrix_T())


def _build_decay_seed(
    n: int,
    mapping: dict[str, list[int]],
    trace_prefix: Sequence[str],
    decay: float,
) -> np.ndarray:
    """Build a seed vector weighted across the full trace prefix.

    The last event gets weight 1.0; the second-to-last gets weight
    ``decay``; the third-to-last gets weight ``decay**2``; and so on.
    Each event's weight is split uniformly across its mapped symbol
    indices. The resulting vector is NOT yet normalized (the caller
    normalizes or falls back to uniform when the total is zero).
    """
    seed = np.zeros(n, dtype=np.float64)
    for step, event in enumerate(reversed(trace_prefix)):
        indices = mapping.get(event, [])
        if indices:
            w = (decay**step) / len(indices)
            for idx in indices:
                seed[idx] += w
    return seed


def query(
    index: Index,
    trace_prefix: Sequence[str],
    *,
    k: int = 10,
    alpha: float = 0.15,
    trace_decay: float | None = None,
) -> list[Hit]:
    """Rank symbols by PPR seeded from the trace prefix.

    By default (``trace_decay=None``) the seed is placed on the symbols
    mapped from the last event in the prefix only. Pass
    ``trace_decay`` in (0, 1) to weight the full prefix with exponential
    recency decay: the last event has weight 1.0, the second-to-last has
    weight ``trace_decay``, the third-to-last ``trace_decay**2``, etc.
    Earlier events therefore contribute but do not dominate.

    If no event in the effective window has a mapping, the seed falls
    back to a uniform distribution (global PageRank).

    Args:
        index: built by ``build_index``.
        trace_prefix: ordered sequence of events; must be non-empty.
        k: number of top hits to return.
        alpha: PPR restart probability (0, 1).
        trace_decay: exponential decay factor for prefix weighting.
            Must be in (0, 1) if provided. ``None`` uses last-event
            seeding only (default, backward-compatible).
    """
    if not trace_prefix:
        raise ValueError("trace_prefix must be non-empty")
    if trace_decay is not None and not 0 < trace_decay < 1:
        raise ValueError("trace_decay must be in (0, 1)")

    n = index.graph.n
    if trace_decay is not None:
        seed = _build_decay_seed(n, index.mapping, trace_prefix, trace_decay)
    else:
        seed_indices = index.mapping.get(trace_prefix[-1], [])
        seed = np.zeros(n, dtype=np.float64)
        if seed_indices:
            for i in seed_indices:
                seed[i] = 1.0 / len(seed_indices)

    if seed.sum() == 0:
        seed[:] = 1.0 / n

    r = personalized_pagerank(index.p_t, seed, alpha=alpha)
    order = np.argsort(-r)
    return [Hit(symbol=index.graph.nodes[int(i)], score=float(r[int(i)])) for i in order[:k]]
