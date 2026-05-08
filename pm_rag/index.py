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


def query(
    index: Index,
    trace_prefix: Sequence[str],
    *,
    k: int = 10,
    alpha: float = 0.15,
) -> list[Hit]:
    """Rank symbols by PPR seeded at the symbols mapped from the trace
    prefix's last event. If the last event has no mapping, fall back to
    uniform seed (i.e. global PageRank).
    """
    if not trace_prefix:
        raise ValueError("trace_prefix must be non-empty")
    last = trace_prefix[-1]
    seed_indices = index.mapping.get(last, [])
    n = index.graph.n
    seed = np.zeros(n, dtype=np.float64)
    if seed_indices:
        for i in seed_indices:
            seed[i] = 1.0 / len(seed_indices)
    else:
        seed[:] = 1.0 / n
    r = personalized_pagerank(index.p_t, seed, alpha=alpha)
    order = np.argsort(-r)
    return [Hit(symbol=index.graph.nodes[int(i)], score=float(r[int(i)])) for i in order[:k]]
