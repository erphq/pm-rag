"""Eval harness for next-event localization.

Definition of the task: given a trace prefix and a code graph, retrieve
the function(s) that emit the next event in the trace. Score: top-k
accuracy, where a hit is "any of the truth symbols appears in the
top-k retrieved symbols."
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from pm_rag.index import Index, query


@dataclass(frozen=True)
class LocalizationCase:
    """One scoring instance: a trace prefix and the activity that follows."""

    prefix: list[str]
    next_event: str


@dataclass(frozen=True)
class LocalizationScore:
    """Top-k accuracy across a set of cases.

    `top_k` maps each `k` to the fraction of cases where any truth
    symbol appears in the top-`k` retrieved. `n` is the count of cases
    that had at least one mapped truth symbol (cases with no mapping
    are unscorable and skipped).
    """

    top_k: dict[int, float]
    n: int


def extract_cases(traces: Iterable[Sequence[str]]) -> list[LocalizationCase]:
    """Build `(prefix, next_event)` pairs from a set of traces.

    For each trace of length L, emit `L - 1` cases: prefix `trace[:k]`
    paired with `trace[k]` for `k` in `1..L-1`. Traces of length 0 or
    1 contribute nothing.
    """
    cases: list[LocalizationCase] = []
    for trace in traces:
        activities = list(trace)
        for k in range(1, len(activities)):
            cases.append(
                LocalizationCase(prefix=activities[:k], next_event=activities[k])
            )
    return cases


def evaluate(
    index: Index,
    cases: Sequence[LocalizationCase],
    *,
    ks: Sequence[int] = (1, 3, 5, 10),
    alpha: float = 0.15,
) -> LocalizationScore:
    """Score `index` on next-event localization.

    For each case, query the index with the prefix and check whether
    any of the truth symbols (the symbols that map to `next_event`)
    appears in the top-k retrieved. Cases whose `next_event` has no
    mapping are skipped (we can't score them).
    """
    if not ks:
        raise ValueError("ks must be non-empty")
    if any(k <= 0 for k in ks):
        raise ValueError("each k must be positive")

    sorted_ks = sorted(set(ks))
    max_k = sorted_ks[-1]

    name_to_idx = {n: i for i, n in enumerate(index.graph.nodes)}
    hits = {k: 0 for k in sorted_ks}
    n = 0
    for c in cases:
        truth_indices = index.mapping.get(c.next_event, [])
        if not truth_indices:
            continue
        truth_set = set(truth_indices)
        ranked = query(index, c.prefix, k=max_k, alpha=alpha)
        ranked_indices = [name_to_idx[h.symbol] for h in ranked]
        n += 1
        for k in sorted_ks:
            if any(idx in truth_set for idx in ranked_indices[:k]):
                hits[k] += 1
    if n == 0:
        return LocalizationScore(top_k={k: 0.0 for k in sorted_ks}, n=0)
    return LocalizationScore(top_k={k: hits[k] / n for k in sorted_ks}, n=n)
