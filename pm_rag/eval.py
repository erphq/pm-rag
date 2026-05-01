"""Evaluation harness for pm-rag (v0.4 skeleton).

See TODO.md for the full implementation plan.

Notes on pm-bench cross-references (not a hard dependency yet):
  - ``pm_bench.split.case_chrono_split`` — chronological train/val/test
    split used to generate (prefix, truth) pairs from an event log.
  - ``pm_bench.score.score_next_event`` — top-k next-event scoring helper
    that can be adapted for symbol-level retrieval evaluation.
"""
from __future__ import annotations

from typing import Any


def evaluate(
    index: Any,
    prefixes: list[list[str]],
    truth: list[str],
    k: int = 1,
) -> dict[str, float]:
    """Evaluate a retrieval *index* against *(prefix, truth)* pairs.

    Parameters
    ----------
    index:
        A ``pm_rag.index.Index`` produced by ``build_index``.
    prefixes:
        List of trace prefixes; each is a list of event-name strings.
    truth:
        Parallel list of ground-truth next events (one per prefix).
    k:
        Top-k cutoff for accuracy calculation.

    Returns
    -------
    dict
        Metrics, e.g. ``{"top_1": 0.0, "top_3": 0.0, "mrr": 0.0}``.

    Raises
    ------
    ValueError
        If ``len(prefixes) != len(truth)``.
    NotImplementedError
        Until the function is implemented.
    """
    if len(prefixes) != len(truth):
        raise ValueError(
            f"prefixes and truth must have the same length "
            f"({len(prefixes)} != {len(truth)})"
        )
    # TODO: for each (prefix, gt) in zip(prefixes, truth):
    #   hits = index.query(prefix, k=k)   # returns list[Hit]
    #   check if gt in {h.symbol for h in hits[:k]}
    # TODO: compute top-1 accuracy, top-k accuracy, MRR
    # TODO: return {"top_1": ..., f"top_{k}": ..., "mrr": ...}
    raise NotImplementedError("evaluate is not yet implemented — see TODO.md")
