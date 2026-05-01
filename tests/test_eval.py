"""TODO tests for pm_rag.eval (v0.4 milestone)."""
from __future__ import annotations

import pytest


def test_evaluate_raises_not_implemented() -> None:
    from pm_rag.eval import evaluate

    with pytest.raises(NotImplementedError):
        evaluate(object(), [], [], k=1)


def test_evaluate_mismatched_lengths_raises_value_error() -> None:
    from pm_rag.eval import evaluate

    with pytest.raises(ValueError):
        evaluate(object(), [["A"]], [], k=1)


def test_evaluate_perfect_index() -> None:
    # TODO: build a synthetic Index where every prefix maps to the correct symbol.
    # result = evaluate(index, prefixes, truth, k=1)
    # assert result["top_1"] == 1.0
    pytest.skip("TODO (v0.4): implement evaluate() first")


def test_evaluate_empty_prefixes() -> None:
    # TODO: evaluate with no prefixes should return all-zero / empty metrics.
    pytest.skip("TODO (v0.4): implement evaluate() first")


def test_evaluate_top_k_accuracy() -> None:
    # TODO: construct a scenario where truth is not top-1 but is top-3.
    # assert result["top_3"] > result["top_1"]
    pytest.skip("TODO (v0.4): implement evaluate() first")


def test_evaluate_mrr() -> None:
    # TODO: verify MRR calculation with a known ranking.
    pytest.skip("TODO (v0.4): implement evaluate() first")
