"""Tests for eval.format_score."""
from __future__ import annotations

import json

import pytest

from pm_rag.eval import LocalizationScore, format_score


def test_format_score_basic_structure() -> None:
    score = LocalizationScore(top_k={1: 0.3333, 3: 0.6667, 5: 1.0}, n=3, mrr=0.5)
    out = format_score(score)
    assert out["task"] == "next-event-localization"
    assert out["n"] == 3
    assert set(out.keys()) == {"task", "n", "top_k", "mrr"}


def test_format_score_top_k_keys_are_strings() -> None:
    score = LocalizationScore(top_k={1: 1.0, 10: 1.0}, n=1, mrr=1.0)
    out = format_score(score)
    assert all(isinstance(k, str) for k in out["top_k"])
    assert "1" in out["top_k"]
    assert "10" in out["top_k"]


def test_format_score_default_decimals_rounds_to_4() -> None:
    score = LocalizationScore(top_k={1: 1.0 / 3}, n=1, mrr=1.0 / 3)
    out = format_score(score)
    assert out["top_k"]["1"] == 0.3333
    assert out["mrr"] == 0.3333


def test_format_score_decimals_param_controls_rounding() -> None:
    score = LocalizationScore(top_k={1: 1.0 / 3}, n=1, mrr=1.0 / 3)
    out = format_score(score, decimals=2)
    assert out["top_k"]["1"] == 0.33
    assert out["mrr"] == 0.33


def test_format_score_alpha_omitted_by_default() -> None:
    score = LocalizationScore(top_k={1: 0.5}, n=2, mrr=0.5)
    out = format_score(score)
    assert "alpha" not in out


def test_format_score_alpha_included_when_passed() -> None:
    score = LocalizationScore(top_k={1: 0.5}, n=2, mrr=0.5)
    out = format_score(score, alpha=0.15)
    assert out["alpha"] == 0.15


def test_format_score_zero_case_no_scorable() -> None:
    score = LocalizationScore(top_k={1: 0.0}, n=0)
    out = format_score(score)
    assert out["n"] == 0
    assert out["mrr"] == 0.0
    assert out["top_k"]["1"] == 0.0


def test_format_score_result_is_json_serialisable() -> None:
    score = LocalizationScore(top_k={1: 0.5, 3: 0.9}, n=5, mrr=0.7)
    out = format_score(score, alpha=0.15)
    serialised = json.dumps(out)
    restored = json.loads(serialised)
    assert restored["task"] == "next-event-localization"
    assert restored["n"] == 5


def test_format_score_high_precision_decimals() -> None:
    score = LocalizationScore(top_k={1: 1.0 / 7}, n=1, mrr=1.0 / 7)
    out = format_score(score, decimals=6)
    assert out["top_k"]["1"] == round(1.0 / 7, 6)
    assert out["mrr"] == round(1.0 / 7, 6)


def test_format_score_alpha_is_passed_through_unchanged() -> None:
    score = LocalizationScore(top_k={1: 0.5}, n=1, mrr=0.5)
    out = format_score(score, alpha=0.42)
    assert out["alpha"] == 0.42
