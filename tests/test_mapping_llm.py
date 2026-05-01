"""Tests for the v0.6 LLM-assisted mapping strategy."""
from __future__ import annotations

import pytest

from pm_rag import compose_mappings, llm_mapping, regex_mapping


def fake_llm(responses: dict[str, str]):
    """A fake LLM that returns canned responses keyed by event name.

    The prompt always contains the event name on its own line; we
    extract it and look up the canned response.
    """

    def llm(prompt: str) -> str:
        for ev, resp in responses.items():
            if f"Event name: {ev}" in prompt:
                return resp
        return "[]"

    return llm


def test_llm_mapping_parses_clean_json_array() -> None:
    symbols = ["a", "b", "c", "d"]
    llm = fake_llm({"foo": "[1, 3]"})
    m = llm_mapping(["foo"], symbols, llm)
    assert m["foo"] == [1, 3]


def test_llm_mapping_handles_empty_array() -> None:
    symbols = ["a", "b"]
    llm = fake_llm({"foo": "[]"})
    m = llm_mapping(["foo"], symbols, llm)
    assert m["foo"] == []


def test_llm_mapping_filters_out_of_range_indices() -> None:
    symbols = ["a", "b"]
    llm = fake_llm({"foo": "[0, 5, -1, 1]"})
    m = llm_mapping(["foo"], symbols, llm)
    assert m["foo"] == [0, 1]


def test_llm_mapping_filters_non_int_values() -> None:
    symbols = ["a", "b"]
    llm = fake_llm({"foo": '[0, "1", 1.5, true, null, 1]'})
    m = llm_mapping(["foo"], symbols, llm)
    assert m["foo"] == [0, 1]


def test_llm_mapping_dedupes_repeated_indices() -> None:
    symbols = ["a", "b", "c"]
    llm = fake_llm({"foo": "[0, 0, 1, 1, 2]"})
    m = llm_mapping(["foo"], symbols, llm)
    assert m["foo"] == [0, 1, 2]


def test_llm_mapping_truncates_to_top_k() -> None:
    symbols = list("abcdefghij")
    llm = fake_llm({"foo": "[0, 1, 2, 3, 4, 5, 6, 7]"})
    m = llm_mapping(["foo"], symbols, llm, top_k=3)
    assert m["foo"] == [0, 1, 2]


def test_llm_mapping_tolerates_prose_around_json() -> None:
    symbols = ["a", "b"]
    llm = fake_llm({"foo": "I think the answer is [1] which is `b`."})
    m = llm_mapping(["foo"], symbols, llm)
    assert m["foo"] == [1]


def test_llm_mapping_tolerates_markdown_fence() -> None:
    symbols = ["a", "b"]
    llm = fake_llm({"foo": "```json\n[0]\n```"})
    m = llm_mapping(["foo"], symbols, llm)
    assert m["foo"] == [0]


def test_llm_mapping_returns_empty_for_invalid_json() -> None:
    symbols = ["a", "b"]
    llm = fake_llm({"foo": "not even close to JSON"})
    m = llm_mapping(["foo"], symbols, llm)
    assert m["foo"] == []


def test_llm_mapping_returns_empty_for_non_array() -> None:
    symbols = ["a", "b"]
    llm = fake_llm({"foo": '{"index": 0}'})
    m = llm_mapping(["foo"], symbols, llm)
    assert m["foo"] == []


def test_llm_mapping_caches_unique_events() -> None:
    seen: list[str] = []

    def llm(prompt: str) -> str:
        seen.append(prompt)
        return "[0]"

    symbols = ["a"]
    llm_mapping(["x", "x", "x"], symbols, llm)
    assert len(seen) == 1


def test_llm_mapping_validates_top_k() -> None:
    with pytest.raises(ValueError):
        llm_mapping(["x"], ["a"], fake_llm({}), top_k=0)


def test_llm_mapping_handles_empty_symbol_list() -> None:
    llm = fake_llm({"foo": "[0]"})
    m = llm_mapping(["foo"], [], llm)
    assert m["foo"] == []


def test_llm_mapping_prompt_contains_event_and_symbols() -> None:
    captured: list[str] = []

    def llm(prompt: str) -> str:
        captured.append(prompt)
        return "[]"

    llm_mapping(["payment_settled"], ["handlers.pay.settled", "utils.fmt"], llm)
    assert "Event name: payment_settled" in captured[0]
    assert "0: handlers.pay.settled" in captured[0]
    assert "1: utils.fmt" in captured[0]


def test_llm_composes_under_compose_mappings() -> None:
    symbols = ["payment_settled_handler", "no_match_for_anything"]
    regex_strategy = lambda evs, syms: regex_mapping(evs, syms)  # noqa: E731

    def llm_strategy(evs: list[str], syms: list[str]):
        return llm_mapping(evs, syms, fake_llm({"new_event": "[1]"}))

    composed = compose_mappings(regex_strategy, llm_strategy)
    m = composed(["payment_settled", "new_event"], symbols)
    # First event hits regex (substring match in symbol 0), second
    # falls through to LLM which picks index 1.
    assert m["payment_settled"] == [0]
    assert m["new_event"] == [1]
