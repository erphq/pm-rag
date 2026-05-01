"""Tests for the v0.5 embedding-based mapping strategy."""
from __future__ import annotations

import hashlib

import pytest

from pm_rag import compose_mappings, embedding_mapping, regex_mapping


def deterministic_embedder(dim: int = 16):
    """A deterministic stand-in embedder for tests.

    Same string → same vector. Two strings sharing a long common
    substring → similar vectors. Pure Python, no dependencies.
    """

    cache: dict[str, list[float]] = {}

    def embed(text: str) -> list[float]:
        if text in cache:
            return cache[text]
        vec = [0.0] * dim
        # Bag-of-overlapping-trigrams hashed into the vector.
        s = text.lower()
        if len(s) < 3:
            s = s + "  "
        for i in range(len(s) - 2):
            tri = s[i : i + 3]
            h = int(hashlib.md5(tri.encode("utf-8")).hexdigest(), 16)
            vec[h % dim] += 1.0
        cache[text] = vec
        return vec

    return embed, cache


def test_embedding_mapping_finds_close_match() -> None:
    embed, _ = deterministic_embedder()
    symbols = [
        "handlers.payment.payment_settled_handler",
        "utils.money.format_amount",
    ]
    m = embedding_mapping(["payment_settled"], symbols, embed, threshold=0.0)
    # `payment_settled` shares many trigrams with the first symbol.
    assert m["payment_settled"]
    assert m["payment_settled"][0] == 0


def test_embedding_mapping_threshold_filters() -> None:
    embed, _ = deterministic_embedder()
    symbols = ["totally.unrelated.zzzz", "another.unrelated.qqqq"]
    m = embedding_mapping(
        ["payment_settled"], symbols, embed, threshold=0.95
    )
    assert m["payment_settled"] == []


def test_embedding_mapping_top_k_caps() -> None:
    embed, _ = deterministic_embedder()
    symbols = [f"payment_settled_handler_{i}" for i in range(20)]
    m = embedding_mapping(["payment_settled"], symbols, embed, top_k=3, threshold=0.0)
    assert len(m["payment_settled"]) <= 3


def test_embedding_mapping_validates_top_k() -> None:
    embed, _ = deterministic_embedder()
    with pytest.raises(ValueError):
        embedding_mapping(["x"], ["y"], embed, top_k=0)


def test_embedding_mapping_rejects_zero_vector() -> None:
    def zero_embed(_text: str) -> list[float]:
        return [0.0, 0.0, 0.0]

    with pytest.raises(ValueError):
        embedding_mapping(["x"], ["y"], zero_embed)


def test_embedding_mapping_caches_unique_events() -> None:
    embed, cache = deterministic_embedder()
    symbols = ["a.b.c"]
    embedding_mapping(["x", "x", "x"], symbols, embed)
    # cache should have 1 event + 1 symbol = 2 entries (since only one
    # unique event "x" was seen)
    assert "x" in cache
    assert "a.b.c" in cache


def test_compose_picks_first_non_empty() -> None:
    def regex_strategy(events: list[str], symbols: list[str]):
        return regex_mapping(events, symbols)

    def fallback_strategy(events: list[str], symbols: list[str]):
        return {ev: [0] for ev in events}

    composed = compose_mappings(regex_strategy, fallback_strategy)
    symbols = ["payment_settled_handler", "fallback_target"]
    m = composed(["payment_settled", "totally_unrelated"], symbols)
    # First event hits regex (substring match), second falls through.
    assert m["payment_settled"] == [0]
    assert m["totally_unrelated"] == [0]


def test_compose_with_embedding_then_regex() -> None:
    embed, _ = deterministic_embedder()

    def emb_strategy(events: list[str], symbols: list[str]):
        return embedding_mapping(events, symbols, embed, threshold=0.95)

    def regex_strategy(events: list[str], symbols: list[str]):
        return regex_mapping(events, symbols)

    composed = compose_mappings(emb_strategy, regex_strategy)
    symbols = ["completely.different.from.search"]
    # Embedding fails the strict threshold, regex still matches the
    # substring.
    m = composed(["different"], symbols)
    assert m["different"] == [0]


def test_compose_empty_strategies() -> None:
    composed = compose_mappings()
    m = composed(["x"], ["y"])
    assert m == {"x": []}
