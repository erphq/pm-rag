"""Event → symbol mapping strategies.

Four strategies are available:

1. ``regex_mapping`` - case-insensitive substring match (the v0
   default, fast and cheap, high precision when log strings actually
   embed function names).
2. ``embedding_mapping`` - user supplies an embedder callable
   ``embed_fn(text) -> Sequence[float]``. The mapping picks the top-k
   symbols by cosine similarity above a threshold. We never bundle a
   model - the caller decides which embedder to plug in (sentence-
   transformers, BAAI/bge-*, OpenAI's API, anything).
3. ``llm_mapping`` - user supplies an LLM callable ``llm_fn(prompt) ->
   str``. The LLM is asked which symbols emit each event and responds
   with a JSON index list.
4. ``manual_mapping`` - explicit symbol-name overrides loaded from a
   dict (or YAML). Use as the last-resort layer in a
   ``compose_mappings`` chain when all automated strategies miss.
5. ``compose_mappings`` - combine strategies in priority order, taking
   the first non-empty result per event.
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


# ---------------------------------------------------------------------
# v0.6: LLM-assisted mapping
# ---------------------------------------------------------------------

import json as _json  # noqa: E402

LlmFn = Callable[[str], str]
"""A pluggable LLM. Takes a prompt, returns the raw text response.

Caller wraps whatever model they want (Anthropic, OpenAI, local llama,
a fake for tests) into this shape. The function does not stream - it
returns the full response as a string.
"""


_PROMPT_TEMPLATE = """\
You are mapping a process-mining event to functions in a codebase.

Event name: {event}

Functions (numbered 0..{last_idx}):
{symbol_block}

Return at most {top_k} indices of functions that most plausibly emit
this event when they run. Rank best-first. If no function plausibly
emits this event, return an empty list.

Respond with a JSON array of integers and nothing else. Examples:
[3, 1, 7]
[]
"""


def llm_mapping(
    events: Iterable[str],
    symbols: list[str],
    llm_fn: LlmFn,
    *,
    top_k: int = 5,
) -> dict[str, list[int]]:
    """Ask an LLM which symbols emit each event.

    The LLM is treated as a black box: given a prompt that lists the
    event and the numbered symbol list, it must respond with a JSON
    array of integers (the symbol indices that plausibly emit that
    event). We tolerate non-JSON or malformed responses by returning
    an empty list for that event - never raising.

    Composes with `regex_mapping` and `embedding_mapping` via
    `compose_mappings`. Use that to stack cheap-and-precise →
    broader-but-fuzzier → LLM-as-fallback.

    Args:
        events: an iterable of event names (duplicates ignored).
        symbols: the symbol list.
        llm_fn: callable ``prompt -> raw response``.
        top_k: maximum symbols returned per event (default 5).

    Returns:
        A dict from event name to a list of symbol indices.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    out: dict[str, list[int]] = {}
    seen: set[str] = set()
    for ev in events:
        if ev in seen:
            continue
        seen.add(ev)
        prompt = _build_llm_prompt(ev, symbols, top_k)
        raw = llm_fn(prompt)
        out[ev] = _parse_indices(raw, len(symbols), top_k)
    return out


def _build_llm_prompt(event: str, symbols: list[str], top_k: int) -> str:
    if not symbols:
        symbol_block = "(none)"
        last_idx = -1
    else:
        symbol_block = "\n".join(f"  {i}: {s}" for i, s in enumerate(symbols))
        last_idx = len(symbols) - 1
    return _PROMPT_TEMPLATE.format(
        event=event, last_idx=last_idx, symbol_block=symbol_block, top_k=top_k
    )


def _parse_indices(raw: str, n_symbols: int, top_k: int) -> list[int]:
    """Tolerantly parse the LLM's response into a list of valid indices.

    Returns an empty list when the response can't be parsed, isn't an
    array, contains non-int values, or is otherwise unusable. Never
    raises.
    """
    if not isinstance(raw, str):
        return []
    text = raw.strip()
    if not text:
        return []
    # Find the first '[ ... ]' substring to tolerate stray prose.
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    candidate = text[start : end + 1]
    try:
        parsed = _json.loads(candidate)
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    out: list[int] = []
    seen: set[int] = set()
    for item in parsed:
        if not isinstance(item, int) or isinstance(item, bool):
            continue
        if item < 0 or item >= n_symbols:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= top_k:
            break
    return out


# ---------------------------------------------------------------------
# Manual / YAML override mapping
# ---------------------------------------------------------------------


def manual_mapping(
    overrides: dict[str, list[str]],
) -> Callable[[Iterable[str], list[str]], dict[str, list[int]]]:
    """Return a mapping strategy backed by an explicit symbol-name table.

    Use as the last layer in a ``compose_mappings`` chain when all
    automated strategies miss an event. The caller provides a dict
    mapping event names to symbol names; the returned strategy resolves
    those names to indices at query time.

    Symbol names absent from the current symbol list are silently
    ignored, so stale YAML override files remain safe after symbols are
    renamed or removed.

    Args:
        overrides: event name to list of symbol names that emit it.

    Returns:
        A strategy callable with the standard ``(events, symbols)``
        signature, composable directly under ``compose_mappings``.
    """

    def _manual(events: Iterable[str], symbols: list[str]) -> dict[str, list[int]]:
        index_of = {s: i for i, s in enumerate(symbols)}
        seen: set[str] = set()
        out: dict[str, list[int]] = {}
        for ev in events:
            if ev in seen:
                continue
            seen.add(ev)
            out[ev] = [index_of[name] for name in overrides.get(ev, []) if name in index_of]
        return out

    return _manual
