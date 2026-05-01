"""pm-rag — process-aware retrieval over code graphs."""
from __future__ import annotations

__version__ = "0.1.0"

from pm_rag.diffusion import personalized_pagerank
from pm_rag.eval import (
    LocalizationCase,
    LocalizationScore,
    evaluate,
    extract_cases,
)
from pm_rag.graph import CodeGraph
from pm_rag.index import Hit, Index, build_index, query
from pm_rag.mapping import (
    EmbedFn,
    LlmFn,
    compose_mappings,
    embedding_mapping,
    llm_mapping,
    regex_mapping,
)

__all__ = [
    "CodeGraph",
    "EmbedFn",
    "Hit",
    "Index",
    "LlmFn",
    "LocalizationCase",
    "LocalizationScore",
    "build_index",
    "compose_mappings",
    "embedding_mapping",
    "evaluate",
    "extract_cases",
    "llm_mapping",
    "personalized_pagerank",
    "query",
    "regex_mapping",
]
