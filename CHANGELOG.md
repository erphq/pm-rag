# Changelog

All notable changes to `pm-rag` are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2026-05-01

### Added
- LLM-assisted event → symbol mapping. `llm_mapping(events,
  symbols, llm_fn, top_k=5)` asks a black-box `LlmFn(prompt) -> str`
  which symbols emit each event, parses a JSON array of indices
  out of the response.
- Tolerant parser: strips prose / markdown around the JSON array,
  filters out-of-range / non-int / duplicate indices, truncates to
  `top_k`, returns `[]` on any parse failure (never raises).
- New `LlmFn` type alias.

## [0.5.0] - 2026-05-01

### Added
- Embedding-based event → symbol mapping. `embedding_mapping(events,
  symbols, embed_fn, threshold=0.5, top_k=5)` ranks symbols by
  cosine similarity above a threshold against a user-supplied
  `EmbedFn`. No bundled model.
- `compose_mappings(*strategies)` stacks strategies; first
  non-empty result per event wins. Lets callers do regex (cheap,
  precise) → embedding (broader recall) → manual override.
- New `EmbedFn` type alias.

## [0.4.0] - 2026-05-01

### Added
- Eval harness for next-event localization.
  `extract_cases(traces)` builds `(prefix, next_event)` pairs.
  `evaluate(index, cases, ks, alpha)` runs each prefix through the
  index and computes top-k accuracy where a hit is "any truth
  symbol appears in top-k retrieved."
- `pm-rag eval` CLI subcommand.
- 10 synthetic traces over the demo graph (happy / refunded /
  fraud-review variants).

## [0.3.0] - 2026-04-30

### Added
- Personalized PageRank diffusion (`personalized_pagerank()`).
  Power iteration `r = α s + (1 - α) P^T r` with restart
  probability biasing the walk toward the seed.
- `index.build_index(graph, events)` builds the joint index;
  `query(index, prefix, k, alpha)` returns ranked `Hit` entries.

## [0.2.0] - 2026-04-30

### Added
- Joint graph builder. `CodeGraph` (nodes + weighted directed
  edges) builds `P^T` for diffusion; rejects out-of-range edges
  and negative weights at construction.

## [0.1.0] - 2026-04-30

### Added
- Initial release. Regex-based event → symbol mapping
  (`regex_mapping`): case-insensitive substring match, regex
  characters in event names are escaped.
- Bundled demo: order-fulfillment graph + 8 events (10 nodes,
  8 edges).
- CLI: `query`, `mapping` commands.
- 19 tests; ruff clean. CI matrix on Python 3.10 / 3.11 / 3.12.

[Unreleased]: https://github.com/erphq/pm-rag/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/erphq/pm-rag/releases/tag/v0.6.0
[0.5.0]: https://github.com/erphq/pm-rag/releases/tag/v0.5.0
[0.4.0]: https://github.com/erphq/pm-rag/releases/tag/v0.4.0
[0.3.0]: https://github.com/erphq/pm-rag/releases/tag/v0.3.0
[0.2.0]: https://github.com/erphq/pm-rag/releases/tag/v0.2.0
[0.1.0]: https://github.com/erphq/pm-rag/releases/tag/v0.1.0
