# Contributing to pm-rag

Thanks for considering a contribution. pm-rag is process-aware
retrieval: code-graph indexing plus event-to-symbol mapping. Tests
hermetic: no real LLM, no real embedder, no API keys required.

## Quickstart

```sh
git clone https://github.com/erphq/pm-rag
cd pm-rag
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
ruff check
pytest -q
```

Both commands must pass before opening a PR.

## Project shape

- `pm_rag/graph.py` - code graph construction and Personalized
  PageRank.
- `pm_rag/mapping/` - event-to-symbol strategies. One file per
  strategy: `regex.py`, `embedding.py`, `llm.py`, plus
  `compose.py` for stacking.
- `pm_rag/eval.py` - top-k accuracy and MRR evaluation harness.
- `pm_rag/cli.py` - the `pm-rag` CLI.
- `tests/` - pytest, one file per module.

## Pluggable functions

Every external dependency is a function the caller passes in:

- `EmbedFn = Callable[[list[str]], list[list[float]]]` - turn strings
  into vectors. Tests use a deterministic in-memory embedder.
- `LlmFn = Callable[[str], str]` - turn a prompt into a string.
  Tests use a deterministic fake that returns a recorded JSON array.
- `EventSource` - your store of events. Tests use `FakeEventSource`.

This is the test contract: never reach for a real OpenAI / Anthropic /
local-model client in `pm_rag/`. If you need one, the caller wires it
in.

## Adding a mapping strategy

1. Create `pm_rag/mapping/<strategy>.py` exporting a
   `<strategy>_mapping(events, symbols, ...)` function returning a
   `dict[event_id, list[symbol_id]]`.
2. If it composes naturally, add an `__all__` entry in
   `pm_rag/mapping/__init__.py`.
3. Test with both deterministic input and an empty-events edge case.
4. If it has a numeric threshold or top_k, write a test pinning the
   expected ordering. **Hand-trace the algorithm before locking the
   assertion**: for diffusion, embedding similarity, and LLM-rerank,
   the seed is rarely top-1.

## Conventions

- Python 3.10+. Type hints required on public functions.
- Ruff config in `pyproject.toml`. Don't disable rules ad-hoc.
- No em dashes in code, comments, or docs.
- Commit messages: `feat(mapping): ...` / `fix(graph): ...` / `docs(...)`.

## Releasing

Releases are tagged on GitHub. There is no PyPI publish workflow yet.
