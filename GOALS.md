# Goals

## North star
Show that retrieval conditioned on process state beats embedding
similarity on tasks where "what's next" matters more than "what's similar".

## v0 success criteria
- Joint graph builder runs end-to-end on a 50k LOC repo + 100k event log
- Event→symbol mapping handles regex + embedding strategies
- PPR diffusion produces stable rankings (deterministic seed)

## v1 success criteria
- Beats baseline embedding RAG by ≥10pp on the next-event localization
  task (defined in pm-bench)
- Used as a retrieval backend in ≥1 agent system

## Architecture decisions
- Python first; Rust hot path only if eval shows we need it
- Reuse `codegraph` for code graph extraction (private dep, public API)
- Reuse `pm-bench` datasets for evaluation
- Diffusion via `scipy.sparse` PPR

## Non-goals
- General-purpose RAG framework
- Inventing new graph algorithms (use known ones, apply them well)
- LLM-from-scratch mapping (we layer on top, we don't replace)

## Out of scope (for now)
- Real-time indexing
- Multi-language code graphs (start with Python + TypeScript)
- Distributed PPR
