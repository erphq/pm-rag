# Goals

## North star
Show that retrieval conditioned on process state beats embedding
similarity on tasks where "what's next" matters more than "what's similar".

## v0 success criteria
- Joint graph builder runs end-to-end on the bundled demo ✅ (50k LOC
  repo target deferred to v1 — needs codegraph integration)
- Event→symbol mapping handles regex ✅; embedding strategy pending v0.5
- PPR diffusion produces stable rankings (deterministic seed) ✅
- Eval harness with reproducible top-k accuracy on a fixed task ✅
  (`pm-rag eval` reports 31% / 71% / 95% / 100% top-1/3/5/10 against
  the demo)

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
