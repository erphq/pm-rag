# v0.4 TODO

## Eval harness

### Core implementation (`pm_rag/eval.py`)
- [ ] Implement `evaluate(index, prefixes, truth, k=1)`:
  - [ ] Top-1 accuracy
  - [ ] Top-k accuracy
  - [ ] MRR (mean reciprocal rank)
- [ ] Reference `pm_bench.split.case_chrono_split` for split generation
- [ ] Reference `pm_bench.score.score_next_event` as a scoring reference
- [ ] Do NOT add `pm-bench` as a hard dependency yet

### CLI (`pm_rag/cli.py`)
- [ ] Implement `cmd_eval`: load trace JSONL, call `evaluate`, print metrics JSON
- [ ] Consider adding `--index` option for a custom graph/index path (future)

### Tests (`tests/test_eval.py`)
- [ ] Implement `test_evaluate_perfect_index`
- [ ] Implement `test_evaluate_empty_prefixes`
- [ ] Implement `test_evaluate_top_k_accuracy`
- [ ] Implement `test_evaluate_mrr`
