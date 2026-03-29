# Phase 1 Summary

This folder stores the Phase 1 comparison snapshot for the MultiplexRAG project.

## Scope

Phase 1 focuses on comparing:

- `sparse` retrieval
- `dense` retrieval
- `hybrid` retrieval
- `router` selection
- `oracle` upper bound

The current experiment uses:

- Corpus: `data/raw/corpus.jsonl`
- Queries: `data/raw/queries.jsonl`
- Train qrels: `data/raw/qrels_train.jsonl`
- Test qrels: `data/raw/qrels_test.jsonl`
- Top-k: `10`

## Main Results

| Method | Recall@10 | MRR@10 | nDCG@10 | Avg Latency (ms) |
| --- | ---: | ---: | ---: | ---: |
| Sparse | 0.7816 | 0.6337 | 0.6644 | 15.52 |
| Dense | 0.7900 | 0.6055 | 0.6479 | 7.59 |
| Hybrid | 0.8329 | 0.6561 | 0.6952 | 23.08 |
| Router | 0.7957 | 0.6515 | 0.6832 | 18.94 |
| Oracle | 0.8762 | 0.7335 | 0.7639 | 20.48 |

## Takeaways

- `hybrid` is the strongest fixed retrieval strategy in this phase.
- `router` nearly matches `hybrid` on MRR@10 while reducing latency.
- `oracle` is still substantially better than the learned router, so there is room to improve routing.
- The current phase supports the proposal claim that query-aware routing is meaningful once dense retrieval becomes competitive.

## Router Snapshot

- Train queries: `809`
- Test queries: `300`
- Train label distribution:
  - `dense`: `101`
  - `hybrid`: `637`
  - `sparse`: `71`
- Router classification accuracy: `0.4133`

## Files

- [phase1_results.json](/home/fangzz/LLM/multiplexrag/phase1/phase1_results.json): structured result snapshot for this phase
- [router_method.md](/home/fangzz/LLM/multiplexrag/phase1/router_method.md): explanation of the current router design
- Original experiment report: [router_report.json](/home/fangzz/LLM/multiplexrag/results/router_report.json)
- Original router predictions: [router_predictions.jsonl](/home/fangzz/LLM/multiplexrag/results/router_predictions.jsonl)
