# MultiplexRAG Project Starter

This project follows the proposal goal: compare Dense/Sparse/Hybrid retrieval and build a query-aware router.

## 1) Environment

```bash
cd /home/fangzz/LLM/multiplexrag
source ../.venv/bin/activate
pip install -r requirements.txt
```

## 2) Project layout

- `data/<dataset>/raw`: corpus + qrels + query files for one dataset
- `data/<dataset>/processed`: indexes for one dataset
- `results/`: retrieval outputs, evaluation summaries, and router artifacts
- `scripts/build_index.py`: build dense and sparse indexes
- `scripts/retrieve.py`: run baseline retrieval
- `scripts/train_router.py`: train query router
- `scripts/evaluate.py`: report Recall@k / nDCG@k / latency / cost
- `docs/roadmap.md`: phased execution checklist

## 3) Suggested execution order

1. Prepare dataset and split by query type.
2. Build BM25 baseline.
3. Build dense baseline.
4. Build hybrid fusion baseline.
5. Train router on query features.
6. Evaluate vs baselines and oracle upper bound.

## 4) Multi-dataset workflow

```bash
python scripts/prepare_data.py --dataset BeIR/scifact --qrels-dataset BeIR/scifact-qrels
python scripts/prepare_data.py --dataset BeIR/fiqa --qrels-dataset BeIR/fiqa-qrels

python scripts/build_index.py --dataset-dir data/scifact
python scripts/retrieve.py --dataset-dir data/scifact --mode sparse --topk 10 --out results/scifact_sparse.jsonl
python scripts/retrieve.py --dataset-dir data/scifact --mode dense --topk 10 --out results/scifact_dense.jsonl
python scripts/retrieve.py --dataset-dir data/scifact --mode hybrid --topk 10 --out results/scifact_hybrid.jsonl
python scripts/evaluate.py --dataset-dir data/scifact --pred results/scifact_hybrid.jsonl
python scripts/train_router.py --dataset-dir data/scifact --report-out results/scifact_router_report.json
```

## 5) Notes

- The loaders normalize BEIR-style `_id` fields into `doc_id` / `query_id`.
- If `sentence-transformers` is unavailable, dense retrieval falls back to an offline hashing-based encoder so experiments still run.
- Existing `data/raw` and `data/processed` paths still work, but `data/<dataset>/...` is the recommended layout for multiple datasets.
