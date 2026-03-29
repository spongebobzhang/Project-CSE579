# MultiplexRAG Project Starter

This project follows the proposal goal: compare Dense/Sparse/Hybrid retrieval and build a query-aware router.

## 1) Environment

```bash
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

1. Prepare dataset and use the provided train/test qrels split.
2. Build BM25 baseline.
3. Build dense baseline.
4. Build hybrid fusion baseline.
5. Evaluate fixed baselines.
6. Train router on query features.
7. Analyze results by query type and compare with the oracle upper bound.

## 4) Multi-dataset workflow

```bash
python scripts/prepare_data.py --dataset BeIR/scifact --qrels-dataset BeIR/scifact-qrels
python scripts/prepare_data.py --dataset BeIR/scidocs --qrels-dataset BeIR/scidocs-qrels

python scripts/build_index.py --dataset-dir data/scifact
python scripts/retrieve.py --dataset-dir data/scifact --mode sparse --topk 10 --out results/scifact/sparse.jsonl
python scripts/retrieve.py --dataset-dir data/scifact --mode dense --topk 10 --out results/scifact/dense.jsonl
python scripts/retrieve.py --dataset-dir data/scifact --mode hybrid --topk 10 --out results/scifact/hybrid.jsonl
python scripts/evaluate.py --dataset-dir data/scifact --pred results/scifact/hybrid.jsonl
python scripts/train_router.py \
  --dataset-dir data/scifact \
  --model-out results/scifact/router.pkl \
  --report-out results/scifact/router_report.json \
  --pred-out results/scifact/router_predictions.jsonl
```

## 5) One-command pipeline

Run the whole pipeline for one prepared dataset:

```bash
python scripts/run_pipeline.py --dataset-dir data/scifact
python scripts/run_pipeline.py --dataset-dir data/scidocs
```

If you only want baselines and evaluation first:

```bash
python scripts/run_pipeline.py --dataset-dir data/scidocs --skip-router
```

If you want the script to prepare a BEIR dataset before running:

```bash
python scripts/run_pipeline.py \
  --dataset-dir data/scifact \
  --prepare-dataset BeIR/scifact \
  --qrels-dataset BeIR/scifact-qrels
```

If a dataset only provides `qrels_test.jsonl`, the pipeline automatically creates a router-only query-level split:

- `qrels_router_train.jsonl`
- `qrels_router_test.jsonl`
- `qrels_router_split_meta.json`

The split protocol is:

- keep the official relevance labels unchanged
- split on unique `query_id`, not individual qrels rows
- use a fixed random seed for reproducibility
- use the generated split only for router training/evaluation
- keep the original `qrels_test.jsonl` untouched for the official dataset artifact

You can control the split with:

```bash
python scripts/run_pipeline.py --dataset-dir data/scidocs --router-train-ratio 0.8 --router-split-seed 42
```

## 6) Router labeling options

By default, router weak labels are created from `MRR@10` only:

```bash
python scripts/train_router.py \
  --dataset-dir data/scidocs \
  --model-out results/scidocs/router.pkl \
  --report-out results/scidocs/router_report.json \
  --pred-out results/scidocs/router_predictions.jsonl
```

You can also train the router with multi-metric weak labels. For example, combine `MRR@10`, `nDCG@10`, and `Recall@10`, and prefer `dense` on ties:

```bash
python scripts/train_router.py \
  --dataset-dir data/scidocs \
  --model-out results/scidocs/router.pkl \
  --report-out results/scidocs/router_report.json \
  --pred-out results/scidocs/router_predictions.jsonl \
  --label-metrics mrr@10 ndcg@10 recall@10 \
  --label-weights 0.5 0.3 0.2 \
  --label-tie-preference dense
```

Useful label-related arguments:

- `--label-metrics`: metrics used to create weak labels
- `--label-weights`: weights aligned with `--label-metrics`
- `--label-tie-preference`: which branch wins when combined scores tie
- `--use-retrieval-features`: add retrieval-confidence features to the router feature vector

The same options now work through the one-command pipeline. For example:

```bash
python scripts/run_pipeline.py \
  --dataset-dir data/scidocs \
  --label-metrics mrr@10 ndcg@10 recall@10 \
  --label-weights 0.5 0.3 0.2 \
  --label-tie-preference dense
```

## 7) Notes

- The loaders normalize BEIR-style `_id` fields into `doc_id` / `query_id`.
- If `sentence-transformers` is unavailable, dense retrieval falls back to an offline hashing-based encoder so experiments still run.
- Existing `data/raw` and `data/processed` paths still work, but `data/<dataset>/...` is the recommended layout for multiple datasets.
- For datasets without an official training qrels split, router experiments use a query-level self-split derived from the official qrels with a fixed random seed.
- The recommended output layout is `results/<dataset>/...`, which matches `scripts/run_pipeline.py`.
