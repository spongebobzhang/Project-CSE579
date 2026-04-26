# Contributions on the `costomato` branch

Three small additions. None of them change retrieval code, router code, or any reported numbers.

## 1. Tests for `eval_utils`

`tests/test_eval_utils.py` — 26 unit tests for `recall@k`, `mrr@k`, `ndcg@k`, and the metric dispatcher. Run with `python -m unittest discover tests`.

Why: every number in the report comes from these four functions. There were zero tests before.

## 2. Result plots

`scripts/plot_results.py` — reads existing JSON in `results/` and writes:

- `results/<dataset>/comparison.png` for SciFact, SCIDOCS, NFCorpus
- `results/cross_dataset_summary.png`

The four PNGs are embedded in `REPORT.md`. No numbers are recomputed.

Why: the report had six dense numeric tables and zero charts. The bar charts make the dataset-aware-routing story easy to see at a glance.

## 3. Dataset stats

`scripts/dataset_stats.py` — walks `data/<dataset>/raw/` and writes `results/dataset_stats.json` and `results/dataset_stats.md`. A new section 4.2 in `REPORT.md` shows the table.

Why: the report described datasets in prose without quoting any numbers. The table also surfaces that FiQA is prepared but not yet evaluated.

## What this branch does not touch

- No changes to `src/multiplexrag/`
- No re-runs, no new experiments, no new dependencies
- No changes to any number already in `REPORT.md`
