# Data Directory Guide

This directory stores all datasets, indexes, and temporary download artifacts used by the MultiplexRAG project.

## Recommended Layout

Each dataset should live in its own subdirectory:

```text
data/
  <dataset_name>/
    raw/
    processed/
```

Example:

```text
data/
  scifact/
    raw/
      corpus.jsonl
      queries.jsonl
      qrels_train.jsonl
      qrels_test.jsonl
    processed/
      corpus.pkl
      dense.pkl
      sparse.pkl
```

This layout allows multiple datasets to coexist without overwriting one another.

## Folder Purposes

### `data/<dataset>/raw`

Stores the raw dataset files used by the retrieval pipeline.

Typical files:

- `corpus.jsonl`: document collection
- `queries.jsonl`: query set
- `qrels_train.jsonl`: training relevance labels
- `qrels_test.jsonl`: test relevance labels

These files are the main source inputs for indexing, retrieval, evaluation, and router training.

### `data/<dataset>/processed`

Stores derived artifacts built from the raw dataset.

Typical files:

- `corpus.pkl`: normalized corpus cache
- `dense.pkl`: dense retriever index / model artifact
- `sparse.pkl`: sparse retriever index artifact

These files can be regenerated from the corresponding `raw/` directory.

## Current Contents

### `data/scifact/`

This is the current dataset-specific directory for the SciFact benchmark.

- `data/scifact/raw/` contains the SciFact corpus, queries, and qrels.
- `data/scifact/processed/` contains the built sparse and dense indexes.

### `data/fiqa/`

This is the current dataset-specific directory for the FiQA benchmark.

- `data/fiqa/raw/` contains the FiQA corpus, queries, and qrels.
- `data/fiqa/processed/` is the intended location for FiQA indexes.
- `data/fiqa/.tmp_beir/` currently stores temporary download artifacts created during dataset preparation.

## Legacy Compatibility Paths

The following directories still exist for backward compatibility with older commands:

- `data/raw/`
- `data/processed/`

They currently contain copies of the SciFact files and indexes. New experiments should prefer the dataset-specific layout under `data/<dataset>/...`.

## Temporary Download Cache

### `data/.tmp_beir/`

This directory contains temporary files created while downloading or unpacking BEIR datasets.

Examples:

- downloaded zip files
- extracted BEIR dataset contents

This directory is not the canonical storage location for experiments. It is only a staging/cache area.

### `data/fiqa/.tmp_beir/`

This is a dataset-local temporary cache created while downloading and unpacking FiQA.

It is also not a canonical experiment directory and can be removed later if you want to clean up disk usage.

## Adding a New Dataset

To add a new dataset such as FiQA, use a new dataset-specific directory:

```bash
python scripts/prepare_data.py --dataset BeIR/fiqa --qrels-dataset BeIR/fiqa-qrels
```

This will create:

```text
data/fiqa/raw/
```

Then build indexes into:

```text
data/fiqa/processed/
```

using:

```bash
python scripts/build_index.py --dataset-dir data/fiqa
```

## Usage Recommendation

For all future experiments, prefer commands that use `--dataset-dir`, for example:

```bash
python scripts/build_index.py --dataset-dir data/scifact
python scripts/retrieve.py --dataset-dir data/scifact --mode hybrid --out results/scifact_hybrid.jsonl
python scripts/evaluate.py --dataset-dir data/scifact --pred results/scifact_hybrid.jsonl
python scripts/train_router.py --dataset-dir data/scifact --report-out results/scifact_router_report.json
```

This keeps experiments reproducible and avoids dataset collisions.
