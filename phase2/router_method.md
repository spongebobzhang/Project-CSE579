# Phase 2 Router Method

## Goal

In Phase 2, the router is treated as a query-aware strategy selector for MultiplexRAG.

It does not retrieve documents by itself. Instead, it predicts which retrieval branch should handle a query:

- `sparse`
- `dense`
- `hybrid`

The learning target is:

`query text -> best retrieval strategy`

So the router sits above the retrievers and decides which retrieval representation is most appropriate for each query.

## Why This Version Matters

The earlier router was a weak baseline built on a very small handcrafted feature set and a nearest-centroid classifier. That version showed that query-aware routing had potential, but it did not make full use of the information already available in the query text.

This Phase 2 version improves that design in three ways:

1. richer query-side features
2. a stronger but still lightweight classifier
3. confidence-aware fallback behavior

The result is a router that is still easy to explain, but is substantially more expressive and more practical.

## Implementation Files

The current Phase 2 router process is mainly implemented in:

- [scripts/train_router.py](/home/zzhan621/CSE579/Project-CSE579/scripts/train_router.py)
- [src/multiplexrag/router.py](/home/zzhan621/CSE579/Project-CSE579/src/multiplexrag/router.py)
- [src/multiplexrag/retrieval.py](/home/zzhan621/CSE579/Project-CSE579/src/multiplexrag/retrieval.py)
- [src/multiplexrag/eval_utils.py](/home/zzhan621/CSE579/Project-CSE579/src/multiplexrag/eval_utils.py)

## End-to-End Process

The Phase 2 router pipeline works as follows:

1. Load corpus, queries, and qrels.
2. Load or build the three retrieval branches: `sparse`, `dense`, and `hybrid`.
3. Run all three branches for each query.
4. Score each branch against qrels using ranking metrics, especially `MRR@10`.
5. Assign each training query a weak supervision label equal to the best-performing branch.
6. Convert each query into a richer feature representation.
7. Train a class-balanced logistic-regression router.
8. Predict a retrieval mode for each test query.
9. If the prediction confidence is low, back off to a default fallback branch.
10. Evaluate the final router decisions against fixed baselines and the oracle upper bound.

So the Phase 2 router is still a supervised strategy selector, but the supervision comes from retrieval outcomes rather than manual query annotations.

## Supervision Process

The router uses weak supervision.

For each query in the training split:

1. retrieve with `sparse`
2. retrieve with `dense`
3. retrieve with `hybrid`
4. compare those ranked lists against ground-truth qrels
5. choose the best branch as the training label

In the current implementation, the label decision is produced by `best_mode_for_query(...)` in [router.py](/home/zzhan621/CSE579/Project-CSE579/src/multiplexrag/router.py).

Default weak-label configuration:

- metric: `MRR@10`
- tie preference: `hybrid`

Optional weak-label settings also support:

- multiple metrics such as `MRR@10`, `nDCG@10`, and `Recall@10`
- custom metric weights
- a different tie preference such as `dense`

This means the router learns branch selection rather than document relevance.

## Query Representation

This version uses a richer query-only representation than Phase 1.

### Numeric Query Features

The numeric feature block includes:

- character length
- token length
- average token length
- unique-token ratio
- digit count
- digit ratio
- acronym count
- acronym ratio
- uppercase ratio
- punctuation count
- punctuation ratio
- punctuation variety
- stopword ratio
- long-token ratio
- short-token ratio
- alphabetic-character ratio
- whether the query contains `?`
- whether it contains `:`
- whether it contains `/`
- whether it contains `-`
- whether it contains parentheses
- whether it starts with a WH-word
- whether it contains comparative cues such as `vs`, `compare`, or `better`

These features help separate short lexical lookup queries from more semantic, question-like, or structurally complex queries.

### Hashed Lexical Features

In addition to numeric statistics, the router also uses hashed text features:

- word unigrams and bigrams
- character n-grams

This adds much more expressiveness without requiring a heavy neural router model or a manually maintained vocabulary.

## Router Model

The current classifier is a class-balanced logistic-regression model.

Reasons for this choice:

- stronger than nearest centroid
- lightweight and fast to train
- stable on small and medium datasets
- easy to explain in a report
- provides class probabilities for confidence-aware routing

Before training, the numeric features are standardized. The lexical hashed features are combined with the numeric block into one sparse feature matrix. The classifier is then trained to predict one of the three routing classes:

- `sparse`
- `dense`
- `hybrid`

## Confidence-Aware Fallback

One important addition in this version is fallback routing.

At inference time:

1. the router predicts class probabilities
2. if the top probability is high enough, the corresponding branch is used
3. if the top probability is too low, the router falls back to a default branch

Current default setting:

- confidence threshold: `0.45`
- fallback branch: `dense`

This design is useful because uncertain router decisions are often where errors happen. Instead of forcing a brittle decision, the system backs off to a stable baseline.

## Dataset Setup

### SciFact

SciFact uses the provided train/test split directly:

- train qrels: `data/scifact/raw/qrels_train.jsonl`
- test qrels: `data/scifact/raw/qrels_test.jsonl`

### SCIDOCS

SCIDOCS does not provide a separate train qrels split in the downloaded setup, so the project uses a query-level self-split:

- source qrels: `data/scidocs/raw/qrels_test.jsonl`
- generated train qrels: `data/scidocs/raw/qrels_router_train.jsonl`
- generated test qrels: `data/scidocs/raw/qrels_router_test.jsonl`

The split is done on unique `query_id` values with a fixed random seed, while keeping the original relevance labels unchanged.

## Efficiency Notes

The training script now reuses cached processed retriever artifacts when available:

- `data/<dataset>/processed/sparse.pkl`
- `data/<dataset>/processed/dense.pkl`

That makes router iteration much faster than rebuilding retrievers from scratch each time.

The pipeline also tracks:

- average latency
- estimated retrieved-context tokens
- estimated total tokens
- estimated cost in USD

These are approximate efficiency signals for comparison across routing choices.

## How To Run

The default router training call uses `MRR@10` as the weak-label metric:

```bash
python scripts/train_router.py \
  --dataset-dir data/scidocs \
  --model-out results/scidocs/router.pkl \
  --report-out results/scidocs/router_report.json \
  --pred-out results/scidocs/router_predictions.jsonl
```

If you want to use multi-metric weak labels, you can combine multiple ranking metrics and choose a tie preference explicitly. For example:

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

Useful router-training options:

- `--label-metrics`: metrics used to create weak labels
- `--label-weights`: weights for the corresponding label metrics
- `--label-tie-preference`: which branch wins if combined label scores tie
- `--use-retrieval-features`: add retrieval-confidence features to the router input representation
- `--router-confidence-threshold`: fallback threshold at inference time
- `--router-fallback-mode`: default branch used when the router is uncertain

The same router options can now also be passed through the one-command pipeline. For example:

```bash
python scripts/run_pipeline.py \
  --dataset-dir data/scidocs \
  --label-metrics mrr@10 ndcg@10 recall@10 \
  --label-weights 0.5 0.3 0.2 \
  --label-tie-preference dense
```

## Current Results Summary

This document tracks two different result snapshots:

- archived default-router snapshot in `phase1/<dataset>/router_report.json`
- newer experimental runs in `results/<dataset>/router_report.json`

The default Phase 2 numbers below refer to the archived default-router snapshot with weak-label rule `MRR@10` and tie preference `hybrid`, not the later multi-metric experiment outputs in `results/`.

### SciFact

- Router accuracy: `0.7333`
- Router `MRR@10`: `0.6608`
- Best fixed baseline `MRR@10`: `0.6563` from `hybrid`
- Oracle `MRR@10`: `0.7337`

Interpretation:

- the upgraded router now slightly beats the best fixed baseline on SciFact
- there is still a clear oracle gap, so routing can improve further

### SCIDOCS

- Router accuracy: `0.5000`
- Router `MRR@10`: `0.3620`
- Best fixed baseline `MRR@10`: `0.3723` from `dense`
- Oracle `MRR@10`: `0.4636`

Interpretation:

- the upgraded router improves over the earlier centroid version
- it still does not beat always-`dense` on SCIDOCS
- the remaining gap suggests the router still needs stronger signals

## Comparison With Multi-Metric Weak Labels

After the default Phase 2 router was established, an alternative labeling strategy was also tested.

Instead of creating weak labels from `MRR@10` only, the alternative version used:

- `MRR@10`
- `nDCG@10`
- `Recall@10`

with weights:

- `0.5`
- `0.3`
- `0.2`

and tie preference set to `dense`.

This was intended to reduce the hybrid bias observed on SCIDOCS and produce more balanced weak labels.

The comparison below uses:

- default snapshot from `phase1/<dataset>/router_report.json`
- multi-metric experimental run from `results/<dataset>/router_report.json`

### SciFact: Default vs Multi-Metric Labels

Default Phase 2 router:

- accuracy: `0.7333`
- `MRR@10`: `0.6608`
- `nDCG@10`: `0.6945`

Multi-metric weak-label router:

- accuracy: `0.7267`
- `MRR@10`: `0.6539`
- `nDCG@10`: `0.6912`

Interpretation:

- the multi-metric version was slightly worse than the default router on SciFact
- the label distribution shifted heavily toward `dense`
- that shift appears to over-correct relative to the earlier hybrid-heavy labeling

### SCIDOCS: Default vs Multi-Metric Labels

Default Phase 2 router:

- accuracy: `0.5000`
- `MRR@10`: `0.3620`
- `nDCG@10`: `0.2277`

Multi-metric weak-label router:

- accuracy: `0.6050`
- `MRR@10`: `0.3623`
- `nDCG@10`: `0.2240`

Interpretation:

- the multi-metric version substantially improved classification accuracy
- it also reduced the earlier hybrid-heavy label skew
- however, the ranking gain was extremely small on `MRR@10`
- `nDCG@10` slightly decreased
- the router still did not beat the fixed `dense` baseline

### Main Takeaway

The multi-metric weak-label strategy is more reasonable from a supervision-design perspective, especially for SCIDOCS where `dense` and `hybrid` tie often under `MRR@10`.

However, in the current experiments:

- it did not outperform the default Phase 2 router on SciFact
- it only produced a marginal `MRR@10` gain on SCIDOCS
- it therefore remains an important ablation and alternative labeling strategy, rather than the new default router configuration

## Main Strengths

- much stronger than the original centroid baseline
- still lightweight and reproducible
- uses only query-side information
- supports uncertainty-aware routing
- already strong enough to beat the best fixed strategy on SciFact

## Current Limitations

- it still relies only on query-side signals
- it does not yet use retrieval-confidence features such as top-score gaps or rank-list overlap
- the fallback policy is global rather than learned per dataset
- SCIDOCS evaluation still depends on a self-split rather than an official train/test split
- the router remains below the oracle upper bound on both datasets

## Next Phase-2 Improvements

The most natural next improvements are:

1. add retrieval-confidence features from the candidate branches
2. add per-query error analysis to explain wrong routing decisions
3. tune or learn fallback behavior per dataset
4. explore cost-aware labeling so the router prefers cheaper branches when quality is close
5. evaluate whether a tree model or small neural router improves over logistic regression without losing interpretability

## Short Summary

The current Phase 2 router is a confidence-aware logistic-regression selector over richer query features and hashed lexical features. It is still simple enough to explain clearly, but it performs much better than the original baseline and gives a more credible foundation for the next stage of MultiplexRAG routing work.
