# Current Router Method

## Goal

The router does not retrieve documents directly. Its job is to choose one retrieval strategy for each query:

- `sparse`
- `dense`
- `hybrid`

## How Supervision Is Created

The router is trained with weak supervision derived from retrieval outcomes.

For each training query:

1. Run all three retrievers.
2. Compare their retrieval quality using `MRR@10`.
3. Assign the query a label equal to the best-performing strategy.

This means the router learns:

`query text -> best retrieval strategy`

It does not learn document relevance directly.

## Current Query Features

The current implementation uses lightweight handcrafted features from the query text:

- character length
- token length
- average token length
- unique-token ratio
- number of digits
- number of acronyms
- whether the query contains a question mark

These features are defined in [router.py](/home/fangzz/LLM/multiplexrag/src/multiplexrag/router.py).

## Current Model

The current router is a centroid-based classifier.

Training:

1. Convert each training query into a feature vector.
2. Group vectors by label: `dense`, `sparse`, `hybrid`.
3. Compute one centroid per label.

Inference:

1. Convert a test query into the same feature vector.
2. Measure its distance to each label centroid.
3. Predict the nearest label.

So the current policy is a simple nearest-centroid routing classifier.

## Why This Is Useful

- very fast
- easy to explain in a proposal or report
- good enough for a first routing baseline

## Current Limitations

- features are shallow
- the classifier is simple
- it does not yet use retrieval-score features such as BM25 top score, dense top score, or overlap between ranked lists
- there is still a large gap between `router` and `oracle`

## Recommended Next Improvements

- add retrieval-confidence features
- add query-type features such as punctuation, percent signs, hyphens, and rare-term statistics
- try a stronger classifier such as logistic regression or random forest
- explore cost-aware labeling so the router prefers cheaper strategies when quality is close
