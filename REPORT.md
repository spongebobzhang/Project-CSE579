# MultiplexRAG Final Report

## Query-Aware Retrieval Routing Across Multiple Retrieval Representations

## 1. Abstract

This project studies whether retrieval can be improved by routing each query to the most suitable representation instead of using one fixed strategy for all queries. The system compares sparse, dense, and hybrid retrieval over the same corpus, then trains a lightweight router to choose among them.

The main finding is that no single retrieval strategy is universally best, and the best router recipe is also dataset-dependent. On SciFact, the router outperformed the strongest fixed baseline. On SCIDOCS, a dataset-specific router with improved weak labeling and retrieval-confidence features slightly outperformed the strongest fixed baseline. On NFCorpus, the key improvement came from aligning the fallback policy with the dataset's strongest fixed retriever. A later Phase 3 refinement showed that lightweight query-to-top-document match features further improved the router on SCIDOCS and NFCorpus, while remaining competitive on SciFact. FiQA served as a useful negative result, showing that a router should not be adopted when it does not beat a strong fixed retriever.

Overall, the project supports the central MultiplexRAG hypothesis: query-aware routing can improve retrieval quality, but it works best when the routing policy is designed and selected with respect to the dataset rather than assumed to be universal.

## 2. Introduction

Modern retrieval systems often rely on one dominant representation. Sparse retrieval is effective for exact lexical overlap, acronyms, and rare terms. Dense retrieval is effective for semantic similarity and paraphrased queries. Hybrid retrieval often performs well overall because it combines lexical and semantic evidence, but it may also carry higher computational cost. This project starts from a simple question: if different query types benefit from different retrieval representations, can a learned router choose the right representation per query and improve over always using one fixed strategy?

The project investigates this question through a MultiplexRAG setting with:

- three retrieval branches: sparse, dense, and hybrid
- a query-aware router that selects one branch per query
- an oracle upper bound that shows the best possible branch choice in hindsight

The central goal is not only to compare fixed retrievers, but to test whether adaptive query-level routing improves the quality-cost tradeoff.

## 3. Problem Statement

The project evaluates five decision policies:

- always sparse
- always dense
- always hybrid
- learned router
- oracle best branch per query

The report focuses on the following research questions:

1. Is the best fixed retriever the same across datasets?
2. Can a learned router beat strong fixed baselines?
3. Do different datasets require different router training recipes?
4. How much performance remains between the learned router and the oracle upper bound?

## 4. Datasets

The final report uses three primary datasets and one supplemental dataset.

### 4.1 Primary Datasets

#### SciFact

SciFact is the most compact dataset in the project and uses the provided train/test split directly. It is a good setting for testing whether query-aware routing can outperform a strong hybrid baseline in a relatively controlled environment.

#### SCIDOCS

SCIDOCS is a larger scientific retrieval dataset. In the available local setup, router training uses a query-level self-split derived from the official relevance judgments. SCIDOCS is important because it behaves differently from SciFact: dense retrieval is stronger than hybrid retrieval, which makes it a strong test of whether routing can adapt to dataset-specific retrieval behavior.

#### NFCorpus

NFCorpus is a biomedical retrieval dataset that matches consumer-style natural-language questions against terminology-heavy medical documents. This creates a strong tension between lexical and semantic matching, making it a useful third primary dataset. Unlike SCIDOCS, it also provides a validation split, which makes router recipe selection cleaner and more realistic.

### 4.2 Supplemental Dataset

#### FiQA

FiQA is included as a supplemental generalization check. It is especially useful because it provides a negative result: although router variants improved over weaker baselines, they still did not beat the strongest fixed dense retriever. This helps clarify that a router should only be adopted when it truly improves retrieval quality.

## 5. Method

The system contains four components:

- sparse retriever
- dense retriever
- hybrid retriever
- query-aware router

The workflow is:

1. run all retrieval branches on training queries
2. compare their rankings against relevance judgments
3. assign each query a weak label equal to the best-performing branch
4. train a router to predict the best branch from the query
5. at inference time, run only the selected branch

The router does not predict relevance directly. Instead, it predicts which retrieval representation is most suitable for the query.

### 5.1 Retrieval Branches

- Sparse retrieval captures exact term overlap and lexical matching.
- Dense retrieval captures semantic similarity and paraphrase matching.
- Hybrid retrieval combines the two signals.

### 5.2 Router Design

The router is a lightweight multiclass classifier that uses:

- numeric query features such as query length, punctuation patterns, acronym ratio, and structural cues
- hashed lexical features over query text
- optional retrieval-confidence features in some configurations

The router also supports confidence-aware fallback. When the router is uncertain, it backs off to a fixed retrieval branch.

### 5.3 Weak Supervision

The router is trained with weak labels rather than manual query-type annotations. For each training query:

1. all retrieval branches are evaluated
2. each branch receives a ranking score
3. the best branch becomes the training label

The baseline label rule uses MRR@10. Later experiments also test multi-metric labels, dense-favoring tie handling, and margin-aware labeling.

## 6. Evaluation Protocol

The main ranking metrics are:

- Recall@10
- MRR@10
- nDCG@10

The report treats MRR@10 as the main top-rank-sensitive metric, while Recall@10 and nDCG@10 provide broader ranking-quality context. Router accuracy against weak labels is also tracked, but ranking quality remains the primary evaluation target.

## 7. Main Results

### 7.1 SciFact

| Method | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: |
| Sparse | 0.7757 | 0.6184 | 0.6523 |
| Dense | 0.7900 | 0.6055 | 0.6479 |
| Hybrid | 0.8306 | 0.6563 | 0.6939 |
| Router | 0.8107 | 0.6608 | 0.6945 |
| Oracle | 0.8662 | 0.7337 | 0.7606 |

SciFact provides the cleanest success case. Hybrid retrieval is the strongest fixed baseline, but the learned router slightly outperforms it on both MRR@10 and nDCG@10. This shows that adaptive routing can improve over a strong fixed retrieval strategy in practice.

A Phase 3 follow-up on SciFact tested the same lightweight query-to-top-document match features that were later successful on the other main datasets. On SciFact, this refinement improved Recall@10, nDCG@10, and router accuracy, while causing a small decrease in MRR@10. So the Phase 3 SciFact result is best interpreted as competitive rather than clearly better, but it still shows that the same router-improvement idea remains viable on the third main dataset.

### 7.2 SCIDOCS

| Method | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: |
| Sparse | 0.1660 | 0.3020 | 0.1669 |
| Dense | 0.2510 | 0.3723 | 0.2307 |
| Hybrid | 0.2428 | 0.3432 | 0.2178 |
| Router | 0.2490 | 0.3740 | 0.2284 |
| Oracle | 0.2635 | 0.4636 | 0.2600 |

SCIDOCS behaves differently from SciFact. Here, dense retrieval is the strongest fixed baseline. The default router was not enough, but after improving the weak-label design and adding retrieval-confidence features, the router slightly outperformed fixed dense on MRR@10. This result is important because it shows that the best router recipe is not universal across datasets.

A later Phase 3 refinement also showed that the SCIDOCS router could be improved further. Adding lightweight query-to-top-document match features increased the router from `MRR@10 = 0.3740` to `0.3771`, while also improving `nDCG@10` from `0.2284` to `0.2306`. This matters because it shows that the router is not only effective, but also improvable through targeted feature design.

### 7.3 NFCorpus

#### Test-Split Comparison

| Method | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: |
| Sparse | 0.1522 | 0.5085 | 0.3062 |
| Dense | 0.1590 | 0.5055 | 0.3191 |
| Hybrid | 0.1639 | 0.5455 | 0.3368 |
| Default Router | 0.1690 | 0.5381 | 0.3408 |
| Oracle | 0.1763 | 0.6124 | 0.3606 |

On the final test split, hybrid retrieval is the strongest fixed baseline. The default query-only router improves Recall@10 and nDCG@10, but it does not exceed fixed hybrid on MRR@10. This means the default router is promising, but not yet the right final policy for this dataset.

#### Validation-Based Recipe Selection

| Configuration | Accuracy | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Retrieval features + dense fallback | 0.5710 | 0.1373 | 0.5184 | 0.3066 |
| Retrieval features + hybrid fallback | 0.6481 | 0.1398 | 0.5241 | 0.3102 |
| Retrieval features + hybrid fallback + threshold 0.5 | 0.6698 | 0.1429 | 0.5222 | 0.3131 |
| Retrieval features + hybrid fallback + multi-metric labels | 0.5123 | 0.1379 | 0.5185 | 0.3132 |
| Fixed Hybrid Baseline | - | 0.1388 | 0.5110 | 0.3084 |

NFCorpus provides a different kind of evidence than SciFact and SCIDOCS. Its main contribution is to show that router quality depends strongly on dataset-aware fallback design. The best fixed retriever on NFCorpus is hybrid, not dense. As a result, using dense as the default fallback was a poor safety policy. Once the fallback was changed to hybrid and retrieval-confidence features were enabled, the router surpassed fixed hybrid on validation metrics. This is strong support for the broader thesis that both retrieval preference and router recipe are dataset-dependent.

Phase 3 produced a further positive result on NFCorpus. When the same lightweight query-to-top-document match features were added to the strongest `hybrid`-fallback router, the validation result improved from `MRR@10 = 0.5241` to `0.5270`, and `nDCG@10` improved from `0.3102` to `0.3144`. This is especially useful because it shows that the same router-improvement idea helped on a second dataset rather than only on SCIDOCS.

## 8. Supplemental Result: FiQA

| Method | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: |
| Sparse | 0.2784 | 0.2713 | 0.2175 |
| Dense | 0.4389 | 0.4463 | 0.3687 |
| Hybrid | 0.4308 | 0.4095 | 0.3442 |
| Best FiQA Router Tested | 0.4322 | 0.4342 | 0.3591 |
| Oracle | 0.4859 | 0.5510 | 0.4311 |

FiQA is a useful negative result. Dense retrieval is the strongest fixed baseline, and the best router variant tested still does not surpass it. This strengthens the report because it shows that the project does not assume a router is always better. Instead, the correct conclusion is conditional: routing should be used when it wins empirically, not by default.

## 9. Cross-Dataset Findings

The project produces four main cross-dataset conclusions.

### 9.1 No Fixed Retriever Is Universally Best

- SciFact favors hybrid retrieval.
- SCIDOCS favors dense retrieval.
- NFCorpus favors hybrid retrieval.
- FiQA favors dense retrieval.

This directly supports the original motivation for MultiplexRAG.

### 9.2 A Router Can Improve Over Strong Baselines

- On SciFact, the router beats fixed hybrid.
- On SCIDOCS, the improved router beats fixed dense.
- On NFCorpus, router gains become convincing only after dataset-specific fallback tuning and retrieval-confidence features are introduced.

So the core project hypothesis is supported, but the supporting evidence is not identical across datasets.

### 9.3 Router Design Must Be Dataset-Aware

The datasets do not merely differ in which fixed retriever is strongest. They also differ in which router improvement matters most.

- SciFact works best with the default Phase 2 router recipe, while the later query-match refinement remains competitive.
- SCIDOCS benefits most from improved weak supervision plus retrieval-confidence features.
- NFCorpus benefits most from retrieval-confidence features plus a hybrid fallback policy.
- FiQA does not yet justify router deployment.

Phase 3 further sharpens this conclusion. Not every richer retrieval feature was helpful, but a more targeted query-match feature family improved the router on both SCIDOCS and NFCorpus and remained competitive on SciFact. So the project does not just show that routing can work. It also shows that the router can be improved through principled, dataset-tested design changes.

This is one of the most important outcomes of the project.

### 9.4 There Is Still A Large Oracle Gap

Across all datasets, the oracle remains much stronger than the learned router. That means the project succeeded in showing the value of adaptive routing, but it also leaves clear room for future improvement.

## 10. Discussion

The final report does not support the simplistic claim that one router design is best everywhere. A more accurate conclusion is that MultiplexRAG works as a dataset-aware retrieval-routing framework.

SciFact provides the cleanest success case: the router beats the best fixed retriever directly. SCIDOCS provides a more nuanced success case: the router only wins after the weak-label design is corrected. NFCorpus contributes a third kind of evidence: success depends heavily on choosing the right fallback policy for the dataset. FiQA then provides an important boundary condition by showing that some datasets still favor a fixed retriever.

Together, these outcomes make the project stronger rather than weaker. They show that the central idea is valid, but only when evaluated carefully and adapted to the retrieval setting.

The Phase 3 refinement strengthens this interpretation further. The project was not limited to a single successful router snapshot. It also demonstrated that the router could be iteratively improved in a measurable way through better feature design, with clear gains appearing on SCIDOCS and NFCorpus and competitive behavior on SciFact.

## 11. Limitations

The project also has several limitations.

- The strongest router setting is not the same across datasets.
- SCIDOCS router training relies on a query-level self-split rather than an official train/test router split.
- Router labels are weak labels derived from retrieval outcomes rather than human query-type annotations.
- The learned router remains below the oracle upper bound on every dataset.
- FiQA still favors a fixed dense retriever.

These limitations do not invalidate the results, but they do define the boundary of the current contribution.

## 12. Conclusion

This project asked whether a query-aware router could improve retrieval over a single fixed strategy by choosing among sparse, dense, and hybrid retrieval. The answer is yes, with an important qualification: the best routing policy depends on the dataset.

SciFact shows that a lightweight router can beat the strongest fixed baseline directly. SCIDOCS shows that routing gains depend on better supervision and retrieval-side evidence. NFCorpus shows that routing gains can depend most strongly on choosing the right fallback policy for the dataset. FiQA shows that routing should not be adopted when a fixed dense retriever remains stronger.

The Phase 3 results strengthen the project's main contribution further. Lightweight query-to-top-document match features improved the router on both SCIDOCS and NFCorpus and remained competitive on SciFact. This shows that the router is not just effective once; it can also be improved through targeted and interpretable feature engineering.

The final conclusion is therefore not that there is one universally best retriever or one universally best router. The real conclusion is that retrieval routing works best when it is treated as a dataset-aware decision problem. That is the central contribution of this MultiplexRAG project.
