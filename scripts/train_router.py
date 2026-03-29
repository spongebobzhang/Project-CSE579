#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args():
    p = argparse.ArgumentParser(description="Train query router for MultiplexRAG.")
    p.add_argument("--dataset-dir", default="", help="Dataset root containing raw/ and processed/ folders")
    p.add_argument("--queries", default="data/raw/queries.jsonl")
    p.add_argument("--corpus", default="data/raw/corpus.jsonl")
    p.add_argument("--train-qrels", default="data/raw/qrels_train.jsonl")
    p.add_argument("--test-qrels", default="data/raw/qrels_test.jsonl")
    p.add_argument("--model-out", default="results/router.pkl")
    p.add_argument("--report-out", default="results/router_report.json")
    p.add_argument("--pred-out", default="results/router_predictions.jsonl")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--policy", choices=["classifier"], default="classifier")
    return p.parse_args()


def collect_query_results(queries: list[dict], retrievers: dict, topk: int) -> dict[str, dict]:
    collected: dict[str, dict] = {}
    for query in queries:
        by_mode = {}
        for mode, retriever in retrievers.items():
            results, latency_ms = retriever.search(query["text"], topk=topk)
            by_mode[mode] = {
                "results": results,
                "doc_ids": [item["doc_id"] for item in results],
                "latency_ms": float(latency_ms),
            }
        collected[query["query_id"]] = by_mode
    return collected


def label_queries(
    queries: list[dict],
    qrels: dict[str, set[str]],
    query_results: dict[str, dict],
    topk: int,
) -> tuple[list[dict], list[str]]:
    from multiplexrag.router import best_mode_for_query

    kept_queries = []
    labels = []
    for query in queries:
        relevant = qrels.get(query["query_id"])
        if not relevant:
            continue
        results_by_mode = {
            mode: payload["doc_ids"]
            for mode, payload in query_results[query["query_id"]].items()
        }
        kept_queries.append(query)
        labels.append(best_mode_for_query(results_by_mode, relevant, k=topk))
    return kept_queries, labels


def summarize_metrics(
    metrics: list[str],
    chosen_doc_ids: dict[str, list[str]],
    chosen_latencies: dict[str, float],
    qrels: dict[str, set[str]],
) -> dict[str, float]:
    from multiplexrag.eval_utils import score_metric

    summary = {metric: 0.0 for metric in metrics}
    counted = 0
    latency_total = 0.0
    for query_id, relevant in qrels.items():
        doc_ids = chosen_doc_ids.get(query_id, [])
        for metric in metrics:
            summary[metric] += score_metric(metric, doc_ids, relevant)
        latency_total += chosen_latencies.get(query_id, 0.0)
        counted += 1
    if counted == 0:
        return {**summary, "queries_evaluated": 0, "avg_latency_ms": 0.0}
    for metric in metrics:
        summary[metric] /= counted
    summary["queries_evaluated"] = counted
    summary["avg_latency_ms"] = latency_total / counted
    return summary


def choose_mode_outputs(
    query_results: dict[str, dict],
    qrels: dict[str, set[str]],
    selection: dict[str, str],
    metrics: list[str],
) -> dict[str, float]:
    chosen_doc_ids = {}
    chosen_latencies = {}
    for query_id in qrels:
        mode = selection[query_id]
        chosen_doc_ids[query_id] = query_results[query_id][mode]["doc_ids"]
        chosen_latencies[query_id] = query_results[query_id][mode]["latency_ms"]
    return summarize_metrics(metrics, chosen_doc_ids, chosen_latencies, qrels)


def main():
    args = parse_args()
    from multiplexrag.data import load_corpus, load_queries, load_qrels
    from multiplexrag.retrieval import DenseRetriever, HybridRetriever, SparseRetriever
    from multiplexrag.router import QueryRouter, best_mode_for_query

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    corpus_path = dataset_dir / "raw" / "corpus.jsonl" if dataset_dir else Path(args.corpus)
    queries_path = dataset_dir / "raw" / "queries.jsonl" if dataset_dir else Path(args.queries)
    train_qrels_path = dataset_dir / "raw" / "qrels_train.jsonl" if dataset_dir else Path(args.train_qrels)
    test_qrels_path = dataset_dir / "raw" / "qrels_test.jsonl" if dataset_dir else Path(args.test_qrels)

    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    train_qrels = load_qrels(train_qrels_path)
    test_qrels = load_qrels(test_qrels_path)
    metrics = [f"recall@{args.topk}", f"mrr@{args.topk}", f"ndcg@{args.topk}"]

    sparse = SparseRetriever().fit(corpus)
    dense = DenseRetriever().fit(corpus)
    hybrid = HybridRetriever(sparse, dense)
    retrievers = {"sparse": sparse, "dense": dense, "hybrid": hybrid}

    query_results = collect_query_results(queries, retrievers, args.topk)
    train_queries, train_labels = label_queries(queries, train_qrels, query_results, args.topk)
    test_queries, test_labels = label_queries(queries, test_qrels, query_results, args.topk)

    router = QueryRouter().fit(train_queries, train_labels)
    preds = router.predict(test_queries)
    labels = sorted(set(train_labels) | set(test_labels))
    test_query_ids = [query["query_id"] for query in test_queries]
    gold_selection = {
        query_id: gold
        for query_id, gold in zip(test_query_ids, test_labels)
    }
    router_selection = {
        query_id: pred
        for query_id, pred in zip(test_query_ids, preds)
    }
    oracle_selection = {}
    for query in test_queries:
        query_id = query["query_id"]
        relevant = test_qrels[query_id]
        results_by_mode = {
            mode: payload["doc_ids"]
            for mode, payload in query_results[query_id].items()
        }
        oracle_selection[query_id] = best_mode_for_query(results_by_mode, relevant, k=args.topk)

    baseline_metrics = {}
    for mode in ("sparse", "dense", "hybrid"):
        selection = {query_id: mode for query_id in test_qrels}
        baseline_metrics[mode] = choose_mode_outputs(query_results, test_qrels, selection, metrics)
    router_metrics = choose_mode_outputs(query_results, test_qrels, router_selection, metrics)
    oracle_metrics = choose_mode_outputs(query_results, test_qrels, oracle_selection, metrics)

    prediction_rows = []
    for query in test_queries:
        query_id = query["query_id"]
        chosen_mode = router_selection[query_id]
        chosen = query_results[query_id][chosen_mode]
        prediction_rows.append(
            {
                "query_id": query_id,
                "query": query["text"],
                "predicted_mode": chosen_mode,
                "oracle_mode": oracle_selection[query_id],
                "gold_mode": gold_selection[query_id],
                "latency_ms": chosen["latency_ms"],
                "results": chosen["results"],
            }
        )

    report = {
        "accuracy": (
            sum(int(gold == pred) for gold, pred in zip(test_labels, preds)) / len(test_labels)
            if test_labels
            else 0.0
        ),
        "labels": labels,
        "test_size": len(test_labels),
        "metrics": metrics,
        "baselines": baseline_metrics,
        "router": router_metrics,
        "oracle": oracle_metrics,
    }

    router.save(args.model_out)
    Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    from multiplexrag.data import write_jsonl

    write_jsonl(args.pred_out, prediction_rows)

    label_counts = {label: train_labels.count(label) for label in sorted(set(train_labels))}
    print(f"Train queries: {len(train_queries)}")
    print(f"Test queries : {len(test_queries)}")
    print(f"Train label distribution: {label_counts}")
    print("\nComparison:")
    for name, payload in (
        ("sparse", baseline_metrics["sparse"]),
        ("dense", baseline_metrics["dense"]),
        ("hybrid", baseline_metrics["hybrid"]),
        ("router", router_metrics),
        ("oracle", oracle_metrics),
    ):
        metric_str = ", ".join(
            f"{metric}={payload[metric]:.4f}" for metric in metrics
        )
        print(f"  {name:>6}: {metric_str}, latency={payload['avg_latency_ms']:.2f} ms")
    print(json.dumps(report, indent=2))
    print(f"Saved model  -> {args.model_out}")
    print(f"Saved report -> {args.report_out}")
    print(f"Saved preds  -> {args.pred_out}")


if __name__ == "__main__":
    main()
