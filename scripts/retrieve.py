#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args():
    p = argparse.ArgumentParser(description="Run retrieval for sparse/dense/hybrid baseline.")
    p.add_argument("--dataset-dir", default="", help="Dataset root containing raw/ and processed/ folders")
    p.add_argument("--queries", default="data/raw/queries.jsonl")
    p.add_argument("--corpus", default="data/raw/corpus.jsonl")
    p.add_argument("--index-dir", default="data/processed")
    p.add_argument("--mode", choices=["sparse", "dense", "hybrid"], default="sparse")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--out", default="results/retrieval.jsonl")
    return p.parse_args()


def load_or_build_retrievers(index_dir: Path, corpus_path: str):
    from multiplexrag.data import load_corpus
    from multiplexrag.retrieval import DenseRetriever, HybridRetriever, SparseRetriever, load_pickle

    sparse_path = index_dir / "sparse.pkl"
    dense_path = index_dir / "dense.pkl"
    if sparse_path.exists() and dense_path.exists():
        sparse = load_pickle(sparse_path)
        dense = load_pickle(dense_path)
    else:
        corpus = load_corpus(corpus_path)
        sparse = SparseRetriever().fit(corpus)
        dense = DenseRetriever().fit(corpus)
    hybrid = HybridRetriever(sparse, dense)
    return sparse, dense, hybrid


def main():
    args = parse_args()
    from multiplexrag.data import load_queries, write_jsonl

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else None
    queries_path = dataset_dir / "raw" / "queries.jsonl" if dataset_dir else Path(args.queries)
    corpus_path = dataset_dir / "raw" / "corpus.jsonl" if dataset_dir else Path(args.corpus)
    index_dir = dataset_dir / "processed" if dataset_dir else Path(args.index_dir)

    queries = load_queries(queries_path)
    sparse, dense, hybrid = load_or_build_retrievers(index_dir, str(corpus_path))
    retriever = {"sparse": sparse, "dense": dense, "hybrid": hybrid}[args.mode]

    rows = []
    total_latency = 0.0
    for query in queries:
        results, latency_ms = retriever.search(query["text"], topk=args.topk)
        total_latency += latency_ms
        rows.append(
            {
                "query_id": query["query_id"],
                "query": query["text"],
                "mode": args.mode,
                "latency_ms": latency_ms,
                "results": results,
            }
        )

    write_jsonl(args.out, rows)
    avg_latency = total_latency / len(rows) if rows else 0.0
    print(f"Queries path: {queries_path}")
    print(f"Wrote {len(rows)} query results -> {args.out}")
    print(f"Average latency: {avg_latency:.2f} ms/query")


if __name__ == "__main__":
    main()
