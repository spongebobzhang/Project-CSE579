#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the full MultiplexRAG pipeline for one dataset."
    )
    p.add_argument(
        "--dataset-dir",
        required=True,
        help="Dataset root such as data/scifact or data/scidocs.",
    )
    p.add_argument(
        "--prepare-dataset",
        default="",
        help="Optional BEIR dataset id such as BeIR/scifact. If set, prepare_data.py runs first.",
    )
    p.add_argument(
        "--qrels-dataset",
        default="",
        help="Optional BEIR qrels dataset id such as BeIR/scifact-qrels.",
    )
    p.add_argument("--topk", type=int, default=10, help="Top-k for retrieval and evaluation.")
    p.add_argument(
        "--label-metrics",
        nargs="+",
        default=[],
        help="Optional weak-label metrics to pass through to train_router.py.",
    )
    p.add_argument(
        "--label-weights",
        nargs="+",
        type=float,
        default=[],
        help="Optional weak-label weights aligned with --label-metrics.",
    )
    p.add_argument(
        "--label-tie-preference",
        choices=["sparse", "dense", "hybrid"],
        default="hybrid",
        help="Which router label wins when weak-label scores tie.",
    )
    p.add_argument(
        "--use-retrieval-features",
        action="store_true",
        help="Pass retrieval-confidence features through to train_router.py.",
    )
    p.add_argument(
        "--router-confidence-threshold",
        type=float,
        default=0.45,
        help="Confidence threshold for router fallback.",
    )
    p.add_argument(
        "--router-fallback-mode",
        choices=["sparse", "dense", "hybrid"],
        default="dense",
        help="Fallback mode used when the router is uncertain.",
    )
    p.add_argument(
        "--router-train-ratio",
        type=float,
        default=0.8,
        help="Train ratio for query-level self-split when a dataset only has test qrels.",
    )
    p.add_argument(
        "--router-split-seed",
        type=int,
        default=42,
        help="Random seed for query-level self-split when a dataset only has test qrels.",
    )
    p.add_argument(
        "--usd-per-1k-tokens",
        type=float,
        default=0.002,
        help="Estimated downstream input price in USD per 1K tokens.",
    )
    p.add_argument(
        "--skip-router",
        action="store_true",
        help="Skip train_router.py and only run build/retrieve/evaluate.",
    )
    return p.parse_args()


def run_step(cmd: list[str]) -> None:
    print("\n[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    dataset_name = dataset_dir.name
    raw_dir = ROOT / dataset_dir / "raw"
    results_dir = ROOT / "results" / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)

    python = sys.executable

    if args.prepare_dataset:
        cmd = [
            python,
            "scripts/prepare_data.py",
            "--dataset",
            args.prepare_dataset,
            "--source",
            "beir-zip",
        ]
        if args.qrels_dataset:
            cmd.extend(["--qrels-dataset", args.qrels_dataset])
        run_step(cmd)

    run_step([python, "scripts/build_index.py", "--dataset-dir", str(dataset_dir)])

    for mode in ("sparse", "dense", "hybrid"):
        pred_path = results_dir / f"{mode}.jsonl"
        run_step(
            [
                python,
                "scripts/retrieve.py",
                "--dataset-dir",
                str(dataset_dir),
                "--mode",
                mode,
                "--topk",
                str(args.topk),
                "--usd-per-1k-tokens",
                str(args.usd_per_1k_tokens),
                "--out",
                str(pred_path),
            ]
        )
        run_step(
            [
                python,
                "scripts/evaluate.py",
                "--dataset-dir",
                str(dataset_dir),
                "--pred",
                str(pred_path),
                "--out",
                str(results_dir / f"eval_{mode}.json"),
            ]
        )

    if not args.skip_router:
        train_qrels_path = raw_dir / "qrels_train.jsonl"
        test_qrels_path = raw_dir / "qrels_test.jsonl"

        if not train_qrels_path.exists():
            router_train_qrels = raw_dir / "qrels_router_train.jsonl"
            router_test_qrels = raw_dir / "qrels_router_test.jsonl"
            router_split_meta = raw_dir / "qrels_router_split_meta.json"
            run_step(
                [
                    python,
                    "scripts/split_qrels.py",
                    "--qrels",
                    str(test_qrels_path),
                    "--train-out",
                    str(router_train_qrels),
                    "--test-out",
                    str(router_test_qrels),
                    "--train-ratio",
                    str(args.router_train_ratio),
                    "--seed",
                    str(args.router_split_seed),
                    "--meta-out",
                    str(router_split_meta),
                ]
            )
            train_qrels_path = router_train_qrels
            test_qrels_path = router_test_qrels

        router_cmd = [
            python,
            "scripts/train_router.py",
            "--dataset-dir",
            str(dataset_dir),
            "--train-qrels",
            str(train_qrels_path),
            "--test-qrels",
            str(test_qrels_path),
            "--topk",
            str(args.topk),
            "--usd-per-1k-tokens",
            str(args.usd_per_1k_tokens),
            "--router-confidence-threshold",
            str(args.router_confidence_threshold),
            "--router-fallback-mode",
            str(args.router_fallback_mode),
            "--model-out",
            str(results_dir / "router.pkl"),
            "--report-out",
            str(results_dir / "router_report.json"),
            "--pred-out",
            str(results_dir / "router_predictions.jsonl"),
        ]
        if args.label_metrics:
            router_cmd.extend(["--label-metrics", *args.label_metrics])
        if args.label_weights:
            router_cmd.extend(["--label-weights", *(str(weight) for weight in args.label_weights)])
        if args.label_tie_preference:
            router_cmd.extend(["--label-tie-preference", args.label_tie_preference])
        if args.use_retrieval_features:
            router_cmd.append("--use-retrieval-features")
        run_step(router_cmd)

    print("\nDone.", flush=True)
    print(f"Dataset  : {dataset_dir}", flush=True)
    print(f"Results  : {results_dir}", flush=True)
    print(f"Top-k    : {args.topk}", flush=True)
    print(f"Router   : {'skipped' if args.skip_router else 'completed'}", flush=True)


if __name__ == "__main__":
    main()
