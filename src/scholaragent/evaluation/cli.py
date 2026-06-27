"""Command-line benchmark runner for ScholarAgent."""

from __future__ import annotations

import argparse
from pathlib import Path

from scholaragent.evaluation import (
    evaluate_benchmark,
    load_benchmark,
)
from scholaragent.retrieval import load_scholarships


def build_parser() -> argparse.ArgumentParser:
    """Create the evaluation command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate BM25 retrieval and eligibility screening."
        )
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path(
            "eval/datasets/synthetic_benchmark.json"
        ),
    )
    parser.add_argument(
        "--scholarships",
        type=Path,
        default=Path(
            "data/demo/synthetic_scholarships.json"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )
    return parser


def main() -> None:
    """Run the selected benchmark and print its metrics."""
    args = build_parser().parse_args()

    benchmark = load_benchmark(args.benchmark)
    scholarships = load_scholarships(args.scholarships)

    summary = evaluate_benchmark(
        benchmark=benchmark,
        scholarships=scholarships,
        k=args.top_k,
    )

    print(f"Benchmark: {summary.benchmark_name}")
    print(f"Cases: {summary.total_cases}")
    print(f"Positive cases: {summary.positive_cases}")
    print(f"No-result cases: {summary.no_result_cases}")
    print()
    print(
        f"BM25 Precision@{summary.k}: "
        f"{summary.bm25_precision_at_k:.4f}"
    )
    print(
        f"BM25 Recall@{summary.k}: "
        f"{summary.bm25_recall_at_k:.4f}"
    )
    print(f"BM25 MRR: {summary.bm25_mrr:.4f}")
    print(
        "BM25 top-1 hit rate: "
        f"{summary.bm25_top1_hit_rate:.4f}"
    )
    print(
        "Screened top-1 hit rate: "
        f"{summary.screened_top1_hit_rate:.4f}"
    )
    print(
        "Eligibility status accuracy: "
        f"{summary.eligibility_status_accuracy:.4f}"
    )

    if summary.no_result_accuracy is not None:
        print(
            "No-result accuracy: "
            f"{summary.no_result_accuracy:.4f}"
        )

    print()
    print("Per-case results:")

    for case in summary.cases:
        print(f"- {case.case_id}")
        print(f"  BM25: {case.bm25_ids}")
        print(f"  Screened: {case.screened_ids}")

        if case.no_result_correct is not None:
            print(
                "  No-result correct: "
                f"{case.no_result_correct}"
            )
        else:
            print(
                f"  Recall@{summary.k}: "
                f"{case.recall_at_k:.4f}"
            )
            print(
                f"  Reciprocal rank: "
                f"{case.reciprocal_rank:.4f}"
            )

        if case.expected_statuses:
            print(
                "  Expected statuses: "
                f"{case.expected_statuses}"
            )
            print(
                "  Predicted statuses: "
                f"{case.predicted_statuses}"
            )


if __name__ == "__main__":
    main()
