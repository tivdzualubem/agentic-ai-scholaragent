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
    print(
        "Actionable screening cases: "
        f"{summary.screened_actionable_cases}"
    )
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
    screened_rate = (
        f"{summary.screened_top1_hit_rate:.4f}"
        if summary.screened_top1_hit_rate is not None
        else "n/a"
    )

    print(
        "Actionable screened top-1 hit rate: "
        f"{screened_rate}"
    )
    eligibility_accuracy = (
        f"{summary.eligibility_status_accuracy:.4f}"
        if summary.eligibility_status_accuracy
        is not None
        else "n/a"
    )

    macro_precision = (
        f"{summary.eligibility_macro_precision:.4f}"
        if summary.eligibility_macro_precision
        is not None
        else "n/a"
    )

    macro_recall = (
        f"{summary.eligibility_macro_recall:.4f}"
        if summary.eligibility_macro_recall
        is not None
        else "n/a"
    )

    macro_f1 = (
        f"{summary.eligibility_macro_f1:.4f}"
        if summary.eligibility_macro_f1
        is not None
        else "n/a"
    )

    weighted_f1 = (
        f"{summary.eligibility_weighted_f1:.4f}"
        if summary.eligibility_weighted_f1
        is not None
        else "n/a"
    )

    print(
        "Eligibility evaluated labels: "
        f"{summary.eligibility_evaluated_labels}"
    )
    print(
        "Eligibility status accuracy: "
        f"{eligibility_accuracy}"
    )
    print(
        "Eligibility macro precision: "
        f"{macro_precision}"
    )
    print(
        "Eligibility macro recall: "
        f"{macro_recall}"
    )
    print(
        "Eligibility macro F1: "
        f"{macro_f1}"
    )
    print(
        "Eligibility weighted F1: "
        f"{weighted_f1}"
    )

    print("Eligibility metrics by status:")

    for status, metrics in (
        summary.eligibility_per_status.items()
    ):
        print(
            f"  {status}: "
            f"support={metrics.support}, "
            f"precision={metrics.precision:.4f}, "
            f"recall={metrics.recall:.4f}, "
            f"f1={metrics.f1:.4f}"
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
