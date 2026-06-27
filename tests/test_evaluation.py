"""Tests for ScholarAgent evaluation components."""

from pathlib import Path

import pytest

from scholaragent.evaluation import (
    evaluate_benchmark,
    load_benchmark,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from scholaragent.retrieval import load_scholarships

BENCHMARK = Path("eval/datasets/synthetic_benchmark.json")
SCHOLARSHIPS = Path(
    "data/demo/synthetic_scholarships.json"
)


def test_information_retrieval_metrics() -> None:
    """Metric implementations return known values."""
    retrieved = ["a", "b", "c"]
    relevant = ["b", "d"]

    assert precision_at_k(
        retrieved,
        relevant,
        k=3,
    ) == pytest.approx(1 / 3)

    assert recall_at_k(
        retrieved,
        relevant,
        k=3,
    ) == pytest.approx(0.5)

    assert reciprocal_rank(
        retrieved,
        relevant,
    ) == pytest.approx(0.5)


def test_benchmark_loads() -> None:
    """The synthetic benchmark satisfies its schema."""
    benchmark = load_benchmark(BENCHMARK)

    assert len(benchmark.cases) == 6
    assert sum(
        case.expect_no_results
        for case in benchmark.cases
    ) == 1


def test_synthetic_evaluation_runs() -> None:
    """The complete development evaluation runs reproducibly."""
    summary = evaluate_benchmark(
        benchmark=load_benchmark(BENCHMARK),
        scholarships=load_scholarships(SCHOLARSHIPS),
        k=3,
    )

    assert summary.total_cases == 6
    assert summary.positive_cases == 5
    assert summary.no_result_cases == 1

    assert 0.0 <= summary.bm25_precision_at_k <= 1.0
    assert 0.0 <= summary.bm25_recall_at_k <= 1.0
    assert 0.0 <= summary.bm25_mrr <= 1.0
    assert 0.0 <= summary.screened_top1_hit_rate <= 1.0

    assert summary.eligibility_status_accuracy == 1.0
    assert summary.no_result_accuracy == 1.0


def test_invalid_k_is_rejected() -> None:
    """Evaluation refuses an invalid retrieval cutoff."""
    with pytest.raises(
        ValueError,
        match="k must be at least 1",
    ):
        evaluate_benchmark(
            benchmark=load_benchmark(BENCHMARK),
            scholarships=load_scholarships(SCHOLARSHIPS),
            k=0,
        )
