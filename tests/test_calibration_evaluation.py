"""Tests for the balanced official calibration benchmark."""

from collections import Counter
from pathlib import Path

from scholaragent.evaluation import (
    evaluate_benchmark,
    load_benchmark,
)
from scholaragent.retrieval import (
    load_scholarships,
)


BENCHMARK_PATH = Path(
    "eval/datasets/"
    "calibration_benchmark.json"
)
CORPUS_PATH = Path(
    "data/calibration/"
    "calibration_scholarships.json"
)


def test_calibration_benchmark_is_balanced() -> None:
    """The calibration partition should contain its frozen design."""
    benchmark = load_benchmark(
        BENCHMARK_PATH
    )

    status_counts = Counter(
        status.value
        for case in benchmark.cases
        for status in case.expected_statuses.values()
    )

    assert len(benchmark.cases) == 24
    assert benchmark.as_of.isoformat() == (
        "2026-05-01"
    )
    assert sum(
        case.expect_no_results
        for case in benchmark.cases
    ) == 4

    assert status_counts == {
        "eligible": 5,
        "potentially_eligible": 5,
        "not_eligible": 5,
        "insufficient_information": 5,
    }


def test_calibration_benchmark_references_frozen_corpus() -> None:
    """Every positive label should reference the six-record corpus."""
    benchmark = load_benchmark(
        BENCHMARK_PATH
    )
    records = load_scholarships(
        CORPUS_PATH
    )

    corpus_ids = {
        record.scholarship_id
        for record in records
    }

    referenced_ids = {
        identifier
        for case in benchmark.cases
        for identifier in (
            list(case.relevant_ids)
            + list(case.expected_statuses)
        )
    }

    assert len(records) == 6
    assert referenced_ids
    assert referenced_ids <= corpus_ids

    for case in benchmark.cases:
        if case.expect_no_results:
            assert case.relevant_ids == []
            assert case.expected_statuses == {}
        else:
            assert len(case.relevant_ids) == 1
            assert len(case.expected_statuses) == 1


def test_calibration_evaluation_reproduces_ground_truth() -> None:
    """The deterministic engine should reproduce calibration labels."""
    summary = evaluate_benchmark(
        benchmark=load_benchmark(
            BENCHMARK_PATH
        ),
        scholarships=load_scholarships(
            CORPUS_PATH
        ),
        k=3,
    )

    assert summary.total_cases == 24
    assert summary.positive_cases == 20
    assert summary.no_result_cases == 4
    assert summary.eligibility_evaluated_labels == 20

    assert summary.bm25_top1_hit_rate == 1.0
    assert summary.bm25_mrr == 1.0
    assert summary.eligibility_status_accuracy == 1.0
    assert summary.eligibility_macro_precision == 1.0
    assert summary.eligibility_macro_recall == 1.0
    assert summary.eligibility_macro_f1 == 1.0
    assert summary.eligibility_weighted_f1 == 1.0
    assert summary.no_result_accuracy == 1.0

    assert {
        status: metrics.support
        for status, metrics
        in summary.eligibility_per_status.items()
    } == {
        "eligible": 5,
        "potentially_eligible": 5,
        "not_eligible": 5,
        "insufficient_information": 5,
    }
