"""Tests for the official-source development benchmark."""

from pathlib import Path

from scholaragent.evaluation import (
    evaluate_benchmark,
    load_benchmark,
)
from scholaragent.retrieval import load_scholarships


BENCHMARK = Path(
    "eval/datasets/official_development_benchmark.json"
)
SCHOLARSHIPS = Path(
    "data/official/official_scholarships.json"
)


def test_official_benchmark_loads() -> None:
    """The official development benchmark should validate."""
    benchmark = load_benchmark(BENCHMARK)

    assert len(benchmark.cases) == 4
    assert sum(
        case.expect_no_results
        for case in benchmark.cases
    ) == 1

    assert {
        status.value
        for case in benchmark.cases
        for status in case.expected_statuses.values()
    } == {
        "potentially_eligible",
        "not_eligible",
    }


def test_official_benchmark_evaluation_is_reproducible() -> None:
    """BM25 and screening should match the manual labels."""
    summary = evaluate_benchmark(
        benchmark=load_benchmark(BENCHMARK),
        scholarships=load_scholarships(SCHOLARSHIPS),
        k=3,
    )

    assert summary.total_cases == 4
    assert summary.positive_cases == 3
    assert summary.no_result_cases == 1
    assert summary.screened_actionable_cases == 2

    assert summary.bm25_recall_at_k == 1.0
    assert summary.bm25_mrr == 1.0
    assert summary.bm25_top1_hit_rate == 1.0
    assert summary.screened_top1_hit_rate == 1.0
    assert summary.eligibility_evaluated_labels == 3
    assert summary.eligibility_status_accuracy == 1.0
    assert summary.eligibility_macro_precision == 1.0
    assert summary.eligibility_macro_recall == 1.0
    assert summary.eligibility_macro_f1 == 1.0
    assert summary.eligibility_weighted_f1 == 1.0
    assert summary.no_result_accuracy == 1.0

    potential = summary.eligibility_per_status[
        "potentially_eligible"
    ]
    ineligible = summary.eligibility_per_status[
        "not_eligible"
    ]

    assert potential.support == 2
    assert potential.true_positives == 2
    assert potential.f1 == 1.0

    assert ineligible.support == 1
    assert ineligible.true_positives == 1
    assert ineligible.f1 == 1.0

    cases = {
        case.case_id: case
        for case in summary.cases
    }

    si_case = cases[
        "si-global-professionals-expired-round"
    ]

    assert si_case.bm25_top1_hit is True
    assert si_case.screened_top1_hit is False
    assert si_case.screened_top1_evaluated is False

    assert si_case.predicted_statuses == {
        "si-global-professionals-2026": "not_eligible"
    }
