"""Run reproducible retrieval and screening evaluations."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from scholaragent.eligibility import (
    EligibilityStatus,
    assess_eligibility,
)
from scholaragent.evaluation.benchmark import BenchmarkDataset
from scholaragent.evaluation.metrics import (
    mean,
    precision_at_k,
    precision_recall_f1,
    recall_at_k,
    reciprocal_rank,
)
from scholaragent.pipeline import search_and_screen
from scholaragent.retrieval import BM25ScholarshipIndex
from scholaragent.schemas import ScholarshipRecord


class CaseEvaluation(BaseModel):
    """Metrics and predictions for one benchmark case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    bm25_ids: list[str]
    screened_ids: list[str]

    precision_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)
    reciprocal_rank: float = Field(ge=0, le=1)

    bm25_top1_hit: bool
    screened_top1_hit: bool
    screened_top1_evaluated: bool
    no_result_correct: bool | None = None

    expected_statuses: dict[str, str]
    predicted_statuses: dict[str, str]


class EligibilityStatusMetrics(BaseModel):
    """One-vs-rest metrics for one eligibility status."""

    model_config = ConfigDict(extra="forbid")

    status: str
    support: int = Field(ge=0)
    predicted_count: int = Field(ge=0)

    true_positives: int = Field(ge=0)
    false_positives: int = Field(ge=0)
    false_negatives: int = Field(ge=0)

    precision: float = Field(ge=0, le=1)
    recall: float = Field(ge=0, le=1)
    f1: float = Field(ge=0, le=1)


class EvaluationSummary(BaseModel):
    """Aggregate benchmark metrics."""

    model_config = ConfigDict(extra="forbid")

    benchmark_name: str
    k: int = Field(ge=1)
    total_cases: int = Field(ge=1)
    positive_cases: int = Field(ge=0)
    no_result_cases: int = Field(ge=0)
    screened_actionable_cases: int = Field(ge=0)

    bm25_precision_at_k: float = Field(ge=0, le=1)
    bm25_recall_at_k: float = Field(ge=0, le=1)
    bm25_mrr: float = Field(ge=0, le=1)
    bm25_top1_hit_rate: float = Field(ge=0, le=1)
    screened_top1_hit_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    eligibility_evaluated_labels: int = Field(ge=0)
    eligibility_status_accuracy: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    eligibility_macro_precision: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    eligibility_macro_recall: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    eligibility_macro_f1: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    eligibility_weighted_f1: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    eligibility_per_status: dict[
        str,
        EligibilityStatusMetrics,
    ]

    no_result_accuracy: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    cases: list[CaseEvaluation]


def evaluate_benchmark(
    *,
    benchmark: BenchmarkDataset,
    scholarships: list[ScholarshipRecord],
    k: int = 3,
) -> EvaluationSummary:
    """Evaluate BM25 retrieval and deterministic screening."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    records_by_id = {
        record.scholarship_id: record
        for record in scholarships
    }
    corpus_ids = set(records_by_id)

    referenced_ids = {
        identifier
        for case in benchmark.cases
        for identifier in (
            list(case.relevant_ids)
            + list(case.expected_statuses)
        )
    }

    unknown_ids = referenced_ids - corpus_ids

    if unknown_ids:
        formatted = ", ".join(sorted(unknown_ids))
        raise ValueError(
            f"Benchmark references unknown scholarship IDs: {formatted}"
        )

    index = BM25ScholarshipIndex(scholarships)

    case_results: list[CaseEvaluation] = []
    precisions: list[float] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    bm25_top1_hits: list[float] = []
    screened_top1_hits: list[float] = []
    status_matches: list[float] = []
    expected_status_labels: list[
        EligibilityStatus
    ] = []
    predicted_status_labels: list[
        EligibilityStatus
    ] = []
    no_result_matches: list[float] = []

    for case in benchmark.cases:
        bm25_results = index.search(case.query, k=k)
        bm25_ids = [
            result.scholarship.scholarship_id
            for result in bm25_results
        ]

        report = search_and_screen(
            query=case.query,
            profile=case.profile,
            index=index,
            k=k,
            as_of=benchmark.as_of,
        )
        screened_ids = [
            result.scholarship.scholarship_id
            for result in report.results
        ]

        predicted_statuses: dict[str, str] = {}

        for identifier, expected_status in (
            case.expected_statuses.items()
        ):
            assessment = assess_eligibility(
                case.profile,
                records_by_id[identifier],
                as_of=benchmark.as_of,
            )
            predicted_statuses[identifier] = assessment.status.value
            expected_status_labels.append(
                expected_status
            )
            predicted_status_labels.append(
                assessment.status
            )
            status_matches.append(
                float(assessment.status is expected_status)
            )

        if case.expect_no_results:
            no_result_correct = not bm25_ids and not screened_ids
            no_result_matches.append(float(no_result_correct))

            case_results.append(
                CaseEvaluation(
                    case_id=case.case_id,
                    bm25_ids=bm25_ids,
                    screened_ids=screened_ids,
                    precision_at_k=0.0,
                    recall_at_k=0.0,
                    reciprocal_rank=0.0,
                    bm25_top1_hit=False,
                    screened_top1_hit=False,
                    screened_top1_evaluated=False,
                    no_result_correct=no_result_correct,
                    expected_statuses={
                        key: value.value
                        for key, value
                        in case.expected_statuses.items()
                    },
                    predicted_statuses=predicted_statuses,
                )
            )
            continue

        precision = precision_at_k(
            bm25_ids,
            case.relevant_ids,
            k=k,
        )
        recall = recall_at_k(
            bm25_ids,
            case.relevant_ids,
            k=k,
        )
        rr = reciprocal_rank(
            bm25_ids,
            case.relevant_ids,
        )

        bm25_top1_hit = bool(
            bm25_ids
            and bm25_ids[0] in case.relevant_ids
        )
        screened_top1_hit = bool(
            screened_ids
            and screened_ids[0] in case.relevant_ids
        )

        expected_status_values = list(
            case.expected_statuses.values()
        )

        screened_top1_evaluated = (
            not expected_status_values
            or any(
                status is not EligibilityStatus.NOT_ELIGIBLE
                for status in expected_status_values
            )
        )

        precisions.append(precision)
        recalls.append(recall)
        reciprocal_ranks.append(rr)
        bm25_top1_hits.append(float(bm25_top1_hit))

        if screened_top1_evaluated:
            screened_top1_hits.append(
                float(screened_top1_hit)
            )

        case_results.append(
            CaseEvaluation(
                case_id=case.case_id,
                bm25_ids=bm25_ids,
                screened_ids=screened_ids,
                precision_at_k=precision,
                recall_at_k=recall,
                reciprocal_rank=rr,
                bm25_top1_hit=bm25_top1_hit,
                screened_top1_hit=screened_top1_hit,
                screened_top1_evaluated=(
                    screened_top1_evaluated
                ),
                expected_statuses={
                    key: value.value
                    for key, value
                    in case.expected_statuses.items()
                },
                predicted_statuses=predicted_statuses,
            )
        )

    positive_count = len(precisions)
    no_result_count = len(no_result_matches)

    eligibility_per_status: dict[
        str,
        EligibilityStatusMetrics,
    ] = {}

    active_status_metrics: list[
        EligibilityStatusMetrics
    ] = []

    for status in EligibilityStatus:
        true_positives = sum(
            expected is status and predicted is status
            for expected, predicted in zip(
                expected_status_labels,
                predicted_status_labels,
                strict=True,
            )
        )

        false_positives = sum(
            expected is not status and predicted is status
            for expected, predicted in zip(
                expected_status_labels,
                predicted_status_labels,
                strict=True,
            )
        )

        false_negatives = sum(
            expected is status and predicted is not status
            for expected, predicted in zip(
                expected_status_labels,
                predicted_status_labels,
                strict=True,
            )
        )

        support = sum(
            expected is status
            for expected in expected_status_labels
        )

        predicted_count = sum(
            predicted is status
            for predicted in predicted_status_labels
        )

        precision, recall, f1 = precision_recall_f1(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
        )

        status_metrics = EligibilityStatusMetrics(
            status=status.value,
            support=support,
            predicted_count=predicted_count,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1=f1,
        )

        eligibility_per_status[
            status.value
        ] = status_metrics

        if support or predicted_count:
            active_status_metrics.append(
                status_metrics
            )

    if active_status_metrics:
        eligibility_macro_precision = mean([
            item.precision
            for item in active_status_metrics
        ])
        eligibility_macro_recall = mean([
            item.recall
            for item in active_status_metrics
        ])
        eligibility_macro_f1 = mean([
            item.f1
            for item in active_status_metrics
        ])
    else:
        eligibility_macro_precision = None
        eligibility_macro_recall = None
        eligibility_macro_f1 = None

    total_status_support = len(
        expected_status_labels
    )

    eligibility_weighted_f1 = (
        sum(
            item.f1 * item.support
            for item in eligibility_per_status.values()
        )
        / total_status_support
        if total_status_support
        else None
    )

    return EvaluationSummary(
        benchmark_name=benchmark.name,
        k=k,
        total_cases=len(benchmark.cases),
        positive_cases=positive_count,
        no_result_cases=no_result_count,
        screened_actionable_cases=len(
            screened_top1_hits
        ),
        bm25_precision_at_k=mean(precisions),
        bm25_recall_at_k=mean(recalls),
        bm25_mrr=mean(reciprocal_ranks),
        bm25_top1_hit_rate=mean(bm25_top1_hits),
        screened_top1_hit_rate=(
            mean(screened_top1_hits)
            if screened_top1_hits
            else None
        ),
        eligibility_evaluated_labels=(
            total_status_support
        ),
        eligibility_status_accuracy=(
            mean(status_matches)
            if status_matches
            else None
        ),
        eligibility_macro_precision=(
            eligibility_macro_precision
        ),
        eligibility_macro_recall=(
            eligibility_macro_recall
        ),
        eligibility_macro_f1=(
            eligibility_macro_f1
        ),
        eligibility_weighted_f1=(
            eligibility_weighted_f1
        ),
        eligibility_per_status=(
            eligibility_per_status
        ),
        no_result_accuracy=(
            mean(no_result_matches)
            if no_result_matches
            else None
        ),
        cases=case_results,
    )
