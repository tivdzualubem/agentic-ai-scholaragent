"""Comparative evaluation of ScholarAgent retrieval strategies."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from scholaragent.evaluation.benchmark import BenchmarkDataset
from scholaragent.evaluation.metrics import (
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


SearchFunction = Callable[[str, int], Sequence[Any]]


class RetrievalCaseResult(BaseModel):
    """Evaluation result for one query and one retriever."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    retrieved_ids: list[str]

    precision_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)
    reciprocal_rank: float = Field(ge=0, le=1)
    top1_hit: bool

    no_result_correct: bool | None = None


class RetrieverMetrics(BaseModel):
    """Aggregate metrics for one retrieval strategy."""

    model_config = ConfigDict(extra="forbid")

    retriever_name: str
    k: int = Field(ge=1)
    positive_cases: int = Field(ge=0)
    no_result_cases: int = Field(ge=0)

    precision_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)
    mrr: float = Field(ge=0, le=1)
    top1_hit_rate: float = Field(ge=0, le=1)
    no_result_accuracy: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    cases: list[RetrievalCaseResult]


class RetrievalComparison(BaseModel):
    """Complete comparison across multiple retrieval strategies."""

    model_config = ConfigDict(extra="forbid")

    benchmark_name: str
    k: int = Field(ge=1)

    embedding_model: str | None = None
    dense_threshold: float | None = Field(
        default=None,
        ge=-1,
        le=1,
    )
    calibration_scope: str | None = None

    retrievers: list[RetrieverMetrics]


def _extract_identifiers(
    results: Sequence[Any],
) -> list[str]:
    """Extract scholarship identifiers from search results."""
    identifiers: list[str] = []

    for result in results:
        scholarship = getattr(result, "scholarship", None)
        identifier = getattr(
            scholarship,
            "scholarship_id",
            None,
        )

        if not isinstance(identifier, str) or not identifier:
            raise ValueError(
                "Retriever returned a result without a valid "
                "scholarship identifier."
            )

        identifiers.append(identifier)

    return identifiers


def evaluate_retriever(
    *,
    retriever_name: str,
    search: SearchFunction,
    benchmark: BenchmarkDataset,
    k: int,
) -> RetrieverMetrics:
    """Evaluate one retriever against a benchmark."""
    if not retriever_name.strip():
        raise ValueError(
            "Retriever name must not be empty."
        )

    if k < 1:
        raise ValueError("k must be at least 1.")

    case_results: list[RetrievalCaseResult] = []

    precisions: list[float] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    top1_hits: list[float] = []
    no_result_matches: list[float] = []

    for case in benchmark.cases:
        retrieved = search(case.query, k)
        retrieved_ids = _extract_identifiers(retrieved)

        if case.expect_no_results:
            no_result_correct = not retrieved_ids
            no_result_matches.append(
                float(no_result_correct)
            )

            case_results.append(
                RetrievalCaseResult(
                    case_id=case.case_id,
                    retrieved_ids=retrieved_ids,
                    precision_at_k=0.0,
                    recall_at_k=0.0,
                    reciprocal_rank=0.0,
                    top1_hit=False,
                    no_result_correct=no_result_correct,
                )
            )
            continue

        precision = precision_at_k(
            retrieved_ids,
            case.relevant_ids,
            k=k,
        )
        recall = recall_at_k(
            retrieved_ids,
            case.relevant_ids,
            k=k,
        )
        rr = reciprocal_rank(
            retrieved_ids,
            case.relevant_ids,
        )

        top1_hit = bool(
            retrieved_ids
            and retrieved_ids[0] in case.relevant_ids
        )

        precisions.append(precision)
        recalls.append(recall)
        reciprocal_ranks.append(rr)
        top1_hits.append(float(top1_hit))

        case_results.append(
            RetrievalCaseResult(
                case_id=case.case_id,
                retrieved_ids=retrieved_ids,
                precision_at_k=precision,
                recall_at_k=recall,
                reciprocal_rank=rr,
                top1_hit=top1_hit,
            )
        )

    if not precisions:
        raise ValueError(
            "Benchmark requires at least one positive case."
        )

    return RetrieverMetrics(
        retriever_name=retriever_name,
        k=k,
        positive_cases=len(precisions),
        no_result_cases=len(no_result_matches),
        precision_at_k=mean(precisions),
        recall_at_k=mean(recalls),
        mrr=mean(reciprocal_ranks),
        top1_hit_rate=mean(top1_hits),
        no_result_accuracy=(
            mean(no_result_matches)
            if no_result_matches
            else None
        ),
        cases=case_results,
    )


def compare_retrievers(
    *,
    benchmark: BenchmarkDataset,
    retrievers: Mapping[str, SearchFunction],
    k: int = 3,
    embedding_model: str | None = None,
    dense_threshold: float | None = None,
    calibration_scope: str | None = None,
) -> RetrievalComparison:
    """Evaluate multiple retrievers using identical cases."""
    if not retrievers:
        raise ValueError(
            "At least one retriever is required."
        )

    metrics = [
        evaluate_retriever(
            retriever_name=name,
            search=search,
            benchmark=benchmark,
            k=k,
        )
        for name, search in retrievers.items()
    ]

    return RetrievalComparison(
        benchmark_name=benchmark.name,
        k=k,
        embedding_model=embedding_model,
        dense_threshold=dense_threshold,
        calibration_scope=calibration_scope,
        retrievers=metrics,
    )
