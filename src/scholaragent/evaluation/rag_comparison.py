"""Compare conventional single-pass RAG with Agentic RAG."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from scholaragent.agentic_rag import (
    AgenticRAGResult,
    run_agentic_rag,
)
from scholaragent.evaluation.benchmark import (
    BenchmarkCase,
    BenchmarkDataset,
)
from scholaragent.evaluation.metrics import mean
from scholaragent.rag_baseline import (
    SinglePassRAGResult,
    TextGenerator,
    run_single_pass_rag,
)
from scholaragent.retrieval import ScholarshipSearchIndex


RAGSystemName = Literal[
    "single_pass_rag",
    "agentic_rag",
]

GeneratorFactory = Callable[
    [RAGSystemName, BenchmarkCase],
    TextGenerator,
]


class RAGCaseResult(BaseModel):
    """Evaluation record for one system on one benchmark case."""

    model_config = ConfigDict(extra="forbid")

    system_name: RAGSystemName
    case_id: str
    expected_relevant_ids: list[str]
    expect_no_results: bool

    status: str
    answer: str
    final_query: str

    latency_seconds: float = Field(ge=0)
    retrieval_calls: int = Field(ge=0)
    generation_calls: int = Field(ge=0)
    query_rewrites: int = Field(ge=0)
    repair_attempts: int = Field(ge=0)
    fallback_used: bool

    citation_passed: bool
    invalid_citation_ids: list[str]
    uncited_bullet_count: int = Field(ge=0)
    citation_errors: list[str]
    audit_history_passed: list[bool]

    grounded_candidate_ids: list[str]
    candidate_roles: dict[str, str]
    cited_scholarship_ids: list[str]

    relevant_grounded: bool | None = None
    relevant_citation: bool | None = None
    positive_completed: bool | None = None
    no_result_correct: bool | None = None


class RAGSystemMetrics(BaseModel):
    """Aggregate metrics for one RAG architecture."""

    model_config = ConfigDict(extra="forbid")

    system_name: RAGSystemName
    generator_name: str

    total_cases: int = Field(ge=1)
    positive_cases: int = Field(ge=0)
    no_result_cases: int = Field(ge=0)

    positive_completion_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    positive_citation_pass_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    positive_relevant_grounding_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    positive_relevant_citation_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    no_result_accuracy: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    mean_latency_seconds: float = Field(ge=0)
    mean_retrieval_calls: float = Field(ge=0)
    mean_generation_calls: float = Field(ge=0)
    mean_query_rewrites: float = Field(ge=0)
    mean_repair_attempts: float = Field(ge=0)
    positive_fallback_rate: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    cases: list[RAGCaseResult]


class RAGComparison(BaseModel):
    """Complete single-pass versus Agentic RAG comparison."""

    model_config = ConfigDict(extra="forbid")

    benchmark_name: str
    benchmark_as_of: date
    top_k: int = Field(ge=1)

    generator_name: str
    embedding_model: str | None = None
    dense_threshold: float | None = Field(
        default=None,
        ge=-1,
        le=1,
    )
    evaluation_scope: str

    systems: list[RAGSystemMetrics]


def _cited_scholarship_ids(
    citation_ids: list[str],
) -> list[str]:
    """Extract scholarship IDs from structured citation IDs."""
    return list(
        dict.fromkeys(
            citation_id.split(":", 1)[0]
            for citation_id in citation_ids
        )
    )


def _case_grounding_data(
    result: SinglePassRAGResult | AgenticRAGResult,
) -> tuple[list[str], dict[str, str]]:
    """Extract grounded candidate identifiers and roles."""
    candidates = result.grounded_report.candidates

    identifiers = [
        candidate.scholarship_id
        for candidate in candidates
    ]

    roles = {
        candidate.scholarship_id: candidate.candidate_role
        for candidate in candidates
    }

    return identifiers, roles


def _positive_completion(
    system_name: RAGSystemName,
    status: str,
) -> bool:
    """Return whether a positive case completed safely."""
    if system_name == "single_pass_rag":
        return status == "completed"

    return status in {
        "completed",
        "completed_fallback",
    }


def _build_case_result(
    *,
    system_name: RAGSystemName,
    case: BenchmarkCase,
    result: SinglePassRAGResult | AgenticRAGResult,
    latency_seconds: float,
) -> RAGCaseResult:
    """Convert a live RAG result into evaluation fields."""
    grounded_ids, candidate_roles = (
        _case_grounding_data(result)
    )

    cited_ids = _cited_scholarship_ids(
        result.citation_audit.cited_ids
    )

    relevant = set(case.relevant_ids)

    if case.expect_no_results:
        relevant_grounded = None
        relevant_citation = None
        positive_completed = None
        no_result_correct = (
            result.status == "abstained"
            and result.generation_calls == 0
            and not grounded_ids
        )
    else:
        relevant_grounded = bool(
            relevant & set(grounded_ids)
        )
        relevant_citation = bool(
            relevant & set(cited_ids)
        )
        positive_completed = _positive_completion(
            system_name,
            result.status,
        )
        no_result_correct = None

    if isinstance(result, AgenticRAGResult):
        final_query = result.final_query
        repair_attempts = result.repair_attempts
        fallback_used = result.fallback_used
        audit_history = [
            audit.passed
            for audit in result.audit_history
        ]
    else:
        final_query = result.query
        repair_attempts = 0
        fallback_used = False
        audit_history = [
            result.citation_audit.passed
        ]

    return RAGCaseResult(
        system_name=system_name,
        case_id=case.case_id,
        expected_relevant_ids=case.relevant_ids,
        expect_no_results=case.expect_no_results,
        status=result.status,
        answer=result.answer,
        final_query=final_query,
        latency_seconds=latency_seconds,
        retrieval_calls=result.retrieval_calls,
        generation_calls=result.generation_calls,
        query_rewrites=result.query_rewrites,
        repair_attempts=repair_attempts,
        fallback_used=fallback_used,
        citation_passed=result.citation_audit.passed,
        invalid_citation_ids=(
            result.citation_audit.invalid_ids
        ),
        uncited_bullet_count=len(
            result.citation_audit.uncited_bullets
        ),
        citation_errors=result.citation_audit.errors,
        audit_history_passed=audit_history,
        grounded_candidate_ids=grounded_ids,
        candidate_roles=candidate_roles,
        cited_scholarship_ids=cited_ids,
        relevant_grounded=relevant_grounded,
        relevant_citation=relevant_citation,
        positive_completed=positive_completed,
        no_result_correct=no_result_correct,
    )


def _optional_boolean_mean(
    values: list[bool],
) -> float | None:
    """Calculate a boolean mean when values are available."""
    if not values:
        return None

    return mean([
        float(value)
        for value in values
    ])


def _summarize_system(
    *,
    system_name: RAGSystemName,
    generator_name: str,
    cases: list[RAGCaseResult],
) -> RAGSystemMetrics:
    """Aggregate case-level RAG measurements."""
    positive = [
        case
        for case in cases
        if not case.expect_no_results
    ]
    no_result = [
        case
        for case in cases
        if case.expect_no_results
    ]

    return RAGSystemMetrics(
        system_name=system_name,
        generator_name=generator_name,
        total_cases=len(cases),
        positive_cases=len(positive),
        no_result_cases=len(no_result),
        positive_completion_rate=(
            _optional_boolean_mean([
                bool(case.positive_completed)
                for case in positive
            ])
        ),
        positive_citation_pass_rate=(
            _optional_boolean_mean([
                case.citation_passed
                for case in positive
            ])
        ),
        positive_relevant_grounding_rate=(
            _optional_boolean_mean([
                bool(case.relevant_grounded)
                for case in positive
            ])
        ),
        positive_relevant_citation_rate=(
            _optional_boolean_mean([
                bool(case.relevant_citation)
                for case in positive
            ])
        ),
        no_result_accuracy=(
            _optional_boolean_mean([
                bool(case.no_result_correct)
                for case in no_result
            ])
        ),
        mean_latency_seconds=mean([
            case.latency_seconds
            for case in cases
        ]),
        mean_retrieval_calls=mean([
            float(case.retrieval_calls)
            for case in cases
        ]),
        mean_generation_calls=mean([
            float(case.generation_calls)
            for case in cases
        ]),
        mean_query_rewrites=mean([
            float(case.query_rewrites)
            for case in cases
        ]),
        mean_repair_attempts=mean([
            float(case.repair_attempts)
            for case in cases
        ]),
        positive_fallback_rate=(
            _optional_boolean_mean([
                case.fallback_used
                for case in positive
            ])
        ),
        cases=cases,
    )


def evaluate_rag_comparison(
    *,
    benchmark: BenchmarkDataset,
    index: ScholarshipSearchIndex,
    generator_factory: GeneratorFactory,
    generator_name: str,
    top_k: int = 3,
    max_retrieval_attempts: int = 2,
    max_generation_attempts: int = 2,
    embedding_model: str | None = None,
    dense_threshold: float | None = None,
    evaluation_scope: str = (
        "Development evaluation; not final publication evidence."
    ),
) -> RAGComparison:
    """Evaluate conventional and Agentic RAG on identical cases."""
    if not generator_name.strip():
        raise ValueError(
            "Generator name must not be empty."
        )

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    if max_retrieval_attempts < 1:
        raise ValueError(
            "max_retrieval_attempts must be at least 1."
        )

    if max_generation_attempts < 1:
        raise ValueError(
            "max_generation_attempts must be at least 1."
        )

    baseline_cases: list[RAGCaseResult] = []
    agentic_cases: list[RAGCaseResult] = []

    for case in benchmark.cases:
        baseline_generator = generator_factory(
            "single_pass_rag",
            case,
        )

        baseline_start = perf_counter()

        baseline = run_single_pass_rag(
            query=case.query,
            profile=case.profile,
            index=index,
            generator=baseline_generator,
            generator_name=generator_name,
            as_of=benchmark.as_of,
            top_k=top_k,
        )

        baseline_latency = (
            perf_counter() - baseline_start
        )

        baseline_cases.append(
            _build_case_result(
                system_name="single_pass_rag",
                case=case,
                result=baseline,
                latency_seconds=baseline_latency,
            )
        )

        agentic_generator = generator_factory(
            "agentic_rag",
            case,
        )

        agentic_start = perf_counter()

        agentic = run_agentic_rag(
            query=case.query,
            profile=case.profile,
            index=index,
            generator=agentic_generator,
            generator_name=generator_name,
            as_of=benchmark.as_of,
            top_k=top_k,
            max_retrieval_attempts=(
                max_retrieval_attempts
            ),
            max_generation_attempts=(
                max_generation_attempts
            ),
        )

        agentic_latency = (
            perf_counter() - agentic_start
        )

        agentic_cases.append(
            _build_case_result(
                system_name="agentic_rag",
                case=case,
                result=agentic,
                latency_seconds=agentic_latency,
            )
        )

    return RAGComparison(
        benchmark_name=benchmark.name,
        benchmark_as_of=benchmark.as_of,
        top_k=top_k,
        generator_name=generator_name,
        embedding_model=embedding_model,
        dense_threshold=dense_threshold,
        evaluation_scope=evaluation_scope,
        systems=[
            _summarize_system(
                system_name="single_pass_rag",
                generator_name=generator_name,
                cases=baseline_cases,
            ),
            _summarize_system(
                system_name="agentic_rag",
                generator_name=generator_name,
                cases=agentic_cases,
            ),
        ],
    )
