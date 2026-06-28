"""Tests for single-pass versus Agentic RAG evaluation."""

from __future__ import annotations

from pathlib import Path

from scholaragent.evaluation import (
    evaluate_rag_comparison,
    load_benchmark,
)
from scholaragent.retrieval import (
    BM25ScholarshipIndex,
    load_scholarships,
)


BENCHMARK = Path(
    "eval/datasets/"
    "official_development_benchmark.json"
)
SCHOLARSHIPS = Path(
    "data/official/"
    "official_scholarships.json"
)


class UncitedGenerator:
    """Always return an answer that fails citation auditing."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        return "This generated answer has no citations."


class FailIfCalledGenerator:
    """Fail when an abstention case incorrectly invokes generation."""

    def __call__(self, prompt: str) -> str:
        raise AssertionError(
            "Generator must not run for the no-result case."
        )


def test_agentic_fallback_outperforms_uncited_baseline() -> None:
    """Verified fallback should recover persistent LLM failures."""
    benchmark = load_benchmark(BENCHMARK)
    index = BM25ScholarshipIndex(
        load_scholarships(SCHOLARSHIPS)
    )

    def factory(system_name, case):
        if case.expect_no_results:
            return FailIfCalledGenerator()

        return UncitedGenerator()

    comparison = evaluate_rag_comparison(
        benchmark=benchmark,
        index=index,
        generator_factory=factory,
        generator_name="deterministic-uncited-generator",
        top_k=3,
        max_retrieval_attempts=2,
        max_generation_attempts=2,
        evaluation_scope="Deterministic unit test.",
    )

    systems = {
        system.system_name: system
        for system in comparison.systems
    }

    baseline = systems["single_pass_rag"]
    agentic = systems["agentic_rag"]

    assert baseline.positive_cases == 3
    assert baseline.no_result_cases == 1
    assert baseline.positive_completion_rate == 0.0
    assert baseline.positive_citation_pass_rate == 0.0
    assert (
        baseline.positive_relevant_grounding_rate
        == 1.0
    )
    assert (
        baseline.positive_relevant_citation_rate
        == 0.0
    )
    assert baseline.no_result_accuracy == 1.0
    assert baseline.positive_fallback_rate == 0.0

    assert agentic.positive_completion_rate == 1.0
    assert agentic.positive_citation_pass_rate == 1.0
    assert (
        agentic.positive_relevant_grounding_rate
        == 1.0
    )
    assert (
        agentic.positive_relevant_citation_rate
        == 1.0
    )
    assert agentic.no_result_accuracy == 1.0
    assert agentic.positive_fallback_rate == 1.0

    si_case = next(
        case
        for case in agentic.cases
        if case.case_id
        == "si-global-professionals-expired-round"
    )

    assert si_case.status == "completed_fallback"
    assert si_case.fallback_used is True
    assert si_case.citation_passed is True
    assert si_case.relevant_citation is True
    assert si_case.candidate_roles[
        "si-global-professionals-2026"
    ] == "explanatory_ineligible"

    no_result = next(
        case
        for case in agentic.cases
        if case.expect_no_results
    )

    assert no_result.status == "abstained"
    assert no_result.generation_calls == 0
    assert no_result.no_result_correct is True
