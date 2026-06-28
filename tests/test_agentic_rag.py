"""Tests for retrieval rewriting and citation repair."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from scholaragent.agentic_rag import (
    run_agentic_rag,
)
from scholaragent.retrieval import (
    BM25ScholarshipIndex,
    load_scholarships,
)
from scholaragent.schemas import StudentProfile

DATASET = Path(
    "data/demo/synthetic_scholarships.json"
)
AS_OF = date(2026, 6, 27)


class SequenceGenerator:
    """Return predetermined answers in sequence."""

    def __init__(
        self,
        answers: list[str],
    ) -> None:
        self.answers = answers
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)

        index = min(
            len(self.prompts) - 1,
            len(self.answers) - 1,
        )

        return self.answers[index]


def build_profile() -> StudentProfile:
    """Return the reusable AI master's profile."""
    return StudentProfile(
        nationality="Nigerian",
        country_of_residence="Finland",
        target_degree_level="master",
        fields_of_study=[
            "Artificial Intelligence",
            "Data Science",
        ],
        gpa=4.2,
        gpa_scale=5.0,
        language_scores={"IELTS": 7.5},
        years_work_experience=1.0,
        preferred_countries=["Finland"],
        requires_full_funding=True,
    )


def build_index() -> BM25ScholarshipIndex:
    """Build the deterministic development index."""
    return BM25ScholarshipIndex(
        load_scholarships(DATASET)
    )


def test_agentic_rag_rewrites_and_repairs() -> None:
    """The workflow should repair an uncited first answer."""
    generator = SequenceGenerator(
        [
            (
                "The Nordic AI scholarship is hosted "
                "in Finland."
            ),
            (
                "- The opportunity is hosted in Finland "
                "[nordic-ai-masters-2027:host_countries].\n"
                "- It is listed as fully funded "
                "[nordic-ai-masters-2027:funding_type]."
            ),
        ]
    )

    result = run_agentic_rag(
        query="scholarship",
        profile=build_profile(),
        index=build_index(),
        generator=generator,
        generator_name="sequence-test-generator",
        as_of=AS_OF,
        top_k=3,
        max_retrieval_attempts=2,
        max_generation_attempts=2,
    )

    assert result.status == "completed"

    assert result.retrieval_calls == 2
    assert result.query_rewrites == 1

    assert result.generation_calls == 2
    assert result.repair_attempts == 1

    assert len(result.audit_history) == 2
    assert result.audit_history[0].passed is False
    assert result.audit_history[1].passed is True
    assert result.citation_audit.passed is True

    assert len(generator.prompts) == 2
    assert (
        "previous answer failed citation verification"
        in generator.prompts[1].casefold()
    )
    assert (
        "[nordic-ai-masters-2027:host_countries]"
        in generator.prompts[1]
    )


def test_agentic_rag_fails_safely_after_budget() -> None:
    """Persistent citation failure should stop at the budget."""
    generator = SequenceGenerator(
        [
            "An uncited answer.",
            "Another uncited answer.",
        ]
    )

    result = run_agentic_rag(
        query=(
            "fully funded artificial intelligence "
            "master's scholarship in Finland"
        ),
        profile=build_profile(),
        index=build_index(),
        generator=generator,
        generator_name="always-invalid-generator",
        as_of=AS_OF,
        top_k=3,
        max_retrieval_attempts=2,
        max_generation_attempts=2,
    )

    assert result.status == "citation_failed"
    assert result.retrieval_calls == 1
    assert result.query_rewrites == 0

    assert result.generation_calls == 2
    assert result.repair_attempts == 1
    assert result.citation_audit.passed is False
    assert len(result.audit_history) == 2


def test_agentic_rag_abstains_before_generation() -> None:
    """Unsupported intent must not invoke the generator."""
    generator = SequenceGenerator(
        ["This answer must never be used."]
    )

    result = run_agentic_rag(
        query=(
            "xylophone archaeology scholarship on Mars"
        ),
        profile=build_profile(),
        index=build_index(),
        generator=generator,
        generator_name="unused-generator",
        as_of=AS_OF,
        top_k=3,
        max_retrieval_attempts=2,
        max_generation_attempts=2,
    )

    assert result.status == "abstained"
    assert result.retrieval_calls == 1
    assert result.generation_calls == 0
    assert result.repair_attempts == 0
    assert result.grounded_report.candidates == []
    assert generator.prompts == []
