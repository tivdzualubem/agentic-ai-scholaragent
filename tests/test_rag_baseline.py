"""Tests for the conventional single-pass RAG baseline."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from scholaragent.rag_baseline import (
    run_single_pass_rag,
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


class CapturingGenerator:
    """Deterministic text generator used in unit tests."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


def build_profile() -> StudentProfile:
    """Return the reusable synthetic AI applicant."""
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


def test_single_pass_rag_generates_cited_answer() -> None:
    """A valid generated answer should pass citation auditing."""
    generator = CapturingGenerator(
        (
            "- The Nordic AI scholarship is hosted in Finland "
            "[nordic-ai-masters-2027:host_countries].\n"
            "- It is listed as fully funded "
            "[nordic-ai-masters-2027:funding_type]."
        )
    )

    result = run_single_pass_rag(
        query=(
            "Find a fully funded artificial intelligence "
            "master's scholarship in Finland"
        ),
        profile=build_profile(),
        index=build_index(),
        generator=generator,
        generator_name="deterministic-test-generator",
        as_of=AS_OF,
        top_k=3,
    )

    assert result.status == "completed"
    assert result.retrieval_calls == 1
    assert result.generation_calls == 1
    assert result.query_rewrites == 0
    assert result.citation_audit.passed is True
    assert len(generator.prompts) == 1

    prompt = generator.prompts[0]

    assert "VERIFIED EVIDENCE" in prompt
    assert (
        "[nordic-ai-masters-2027:host_countries]"
        in prompt
    )
    assert "https://example.org/nordic-ai-masters" in prompt


def test_single_pass_rag_detects_invalid_citation() -> None:
    """Invented citation identifiers should fail the audit."""
    generator = CapturingGenerator(
        "- This statement is unsupported [invented-source:deadline]."
    )

    result = run_single_pass_rag(
        query=(
            "Find a fully funded artificial intelligence "
            "master's scholarship in Finland"
        ),
        profile=build_profile(),
        index=build_index(),
        generator=generator,
        generator_name="invalid-citation-generator",
        as_of=AS_OF,
        top_k=3,
    )

    assert result.status == "citation_failed"
    assert result.citation_audit.passed is False
    assert result.citation_audit.invalid_ids == [
        "invented-source:deadline"
    ]


def test_single_pass_rag_abstains_without_generation() -> None:
    """Unsupported queries should not invoke the LLM."""
    generator = CapturingGenerator(
        "This answer must never be generated."
    )

    result = run_single_pass_rag(
        query=(
            "xylophone archaeology scholarship on Mars"
        ),
        profile=build_profile(),
        index=build_index(),
        generator=generator,
        generator_name="unused-generator",
        as_of=AS_OF,
        top_k=3,
    )

    assert result.status == "abstained"
    assert result.retrieval_calls == 1
    assert result.generation_calls == 0
    assert result.query_rewrites == 0
    assert result.grounded_report.candidates == []
    assert result.citation_audit.passed is True
    assert generator.prompts == []


def test_single_pass_rag_rejects_partially_uncited_bullets() -> None:
    """One citation must not validate other uncited bullets."""
    generator = CapturingGenerator(
        (
            "- The scholarship is hosted in Finland "
            "[nordic-ai-masters-2027:host_countries].\n"
            "- It includes additional unsupported information."
        )
    )

    result = run_single_pass_rag(
        query=(
            "Find a fully funded artificial intelligence "
            "master's scholarship in Finland"
        ),
        profile=build_profile(),
        index=build_index(),
        generator=generator,
        generator_name="partial-citation-generator",
        as_of=AS_OF,
        top_k=3,
    )

    assert result.status == "citation_failed"
    assert result.citation_audit.passed is False
    assert result.citation_audit.bullet_count == 2
    assert len(
        result.citation_audit.uncited_bullets
    ) == 1
