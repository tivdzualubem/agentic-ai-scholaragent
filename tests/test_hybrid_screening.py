"""Integration tests for hybrid retrieval and screening."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

from scholaragent.agents.scholar_graph import (
    ScholarAgentOutcome,
    build_scholar_agent_graph,
)
from scholaragent.eligibility import EligibilityStatus
from scholaragent.pipeline import search_and_screen
from scholaragent.retrieval import (
    HybridScholarshipIndex,
    load_scholarships,
)
from scholaragent.schemas import StudentProfile

DATASET = Path(
    "data/demo/synthetic_scholarships.json"
)
AS_OF = date(2026, 6, 27)


class KeywordEmbeddingProvider:
    """Deterministic semantic embeddings for integration tests."""

    GROUPS = (
        (
            "artificial intelligence",
            "machine learning",
            "data science",
            "computer science",
        ),
        (
            "civil engineering",
            "structural engineering",
        ),
        (
            "public health",
            "global health",
        ),
        (
            "renewable energy",
            "sustainable energy",
        ),
        (
            "finland",
            "nordic",
        ),
        (
            "germany",
            "european",
        ),
        (
            "netherlands",
            "dutch",
        ),
        (
            "sweden",
            "scandinavian",
        ),
    )

    def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        vectors: list[list[float]] = []

        for text in texts:
            lowered = text.casefold()

            vector = [
                float(
                    sum(
                        lowered.count(keyword)
                        for keyword in group
                    )
                )
                for group in self.GROUPS
            ]

            vector.append(1.0)
            vectors.append(vector)

        return vectors


def build_profile() -> StudentProfile:
    """Return the reusable AI master's applicant profile."""
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


def build_hybrid_index() -> HybridScholarshipIndex:
    """Build the deterministic hybrid development index."""
    return HybridScholarshipIndex(
        load_scholarships(DATASET),
        KeywordEmbeddingProvider(),
    )


def test_hybrid_retrieval_can_use_screening_pipeline() -> None:
    """Hybrid results should pass through eligibility screening."""
    report = search_and_screen(
        query=(
            "fully funded artificial intelligence "
            "master's scholarship in Finland"
        ),
        profile=build_profile(),
        index=build_hybrid_index(),
        k=3,
        as_of=AS_OF,
    )

    assert report.results

    first = report.results[0]

    assert (
        first.scholarship.scholarship_id
        == "nordic-ai-masters-2027"
    )
    assert (
        first.assessment.status
        is EligibilityStatus.ELIGIBLE
    )


def test_langgraph_accepts_hybrid_retriever() -> None:
    """The bounded agent graph should accept the hybrid index."""
    graph = build_scholar_agent_graph(
        build_hybrid_index()
    )

    state = graph.invoke(
        {
            "original_query": (
                "fully funded artificial intelligence "
                "master's scholarship in Finland"
            ),
            "profile": build_profile(),
            "as_of": AS_OF,
            "top_k": 3,
            "max_attempts": 2,
        }
    )

    outcome = ScholarAgentOutcome.model_validate(
        state["outcome"]
    )

    assert outcome.status == "completed"
    assert outcome.attempts == 1
    assert outcome.candidates
    assert (
        outcome.candidates[0].scholarship_id
        == "nordic-ai-masters-2027"
    )
