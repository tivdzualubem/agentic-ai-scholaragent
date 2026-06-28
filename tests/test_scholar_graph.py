"""Tests for the bounded ScholarAgent LangGraph workflow."""

from datetime import date
from pathlib import Path

from scholaragent.agents.scholar_graph import (
    ScholarAgentOutcome,
    build_scholar_agent_graph,
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


def build_profile() -> StudentProfile:
    """Create a reusable synthetic student profile."""
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


def build_graph():
    """Create the graph over the fictional development corpus."""
    records = load_scholarships(DATASET)
    index = BM25ScholarshipIndex(records)

    return build_scholar_agent_graph(index)


def test_specific_query_completes_without_rewrite() -> None:
    """A well-specified matching query should finish in one pass."""
    graph = build_graph()

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
    assert outcome.rewrites == []
    assert outcome.candidates
    assert (
        outcome.candidates[0].scholarship_id
        == "nordic-ai-masters-2027"
    )


def test_generic_query_is_rewritten_from_profile() -> None:
    """A generic query should receive one bounded profile rewrite."""
    graph = build_graph()

    state = graph.invoke(
        {
            "original_query": "scholarship",
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
    assert outcome.attempts == 2
    assert len(outcome.rewrites) == 1
    assert "Artificial Intelligence" in outcome.final_query
    assert "Finland" in outcome.final_query
    assert (
        outcome.candidates[0].scholarship_id
        == "nordic-ai-masters-2027"
    )


def test_out_of_domain_query_abstains_without_drift() -> None:
    """Explicit unsupported intent must not be replaced by profile intent."""
    graph = build_graph()

    state = graph.invoke(
        {
            "original_query": (
                "xylophone archaeology scholarship on Mars"
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

    assert outcome.status == "abstained"
    assert outcome.attempts == 1
    assert outcome.rewrites == []
    assert outcome.candidates == []
    assert outcome.evidence_sufficient is False
