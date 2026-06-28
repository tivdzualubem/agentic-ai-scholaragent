"""Tests for evidence grounding and citation verification."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from scholaragent.agents.scholar_graph import (
    ScholarAgentOutcome,
    build_scholar_agent_graph,
)
from scholaragent.grounding import (
    build_grounded_candidate,
    build_grounded_report,
    verify_candidate_citations,
)
from scholaragent.pipeline import search_and_screen
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


def build_search_report():
    """Run the deterministic development search."""
    records = load_scholarships(DATASET)
    index = BM25ScholarshipIndex(records)

    return search_and_screen(
        query=(
            "fully funded artificial intelligence "
            "master's scholarship in Finland"
        ),
        profile=build_profile(),
        index=index,
        k=3,
        as_of=AS_OF,
    )


def test_grounded_report_contains_verified_citations() -> None:
    """Every factual claim should resolve to verified evidence."""
    grounded = build_grounded_report(
        build_search_report(),
        as_of=AS_OF,
    )

    assert grounded.candidates
    assert grounded.all_citations_verified is True

    candidate = grounded.candidates[0]

    assert candidate.verification.passed is True
    assert candidate.verification.errors == []
    assert candidate.claims
    assert candidate.evidence

    evidence_ids = {
        evidence.citation_id
        for evidence in candidate.evidence
    }

    for claim in candidate.claims:
        assert claim.citation_ids
        assert set(claim.citation_ids) <= evidence_ids


def test_citation_verifier_detects_tampering() -> None:
    """Modified evidence must fail deterministic verification."""
    report = build_search_report()
    screened = report.results[0]

    candidate = build_grounded_candidate(screened)

    tampered_evidence = list(candidate.evidence)
    tampered_evidence[0] = (
        tampered_evidence[0].model_copy(
            update={
                "text": "Tampered unsupported evidence."
            }
        )
    )

    tampered_candidate = candidate.model_copy(
        update={"evidence": tampered_evidence}
    )

    verification = verify_candidate_citations(
        tampered_candidate,
        screened.scholarship,
    )

    assert verification.passed is False
    assert any(
        "does not match the source" in error
        for error in verification.errors
    )


def test_graph_outcome_contains_grounded_report() -> None:
    """Completed graph executions should expose cited evidence."""
    records = load_scholarships(DATASET)
    graph = build_scholar_agent_graph(
        BM25ScholarshipIndex(records)
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
    assert outcome.grounded_report is not None
    assert (
        outcome.grounded_report.all_citations_verified
        is True
    )

    first = outcome.grounded_report.candidates[0]

    assert (
        first.scholarship_id
        == "nordic-ai-masters-2027"
    )
    assert first.official_url.startswith("https://")
    assert first.verification.passed is True


def test_abstained_graph_has_no_grounded_candidates() -> None:
    """Abstention must not manufacture claims or citations."""
    records = load_scholarships(DATASET)
    graph = build_scholar_agent_graph(
        BM25ScholarshipIndex(records)
    )

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
    assert outcome.grounded_report is not None
    assert outcome.grounded_report.candidates == []
    assert (
        outcome.grounded_report.all_citations_verified
        is False
    )


def test_manual_requirements_receive_verified_citations() -> None:
    """Manual-review conditions must appear as grounded evidence."""
    screened = build_search_report().results[0]

    scholarship = screened.scholarship.model_copy(
        update={
            "manual_review_requirements": [
                "Confirm admission before applying."
            ]
        }
    )

    updated_screened = screened.model_copy(
        update={"scholarship": scholarship}
    )

    candidate = build_grounded_candidate(
        updated_screened
    )

    citation_id = (
        f"{scholarship.scholarship_id}:"
        "manual_review_requirements"
    )

    assert candidate.verification.passed is True

    assert any(
        evidence.citation_id == citation_id
        for evidence in candidate.evidence
    )

    assert any(
        citation_id in claim.citation_ids
        for claim in candidate.claims
    )
