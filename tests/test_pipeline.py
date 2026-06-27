"""Tests for integrated retrieval and eligibility screening."""

from datetime import date
from pathlib import Path

import pytest

from scholaragent.eligibility import EligibilityStatus
from scholaragent.pipeline import search_and_screen
from scholaragent.retrieval import (
    BM25ScholarshipIndex,
    load_scholarships,
)
from scholaragent.schemas import StudentProfile

DATASET = Path("data/demo/synthetic_scholarships.json")
AS_OF = date(2026, 6, 27)


@pytest.fixture
def profile() -> StudentProfile:
    """Return the standard development applicant profile."""
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
        years_work_experience=1,
        preferred_countries=["Finland", "Germany"],
        requires_full_funding=True,
    )


@pytest.fixture
def index() -> BM25ScholarshipIndex:
    """Build the development BM25 index."""
    return BM25ScholarshipIndex(load_scholarships(DATASET))


def test_matching_scholarship_is_ranked_first(
    profile: StudentProfile,
    index: BM25ScholarshipIndex,
) -> None:
    """The matching AI scholarship should lead the final ranking."""
    report = search_and_screen(
        query=(
            "fully funded master artificial intelligence "
            "data science Finland"
        ),
        profile=profile,
        index=index,
        k=5,
        as_of=AS_OF,
    )

    assert report.results
    assert (
        report.results[0].scholarship.scholarship_id
        == "nordic-ai-masters-2027"
    )
    assert (
        report.results[0].assessment.status
        is EligibilityStatus.ELIGIBLE
    )


def test_screening_can_correct_lexical_ranking(
    profile: StudentProfile,
    index: BM25ScholarshipIndex,
) -> None:
    """An ineligible lexical leader should move below an eligible result."""
    report = search_and_screen(
        query=(
            "fully funded artificial intelligence "
            "doctoral phd Norway"
        ),
        profile=profile,
        index=index,
        k=5,
        as_of=AS_OF,
    )

    african_result = next(
        item
        for item in report.results
        if item.scholarship.scholarship_id
        == "african-women-stem-phd-2027"
    )
    nordic_result = next(
        item
        for item in report.results
        if item.scholarship.scholarship_id
        == "nordic-ai-masters-2027"
    )

    assert african_result.retrieval_rank < nordic_result.retrieval_rank
    assert african_result.assessment.status is EligibilityStatus.NOT_ELIGIBLE
    assert nordic_result.assessment.status is EligibilityStatus.ELIGIBLE
    assert nordic_result.final_rank < african_result.final_rank


def test_report_retains_retrieval_evidence(
    profile: StudentProfile,
    index: BM25ScholarshipIndex,
) -> None:
    """The report preserves lexical ranks and scores for evaluation."""
    report = search_and_screen(
        query="master data science Europe",
        profile=profile,
        index=index,
        k=3,
        as_of=AS_OF,
    )

    assert report.retrieved_count <= 3
    assert all(item.retrieval_rank >= 1 for item in report.results)
    assert all(
        isinstance(item.retrieval_score, float)
        for item in report.results
    )


def test_no_overlap_returns_empty_report(
    profile: StudentProfile,
    index: BM25ScholarshipIndex,
) -> None:
    """No lexical overlap should produce an explicit empty result set."""
    report = search_and_screen(
        query="xylophone archaeology mars",
        profile=profile,
        index=index,
        k=3,
        as_of=AS_OF,
    )

    assert report.retrieved_count == 0
    assert report.results == []


def test_invalid_top_k_is_rejected(
    profile: StudentProfile,
    index: BM25ScholarshipIndex,
) -> None:
    """Invalid retrieval limits are not silently accepted."""
    with pytest.raises(ValueError, match="k must be at least 1"):
        search_and_screen(
            query="data science",
            profile=profile,
            index=index,
            k=0,
            as_of=AS_OF,
        )


def test_query_relevance_is_preserved_among_viable_candidates(
    profile: StudentProfile,
    index: BM25ScholarshipIndex,
) -> None:
    """Manual review must not make an unrelated scholarship rank first."""
    netherlands_profile = profile.model_copy(
        update={
            "preferred_countries": ["Netherlands"],
            "requires_full_funding": False,
            "gpa": 3.8,
            "gpa_scale": 4.0,
        }
    )

    report = search_and_screen(
        query=(
            "data science excellence master's scholarship "
            "in the Netherlands"
        ),
        profile=netherlands_profile,
        index=index,
        k=3,
        as_of=AS_OF,
    )

    assert report.results
    assert (
        report.results[0].scholarship.scholarship_id
        == "netherlands-data-excellence-2027"
    )
    assert (
        report.results[0].assessment.status
        is EligibilityStatus.POTENTIALLY_ELIGIBLE
    )
