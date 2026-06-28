"""Tests for deterministic scholarship eligibility screening."""

from datetime import date
from pathlib import Path

from scholaragent.eligibility import (
    EligibilityStatus,
    assess_eligibility,
)
from scholaragent.retrieval import load_scholarships
from scholaragent.schemas import StudentProfile

DATASET = Path("data/demo/synthetic_scholarships.json")
AS_OF = date(2026, 6, 27)


def _records_by_id():
    return {
        item.scholarship_id: item
        for item in load_scholarships(DATASET)
    }


def _ai_profile(**updates):
    values = {
        "nationality": "Nigerian",
        "country_of_residence": "Finland",
        "target_degree_level": "master",
        "fields_of_study": [
            "Artificial Intelligence",
            "Data Science",
        ],
        "gpa": 4.2,
        "gpa_scale": 5.0,
        "language_scores": {"IELTS": 7.5},
        "years_work_experience": 1,
        "preferred_countries": ["Finland", "Germany"],
        "requires_full_funding": True,
    }
    values.update(updates)
    return StudentProfile(**values)


def test_matching_ai_scholarship_is_eligible() -> None:
    """The complete matching profile satisfies the Nordic AI record."""
    scholarship = _records_by_id()["nordic-ai-masters-2027"]

    assessment = assess_eligibility(
        _ai_profile(),
        scholarship,
        as_of=AS_OF,
    )

    assert assessment.status is EligibilityStatus.ELIGIBLE
    assert assessment.hard_failures == []
    assert assessment.missing_information == []


def test_field_mismatch_is_not_eligible() -> None:
    """A lexical result can still fail structured eligibility."""
    scholarship = _records_by_id()[
        "european-civil-engineering-award-2027"
    ]

    assessment = assess_eligibility(
        _ai_profile(),
        scholarship,
        as_of=AS_OF,
    )

    assert assessment.status is EligibilityStatus.NOT_ELIGIBLE
    assert any(
        "fields of study" in reason
        for reason in assessment.hard_failures
    )


def test_missing_gpa_produces_insufficient_information() -> None:
    """Missing applicant evidence must not be treated as eligibility."""
    scholarship = _records_by_id()["nordic-ai-masters-2027"]

    assessment = assess_eligibility(
        _ai_profile(gpa=None, gpa_scale=None),
        scholarship,
        as_of=AS_OF,
    )

    assert (
        assessment.status
        is EligibilityStatus.INSUFFICIENT_INFORMATION
    )
    assert assessment.missing_information


def test_regional_nationality_rule_requires_manual_review() -> None:
    """Broad regional rules are not guessed by the deterministic engine."""
    scholarship = _records_by_id()["african-women-stem-phd-2027"]

    profile = _ai_profile(
        target_degree_level="phd",
        fields_of_study=["Artificial Intelligence"],
        gpa=None,
        gpa_scale=None,
    )

    assessment = assess_eligibility(
        profile,
        scholarship,
        as_of=AS_OF,
    )

    assert (
        assessment.status
        is EligibilityStatus.POTENTIALLY_ELIGIBLE
    )
    assert assessment.manual_review_items


def test_funding_preference_is_reported_separately() -> None:
    """A preference mismatch is not mislabeled as formal ineligibility."""
    scholarship = _records_by_id()[
        "netherlands-data-excellence-2027"
    ]

    assessment = assess_eligibility(
        _ai_profile(preferred_countries=["Netherlands"]),
        scholarship,
        as_of=AS_OF,
    )

    assert assessment.preference_warnings
    assert any(
        "full funding" in warning
        for warning in assessment.preference_warnings
    )


def test_recorded_manual_requirements_prevent_full_eligibility() -> None:
    """Unmodelled official conditions require manual verification."""
    scholarship = _records_by_id()[
        "nordic-ai-masters-2027"
    ].model_copy(
        update={
            "manual_review_requirements": [
                "Confirm prior admission to the programme.",
                "Verify programme-specific document requirements.",
            ]
        }
    )

    assessment = assess_eligibility(
        _ai_profile(),
        scholarship,
        as_of=AS_OF,
    )

    assert (
        assessment.status
        is EligibilityStatus.POTENTIALLY_ELIGIBLE
    )
    assert assessment.hard_failures == []
    assert assessment.manual_review_items == [
        "Confirm prior admission to the programme.",
        "Verify programme-specific document requirements.",
    ]
