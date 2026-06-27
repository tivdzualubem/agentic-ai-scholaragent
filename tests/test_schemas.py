"""Tests for ScholarAgent's core domain schemas."""

from datetime import date

import pytest
from pydantic import ValidationError

from scholaragent.schemas import (
    DegreeLevel,
    FundingType,
    ScholarshipRecord,
    StudentProfile,
)


def test_student_profile_normalizes_values() -> None:
    """Profile text and repeated list values are normalized."""
    profile = StudentProfile(
        nationality=" Nigerian ",
        country_of_residence=" Finland ",
        target_degree_level="master",
        fields_of_study=[
            "Artificial Intelligence",
            " artificial intelligence ",
            "Data Science",
            "",
        ],
        gpa=4.2,
        gpa_scale=5.0,
        language_scores={" ielts ": 7.5},
        preferred_countries=["Germany", " germany ", "Finland"],
    )

    assert profile.nationality == "Nigerian"
    assert profile.target_degree_level is DegreeLevel.MASTER
    assert profile.fields_of_study == [
        "Artificial Intelligence",
        "Data Science",
    ]
    assert profile.language_scores == {"IELTS": 7.5}
    assert profile.preferred_countries == ["Germany", "Finland"]


def test_student_profile_requires_complete_gpa_pair() -> None:
    """A GPA without its scale is ambiguous and must be rejected."""
    with pytest.raises(
        ValidationError,
        match="gpa and gpa_scale",
    ):
        StudentProfile(
            nationality="Nigerian",
            target_degree_level="master",
            fields_of_study=["Artificial Intelligence"],
            gpa=4.2,
        )


def test_student_profile_rejects_gpa_above_scale() -> None:
    """A GPA cannot exceed its declared scale."""
    with pytest.raises(
        ValidationError,
        match="gpa must not exceed gpa_scale",
    ):
        StudentProfile(
            nationality="Nigerian",
            target_degree_level="master",
            fields_of_study=["Artificial Intelligence"],
            gpa=6.0,
            gpa_scale=5.0,
        )


def test_scholarship_record_accepts_valid_data() -> None:
    """A complete scholarship record is parsed into typed fields."""
    scholarship = ScholarshipRecord(
        scholarship_id="example-global-masters-2027",
        title="Example Global Master's Scholarship",
        provider="Example University",
        official_url="https://example.edu/scholarships/global-masters",
        host_countries=["Finland"],
        degree_levels=["master"],
        eligible_nationalities=["All international students"],
        eligible_fields=["Artificial Intelligence", "Data Science"],
        minimum_gpa=4.0,
        gpa_scale=5.0,
        language_requirements={"ielts": 6.5},
        funding_type="fully_funded",
        deadline="2027-01-15",
        application_year=2027,
        source_last_checked="2026-06-27",
        eligibility_text="Applicants must satisfy the listed requirements.",
    )

    assert scholarship.degree_levels == [DegreeLevel.MASTER]
    assert scholarship.funding_type is FundingType.FULLY_FUNDED
    assert scholarship.deadline == date(2027, 1, 15)
    assert scholarship.language_requirements == {"IELTS": 6.5}


def test_scholarship_record_rejects_invalid_identifier() -> None:
    """Scholarship identifiers must be stable URL-safe slugs."""
    with pytest.raises(ValidationError):
        ScholarshipRecord(
            scholarship_id="Invalid Scholarship ID",
            title="Invalid Scholarship",
            provider="Example University",
            official_url="https://example.edu/scholarship",
            host_countries=["Finland"],
            degree_levels=["master"],
            source_last_checked="2026-06-27",
            eligibility_text="Example requirements.",
        )


def test_models_reject_unknown_fields() -> None:
    """Unexpected fields are rejected instead of silently ignored."""
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        StudentProfile(
            nationality="Nigerian",
            target_degree_level="master",
            fields_of_study=["Artificial Intelligence"],
            private_note="must not be silently accepted",
        )
