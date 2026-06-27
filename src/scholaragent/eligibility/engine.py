"""Deterministic scholarship-eligibility screening rules."""

from __future__ import annotations

from datetime import date
from typing import Iterable

from scholaragent.eligibility.models import (
    EligibilityAssessment,
    EligibilityStatus,
)
from scholaragent.schemas import ScholarshipRecord, StudentProfile


GLOBAL_NATIONALITY_PHRASES = {
    "all international students",
    "international students",
    "all nationalities",
    "students of all nationalities",
    "open to all nationalities",
}

BROAD_NATIONALITY_MARKERS = {
    "african",
    "countries",
    "country",
    "non-eu",
    "developing",
    "low-income",
    "middle-income",
    "applicants from",
}


def _normalize(value: str) -> str:
    """Normalize text for case-insensitive comparisons."""
    return " ".join(value.casefold().split())


def _overlaps(left: Iterable[str], right: Iterable[str]) -> bool:
    """Return whether two text collections overlap semantically enough."""
    normalized_left = [_normalize(item) for item in left]
    normalized_right = [_normalize(item) for item in right]

    for first in normalized_left:
        for second in normalized_right:
            if first == second or first in second or second in first:
                return True

    return False


def _assess_nationality(
    profile: StudentProfile,
    scholarship: ScholarshipRecord,
    *,
    passed: list[str],
    failures: list[str],
    manual: list[str],
) -> None:
    """Evaluate structured nationality requirements conservatively."""
    requirements = [
        _normalize(item)
        for item in scholarship.eligible_nationalities
    ]

    if not requirements:
        manual.append(
            "No structured nationality rule is available; inspect the "
            "official eligibility text."
        )
        return

    if any(item in GLOBAL_NATIONALITY_PHRASES for item in requirements):
        passed.append("Nationality requirement allows international students.")
        return

    nationality = _normalize(profile.nationality)

    if nationality in requirements:
        passed.append(
            f"Applicant nationality '{profile.nationality}' is explicitly allowed."
        )
        return

    has_broad_rule = any(
        marker in requirement
        for requirement in requirements
        for marker in BROAD_NATIONALITY_MARKERS
    )

    if has_broad_rule:
        manual.append(
            "Nationality is governed by a regional or economic-group rule "
            "that requires authoritative verification."
        )
        return

    failures.append(
        f"Applicant nationality '{profile.nationality}' is not listed among "
        "the eligible nationalities."
    )


def _assess_degree(
    profile: StudentProfile,
    scholarship: ScholarshipRecord,
    *,
    passed: list[str],
    failures: list[str],
) -> None:
    """Check the target degree level."""
    if profile.target_degree_level in scholarship.degree_levels:
        passed.append(
            f"Target degree level '{profile.target_degree_level.value}' matches."
        )
    else:
        allowed = ", ".join(
            level.value for level in scholarship.degree_levels
        )
        failures.append(
            f"Target degree level '{profile.target_degree_level.value}' "
            f"is not accepted; allowed levels: {allowed}."
        )


def _assess_field(
    profile: StudentProfile,
    scholarship: ScholarshipRecord,
    *,
    passed: list[str],
    failures: list[str],
) -> None:
    """Check whether the applicant's field overlaps an eligible field."""
    if not scholarship.eligible_fields:
        passed.append("No structured field-of-study restriction is recorded.")
        return

    if _overlaps(
        profile.fields_of_study,
        scholarship.eligible_fields,
    ):
        passed.append("At least one field of study matches.")
    else:
        failures.append(
            "The applicant's fields of study do not match the recorded "
            "eligible fields."
        )


def _assess_gpa(
    profile: StudentProfile,
    scholarship: ScholarshipRecord,
    *,
    passed: list[str],
    failures: list[str],
    missing: list[str],
) -> None:
    """Compare GPAs after converting both to proportions of their scales."""
    if scholarship.minimum_gpa is None:
        passed.append("No structured minimum GPA is recorded.")
        return

    if profile.gpa is None or profile.gpa_scale is None:
        missing.append(
            "Applicant GPA and GPA scale are required for this scholarship."
        )
        return

    assert scholarship.gpa_scale is not None

    applicant_ratio = profile.gpa / profile.gpa_scale
    required_ratio = scholarship.minimum_gpa / scholarship.gpa_scale

    if applicant_ratio + 1e-9 >= required_ratio:
        passed.append(
            "Applicant GPA satisfies the normalized minimum requirement."
        )
    else:
        failures.append(
            "Applicant GPA is below the normalized minimum requirement."
        )


def _assess_work_experience(
    profile: StudentProfile,
    scholarship: ScholarshipRecord,
    *,
    passed: list[str],
    failures: list[str],
) -> None:
    """Check required years of professional experience."""
    required = scholarship.minimum_work_experience_years

    if required is None:
        passed.append("No minimum work-experience requirement is recorded.")
        return

    if profile.years_work_experience >= required:
        passed.append(
            f"Applicant meets the {required:g}-year work-experience requirement."
        )
    else:
        failures.append(
            f"Scholarship requires {required:g} years of work experience; "
            f"applicant provided {profile.years_work_experience:g}."
        )


def _assess_language(
    profile: StudentProfile,
    scholarship: ScholarshipRecord,
    *,
    passed: list[str],
    failures: list[str],
    missing: list[str],
) -> None:
    """Check each structured language-test requirement."""
    if not scholarship.language_requirements:
        passed.append("No structured language-score requirement is recorded.")
        return

    for test_name, required_score in (
        scholarship.language_requirements.items()
    ):
        applicant_score = profile.language_scores.get(test_name)

        if applicant_score is None:
            missing.append(
                f"Applicant score for {test_name} is required."
            )
        elif applicant_score >= required_score:
            passed.append(
                f"{test_name} score {applicant_score:g} meets the "
                f"minimum {required_score:g}."
            )
        else:
            failures.append(
                f"{test_name} score {applicant_score:g} is below the "
                f"minimum {required_score:g}."
            )


def _assess_deadline(
    scholarship: ScholarshipRecord,
    *,
    as_of: date,
    passed: list[str],
    failures: list[str],
) -> None:
    """Reject scholarship rounds whose recorded deadline has passed."""
    if scholarship.deadline is None:
        passed.append("No structured deadline is available.")
        return

    if scholarship.deadline >= as_of:
        passed.append(
            f"Deadline {scholarship.deadline.isoformat()} has not passed."
        )
    else:
        failures.append(
            f"Deadline {scholarship.deadline.isoformat()} has passed."
        )


def _assess_preferences(
    profile: StudentProfile,
    scholarship: ScholarshipRecord,
    *,
    warnings: list[str],
) -> None:
    """Record preference mismatches without treating them as eligibility rules."""
    if (
        profile.requires_full_funding
        and scholarship.funding_type.value != "fully_funded"
    ):
        warnings.append(
            "Applicant requires full funding, but this scholarship is "
            f"recorded as '{scholarship.funding_type.value}'."
        )

    if (
        profile.preferred_countries
        and not _overlaps(
            profile.preferred_countries,
            scholarship.host_countries,
        )
    ):
        warnings.append(
            "Host country is outside the applicant's preferred countries."
        )


def assess_eligibility(
    profile: StudentProfile,
    scholarship: ScholarshipRecord,
    *,
    as_of: date | None = None,
) -> EligibilityAssessment:
    """Produce an explainable deterministic eligibility assessment."""
    evaluation_date = as_of or date.today()

    passed: list[str] = []
    failures: list[str] = []
    missing: list[str] = []
    manual: list[str] = []
    warnings: list[str] = []

    _assess_degree(
        profile,
        scholarship,
        passed=passed,
        failures=failures,
    )
    _assess_field(
        profile,
        scholarship,
        passed=passed,
        failures=failures,
    )
    _assess_nationality(
        profile,
        scholarship,
        passed=passed,
        failures=failures,
        manual=manual,
    )
    _assess_gpa(
        profile,
        scholarship,
        passed=passed,
        failures=failures,
        missing=missing,
    )
    _assess_work_experience(
        profile,
        scholarship,
        passed=passed,
        failures=failures,
    )
    _assess_language(
        profile,
        scholarship,
        passed=passed,
        failures=failures,
        missing=missing,
    )
    _assess_deadline(
        scholarship,
        as_of=evaluation_date,
        passed=passed,
        failures=failures,
    )
    _assess_preferences(
        profile,
        scholarship,
        warnings=warnings,
    )

    if failures:
        status = EligibilityStatus.NOT_ELIGIBLE
    elif missing:
        status = EligibilityStatus.INSUFFICIENT_INFORMATION
    elif manual:
        status = EligibilityStatus.POTENTIALLY_ELIGIBLE
    else:
        status = EligibilityStatus.ELIGIBLE

    return EligibilityAssessment(
        scholarship_id=scholarship.scholarship_id,
        scholarship_title=scholarship.title,
        status=status,
        passed_checks=passed,
        hard_failures=failures,
        missing_information=missing,
        manual_review_items=manual,
        preference_warnings=warnings,
    )
