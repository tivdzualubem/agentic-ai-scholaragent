"""Command-line demonstration of deterministic eligibility screening."""

from datetime import date
from pathlib import Path

from scholaragent.eligibility import assess_eligibility
from scholaragent.retrieval import load_scholarships
from scholaragent.schemas import StudentProfile


def main() -> None:
    """Assess one example applicant against the development dataset."""
    profile = StudentProfile(
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

    scholarships = load_scholarships(
        Path("data/demo/synthetic_scholarships.json")
    )

    print("Applicant profile:")
    print(profile.model_dump_json(indent=2))
    print()

    for scholarship in scholarships:
        assessment = assess_eligibility(
            profile,
            scholarship,
            as_of=date(2026, 6, 27),
        )

        print(f"{scholarship.title}")
        print(f"Status: {assessment.status.value}")

        for failure in assessment.hard_failures:
            print(f"  FAIL: {failure}")

        for missing in assessment.missing_information:
            print(f"  MISSING: {missing}")

        for item in assessment.manual_review_items:
            print(f"  REVIEW: {item}")

        for warning in assessment.preference_warnings:
            print(f"  PREFERENCE: {warning}")

        print()


if __name__ == "__main__":
    main()
