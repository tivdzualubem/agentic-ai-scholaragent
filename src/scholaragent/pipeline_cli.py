"""Command-line demonstration of integrated retrieval and screening."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from scholaragent.pipeline import search_and_screen
from scholaragent.retrieval import (
    BM25ScholarshipIndex,
    load_scholarships,
)
from scholaragent.schemas import StudentProfile


def build_parser() -> argparse.ArgumentParser:
    """Create the pipeline command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve scholarships with BM25 and screen them against "
            "a structured student profile."
        )
    )
    parser.add_argument(
        "query",
        help="Scholarship information need.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/demo/synthetic_scholarships.json"),
        help="Path to the scholarship dataset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of lexical candidates.",
    )
    return parser


def main() -> None:
    """Run the integrated development demonstration."""
    args = build_parser().parse_args()

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

    records = load_scholarships(args.data)
    index = BM25ScholarshipIndex(records)

    report = search_and_screen(
        query=args.query,
        profile=profile,
        index=index,
        k=args.top_k,
        as_of=date(2026, 6, 27),
    )

    print(f"Query: {report.query}")
    print(f"Candidates retrieved: {report.retrieved_count}")
    print()

    if not report.results:
        print("No lexical candidates were found.")
        return

    for item in report.results:
        scholarship = item.scholarship
        assessment = item.assessment

        print(
            f"{item.final_rank}. {scholarship.title}"
        )
        print(
            f"   Status: {assessment.status.value}"
        )
        print(
            f"   Retrieval rank: {item.retrieval_rank}"
        )
        print(
            f"   BM25 score: {item.retrieval_score:.4f}"
        )
        print(
            "   Country: "
            + ", ".join(scholarship.host_countries)
        )
        print(
            f"   Source: {scholarship.official_url}"
        )

        for failure in assessment.hard_failures:
            print(f"   Failure: {failure}")

        for missing in assessment.missing_information:
            print(f"   Missing: {missing}")

        for review in assessment.manual_review_items:
            print(f"   Review: {review}")

        for warning in assessment.preference_warnings:
            print(f"   Preference: {warning}")

        print()


if __name__ == "__main__":
    main()
