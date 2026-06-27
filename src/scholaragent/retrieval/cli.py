"""Command-line interface for the BM25 scholarship baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

from scholaragent.retrieval import (
    BM25ScholarshipIndex,
    load_scholarships,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the search command-line parser."""
    parser = argparse.ArgumentParser(
        description="Search a validated scholarship JSON dataset with BM25."
    )
    parser.add_argument("query", help="Scholarship information need.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/demo/synthetic_scholarships.json"),
        help="Path to a scholarship JSON dataset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Maximum number of results to display.",
    )
    return parser


def main() -> None:
    """Run the BM25 scholarship-search demonstration."""
    args = build_parser().parse_args()

    scholarships = load_scholarships(args.data)
    index = BM25ScholarshipIndex(scholarships)
    results = index.search(args.query, k=args.top_k)

    print(f"Query: {args.query}")
    print(f"Dataset: {args.data}")
    print()

    if not results:
        print("No lexical matches were found.")
        return

    for result in results:
        scholarship = result.scholarship

        print(
            f"{result.rank}. {scholarship.title} "
            f"(score={result.score:.4f})"
        )
        print(f"   Provider: {scholarship.provider}")
        print(
            "   Country: "
            + ", ".join(scholarship.host_countries)
        )
        print(
            "   Degree: "
            + ", ".join(
                level.value for level in scholarship.degree_levels
            )
        )
        print(f"   Funding: {scholarship.funding_type.value}")
        print(f"   Source: {scholarship.official_url}")
        print()


if __name__ == "__main__":
    main()
