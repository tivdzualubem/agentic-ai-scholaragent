"""Tests for the BM25 scholarship baseline."""

from pathlib import Path

import pytest

from scholaragent.retrieval import (
    BM25ScholarshipIndex,
    ScholarshipDataError,
    load_scholarships,
)

DATASET = Path("data/demo/synthetic_scholarships.json")


def test_demo_dataset_loads() -> None:
    """All synthetic development records satisfy the domain schema."""
    scholarships = load_scholarships(DATASET)

    assert len(scholarships) == 6
    assert len({
        scholarship.scholarship_id
        for scholarship in scholarships
    }) == 6


def test_ai_finland_query_ranks_expected_scholarship_first() -> None:
    """Relevant field, country, degree and funding terms affect ranking."""
    index = BM25ScholarshipIndex(load_scholarships(DATASET))

    results = index.search(
        "fully funded master artificial intelligence "
        "data science Finland",
        k=3,
    )

    assert results
    assert (
        results[0].scholarship.scholarship_id
        == "nordic-ai-masters-2027"
    )


def test_unknown_vocabulary_returns_no_results() -> None:
    """A query with no corpus overlap must not return arbitrary records."""
    index = BM25ScholarshipIndex(load_scholarships(DATASET))

    assert index.search("xylophone archaeology scholarship on Mars", k=3) == []


def test_empty_query_is_rejected() -> None:
    """Blank searches are invalid."""
    index = BM25ScholarshipIndex(load_scholarships(DATASET))

    with pytest.raises(ValueError, match="must not be empty"):
        index.search("   ")


def test_invalid_dataset_shape_is_rejected(tmp_path: Path) -> None:
    """The loader rejects JSON objects instead of silently accepting them."""
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text('{"title": "not a list"}', encoding="utf-8")

    with pytest.raises(
        ScholarshipDataError,
        match="must contain a JSON list",
    ):
        load_scholarships(invalid_file)


def test_generic_domain_terms_do_not_force_results() -> None:
    """Generic scholarship vocabulary must not create false matches."""
    index = BM25ScholarshipIndex(load_scholarships(DATASET))

    assert index.search(
        "scholarship fellowship grant",
        k=3,
    ) == []


def test_two_document_corpus_uses_discriminative_terms() -> None:
    """Distinctive terms should rank correctly in a tiny corpus."""
    from datetime import date

    from scholaragent.schemas import ScholarshipRecord

    records = [
        ScholarshipRecord(
            scholarship_id="alpha-academic-award",
            title="Alpha Academic Excellence Award",
            provider="Alpha University",
            official_url=(
                "https://example.org/alpha-academic-award"
            ),
            host_countries=["Australia"],
            degree_levels=["master"],
            source_last_checked=date(2026, 6, 28),
            eligibility_text=(
                "International master applicants require "
                "a GPA of 6.7 and receive a tuition fee "
                "reduction."
            ),
        ),
        ScholarshipRecord(
            scholarship_id="beta-research-award",
            title="Beta Research Award",
            provider="Beta University",
            official_url=(
                "https://example.org/beta-research-award"
            ),
            host_countries=["Australia"],
            degree_levels=["phd"],
            source_last_checked=date(2026, 6, 28),
            eligibility_text=(
                "International PhD applicants receive "
                "a research stipend."
            ),
        ),
    ]

    index = BM25ScholarshipIndex(records)

    results = index.search(
        (
            "Alpha Academic Excellence GPA 6.7 "
            "master tuition fee reduction"
        ),
        k=2,
    )

    assert len(results) == 2
    assert (
        results[0].scholarship.scholarship_id
        == "alpha-academic-award"
    )
    assert results[0].score > results[1].score
    assert results[0].score > 0
