"""Validation tests for the official-source development corpus."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from scholaragent.eligibility import (
    EligibilityStatus,
    assess_eligibility,
)
from scholaragent.ingestion import (
    load_source_snapshot,
)
from scholaragent.retrieval import (
    BM25ScholarshipIndex,
    load_scholarships,
)
from scholaragent.schemas import StudentProfile


DATASET = Path(
    "data/official/official_scholarships.json"
)
MANIFEST = Path(
    "data/official/source_manifest.json"
)
AS_OF = date(2026, 6, 28)


def _profile() -> StudentProfile:
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
        preferred_countries=[
            "Sweden",
            "Netherlands",
        ],
        requires_full_funding=True,
    )


def test_official_dataset_loads_three_records() -> None:
    """The initial official corpus should be valid and unique."""
    records = load_scholarships(DATASET)

    assert {
        record.scholarship_id
        for record in records
    } == {
        "erasmus-mundus-joint-masters",
        "si-global-professionals-2026",
        "university-twente-scholarship-2027",
    }

    assert all(
        record.manual_review_requirements
        for record in records
    )


def test_official_records_are_screened_conservatively() -> None:
    """Unverified conditions must prevent full eligibility."""
    records = {
        record.scholarship_id: record
        for record in load_scholarships(DATASET)
    }

    results = {
        scholarship_id: assess_eligibility(
            _profile(),
            record,
            as_of=AS_OF,
        )
        for scholarship_id, record in records.items()
    }

    assert (
        results[
            "erasmus-mundus-joint-masters"
        ].status
        is EligibilityStatus.POTENTIALLY_ELIGIBLE
    )

    assert (
        results[
            "si-global-professionals-2026"
        ].status
        is EligibilityStatus.NOT_ELIGIBLE
    )

    assert any(
        "has passed" in failure
        for failure in results[
            "si-global-professionals-2026"
        ].hard_failures
    )

    assert (
        results[
            "university-twente-scholarship-2027"
        ].status
        is EligibilityStatus.POTENTIALLY_ELIGIBLE
    )


def test_official_manifest_matches_saved_snapshots() -> None:
    """Every corpus record must link to a validated snapshot."""
    manifest = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )

    records = load_scholarships(DATASET)
    record_ids = {
        record.scholarship_id
        for record in records
    }

    entries = manifest["records"]

    assert {
        entry["scholarship_id"]
        for entry in entries
    } == record_ids

    for entry in entries:
        snapshot = load_source_snapshot(
            entry["snapshot_path"]
        )

        assert (
            entry["content_sha256"]
            == snapshot.content_sha256
        )
        assert (
            entry["source_last_checked"]
            == snapshot.source_last_checked.isoformat()
        )
        assert (
            entry["source_url"].rstrip("/")
            == str(snapshot.source_url).rstrip("/")
        )


def test_official_bm25_queries_rank_expected_sources() -> None:
    """Source-specific queries should retrieve the correct record."""
    index = BM25ScholarshipIndex(
        load_scholarships(DATASET)
    )

    expectations = {
        (
            "Erasmus Mundus joint master worldwide "
            "travel visa living allowance"
        ): "erasmus-mundus-joint-masters",
        (
            "Sweden global professionals leadership "
            "work experience monthly allowance"
        ): "si-global-professionals-2026",
        (
            "University Twente non EU MSc IELTS "
            "22000 scholarship"
        ): "university-twente-scholarship-2027",
    }

    for query, expected_id in expectations.items():
        results = index.search(query, k=1)

        assert results
        assert (
            results[0].scholarship.scholarship_id
            == expected_id
        )
