"""Validation tests for the official calibration corpus."""

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
    "data/calibration/"
    "calibration_scholarships.json"
)
MANIFEST = Path(
    "data/calibration/"
    "source_manifest.json"
)
EVALUATION_DATE = date(2026, 5, 1)

ADELAIDE_ID = (
    "adelaide-academic-excellence-"
    "scholarship-2026"
)


def _profile(
    *,
    gpa: float | None,
    gpa_scale: float | None,
) -> StudentProfile:
    return StudentProfile(
        nationality="Nigerian",
        country_of_residence="Finland",
        target_degree_level="master",
        fields_of_study=[
            "Artificial Intelligence",
        ],
        gpa=gpa,
        gpa_scale=gpa_scale,
        language_scores={
            "IELTS": 7.5,
        },
        years_work_experience=1,
        preferred_countries=[
            "Australia",
        ],
        requires_full_funding=False,
    )


def test_calibration_dataset_contains_adelaide_record() -> None:
    """The Adelaide calibration record should validate."""
    records = load_scholarships(DATASET)

    assert {
        record.scholarship_id
        for record in records
    } == {
        ADELAIDE_ID,
    }

    record = records[0]

    assert {
        level.value
        for level in record.degree_levels
    } == {
        "bachelor",
        "master",
    }
    assert record.funding_type.value == "tuition_only"
    assert record.minimum_gpa == 6.7
    assert record.gpa_scale == 7.0
    assert record.deadline.isoformat() == "2026-05-22"
    assert record.manual_review_requirements


def test_calibration_manifest_matches_snapshot() -> None:
    """The record should map to its validated source snapshot."""
    manifest = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )
    records = load_scholarships(DATASET)

    assert manifest["partition"] == "calibration"
    assert {
        entry["scholarship_id"]
        for entry in manifest["records"]
    } == {
        record.scholarship_id
        for record in records
    }

    entry = manifest["records"][0]
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


def test_adelaide_gpa_cases_cover_three_statuses() -> None:
    """GPA evidence should create deterministic status cases."""
    record = load_scholarships(DATASET)[0]

    high_gpa = assess_eligibility(
        _profile(
            gpa=6.8,
            gpa_scale=7.0,
        ),
        record,
        as_of=EVALUATION_DATE,
    )
    missing_gpa = assess_eligibility(
        _profile(
            gpa=None,
            gpa_scale=None,
        ),
        record,
        as_of=EVALUATION_DATE,
    )
    low_gpa = assess_eligibility(
        _profile(
            gpa=6.5,
            gpa_scale=7.0,
        ),
        record,
        as_of=EVALUATION_DATE,
    )

    assert (
        high_gpa.status
        is EligibilityStatus.POTENTIALLY_ELIGIBLE
    )
    assert (
        missing_gpa.status
        is EligibilityStatus.INSUFFICIENT_INFORMATION
    )
    assert (
        low_gpa.status
        is EligibilityStatus.NOT_ELIGIBLE
    )


def test_bm25_retrieves_adelaide_record() -> None:
    """A source-specific query should rank Adelaide first."""
    index = BM25ScholarshipIndex(
        load_scholarships(DATASET)
    )

    results = index.search(
        (
            "Adelaide Australia international "
            "bachelor master GPA 6.7 tuition "
            "fee reduction"
        ),
        k=1,
    )

    assert results
    assert (
        results[0].scholarship.scholarship_id
        == ADELAIDE_ID
    )
