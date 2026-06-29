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
    "data/calibration/source_manifest.json"
)

ADELAIDE_ID = (
    "adelaide-academic-excellence-"
    "scholarship-2026"
)
ANU_ID = "anu-phd-scholarship"
FERGUSON_ID = (
    "allan-nesta-ferguson-"
    "masters-scholarships-2026"
)
ETH_ID = (
    "eth-excellence-scholarship-"
    "opportunity-programme-2027"
)
KTH_ID = "kth-scholarship-2026"
AUCKLAND_ID = (
    "university-of-auckland-"
    "doctoral-scholarship-2026"
)

EXPECTED_IDS = {
    ADELAIDE_ID,
    ANU_ID,
    FERGUSON_ID,
    ETH_ID,
    KTH_ID,
    AUCKLAND_ID,
}


def _records():
    return {
        record.scholarship_id: record
        for record in load_scholarships(DATASET)
    }


def _profile(
    *,
    degree: str,
    field: str,
    gpa: float | None = 4.5,
    gpa_scale: float | None = 5.0,
    requires_full_funding: bool = False,
) -> StudentProfile:
    return StudentProfile(
        nationality="Nigerian",
        country_of_residence="Finland",
        target_degree_level=degree,
        fields_of_study=[field],
        gpa=gpa,
        gpa_scale=gpa_scale,
        language_scores={
            "IELTS": 7.5,
        },
        years_work_experience=2,
        preferred_countries=[],
        requires_full_funding=(
            requires_full_funding
        ),
    )


def test_calibration_dataset_has_six_diverse_records() -> None:
    """The frozen calibration corpus should contain six records."""
    records = _records()

    assert set(records) == EXPECTED_IDS

    degree_levels = {
        level.value
        for record in records.values()
        for level in record.degree_levels
    }
    funding_types = {
        record.funding_type.value
        for record in records.values()
    }
    host_countries = {
        country
        for record in records.values()
        for country in record.host_countries
    }

    assert {
        "bachelor",
        "master",
        "phd",
    }.issubset(degree_levels)

    assert {
        "fully_funded",
        "partially_funded",
        "tuition_only",
    }.issubset(funding_types)

    assert len(host_countries) >= 5


def test_calibration_manifest_matches_all_snapshots() -> None:
    """Every record should map to one validated snapshot."""
    manifest = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )
    records = _records()

    assert manifest["partition"] == "calibration"
    assert {
        entry["scholarship_id"]
        for entry in manifest["records"]
    } == set(records)

    assert len(manifest["records"]) == 6

    for entry in manifest["records"]:
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


def test_current_records_preserve_manual_conditions() -> None:
    """Unmodelled official conditions should remain explicit."""
    records = _records()

    assert all(
        record.manual_review_requirements
        for record in records.values()
    )


def test_adelaide_gpa_cases_cover_three_statuses() -> None:
    """Adelaide GPA evidence should create three outcomes."""
    record = _records()[ADELAIDE_ID]
    evaluation_date = date(2026, 5, 1)

    high = assess_eligibility(
        _profile(
            degree="master",
            field="Artificial Intelligence",
            gpa=6.8,
            gpa_scale=7.0,
        ),
        record,
        as_of=evaluation_date,
    )
    missing = assess_eligibility(
        _profile(
            degree="master",
            field="Artificial Intelligence",
            gpa=None,
            gpa_scale=None,
        ),
        record,
        as_of=evaluation_date,
    )
    low = assess_eligibility(
        _profile(
            degree="master",
            field="Artificial Intelligence",
            gpa=6.5,
            gpa_scale=7.0,
        ),
        record,
        as_of=evaluation_date,
    )

    assert (
        high.status
        is EligibilityStatus.POTENTIALLY_ELIGIBLE
    )
    assert (
        missing.status
        is EligibilityStatus.INSUFFICIENT_INFORMATION
    )
    assert (
        low.status
        is EligibilityStatus.NOT_ELIGIBLE
    )


def test_anu_degree_cases_are_conservative() -> None:
    """ANU should distinguish matching and wrong degrees."""
    record = _records()[ANU_ID]

    matching = assess_eligibility(
        _profile(
            degree="phd",
            field="Artificial Intelligence",
        ),
        record,
        as_of=date(2026, 6, 28),
    )
    wrong = assess_eligibility(
        _profile(
            degree="master",
            field="Artificial Intelligence",
        ),
        record,
        as_of=date(2026, 6, 28),
    )

    assert (
        matching.status
        is EligibilityStatus.POTENTIALLY_ELIGIBLE
    )
    assert (
        wrong.status
        is EligibilityStatus.NOT_ELIGIBLE
    )


def test_ferguson_cases_include_deadline_failure() -> None:
    """Ferguson should support active and expired cases."""
    record = _records()[FERGUSON_ID]
    profile = _profile(
        degree="master",
        field="International Development",
        requires_full_funding=True,
    )

    active = assess_eligibility(
        profile,
        record,
        as_of=date(2026, 5, 20),
    )
    expired = assess_eligibility(
        profile,
        record,
        as_of=date(2026, 6, 28),
    )

    assert (
        active.status
        is EligibilityStatus.POTENTIALLY_ELIGIBLE
    )
    assert (
        expired.status
        is EligibilityStatus.NOT_ELIGIBLE
    )


def test_eth_master_case_is_potentially_eligible() -> None:
    """ETH should remain manual because top-ten status is unverified."""
    record = _records()[ETH_ID]

    result = assess_eligibility(
        _profile(
            degree="master",
            field="Computer Science",
            requires_full_funding=True,
        ),
        record,
        as_of=date(2026, 11, 15),
    )

    assert (
        result.status
        is EligibilityStatus.POTENTIALLY_ELIGIBLE
    )
    assert result.manual_review_items


def test_kth_cases_include_expired_round() -> None:
    """KTH should distinguish an active and expired round."""
    record = _records()[KTH_ID]
    profile = _profile(
        degree="master",
        field="Data Science",
        requires_full_funding=False,
    )

    active = assess_eligibility(
        profile,
        record,
        as_of=date(2026, 1, 10),
    )
    expired = assess_eligibility(
        profile,
        record,
        as_of=date(2026, 6, 28),
    )

    assert (
        active.status
        is EligibilityStatus.POTENTIALLY_ELIGIBLE
    )
    assert (
        expired.status
        is EligibilityStatus.NOT_ELIGIBLE
    )


def test_auckland_doctoral_degree_cases() -> None:
    """Auckland should distinguish doctoral and master profiles."""
    record = _records()[AUCKLAND_ID]

    doctoral = assess_eligibility(
        _profile(
            degree="phd",
            field="Artificial Intelligence",
            requires_full_funding=True,
        ),
        record,
        as_of=date(2026, 6, 28),
    )
    master = assess_eligibility(
        _profile(
            degree="master",
            field="Artificial Intelligence",
            requires_full_funding=True,
        ),
        record,
        as_of=date(2026, 6, 28),
    )

    assert (
        doctoral.status
        is EligibilityStatus.POTENTIALLY_ELIGIBLE
    )
    assert (
        master.status
        is EligibilityStatus.NOT_ELIGIBLE
    )


def test_bm25_ranks_each_calibration_record_first() -> None:
    """Distinctive source queries should retrieve the correct record."""
    index = BM25ScholarshipIndex(
        list(_records().values())
    )

    expectations = {
        (
            "Adelaide Academic Excellence GPA 6.7 "
            "bachelor master tuition reduction"
        ): ADELAIDE_ID,
        (
            "ANU PhD Australia research stipend "
            "first class honours"
        ): ANU_ID,
        (
            "Sheffield Ferguson International "
            "Development full tuition maintenance"
        ): FERGUSON_ID,
        (
            "ETH ESOP Switzerland master top 10 "
            "CHF tuition waiver"
        ): ETH_ID,
        (
            "KTH Sweden master full tuition "
            "fee-paying first priority"
        ): KTH_ID,
        (
            "Auckland New Zealand doctoral stipend "
            "tuition 42 months health insurance"
        ): AUCKLAND_ID,
    }

    for query, expected_id in expectations.items():
        results = index.search(
            query,
            k=1,
        )

        assert results
        assert (
            results[0].scholarship.scholarship_id
            == expected_id
        )
        assert results[0].score > 0


def test_verified_anu_requirements_enable_eligible_status() -> None:
    """Adjudicated ANU evidence should cover the eligible class."""
    record = _records()[ANU_ID]

    profile = _profile(
        degree="phd",
        field="Artificial Intelligence",
        requires_full_funding=False,
    ).model_copy(
        update={
            "verified_manual_requirements": {
                ANU_ID: list(
                    record.manual_review_requirements
                ),
            },
        }
    )

    result = assess_eligibility(
        profile,
        record,
        as_of=date(2026, 6, 28),
    )

    assert result.status is EligibilityStatus.ELIGIBLE
    assert result.hard_failures == []
    assert result.missing_information == []
    assert result.manual_review_items == []
    assert len(
        [
            check
            for check in result.passed_checks
            if check.startswith(
                "Verified manual requirement:"
            )
        ]
    ) == len(record.manual_review_requirements)
