"""Validation tests for the official held-out corpus."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from scholaragent.eligibility import (
    EligibilityStatus,
    assess_eligibility,
)
from scholaragent.ingestion import (
    load_source_snapshot,
)
from scholaragent.retrieval import (
    load_scholarships,
)
from scholaragent.schemas import StudentProfile


DATASET = Path(
    "data/held_out/held_out_scholarships.json"
)
MANIFEST = Path(
    "data/held_out/source_manifest.json"
)
SETTINGS = Path(
    "eval/config/frozen_retrieval_settings.json"
)

BRISTOL_ID = (
    "bristol-think-big-scholarships-2026"
)
GLASGOW_ID = (
    "glasgow-global-leadership-scholarship-2026"
)
AALTO_ID = (
    "aalto-university-excellence-scholarship"
)
WAIKATO_ID = (
    "waikato-vice-chancellors-international-"
    "excellence-scholarship"
)
MANITOBA_ID = (
    "university-of-manitoba-graduate-fellowship"
)
MAASTRICHT_ID = (
    "maastricht-university-nl-high-potential-"
    "scholarship-2026"
)

EXPECTED_IDS = {
    BRISTOL_ID,
    GLASGOW_ID,
    AALTO_ID,
    WAIKATO_ID,
    MANITOBA_ID,
    MAASTRICHT_ID,
}


def _records():
    return {
        record.scholarship_id: record
        for record in load_scholarships(DATASET)
    }


def _profile(
    *,
    degree: str,
    field: str = "Artificial Intelligence",
    gpa: float | None = 4.0,
    gpa_scale: float | None = 5.0,
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
        requires_full_funding=False,
    )


def test_held_out_dataset_has_six_diverse_records() -> None:
    """The frozen test corpus should contain six identities."""
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

    assert len(host_countries) == 5


def test_held_out_manifest_matches_all_snapshots() -> None:
    """Every record should map to one immutable source snapshot."""
    manifest = json.loads(
        MANIFEST.read_text(encoding="utf-8")
    )
    records = _records()

    assert manifest["partition"] == "held_out_test"
    assert manifest["dataset"] == str(DATASET)

    assert {
        entry["scholarship_id"]
        for entry in manifest["records"]
    } == set(records)

    assert len(manifest["records"]) == 6

    for entry in manifest["records"]:
        snapshot = load_source_snapshot(
            entry["snapshot_path"]
        )
        record = records[
            entry["scholarship_id"]
        ]

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
        assert (
            str(record.official_url).rstrip("/")
            == str(snapshot.source_url).rstrip("/")
        )
        assert (
            record.source_last_checked
            == snapshot.source_last_checked
        )


def test_snapshots_retain_required_official_evidence() -> None:
    """Committed snapshots should contain the encoded source facts."""
    required_phrases = {
        (
            "bristol-think-big-scholarships-2026.json"
        ): [
            "funding towards tuition fees",
            "undergraduate or postgraduate",
            "10 April 2026",
        ],
        (
            "glasgow-global-leadership-"
            "scholarship-2026.json"
        ): [
            "postgraduate taught Masters",
            "tuition fees discount",
            "17th July 2026",
        ],
        (
            "aalto-university-excellence-"
            "scholarship.json"
        ): [
            "covering the full tuition fee",
            "do not cover living costs",
            "granted on a competitive basis",
        ],
        (
            "waikato-vice-chancellors-"
            "international-excellence-scholarship.json"
        ): [
            "Up to NZD$15,000 towards tuition fees",
            "all areas of study",
            "enrolling for the first time",
        ],
        (
            "university-of-manitoba-"
            "graduate-fellowship.json"
        ): [
            "$20,000 per year",
            "$25,000 per year",
            "minimum admission GPA of 3.0",
        ],
        (
            "maastricht-university-nl-high-"
            "potential-scholarship-2026.json"
        ): [
            "offers 21 full scholarships",
            "GPA of 7.5 out of 10 or higher",
            "1 February 2026",
        ],
    }

    root = Path("data/sources/held_out")

    for filename, phrases in required_phrases.items():
        snapshot = load_source_snapshot(
            root / filename
        )
        normalized = snapshot.text.casefold()

        for phrase in phrases:
            assert phrase.casefold() in normalized


def test_unmodelled_conditions_remain_manual() -> None:
    """Official conditions outside the schema must stay explicit."""
    records = _records()

    assert all(
        record.manual_review_requirements
        for record in records.values()
    )

    manitoba = records[MANITOBA_ID]

    assert manitoba.minimum_gpa is None
    assert manitoba.gpa_scale is None
    assert any(
        "does not state the GPA scale"
        in requirement
        for requirement
        in manitoba.manual_review_requirements
    )


def test_structured_funding_and_deadlines_are_conservative() -> None:
    """Only directly supported universal deadlines are structured."""
    records = _records()

    assert records[BRISTOL_ID].deadline == date(
        2026,
        4,
        10,
    )
    assert records[GLASGOW_ID].deadline == date(
        2026,
        7,
        17,
    )
    assert records[MAASTRICHT_ID].deadline == date(
        2026,
        2,
        1,
    )

    assert records[AALTO_ID].deadline is None
    assert records[WAIKATO_ID].deadline is None
    assert records[MANITOBA_ID].deadline is None

    assert (
        records[MAASTRICHT_ID].funding_type.value
        == "fully_funded"
    )
    assert (
        records[MANITOBA_ID].funding_type.value
        == "partially_funded"
    )

    for identifier in {
        BRISTOL_ID,
        GLASGOW_ID,
        AALTO_ID,
        WAIKATO_ID,
    }:
        assert (
            records[identifier].funding_type.value
            == "tuition_only"
        )


def test_maastricht_gpa_supports_three_statuses() -> None:
    """The explicit 7.5/10 rule should support three outcomes."""
    record = _records()[MAASTRICHT_ID]
    evaluation_date = date(2026, 1, 15)

    high = assess_eligibility(
        _profile(
            degree="master",
            gpa=8.0,
            gpa_scale=10.0,
        ),
        record,
        as_of=evaluation_date,
    )
    missing = assess_eligibility(
        _profile(
            degree="master",
            gpa=None,
            gpa_scale=None,
        ),
        record,
        as_of=evaluation_date,
    )
    low = assess_eligibility(
        _profile(
            degree="master",
            gpa=7.0,
            gpa_scale=10.0,
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


@pytest.mark.parametrize(
    (
        "scholarship_id",
        "matching_degree",
        "wrong_degree",
    ),
    [
        (
            BRISTOL_ID,
            "master",
            "phd",
        ),
        (
            GLASGOW_ID,
            "master",
            "bachelor",
        ),
        (
            AALTO_ID,
            "bachelor",
            "phd",
        ),
        (
            WAIKATO_ID,
            "master",
            "phd",
        ),
        (
            MANITOBA_ID,
            "phd",
            "bachelor",
        ),
    ],
)
def test_degree_rules_are_conservative(
    scholarship_id: str,
    matching_degree: str,
    wrong_degree: str,
) -> None:
    """Matching profiles remain manual and wrong degrees fail."""
    record = _records()[scholarship_id]
    evaluation_date = date(2026, 1, 15)

    matching = assess_eligibility(
        _profile(
            degree=matching_degree,
        ),
        record,
        as_of=evaluation_date,
    )
    wrong = assess_eligibility(
        _profile(
            degree=wrong_degree,
        ),
        record,
        as_of=evaluation_date,
    )

    assert (
        matching.status
        is EligibilityStatus.POTENTIALLY_ELIGIBLE
    )
    assert matching.manual_review_items

    assert (
        wrong.status
        is EligibilityStatus.NOT_ELIGIBLE
    )
    assert wrong.hard_failures


def test_explicit_deadlines_support_expired_outcomes() -> None:
    """Recorded 2026 deadlines should cause deterministic rejection."""
    records = _records()
    profile = _profile(
        degree="master",
    )

    bristol = assess_eligibility(
        profile,
        records[BRISTOL_ID],
        as_of=date(2026, 4, 11),
    )
    glasgow = assess_eligibility(
        profile,
        records[GLASGOW_ID],
        as_of=date(2026, 7, 18),
    )
    maastricht = assess_eligibility(
        _profile(
            degree="master",
            gpa=8.0,
            gpa_scale=10.0,
        ),
        records[MAASTRICHT_ID],
        as_of=date(2026, 2, 2),
    )

    assert (
        bristol.status
        is EligibilityStatus.NOT_ELIGIBLE
    )
    assert (
        glasgow.status
        is EligibilityStatus.NOT_ELIGIBLE
    )
    assert (
        maastricht.status
        is EligibilityStatus.NOT_ELIGIBLE
    )


def test_frozen_settings_remain_pre_evaluation() -> None:
    """Source construction must not alter calibrated parameters."""
    settings = json.loads(
        SETTINGS.read_text(encoding="utf-8")
    )

    assert settings["status"] == (
        "frozen_after_calibration"
    )
    assert settings["dense_threshold"] == 0.60
    assert settings["top_k"] == 3
    assert settings["candidate_k"] == 9
    assert settings["rrf_constant"] == 60
    assert settings["held_out_test_used"] is False
