"""Validation tests for the frozen held-out benchmark."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scholaragent.eligibility import assess_eligibility
from scholaragent.evaluation import load_benchmark
from scholaragent.retrieval import load_scholarships


BENCHMARK_PATH = Path(
    "eval/datasets/held_out_benchmark.json"
)
HELD_OUT_CORPUS = Path(
    "data/held_out/held_out_scholarships.json"
)
CALIBRATION_CORPUS = Path(
    "data/calibration/calibration_scholarships.json"
)
DEVELOPMENT_CORPUS = Path(
    "data/official/official_scholarships.json"
)
SETTINGS_PATH = Path(
    "eval/config/frozen_retrieval_settings.json"
)


def test_held_out_benchmark_is_balanced() -> None:
    """The held-out partition should match its frozen design."""
    benchmark = load_benchmark(
        BENCHMARK_PATH
    )

    status_counts = Counter(
        status.value
        for case in benchmark.cases
        for status in case.expected_statuses.values()
    )

    assert benchmark.as_of.isoformat() == (
        "2026-01-15"
    )
    assert len(benchmark.cases) == 24

    assert sum(
        not case.expect_no_results
        for case in benchmark.cases
    ) == 20

    assert sum(
        case.expect_no_results
        for case in benchmark.cases
    ) == 4

    assert status_counts == {
        "eligible": 5,
        "potentially_eligible": 5,
        "not_eligible": 5,
        "insufficient_information": 5,
    }


def test_held_out_identities_do_not_leak() -> None:
    """Held-out identities must be disjoint from prior partitions."""
    held_out_ids = {
        record.scholarship_id
        for record in load_scholarships(
            HELD_OUT_CORPUS
        )
    }
    calibration_ids = {
        record.scholarship_id
        for record in load_scholarships(
            CALIBRATION_CORPUS
        )
    }
    development_ids = {
        record.scholarship_id
        for record in load_scholarships(
            DEVELOPMENT_CORPUS
        )
    }

    assert len(held_out_ids) == 6
    assert held_out_ids.isdisjoint(
        calibration_ids
    )
    assert held_out_ids.isdisjoint(
        development_ids
    )


def test_benchmark_references_only_held_out_records() -> None:
    """Every positive case should reference one held-out identity."""
    benchmark = load_benchmark(
        BENCHMARK_PATH
    )
    corpus_ids = {
        record.scholarship_id
        for record in load_scholarships(
            HELD_OUT_CORPUS
        )
    }

    referenced_ids = {
        identifier
        for case in benchmark.cases
        for identifier in (
            list(case.relevant_ids)
            + list(case.expected_statuses)
        )
    }

    assert referenced_ids == corpus_ids

    for case in benchmark.cases:
        if case.expect_no_results:
            assert case.relevant_ids == []
            assert case.expected_statuses == {}
        else:
            assert len(case.relevant_ids) == 1
            assert len(case.expected_statuses) == 1


def test_deterministic_labels_match_frozen_ground_truth() -> None:
    """Eligibility rules should reproduce all twenty labels."""
    benchmark = load_benchmark(
        BENCHMARK_PATH
    )
    records = {
        record.scholarship_id: record
        for record in load_scholarships(
            HELD_OUT_CORPUS
        )
    }

    evaluated = 0

    for case in benchmark.cases:
        for identifier, expected in (
            case.expected_statuses.items()
        ):
            assessment = assess_eligibility(
                case.profile,
                records[identifier],
                as_of=benchmark.as_of,
            )

            assert assessment.status is expected
            evaluated += 1

    assert evaluated == 20


def test_eligible_cases_use_scoped_verified_evidence() -> None:
    """Eligible labels must rely on exact scholarship-scoped evidence."""
    benchmark = load_benchmark(
        BENCHMARK_PATH
    )

    eligible_cases = [
        case
        for case in benchmark.cases
        if any(
            status.value == "eligible"
            for status
            in case.expected_statuses.values()
        )
    ]

    assert len(eligible_cases) == 5

    for case in eligible_cases:
        assert len(case.relevant_ids) == 1

        identifier = case.relevant_ids[0]
        verified = (
            case.profile.verified_manual_requirements
        )

        assert set(verified) == {
            identifier,
        }
        assert verified[identifier]


def test_frozen_settings_precede_held_out_evaluation() -> None:
    """Benchmark creation must not change calibrated settings."""
    settings = json.loads(
        SETTINGS_PATH.read_text(encoding="utf-8")
    )

    assert settings["status"] == (
        "frozen_after_calibration"
    )
    assert settings["dense_threshold"] == 0.60
    assert settings["top_k"] == 3
    assert settings["candidate_k"] == 9
    assert settings["rrf_constant"] == 60
    assert settings["held_out_test_used"] is False
