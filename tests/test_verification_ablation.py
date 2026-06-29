"""Regression tests for the verification-stage ablation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


RESULT_PATH = Path(
    "eval/results/held_out_verification_ablation.json"
)


def _load(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_verification_ablation_is_trace_derived() -> None:
    """The artifact must not claim a new experiment."""
    payload = _load(RESULT_PATH)

    assert payload["artifact_type"] == (
        "trace_derived_held_out_verification_ablation"
    )
    assert payload["partition"] == "held_out_test"
    assert payload["evaluation_status"] == (
        "post_hoc_trace_derived_counterfactual"
    )

    assert payload["new_llm_calls_performed"] is False
    assert payload["held_out_tuning_performed"] is False
    assert (
        payload["parameter_selection_after_test"]
        is False
    )
    assert (
        payload["independent_confirmatory_evidence"]
        is False
    )


def test_verification_ablation_provenance() -> None:
    """Every source artifact must retain its recorded hash."""
    payload = _load(RESULT_PATH)

    for entry in payload["provenance"].values():
        path = Path(entry["path"])

        assert path.is_file()
        assert entry["sha256"] == _sha256(path)


def test_all_first_pass_audits_were_reproduced() -> None:
    """Stored first-pass outputs must match agentic audits."""
    validation = _load(RESULT_PATH)[
        "trace_validation"
    ]

    assert validation["positive_cases_checked"] == 20
    assert validation[
        "matching_first_pass_audits"
    ] == 20
    assert validation[
        "all_first_pass_audits_reproduced"
    ] is True


def test_without_verification_metrics() -> None:
    """Unverified generation should expose unsafe acceptance."""
    metrics = _load(RESULT_PATH)[
        "without_verification"
    ]

    assert metrics[
        "positive_raw_acceptance_rate"
    ] == pytest.approx(1.0)

    assert metrics[
        "positive_posthoc_citation_pass_rate"
    ] == pytest.approx(0.0)

    assert metrics[
        "positive_relevant_grounding_rate"
    ] == pytest.approx(0.95)

    assert metrics[
        "positive_relevant_citation_rate"
    ] == pytest.approx(0.05)

    assert metrics[
        "unsafe_acceptance_rate"
    ] == pytest.approx(1.0)

    assert metrics[
        "no_result_accuracy"
    ] == pytest.approx(1.0)

    assert metrics[
        "mean_generation_calls"
    ] == pytest.approx(20 / 24)

    assert metrics[
        "mean_repair_attempts"
    ] == pytest.approx(0.0)

    assert metrics[
        "positive_fallback_rate"
    ] == pytest.approx(0.0)


def test_with_verification_metrics() -> None:
    """Verification and fallback should recover safe outputs."""
    metrics = _load(RESULT_PATH)[
        "with_verification"
    ]

    assert metrics[
        "positive_verified_completion_rate"
    ] == pytest.approx(1.0)

    assert metrics[
        "positive_citation_pass_rate"
    ] == pytest.approx(1.0)

    assert metrics[
        "positive_relevant_grounding_rate"
    ] == pytest.approx(0.95)

    assert metrics[
        "positive_relevant_citation_rate"
    ] == pytest.approx(0.70)

    assert metrics[
        "no_result_accuracy"
    ] == pytest.approx(1.0)

    assert metrics[
        "mean_generation_calls"
    ] == pytest.approx(40 / 24)

    assert metrics[
        "mean_repair_attempts"
    ] == pytest.approx(20 / 24)

    assert metrics[
        "positive_fallback_rate"
    ] == pytest.approx(1.0)


def test_verification_intervention_summary() -> None:
    """Fallback, not LLM repair, supplied the recovery."""
    intervention = _load(RESULT_PATH)[
        "intervention_summary"
    ]

    assert intervention[
        "positive_first_pass_audit_failures"
    ] == 20

    assert intervention[
        "verification_intervention_rate"
    ] == pytest.approx(1.0)

    assert intervention[
        "verified_safe_recoveries"
    ] == 20

    assert intervention[
        "successful_llm_repairs"
    ] == 0

    assert intervention[
        "deterministic_fallback_recoveries"
    ] == 20


def test_verification_metric_deltas() -> None:
    """The artifact should retain measured intervention deltas."""
    deltas = _load(RESULT_PATH)[
        "metric_deltas_with_minus_without"
    ]

    assert deltas[
        "citation_pass_rate"
    ] == pytest.approx(1.0)

    assert deltas[
        "relevant_citation_rate"
    ] == pytest.approx(0.65)

    assert deltas[
        "mean_generation_calls"
    ] == pytest.approx(20 / 24)

    assert deltas[
        "positive_fallback_rate"
    ] == pytest.approx(1.0)


def test_verification_ablation_case_balance() -> None:
    """The artifact must include all benchmark cases."""
    cases = _load(RESULT_PATH)["cases"]

    assert len(cases) == 24

    positive = [
        case
        for case in cases
        if not case["expect_no_results"]
    ]

    unsupported = [
        case
        for case in cases
        if case["expect_no_results"]
    ]

    assert len(positive) == 20
    assert len(unsupported) == 4

    assert all(
        case["accepted_without_verification"]
        for case in positive
    )

    assert all(
        not case[
            "without_verification_posthoc_audit_passed"
        ]
        for case in positive
    )

    assert all(
        case["with_verification_citation_passed"]
        for case in positive
    )

    assert all(
        case["with_verification_fallback_used"]
        for case in positive
    )

    assert all(
        case["no_result_correct"] is True
        for case in unsupported
    )
