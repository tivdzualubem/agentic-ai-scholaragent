"""Regression tests for held-out runtime and cost reporting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


RESULT_PATH = Path(
    "eval/results/held_out_runtime_cost_summary.json"
)


def load_result() -> dict:
    """Load the tracked runtime-cost artifact."""
    return json.loads(
        RESULT_PATH.read_text(encoding="utf-8")
    )


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def systems_by_name() -> dict[str, dict]:
    """Index system summaries by system name."""
    return {
        item["system_name"]: item
        for item in load_result()["systems"]
    }


def test_runtime_cost_artifact_metadata() -> None:
    payload = load_result()

    assert payload["artifact_type"] == (
        "held_out_runtime_and_cost_summary"
    )
    assert payload["partition"] == "held_out_test"

    constraints = payload["evaluation_constraints"]

    assert constraints[
        "held_out_tuning_performed"
    ] is False
    assert constraints[
        "parameter_selection_after_test"
    ] is False
    assert constraints[
        "new_model_calls_for_this_artifact"
    ] is False
    assert constraints[
        "independent_confirmatory_evidence"
    ] is False


def test_runtime_cost_source_provenance() -> None:
    source = load_result()["source"]
    path = Path(source["path"])

    assert path.is_file()
    assert source["sha256"] == file_sha256(path)


def test_single_pass_runtime_summary() -> None:
    single = systems_by_name()["single_pass_rag"]

    assert single["total_cases"] == 24
    assert single["positive_cases"] == 20
    assert single["no_result_cases"] == 4
    assert single["total_retrieval_calls"] == 24
    assert single["total_generation_calls"] == 20

    assert single[
        "total_latency_minutes"
    ] == pytest.approx(56.100574)

    assert single[
        "mean_latency_seconds"
    ] == pytest.approx(140.251435)

    assert single[
        "direct_hosted_api_fee_usd"
    ] == 0.0


def test_agentic_runtime_summary() -> None:
    agentic = systems_by_name()["agentic_rag"]

    assert agentic["total_cases"] == 24
    assert agentic["positive_cases"] == 20
    assert agentic["no_result_cases"] == 4
    assert agentic["total_retrieval_calls"] == 24
    assert agentic["total_generation_calls"] == 40

    assert agentic[
        "total_latency_minutes"
    ] == pytest.approx(117.225518)

    assert agentic[
        "mean_latency_seconds"
    ] == pytest.approx(293.063795)

    assert agentic[
        "direct_hosted_api_fee_usd"
    ] == 0.0


def test_agentic_runtime_overhead() -> None:
    ratios = load_result()["comparison"][
        "agentic_to_single_pass_ratios"
    ]

    assert ratios["total_latency"] == pytest.approx(
        2.08956
    )
    assert ratios["generation_calls"] == 2.0


def test_cost_interpretation_is_bounded() -> None:
    accounting = load_result()["cost_accounting"]

    assert accounting[
        "direct_hosted_model_api_fee_usd"
    ] == 0.0

    assert "electricity consumption" in accounting[
        "not_monetized"
    ]
    assert "hardware purchase or depreciation" in (
        accounting["not_monetized"]
    )

    assert "direct hosted API fees only" in (
        accounting["interpretation"]
    )


def test_transport_timeout_is_recorded() -> None:
    failures = load_result()[
        "transport_failure_summary"
    ]

    assert failures["count"] == 1

    event = failures["events"][0]

    assert event["system_name"] == "agentic_rag"
    assert event["generation_call"] == 2
    assert event["timeout_seconds"] == 900.0
    assert event["failure_marker"] == (
        "GENERATION_FAILED_DUE_TO_OLLAMA_TRANSPORT_ERROR"
    )
