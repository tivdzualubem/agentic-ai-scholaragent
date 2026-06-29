"""Regression tests for final held-out RAG artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


COMPARISON_PATH = Path(
    "eval/results/held_out_rag_comparison.json"
)
ABLATION_PATH = Path(
    "eval/results/held_out_rag_ablation.json"
)
CHECKPOINT_PATH = Path(
    "eval/results/held_out_rag_checkpoint.json"
)
SETTINGS_PATH = Path(
    "eval/config/frozen_rag_settings.json"
)


def _load(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _systems() -> dict[str, dict]:
    payload = _load(COMPARISON_PATH)

    return {
        system["system_name"]: system
        for system
        in payload["comparison"]["systems"]
    }


def test_final_artifacts_are_held_out_results() -> None:
    """Artifacts must be marked as final held-out outputs."""
    comparison = _load(COMPARISON_PATH)
    ablation = _load(ABLATION_PATH)

    assert comparison["artifact_type"] == (
        "final_held_out_rag_comparison"
    )
    assert ablation["artifact_type"] == (
        "final_held_out_rag_ablation"
    )

    for payload in [comparison, ablation]:
        assert payload["partition"] == "held_out_test"
        assert payload["held_out_rag_test_used"] is True
        assert (
            payload["held_out_rag_test_completed"]
            is True
        )
        assert (
            payload["held_out_tuning_performed"]
            is False
        )
        assert (
            payload["parameter_selection_after_test"]
            is False
        )

        assert payload["configuration"]["sha256"] == (
            _sha256(SETTINGS_PATH)
        )


def test_checkpoint_contains_every_case() -> None:
    """The resumable execution must contain all cases."""
    checkpoint = _load(CHECKPOINT_PATH)

    assert len(checkpoint["baseline_cases"]) == 24
    assert len(checkpoint["agentic_cases"]) == 24
    assert len(
        checkpoint["agentic_raw_results"]
    ) == 24

    assert (
        checkpoint["primary_execution_completed"]
        is True
    )


def test_single_pass_rag_results() -> None:
    """Single-pass RAG should preserve observed metrics."""
    baseline = _systems()["single_pass_rag"]

    assert baseline["total_cases"] == 24
    assert baseline["positive_cases"] == 20
    assert baseline["no_result_cases"] == 4

    assert baseline[
        "positive_completion_rate"
    ] == pytest.approx(0.0)

    assert baseline[
        "positive_citation_pass_rate"
    ] == pytest.approx(0.0)

    assert baseline[
        "positive_relevant_grounding_rate"
    ] == pytest.approx(0.95)

    assert baseline[
        "positive_relevant_citation_rate"
    ] == pytest.approx(0.05)

    assert baseline[
        "no_result_accuracy"
    ] == pytest.approx(1.0)

    assert baseline[
        "mean_retrieval_calls"
    ] == pytest.approx(1.0)

    assert baseline[
        "mean_generation_calls"
    ] == pytest.approx(20 / 24)


def test_agentic_rag_results() -> None:
    """Agentic recovery metrics should remain explicit."""
    agentic = _systems()["agentic_rag"]

    assert agentic["total_cases"] == 24
    assert agentic["positive_cases"] == 20
    assert agentic["no_result_cases"] == 4

    assert agentic[
        "positive_completion_rate"
    ] == pytest.approx(1.0)

    assert agentic[
        "positive_citation_pass_rate"
    ] == pytest.approx(1.0)

    assert agentic[
        "positive_relevant_grounding_rate"
    ] == pytest.approx(0.95)

    assert agentic[
        "positive_relevant_citation_rate"
    ] == pytest.approx(0.70)

    assert agentic[
        "no_result_accuracy"
    ] == pytest.approx(1.0)

    assert agentic[
        "positive_fallback_rate"
    ] == pytest.approx(1.0)

    assert agentic[
        "mean_retrieval_calls"
    ] == pytest.approx(1.0)

    assert agentic[
        "mean_generation_calls"
    ] == pytest.approx(40 / 24)

    assert agentic[
        "mean_query_rewrites"
    ] == pytest.approx(0.0)

    assert agentic[
        "mean_repair_attempts"
    ] == pytest.approx(20 / 24)


def test_citation_pass_and_relevance_are_distinct() -> None:
    """Citation validity must not be confused with relevance."""
    agentic = _systems()["agentic_rag"]

    assert agentic[
        "positive_citation_pass_rate"
    ] == pytest.approx(1.0)

    assert agentic[
        "positive_relevant_citation_rate"
    ] == pytest.approx(0.70)

    assert (
        agentic["positive_relevant_citation_rate"]
        < agentic["positive_citation_pass_rate"]
    )


def test_query_rewriting_was_not_invoked() -> None:
    """No held-out case should use query rewriting."""
    ablation = _load(ABLATION_PATH)
    rewrite = ablation[
        "query_rewriting_ablation"
    ]

    assert rewrite[
        "observed_rewrite_case_count"
    ] == 0

    assert rewrite[
        "observed_rewrite_cases"
    ] == []

    assert (
        rewrite["counterfactual_rerun_performed"]
        is False
    )


def test_citation_repair_added_no_successes() -> None:
    """TinyLlama repair attempts should remain unsuccessful."""
    ablation = _load(ABLATION_PATH)
    repair = ablation[
        "citation_repair_ablation"
    ]

    assert repair["initial_audit_failures"] == 20
    assert repair["successful_llm_repairs"] == 0

    assert (
        repair["fallback_completions_after_repairs"]
        == 20
    )

    full = repair["full_agentic_metrics"]
    no_repair = repair[
        "no_repair_counterfactual"
    ]

    assert no_repair[
        "positive_completion_rate"
    ] == pytest.approx(
        full["positive_completion_rate"]
    )

    assert no_repair[
        "positive_citation_pass_rate"
    ] == pytest.approx(
        full["positive_citation_pass_rate"]
    )

    assert no_repair[
        "mean_generation_calls"
    ] < full["mean_generation_calls"]


def test_fallback_was_required_for_completion() -> None:
    """Removing fallback should remove safe completion."""
    ablation = _load(ABLATION_PATH)
    fallback = ablation[
        "deterministic_fallback_ablation"
    ]

    without_fallback = fallback[
        "no_fallback_counterfactual"
    ]

    assert without_fallback[
        "positive_completion_rate"
    ] == pytest.approx(0.0)

    assert without_fallback[
        "positive_citation_pass_rate"
    ] == pytest.approx(0.0)


def test_transport_failure_is_disclosed() -> None:
    """The Manitoba timeout must remain visible."""
    comparison = _load(COMPARISON_PATH)
    summary = comparison[
        "transport_failure_summary"
    ]

    assert summary["count"] == 1
    assert len(summary["events"]) == 1

    event = summary["events"][0]

    assert event["case_id"] == (
        "eligible-manitoba-phd-verified"
    )
    assert event["system_name"] == "agentic_rag"
    assert event["generation_call"] == 2
    assert event["timeout_seconds"] == 900.0

    assert (
        event["failure_marker"]
        == (
            "GENERATION_FAILED_DUE_TO_"
            "OLLAMA_TRANSPORT_ERROR"
        )
    )
