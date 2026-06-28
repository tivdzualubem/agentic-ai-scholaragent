"""Tests for the preregistered and amended RAG configuration."""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from scholaragent.retrieval import HybridScholarshipIndex


SETTINGS_PATH = Path(
    "eval/config/frozen_rag_settings.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def _settings() -> dict:
    return json.loads(
        SETTINGS_PATH.read_text(encoding="utf-8")
    )


def test_rag_configuration_records_partial_start() -> None:
    """The operational amendment must be disclosed."""
    settings = _settings()

    assert settings["status"] == (
        "operationally_amended_after_partial_start"
    )
    assert settings["partition"] == "held_out_test"
    assert settings["held_out_rag_test_used"] is True
    assert settings["held_out_rag_test_completed"] is False

    assert (
        settings[
            "operational_parameter_amendment_after_partial_start"
        ]
        is True
    )

    assert (
        settings["parameter_selection_after_test"]
        is False
    )
    assert settings["held_out_tuning_permitted"] is False


def test_only_transport_timeout_was_amended() -> None:
    """The amendment should not change model behavior."""
    settings = _settings()
    generator = settings["generator"]
    amendment = settings["operational_amendment"]

    assert generator["model"] == "tinyllama:latest"
    assert generator["temperature"] == 0.0
    assert generator["timeout_seconds"] == 900.0

    assert amendment["amendment_type"] == (
        "transport_timeout_only"
    )
    assert amendment["original_timeout_seconds"] == 240.0
    assert amendment["amended_timeout_seconds"] == 900.0

    state = amendment[
        "execution_state_before_amendment"
    ]

    assert state["completed_baseline_cases"] == 1
    assert state["completed_agentic_cases"] == 0
    assert state["final_comparison_generated"] is False
    assert state["final_ablation_generated"] is False
    assert state["aggregate_rag_metrics_observed"] is False
    assert state["partial_case_output_observed"] is True


def test_rag_configuration_preserves_frozen_parameters() -> None:
    """Retrieval and agent budgets must remain fixed."""
    settings = _settings()

    retrieval = settings["retrieval"]

    assert retrieval[
        "embedding_model"
    ] == "nomic-embed-text:latest"

    assert retrieval["embedding_dimension"] == 768
    assert retrieval["dense_threshold"] == 0.60
    assert retrieval["top_k"] == 3
    assert retrieval["candidate_k"] == 9
    assert retrieval["rrf_constant"] == 60

    assert settings["agent_budgets"] == {
        "max_retrieval_attempts": 2,
        "max_generation_attempts": 2,
    }


def test_hybrid_defaults_resolve_to_frozen_values() -> None:
    """Implicit hybrid defaults must remain unchanged."""
    settings = _settings()

    constructor = inspect.signature(
        HybridScholarshipIndex
    )
    search = inspect.signature(
        HybridScholarshipIndex.search
    )

    assert constructor.parameters[
        "rrf_constant"
    ].default == settings["retrieval"]["rrf_constant"]

    assert search.parameters[
        "candidate_k"
    ].default is None

    top_k = settings["retrieval"]["top_k"]

    assert max(top_k * 3, top_k) == (
        settings["retrieval"]["candidate_k"]
    )


def test_frozen_rag_input_hashes_match() -> None:
    """Preregistered data inputs should remain immutable."""
    settings = _settings()

    for entry in settings["inputs"].values():
        path = Path(entry["path"])

        assert path.is_file()
        assert entry["sha256"] == _sha256(path)


def test_ablation_rules_remain_preregistered() -> None:
    """The operational amendment must not alter ablations."""
    settings = _settings()
    plan = settings["ablation_plan"]

    assert set(plan) == {
        "retrieval",
        "query_rewriting",
        "citation_repair",
        "deterministic_fallback",
    }

    assert plan["retrieval"]["no_rerun"] is True

    assert (
        plan["query_rewriting"][
            "selection_rule_frozen_before_test"
        ]
        is True
    )

    assert (
        plan["citation_repair"]["trace_derived"]
        is True
    )

    assert (
        plan["deterministic_fallback"]["trace_derived"]
        is True
    )
