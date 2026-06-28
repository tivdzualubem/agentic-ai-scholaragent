"""Tests for the preregistered held-out RAG configuration."""

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


def test_rag_configuration_is_frozen_before_test() -> None:
    """The held-out RAG test must remain unopened."""
    settings = _settings()

    assert settings["status"] == (
        "frozen_before_held_out_rag"
    )
    assert settings["partition"] == "held_out_test"
    assert settings["held_out_rag_test_used"] is False
    assert (
        settings["parameter_selection_after_test"]
        is False
    )
    assert settings["held_out_tuning_permitted"] is False


def test_rag_configuration_uses_frozen_parameters() -> None:
    """RAG retrieval and generation settings should be fixed."""
    settings = _settings()

    assert settings["generator"] == {
        "model": "tinyllama:latest",
        "temperature": 0.0,
        "timeout_seconds": 240.0,
    }

    assert settings["retrieval"][
        "embedding_model"
    ] == "nomic-embed-text:latest"

    assert settings["retrieval"][
        "embedding_dimension"
    ] == 768

    assert settings["retrieval"][
        "dense_threshold"
    ] == 0.60

    assert settings["retrieval"]["top_k"] == 3
    assert settings["retrieval"]["candidate_k"] == 9
    assert settings["retrieval"]["rrf_constant"] == 60

    assert settings["agent_budgets"] == {
        "max_retrieval_attempts": 2,
        "max_generation_attempts": 2,
    }


def test_hybrid_defaults_resolve_to_frozen_values() -> None:
    """Implicit hybrid defaults must equal the recorded settings."""
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
    """All preregistered inputs should remain immutable."""
    settings = _settings()

    for entry in settings["inputs"].values():
        path = Path(entry["path"])

        assert path.is_file()
        assert entry["sha256"] == _sha256(path)


def test_ablation_rules_are_preregistered() -> None:
    """Every promised component should have a fixed analysis rule."""
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
