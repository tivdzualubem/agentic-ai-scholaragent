"""Regression tests for the supplemental Recall@5 artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


RESULT_PATH = Path(
    "eval/results/"
    "held_out_retrieval_recall5_supplement.json"
)


def _load(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_recall5_artifact_is_descriptive_only() -> None:
    """The supplement must not claim confirmatory status."""
    payload = _load(RESULT_PATH)

    assert payload["artifact_type"] == (
        "supplemental_held_out_retrieval_recall_at_5"
    )
    assert payload["partition"] == "held_out_test"
    assert payload["evaluation_status"] == (
        "post_hoc_descriptive_supplement"
    )
    assert payload["confirmatory_status"] == (
        "descriptive_only_not_independent_confirmatory_evidence"
    )

    assert payload["held_out_tuning_performed"] is False
    assert payload["parameter_selection_after_test"] is False
    assert payload["used_for_model_selection"] is False
    assert payload["replaces_primary_recall_at_3"] is False


def test_recall5_inputs_preserve_provenance() -> None:
    """All referenced inputs must retain their recorded hashes."""
    payload = _load(RESULT_PATH)

    for key in [
        "benchmark",
        "corpus",
        "primary_frozen_result",
    ]:
        entry = payload[key]
        path = Path(entry["path"])

        assert path.is_file()
        assert entry["sha256"] == _sha256(path)

    frozen = payload["frozen_settings"]
    frozen_path = Path(frozen["path"])

    assert frozen_path.is_file()
    assert frozen["sha256"] == _sha256(frozen_path)

    assert frozen["embedding_model"] == (
        "nomic-embed-text:latest"
    )
    assert frozen["dense_threshold"] == pytest.approx(
        0.60
    )
    assert frozen["primary_top_k"] == 3
    assert frozen["supplemental_top_k"] == 5
    assert frozen["candidate_k"] == 9
    assert frozen["rrf_constant"] == 60


def test_supplemental_recall5_metrics() -> None:
    """Recall@5 metrics should remain reproducible."""
    payload = _load(RESULT_PATH)
    results = payload["results"]

    assert results["k"] == 5
    assert results["dense_threshold"] == pytest.approx(
        0.60
    )

    metrics = {
        item["retriever_name"]: item
        for item in results["retrievers"]
    }

    assert set(metrics) == {
        "bm25",
        "dense",
        "hybrid_rrf",
    }

    bm25 = metrics["bm25"]
    dense = metrics["dense"]
    hybrid = metrics["hybrid_rrf"]

    assert bm25["precision_at_k"] == pytest.approx(
        0.20
    )
    assert bm25["recall_at_k"] == pytest.approx(1.0)
    assert bm25["mrr"] == pytest.approx(1.0)
    assert bm25["top1_hit_rate"] == pytest.approx(1.0)
    assert bm25["no_result_accuracy"] == pytest.approx(
        1.0
    )

    assert dense["precision_at_k"] == pytest.approx(
        0.19
    )
    assert dense["recall_at_k"] == pytest.approx(
        0.95
    )
    assert dense["mrr"] == pytest.approx(
        0.6041666666666666
    )
    assert dense["top1_hit_rate"] == pytest.approx(
        0.40
    )
    assert dense["no_result_accuracy"] == pytest.approx(
        1.0
    )

    assert hybrid["precision_at_k"] == pytest.approx(
        0.20
    )
    assert hybrid["recall_at_k"] == pytest.approx(
        1.0
    )
    assert hybrid["mrr"] == pytest.approx(0.875)
    assert hybrid["top1_hit_rate"] == pytest.approx(
        0.75
    )
    assert hybrid["no_result_accuracy"] == pytest.approx(
        1.0
    )


def test_recall5_does_not_replace_primary_recall3() -> None:
    """The primary Recall@3 artifact must remain unchanged."""
    payload = _load(RESULT_PATH)

    primary_path = Path(
        payload["primary_frozen_result"]["path"]
    )
    primary = _load(primary_path)

    assert primary["results"]["k"] == 3
    assert payload["results"]["k"] == 5
    assert payload["replaces_primary_recall_at_3"] is False
