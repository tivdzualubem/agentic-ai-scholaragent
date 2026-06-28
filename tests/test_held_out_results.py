"""Regression tests for final frozen held-out results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


RETRIEVAL_PATH = Path(
    "eval/results/held_out_retrieval_comparison.json"
)
ELIGIBILITY_PATH = Path(
    "eval/results/held_out_eligibility_evaluation.json"
)


def _load(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_held_out_results_preserve_frozen_provenance() -> None:
    """Final artifacts should reference unchanged frozen inputs."""
    for result_path in [
        RETRIEVAL_PATH,
        ELIGIBILITY_PATH,
    ]:
        payload = _load(result_path)

        assert payload["partition"] == "held_out_test"
        assert (
            payload["parameter_selection_after_test"]
            is False
        )
        assert (
            payload["held_out_tuning_performed"]
            is False
        )

        frozen = payload["frozen_settings"]

        assert frozen["dense_threshold"] == pytest.approx(
            0.60
        )
        assert frozen["top_k"] == 3
        assert frozen["candidate_k"] == 9
        assert frozen["rrf_constant"] == 60
        assert frozen["embedding_dimension"] == 768

        for key in [
            "benchmark",
            "corpus",
            "frozen_settings",
        ]:
            referenced = Path(
                payload[key]["path"]
            )

            assert referenced.is_file()
            assert payload[key]["sha256"] == _sha256(
                referenced
            )


def test_final_held_out_retrieval_metrics() -> None:
    """Frozen retrieval metrics should remain reproducible."""
    payload = _load(RETRIEVAL_PATH)
    results = payload["results"]

    assert results["k"] == 3
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
        1 / 3
    )
    assert bm25["recall_at_k"] == pytest.approx(1.0)
    assert bm25["mrr"] == pytest.approx(1.0)
    assert bm25["top1_hit_rate"] == pytest.approx(1.0)
    assert bm25["no_result_accuracy"] == pytest.approx(
        1.0
    )

    assert dense["precision_at_k"] == pytest.approx(
        0.30
    )
    assert dense["recall_at_k"] == pytest.approx(0.90)
    assert dense["mrr"] == pytest.approx(
        0.5916666666666667
    )
    assert dense["top1_hit_rate"] == pytest.approx(
        0.40
    )
    assert dense["no_result_accuracy"] == pytest.approx(
        1.0
    )

    assert hybrid["precision_at_k"] == pytest.approx(
        1 / 3
    )
    assert hybrid["recall_at_k"] == pytest.approx(1.0)
    assert hybrid["mrr"] == pytest.approx(0.875)
    assert hybrid["top1_hit_rate"] == pytest.approx(
        0.75
    )
    assert hybrid["no_result_accuracy"] == pytest.approx(
        1.0
    )

    assert bm25["mrr"] > hybrid["mrr"] > dense["mrr"]
    assert (
        bm25["top1_hit_rate"]
        > hybrid["top1_hit_rate"]
        > dense["top1_hit_rate"]
    )


def test_final_held_out_eligibility_metrics() -> None:
    """All balanced deterministic labels should be reproduced."""
    payload = _load(ELIGIBILITY_PATH)
    results = payload["results"]

    assert results["total_cases"] == 24
    assert results["positive_cases"] == 20
    assert results["no_result_cases"] == 4
    assert results["eligibility_evaluated_labels"] == 20

    assert results[
        "eligibility_status_accuracy"
    ] == pytest.approx(1.0)

    assert results[
        "eligibility_macro_precision"
    ] == pytest.approx(1.0)

    assert results[
        "eligibility_macro_recall"
    ] == pytest.approx(1.0)

    assert results[
        "eligibility_macro_f1"
    ] == pytest.approx(1.0)

    assert results[
        "eligibility_weighted_f1"
    ] == pytest.approx(1.0)

    support = {
        status: item["support"]
        for status, item
        in results["eligibility_per_status"].items()
    }

    assert support == {
        "eligible": 5,
        "potentially_eligible": 5,
        "not_eligible": 5,
        "insufficient_information": 5,
    }
