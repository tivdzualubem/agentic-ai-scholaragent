"""Tests for calibration-selected retrieval defaults."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scholaragent.evaluation.comparison_cli import (
    build_parser as build_retrieval_parser,
)
from scholaragent.evaluation.defaults import (
    CALIBRATED_DENSE_THRESHOLD,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RETRIEVAL_TOP_K,
)
from scholaragent.evaluation.rag_comparison_cli import (
    build_parser as build_rag_parser,
)


SETTINGS_PATH = Path(
    "eval/config/frozen_retrieval_settings.json"
)
SWEEP_PATH = Path(
    "eval/results/"
    "calibration_retrieval_threshold_sweep.json"
)


def _load(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def test_frozen_settings_match_calibration_artifact() -> None:
    """Frozen settings should reproduce the selected sweep row."""
    settings = _load(SETTINGS_PATH)
    sweep = _load(SWEEP_PATH)

    assert settings["status"] == (
        "frozen_after_calibration"
    )
    assert settings["held_out_test_used"] is False
    assert sweep["held_out_test_used"] is False

    assert settings["dense_threshold"] == pytest.approx(
        CALIBRATED_DENSE_THRESHOLD
    )
    assert sweep["selected_threshold"] == pytest.approx(
        CALIBRATED_DENSE_THRESHOLD
    )

    assert settings["embedding_model"] == (
        DEFAULT_EMBEDDING_MODEL
    )
    assert settings["embedding_dimension"] == 768
    assert settings["top_k"] == DEFAULT_RETRIEVAL_TOP_K
    assert settings["candidate_k"] == 9
    assert settings["rrf_constant"] == 60

    assert len(sweep["threshold_grid"]) == 13
    assert len(sweep["sweep_results"]) == 13

    selected = sweep["selected_metrics"]

    assert selected["threshold"] == pytest.approx(0.60)
    assert selected["selection_score"] == pytest.approx(
        0.85
    )
    assert selected["dense"]["top1_hit_rate"] == (
        pytest.approx(0.50)
    )
    assert selected["dense"]["no_result_accuracy"] == (
        pytest.approx(1.0)
    )
    assert selected["hybrid_rrf"][
        "top1_hit_rate"
    ] == pytest.approx(0.90)
    assert selected["hybrid_rrf"][
        "no_result_accuracy"
    ] == pytest.approx(1.0)


def test_frozen_input_hashes_match_current_files() -> None:
    """Frozen configuration should reference unchanged inputs."""
    settings = _load(SETTINGS_PATH)

    benchmark = Path(
        settings["calibration_benchmark"]["path"]
    )
    corpus = Path(
        settings["calibration_corpus"]["path"]
    )

    assert benchmark.is_file()
    assert corpus.is_file()

    assert settings["calibration_benchmark"][
        "sha256"
    ] == _sha256(benchmark)

    assert settings["calibration_corpus"][
        "sha256"
    ] == _sha256(corpus)


def test_cli_defaults_use_frozen_calibration_settings() -> None:
    """Retrieval and RAG CLIs should share frozen defaults."""
    retrieval_args = (
        build_retrieval_parser().parse_args([])
    )
    rag_args = build_rag_parser().parse_args([])

    for args in [
        retrieval_args,
        rag_args,
    ]:
        assert args.dense_threshold == pytest.approx(
            CALIBRATED_DENSE_THRESHOLD
        )
        assert args.embedding_model == (
            DEFAULT_EMBEDDING_MODEL
        )
        assert args.top_k == DEFAULT_RETRIEVAL_TOP_K
