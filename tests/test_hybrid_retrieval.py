"""Tests for hybrid scholarship retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from scholaragent.retrieval import (
    HybridScholarshipIndex,
    load_scholarships,
)

DATASET = Path(
    "data/demo/synthetic_scholarships.json"
)


class KeywordEmbeddingProvider:
    """Deterministic semantic vectors for unit tests."""

    GROUPS = (
        (
            "artificial intelligence",
            "machine learning",
            "data science",
            "computer science",
        ),
        (
            "civil engineering",
            "structural engineering",
            "construction",
        ),
        (
            "public health",
            "global health",
            "healthcare",
        ),
        (
            "renewable energy",
            "sustainable energy",
            "clean energy",
        ),
        (
            "finland",
            "nordic",
        ),
        (
            "germany",
            "european",
        ),
        (
            "netherlands",
            "dutch",
        ),
        (
            "sweden",
            "scandinavian",
        ),
    )

    def embed(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        vectors: list[list[float]] = []

        for text in texts:
            lowered = text.casefold()

            vector = [
                float(
                    sum(
                        lowered.count(keyword)
                        for keyword in group
                    )
                )
                for group in self.GROUPS
            ]

            vector.append(1.0)
            vectors.append(vector)

        return vectors


def build_index() -> HybridScholarshipIndex:
    """Create the deterministic development hybrid index."""
    return HybridScholarshipIndex(
        load_scholarships(DATASET),
        KeywordEmbeddingProvider(),
    )


def test_hybrid_index_returns_expected_ai_result() -> None:
    """Lexical and semantic evidence should agree on the AI result."""
    index = build_index()

    results = index.search(
        "machine learning and data science "
        "master's funding in Finland",
        k=3,
    )

    assert len(results) == 3
    assert (
        results[0].scholarship.scholarship_id
        == "nordic-ai-masters-2027"
    )

    assert results[0].bm25_rank is not None
    assert results[0].dense_rank is not None
    assert results[0].score > 0


def test_rrf_rewards_results_found_by_both_retrievers() -> None:
    """A record present in both rankings gets both RRF contributions."""
    index = build_index()

    result = index.search(
        "civil and structural engineering "
        "scholarship in Germany",
        k=1,
    )[0]

    assert (
        result.scholarship.scholarship_id
        == "european-civil-engineering-award-2027"
    )
    assert result.bm25_rank is not None
    assert result.dense_rank is not None

    expected_score = (
        1 / (index.rrf_constant + result.bm25_rank)
        + 1 / (index.rrf_constant + result.dense_rank)
    )

    assert result.score == pytest.approx(
        expected_score
    )


def test_hybrid_search_validates_arguments() -> None:
    """Hybrid search rejects invalid inputs."""
    index = build_index()

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        index.search(" ")

    with pytest.raises(
        ValueError,
        match="k must be at least 1",
    ):
        index.search("AI scholarship", k=0)

    with pytest.raises(
        ValueError,
        match="candidate_k",
    ):
        index.search(
            "AI scholarship",
            k=3,
            candidate_k=2,
        )


def test_hybrid_index_metadata() -> None:
    """The hybrid index exposes its corpus configuration."""
    index = build_index()

    assert index.size == 6
    assert index.dense_dimension == 9
    assert index.rrf_constant == 60
