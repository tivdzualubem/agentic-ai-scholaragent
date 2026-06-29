"""Tests for dense scholarship retrieval."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from scholaragent.retrieval import (
    DenseScholarshipIndex,
    cosine_similarity,
    load_scholarships,
    scholarship_embedding_text,
)

DATASET = Path(
    "data/demo/synthetic_scholarships.json"
)


class KeywordEmbeddingProvider:
    """Small deterministic embedding provider for unit tests."""

    KEYWORD_GROUPS = (
        (
            "artificial intelligence",
            "data science",
            "computer science",
        ),
        (
            "civil engineering",
            "structural engineering",
        ),
        (
            "public health",
            "global health",
        ),
        (
            "renewable energy",
            "sustainable energy",
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
                for group in self.KEYWORD_GROUPS
            ]

            vector.append(1.0)
            vectors.append(vector)

        return vectors


def test_cosine_similarity_known_values() -> None:
    """Cosine similarity returns known geometric values."""
    assert cosine_similarity(
        [1.0, 0.0],
        [1.0, 0.0],
    ) == pytest.approx(1.0)

    assert cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    ) == pytest.approx(0.0)


def test_scholarship_embedding_text_contains_identity() -> None:
    """Embedding text preserves important record information."""
    record = load_scholarships(DATASET)[0]
    text = scholarship_embedding_text(record)

    assert record.scholarship_id in text
    assert record.title in text


def test_dense_index_returns_semantic_match() -> None:
    """A relevant AI scholarship ranks first for an AI query."""
    records = load_scholarships(DATASET)

    index = DenseScholarshipIndex(
        records,
        KeywordEmbeddingProvider(),
    )

    results = index.search(
        "artificial intelligence and data science "
        "master's scholarship in Finland",
        k=3,
    )

    assert index.size == len(records)
    assert index.dimension == 8
    assert len(results) == 3

    assert (
        results[0].scholarship.scholarship_id
        == "nordic-ai-masters-2027"
    )

    assert results[0].rank == 1
    assert results[0].score >= results[1].score


def test_dense_index_validates_search_arguments() -> None:
    """Dense search rejects empty queries and invalid k."""
    index = DenseScholarshipIndex(
        load_scholarships(DATASET),
        KeywordEmbeddingProvider(),
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        index.search("   ")

    with pytest.raises(
        ValueError,
        match="k must be at least 1",
    ):
        index.search("artificial intelligence", k=0)
