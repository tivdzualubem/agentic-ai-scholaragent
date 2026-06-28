"""In-memory dense-vector retrieval for scholarship records."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from scholaragent.retrieval.embeddings import EmbeddingProvider
from scholaragent.schemas import ScholarshipRecord


class DenseSearchResult(BaseModel):
    """One ranked dense-retrieval result."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    scholarship: ScholarshipRecord
    score: float
    rank: int = Field(ge=1)


def _flatten_value(
    value: Any,
    *,
    prefix: str = "",
) -> list[str]:
    """Convert nested scholarship data into embedding text."""
    if value is None:
        return []

    if isinstance(value, Enum):
        value = value.value

    if isinstance(value, Mapping):
        parts: list[str] = []

        for key, nested_value in value.items():
            field_name = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )
            parts.extend(
                _flatten_value(
                    nested_value,
                    prefix=field_name,
                )
            )

        return parts

    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        parts: list[str] = []

        for item in value:
            parts.extend(
                _flatten_value(
                    item,
                    prefix=prefix,
                )
            )

        return parts

    text = str(value).strip()

    if not text:
        return []

    if prefix:
        return [f"{prefix}: {text}"]

    return [text]


def scholarship_embedding_text(
    scholarship: ScholarshipRecord,
) -> str:
    """Create stable semantic text from a scholarship record."""
    data = scholarship.model_dump(
        mode="json",
        exclude_none=True,
    )

    parts = _flatten_value(data)

    if not parts:
        raise ValueError(
            "Scholarship record produced no embedding text."
        )

    return "\n".join(parts)


def cosine_similarity(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    """Calculate cosine similarity between equal-size vectors."""
    if not left or not right:
        raise ValueError(
            "Cosine similarity requires non-empty vectors."
        )

    if len(left) != len(right):
        raise ValueError(
            "Cosine similarity requires equal dimensions."
        )

    dot_product = math.fsum(
        left_value * right_value
        for left_value, right_value in zip(
            left,
            right,
            strict=True,
        )
    )

    left_norm = math.sqrt(
        math.fsum(value * value for value in left)
    )
    right_norm = math.sqrt(
        math.fsum(value * value for value in right)
    )

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)


class DenseScholarshipIndex:
    """Precomputed in-memory dense index for scholarship records."""

    def __init__(
        self,
        scholarships: Sequence[ScholarshipRecord],
        embedder: EmbeddingProvider,
    ) -> None:
        self._scholarships = list(scholarships)
        self._embedder = embedder

        if not self._scholarships:
            raise ValueError(
                "Dense index requires at least one scholarship."
            )

        corpus_texts = [
            scholarship_embedding_text(record)
            for record in self._scholarships
        ]

        vectors = self._embedder.embed(corpus_texts)

        if len(vectors) != len(self._scholarships):
            raise ValueError(
                "Embedding provider returned an unexpected "
                "number of corpus vectors."
            )

        dimensions = {
            len(vector)
            for vector in vectors
        }

        if not dimensions or 0 in dimensions:
            raise ValueError(
                "Corpus embeddings must be non-empty."
            )

        if len(dimensions) != 1:
            raise ValueError(
                "Corpus embedding dimensions are inconsistent."
            )

        self._dimension = dimensions.pop()
        self._vectors = vectors

    @property
    def dimension(self) -> int:
        """Return the index embedding dimension."""
        return self._dimension

    @property
    def size(self) -> int:
        """Return the number of indexed scholarships."""
        return len(self._scholarships)

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        minimum_score: float | None = None,
    ) -> list[DenseSearchResult]:
        """Return the scholarships most similar to a query."""
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("Search query must not be empty.")

        if k < 1:
            raise ValueError("k must be at least 1.")

        query_vectors = self._embedder.embed(
            [normalized_query]
        )

        if len(query_vectors) != 1:
            raise ValueError(
                "Embedding provider must return one query vector."
            )

        query_vector = query_vectors[0]

        if len(query_vector) != self._dimension:
            raise ValueError(
                "Query and corpus embedding dimensions differ."
            )

        scored_records = [
            (
                cosine_similarity(
                    query_vector,
                    corpus_vector,
                ),
                record,
            )
            for record, corpus_vector in zip(
                self._scholarships,
                self._vectors,
                strict=True,
            )
        ]

        scored_records.sort(
            key=lambda item: (
                -item[0],
                item[1].scholarship_id,
            )
        )

        if minimum_score is not None:
            scored_records = [
                item
                for item in scored_records
                if item[0] >= minimum_score
            ]

        return [
            DenseSearchResult(
                scholarship=record,
                score=score,
                rank=rank,
            )
            for rank, (score, record) in enumerate(
                scored_records[:k],
                start=1,
            )
        ]
