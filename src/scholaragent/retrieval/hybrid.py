"""Hybrid BM25 and dense retrieval using Reciprocal Rank Fusion."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from scholaragent.retrieval.bm25 import BM25ScholarshipIndex
from scholaragent.retrieval.dense import DenseScholarshipIndex
from scholaragent.retrieval.embeddings import EmbeddingProvider
from scholaragent.schemas import ScholarshipRecord


class HybridSearchResult(BaseModel):
    """One result produced by hybrid rank fusion."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    scholarship: ScholarshipRecord
    score: float = Field(gt=0)
    rank: int = Field(ge=1)

    bm25_rank: int | None = Field(default=None, ge=1)
    dense_rank: int | None = Field(default=None, ge=1)

    bm25_score: float | None = None
    dense_score: float | None = None


class HybridScholarshipIndex:
    """Combine lexical and semantic rankings through RRF."""

    def __init__(
        self,
        scholarships: Sequence[ScholarshipRecord],
        embedder: EmbeddingProvider,
        *,
        rrf_constant: int = 60,
        minimum_dense_score: float | None = None,
    ) -> None:
        records = list(scholarships)

        if not records:
            raise ValueError(
                "Hybrid index requires at least one scholarship."
            )

        if rrf_constant < 1:
            raise ValueError(
                "rrf_constant must be at least 1."
            )

        if (
            minimum_dense_score is not None
            and not -1.0 <= minimum_dense_score <= 1.0
        ):
            raise ValueError(
                "minimum_dense_score must be between -1 and 1."
            )

        self._records = records
        self._rrf_constant = rrf_constant
        self._minimum_dense_score = minimum_dense_score
        self._bm25 = BM25ScholarshipIndex(records)
        self._dense = DenseScholarshipIndex(
            records,
            embedder,
        )

    @property
    def size(self) -> int:
        """Return the number of indexed scholarships."""
        return len(self._records)

    @property
    def dense_dimension(self) -> int:
        """Return the dense embedding dimension."""
        return self._dense.dimension

    @property
    def rrf_constant(self) -> int:
        """Return the RRF smoothing constant."""
        return self._rrf_constant

    @property
    def minimum_dense_score(self) -> float | None:
        """Return the configured semantic abstention threshold."""
        return self._minimum_dense_score

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        candidate_k: int | None = None,
        minimum_dense_score: float | None = None,
    ) -> list[HybridSearchResult]:
        """Retrieve and fuse BM25 and dense candidates."""
        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("Search query must not be empty.")

        if k < 1:
            raise ValueError("k must be at least 1.")

        if candidate_k is None:
            candidate_k = max(k * 3, k)

        if candidate_k < k:
            raise ValueError(
                "candidate_k must be greater than or equal to k."
            )

        bm25_results = self._bm25.search(
            normalized_query,
            k=candidate_k,
        )

        effective_minimum_dense_score = (
            self._minimum_dense_score
            if minimum_dense_score is None
            else minimum_dense_score
        )

        if (
            effective_minimum_dense_score is not None
            and not -1.0
            <= effective_minimum_dense_score
            <= 1.0
        ):
            raise ValueError(
                "minimum_dense_score must be between -1 and 1."
            )

        dense_results = self._dense.search(
            normalized_query,
            k=candidate_k,
            minimum_score=effective_minimum_dense_score,
        )

        fused: dict[str, dict[str, object]] = {}

        for result in bm25_results:
            identifier = result.scholarship.scholarship_id

            fused[identifier] = {
                "scholarship": result.scholarship,
                "score": (
                    1.0
                    / (
                        self._rrf_constant
                        + result.rank
                    )
                ),
                "bm25_rank": result.rank,
                "dense_rank": None,
                "bm25_score": result.score,
                "dense_score": None,
            }

        for result in dense_results:
            identifier = result.scholarship.scholarship_id
            contribution = (
                1.0
                / (
                    self._rrf_constant
                    + result.rank
                )
            )

            if identifier not in fused:
                fused[identifier] = {
                    "scholarship": result.scholarship,
                    "score": contribution,
                    "bm25_rank": None,
                    "dense_rank": result.rank,
                    "bm25_score": None,
                    "dense_score": result.score,
                }
            else:
                fused[identifier]["score"] = (
                    float(fused[identifier]["score"])
                    + contribution
                )
                fused[identifier]["dense_rank"] = (
                    result.rank
                )
                fused[identifier]["dense_score"] = (
                    result.score
                )

        ranked = sorted(
            fused.values(),
            key=lambda item: (
                -float(item["score"]),
                str(
                    item["scholarship"].scholarship_id
                ),
            ),
        )

        return [
            HybridSearchResult(
                scholarship=item["scholarship"],
                score=float(item["score"]),
                rank=rank,
                bm25_rank=item["bm25_rank"],
                dense_rank=item["dense_rank"],
                bm25_score=item["bm25_score"],
                dense_score=item["dense_score"],
            )
            for rank, item in enumerate(
                ranked[:k],
                start=1,
            )
        ]
