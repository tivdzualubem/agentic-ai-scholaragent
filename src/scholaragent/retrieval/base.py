"""Shared structural interfaces for scholarship retrievers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from scholaragent.schemas import ScholarshipRecord


class ScholarshipSearchResult(Protocol):
    """Minimum result structure required by the screening pipeline."""

    scholarship: ScholarshipRecord
    score: float
    rank: int


class ScholarshipSearchIndex(Protocol):
    """Retriever interface accepted by ScholarAgent components."""

    def search(
        self,
        query: str,
        *,
        k: int = 5,
    ) -> Sequence[ScholarshipSearchResult]:
        """Return ranked scholarship records for a query."""
