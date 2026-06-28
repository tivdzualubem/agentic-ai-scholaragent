"""Lexical BM25 scholarship-retrieval baseline."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from scholaragent.schemas import ScholarshipRecord

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Terms that carry little or no discriminative value in this domain.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "s",
        "that",
        "the",
        "to",
        "with",
        "award",
        "awards",
        "fellowship",
        "fellowships",
        "grant",
        "grants",
        "scholarship",
        "scholarships",
    }
)


def tokenize(text: str) -> list[str]:
    """Return informative lowercase alphanumeric BM25 tokens."""
    return [
        token
        for token in TOKEN_PATTERN.findall(text.casefold())
        if token not in STOPWORDS
    ]


def scholarship_to_text(record: ScholarshipRecord) -> str:
    """Convert searchable scholarship fields into one document."""
    degree_levels = " ".join(level.value for level in record.degree_levels)

    parts = [
        record.title,
        record.provider,
        " ".join(record.host_countries),
        degree_levels,
        " ".join(record.eligible_nationalities),
        " ".join(record.eligible_fields),
        " ".join(record.manual_review_requirements),
        record.funding_type.value,
        record.eligibility_text,
    ]

    return " ".join(part for part in parts if part)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One ranked scholarship-search result."""

    rank: int
    score: float
    scholarship: ScholarshipRecord


class BM25ScholarshipIndex:
    """A reproducible lexical-search baseline over scholarship records."""

    def __init__(self, scholarships: list[ScholarshipRecord]) -> None:
        if not scholarships:
            raise ValueError(
                "At least one scholarship is required to build the index."
            )

        self._scholarships = list(scholarships)
        self._tokenized_corpus = [
            tokenize(scholarship_to_text(record))
            for record in self._scholarships
        ]
        self._vocabulary = {
            token
            for document in self._tokenized_corpus
            for token in document
        }
        self._index = BM25Okapi(self._tokenized_corpus)

    def search(self, query: str, *, k: int = 5) -> list[SearchResult]:
        """Return the top-k lexical matches for a query."""
        if not query.strip():
            raise ValueError("Search query must not be empty.")

        if k < 1:
            raise ValueError("k must be at least 1.")

        query_tokens = tokenize(query)

        # A query containing only generic terms carries no searchable intent.
        if not query_tokens:
            return []

        if not self._vocabulary.intersection(query_tokens):
            return []

        scores = self._index.get_scores(query_tokens)

        ranked_positions = sorted(
            range(len(self._scholarships)),
            key=lambda position: float(scores[position]),
            reverse=True,
        )[: min(k, len(self._scholarships))]

        return [
            SearchResult(
                rank=rank,
                score=float(scores[position]),
                scholarship=self._scholarships[position],
            )
            for rank, position in enumerate(ranked_positions, start=1)
        ]
