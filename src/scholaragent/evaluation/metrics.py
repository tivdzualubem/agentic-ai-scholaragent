"""Information-retrieval metrics used by ScholarAgent."""

from __future__ import annotations

from collections.abc import Sequence


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    *,
    k: int,
) -> float:
    """Calculate precision among the first k retrieved identifiers."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    top_k = list(retrieved_ids[:k])
    relevant = set(relevant_ids)

    if not top_k:
        return 0.0

    matches = sum(identifier in relevant for identifier in top_k)
    return matches / k


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
    *,
    k: int,
) -> float:
    """Calculate the proportion of relevant records found in the top k."""
    if k < 1:
        raise ValueError("k must be at least 1.")

    relevant = set(relevant_ids)

    if not relevant:
        raise ValueError(
            "recall_at_k requires at least one relevant identifier."
        )

    retrieved = set(retrieved_ids[:k])
    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ids: Sequence[str],
) -> float:
    """Return the reciprocal rank of the first relevant result."""
    relevant = set(relevant_ids)

    if not relevant:
        raise ValueError(
            "reciprocal_rank requires at least one relevant identifier."
        )

    for rank, identifier in enumerate(retrieved_ids, start=1):
        if identifier in relevant:
            return 1.0 / rank

    return 0.0


def mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean of a non-empty sequence."""
    if not values:
        raise ValueError("mean requires at least one value.")

    return sum(values) / len(values)
